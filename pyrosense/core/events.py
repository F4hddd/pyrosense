"""
Stage 4: persistence, and the event lifecycle.

A detector that alarms on a single frame is a detector nobody keeps switched on.
Every real installation needs the same three things, and they are all here:

  persistence  the signal must hold above threshold for N seconds before anything
               leaves the building. One bright frame is a glitch, not a fire.
  hysteresis   once alarming, we drop out at a *lower* threshold, so a flame that
               dips for half a second does not re-trigger the whole pipeline.
  cooldown     one fire produces one incident, not 400 pushes. Re-alerts only on
               escalation or after the cooldown window.

An Event carries its own evidence with it: the pre-roll clip (we keep a rolling
buffer, so the clip starts *before* ignition), a full-resolution still, the
feature vector, and the adjudication text. That bundle is what a fire officer or
an insurer will want afterwards, and it is much easier to capture now than to
reconstruct later.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np


class AlarmRule:
    """Sustained-threshold rule with hysteresis. Feed it (t, score) per frame."""

    def __init__(self, threshold: float = 0.72, sustain_s: float = 2.5,
                 release: float = 0.45, release_s: float = 6.0):
        self.threshold = threshold
        self.sustain_s = sustain_s
        self.release = release
        self.release_s = release_s
        self._above_since: Optional[float] = None
        self._below_since: Optional[float] = None
        self.active = False

    def update(self, t: float, score: float) -> bool:
        """Returns True on the transition into alarm (the rising edge only)."""
        rising = False
        if score >= self.threshold:
            self._below_since = None
            if self._above_since is None:
                self._above_since = t
            if not self.active and (t - self._above_since) >= self.sustain_s:
                self.active = True
                rising = True
        else:
            self._above_since = None
            if self.active:
                if score < self.release:
                    if self._below_since is None:
                        self._below_since = t
                    elif (t - self._below_since) >= self.release_s:
                        self.active = False
                        self._below_since = None
                else:
                    self._below_since = None
        return rising


@dataclass
class Event:
    id: str
    camera: str
    kind: str                  # flame | smoke
    score: float
    started: float
    box: tuple
    reason: str = ""
    features: dict = field(default_factory=dict)
    stage: str = "stage1"      # stage1 | neural | vlm
    verdict: str = "pending"   # pending | confirmed | dismissed
    adjudication: str = ""
    snapshot: str = ""
    clip: str = ""
    clip_seconds: float = 0.0
    zone: str = ""
    acknowledged: bool = False
    escalated: bool = False
    # False when stage 3 filtered it out. The event is still recorded - it is
    # the evidence for tuning a site - but it never alerted anyone.
    alerted: bool = True

    def to_json(self) -> dict:
        d = asdict(self)
        d["started_iso"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started))
        return d


class RollingClip:
    """Keeps the last N seconds of frames so an alert clip can begin *before*
    the moment of detection. Operators consistently ask 'what happened just
    before' - having the answer already in RAM costs a few MB per camera."""

    def __init__(self, seconds: float = 10.0, fps: int = 12):
        self.buf: deque = deque(maxlen=int(seconds * fps))
        self.fps = fps

    def push(self, frame: np.ndarray) -> None:
        self.buf.append(frame.copy())

    def snapshot_buffer(self) -> list:
        """Freeze the pre-roll at this instant.

        Taken the moment an event fires, before the follow-up thread goes off to
        do adjudication. Without this the ring buffer keeps rolling during those
        seconds and the 'before' footage quietly scrolls out of the clip - you end
        up with a recording that starts after the thing you wanted to see."""
        return list(self.buf)

    def write(self, path: str, tail: list[np.ndarray] | None = None) -> str:
        return self.write_frames(list(self.buf) + list(tail or []), path, self.fps)

    @staticmethod
    def write_frames(frames: list, path: str, fps: int = 12) -> str:
        if not frames:
            return ""
        h, w = frames[0].shape[:2]
        # mp4v first because every browser can play it inline; MJPG/avi is the
        # fallback when the OpenCV build has no mp4 encoder.
        for fourcc, ext in (("mp4v", ".mp4"), ("MJPG", ".avi")):
            p = path.rsplit(".", 1)[0] + ext
            vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if not vw.isOpened():
                continue
            for f in frames:
                vw.write(f)
            vw.release()
            return p
        return ""


class EventStore:
    """Flat-file event log. Deliberately not a database: a fire-safety appliance
    on a factory LAN should survive a power cut with no recovery procedure, and
    an operator should be able to hand a directory to an investigator."""

    def __init__(self, root: str = "data/events"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root, "media"), exist_ok=True)
        self.events: list[Event] = []
        self._load()

    def _path(self) -> str:
        return os.path.join(self.root, "events.jsonl")

    def _load(self) -> None:
        p = self._path()
        if not os.path.exists(p):
            return
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    d.pop("started_iso", None)
                    d["box"] = tuple(d.get("box") or ())
                    self.events.append(Event(**d))
                except Exception:
                    continue

    def add(self, ev: Event) -> Event:
        self.events.append(ev)
        with open(self._path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(ev.to_json()) + "\n")
        return ev

    def update(self, ev: Event) -> None:
        """Rewrite the log. Fine at event volumes (a busy site sees single digits
        per day); if that ever changes, this is the one place to swap in sqlite."""
        with open(self._path(), "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e.to_json()) + "\n")

    def recent(self, n: int = 50) -> list[dict]:
        return [e.to_json() for e in self.events[-n:]][::-1]

    def media_path(self, ev_id: str, suffix: str) -> str:
        return os.path.join(self.root, "media", f"{ev_id}{suffix}")


class Debouncer:
    """One incident per fire. Suppresses repeats inside the cooldown window
    unless the score climbs materially, which we treat as escalation."""

    def __init__(self, cooldown_s: float = 180.0, escalate_delta: float = 0.15):
        self.cooldown_s = cooldown_s
        self.escalate_delta = escalate_delta
        self._last: dict[str, tuple[float, float]] = {}   # key -> (t, score)

    def should_emit(self, key: str, t: float, score: float) -> tuple[bool, bool]:
        """Returns (emit, is_escalation)."""
        prev = self._last.get(key)
        if prev is None or (t - prev[0]) > self.cooldown_s:
            self._last[key] = (t, score)
            return True, False
        if score >= prev[1] + self.escalate_delta:
            self._last[key] = (t, score)
            return True, True
        return False, False


def new_event_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
