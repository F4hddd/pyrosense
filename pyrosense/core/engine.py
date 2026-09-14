"""
The runtime: one worker per camera, plus a supervisor.

Threads, not asyncio, and not one process per camera:

  The work is OpenCV and numpy, which release the GIL for essentially the whole
  of their runtime. Threads therefore get real parallelism here, while sharing
  the event store, the dispatcher and the adjudicator budget without any IPC.

  Decoding happens in ffmpeg subprocesses, so the heaviest work is already off
  in its own process and on its own core.

  A camera worker that dies must not take the others with it, so every worker
  loop is wrapped and restarted by the supervisor with backoff.

The one rule the whole file is built around: nothing slow ever runs on the
detection path. Adjudication (1-2s of network), clip encoding and alert delivery
all happen on separate threads. The camera loop keeps consuming frames while a
fire it already found is being verified and reported.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..cameras.source import BaseSource, make_source
from ..detect.cascade import Cascade, Assessment
from ..detect.vlm import Adjudicator
from ..detect.neural import NeuralVerifier
from .events import (AlarmRule, Debouncer, Event, EventStore, RollingClip,
                     new_event_id)
from .zones import ZoneSet


@dataclass
class CameraConfig:
    name: str
    source: dict
    zones: list = field(default_factory=list)
    flame_threshold: float = 0.72
    smoke_threshold: float = 0.70
    flame_sustain: float = 2.5
    smoke_sustain: float = 1.0
    detect_smoke: bool = True
    work_width: int = 480
    motion_threshold: float = 0.0008
    cooldown_s: float = 180.0
    enabled: bool = True
    location: str = ""
    postroll_s: float = 8.0


class CameraWorker(threading.Thread):
    def __init__(self, cfg: CameraConfig, engine: "Engine"):
        super().__init__(daemon=True, name=f"cam-{cfg.name}")
        self.cfg = cfg
        self.engine = engine
        self.postroll_s = float(getattr(cfg, "postroll_s", 8.0))
        self.source: BaseSource | None = None
        self.zones = ZoneSet.from_config(cfg.zones)
        self.cascade = Cascade(work_width=cfg.work_width,
                               detect_smoke=cfg.detect_smoke,
                               motion_threshold=cfg.motion_threshold,
                               neural=engine.neural if engine.neural.available else None)
        self.rules = {
            "flame": AlarmRule(cfg.flame_threshold, cfg.flame_sustain),
            "smoke": AlarmRule(cfg.smoke_threshold, cfg.smoke_sustain),
        }
        self.clip = RollingClip(seconds=10.0, fps=12)
        self.debounce = Debouncer(cooldown_s=cfg.cooldown_s)
        # Post-roll collectors. An event registers one and the capture loop feeds
        # it for a few seconds, so the saved clip shows what happened AFTER
        # detection as well as before. A pre-roll-only recording answers "what led
        # up to this" but not "did it spread", which is the question an operator
        # reviewing an alert actually has.
        self._collectors: list = []
        self._coll_lock = threading.Lock()
        # Regions the adjudicator confidently dismissed: (kind, box, score, until).
        self._dismissed: list = []
        # kind -> time stage 3 last filtered it. While set, the raw stage-1 alarm
        # for that kind is not reported as an alarm: the AI has already said what
        # it is, so the dashboard must not keep it lit as a fire.
        self._filtered_at: dict = {}
        self._stop = threading.Event()

        # shared, read by the web layer
        self.latest: np.ndarray | None = None
        self.latest_annotated: np.ndarray | None = None
        self.last_result = None
        self.score = {"flame": 0.0, "smoke": 0.0}
        self.history: list[tuple[float, float, float]] = []   # (t, flame, smoke)
        self.frames = 0
        self.gated = 0
        self.err = ""
        self.proc_ms = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self.source = make_source({**self.cfg.source, "name": self.cfg.name})
                self.source.start()
                self.err = ""
                backoff = 1.0
                self._loop()
            except Exception as e:
                self.err = f"{type(e).__name__}: {e}"[:200]
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                if self.source:
                    try:
                        self.source.stop()
                    except Exception:
                        pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            frame = self.source.read() if self.source else None
            if frame is None:
                time.sleep(0.01)
                continue
            self.frames += 1
            now = time.time()

            t0 = time.perf_counter()
            zm = None
            if self.zones.zones:
                h = int(frame.shape[0] * self.cfg.work_width / frame.shape[1])
                zm = self.zones.mask(self.cfg.work_width, h, now)
            self.cascade.zone_mask = zm

            res = self.cascade(frame, now=now)
            self.proc_ms = 0.85 * self.proc_ms + 0.15 * (time.perf_counter() - t0) * 1000

            if res.gated:
                self.gated += 1

            # Per-zone sensitivity is applied here, by shifting the score rather
            # than the threshold. Same effect on the alarm decision, but the raw
            # score stays intact in the event record - so an investigator can see
            # what the detector actually measured, not what a zone rule left of it.
            wh = int(frame.shape[0] * self.cfg.work_width / frame.shape[1])
            with self._lock:
                self.latest = frame
                self.last_result = res
                for kind in ("flame", "smoke"):
                    live = []
                    for a in res.assessments:
                        if a.kind != kind or a.vetoed:
                            continue
                        d = self.zones.threshold_delta(
                            a.box[0] + a.box[2] / 2, a.box[1] + a.box[3] / 2,
                            self.cfg.work_width, wh) if self.zones.zones else 0.0
                        live.append(max(0.0, a.score - d))
                    self.score[kind] = max(live) if live else 0.0
                self.history.append((now, self.score["flame"], self.score["smoke"]))
                if len(self.history) > 900:
                    del self.history[:-900]
                self.latest_annotated = self._annotate(res)

            self.clip.push(frame)
            if self._collectors:
                with self._coll_lock:
                    for c in list(self._collectors):
                        if now <= c["until"]:
                            c["frames"].append(frame.copy())
                        else:
                            c["done"] = True
                            self._collectors.remove(c)

            for kind, rule in self.rules.items():
                if rule.update(now, self.score[kind]):
                    best = max((a for a in res.assessments
                                if a.kind == kind and not a.vetoed),
                               key=lambda a: a.score, default=None)
                    if best is not None:
                        self._raise(best, frame, now)

    # ----------------------------------------------------------------- alarm
    def _raise(self, a: Assessment, frame: np.ndarray, now: float) -> None:
        key = f"{self.cfg.name}:{a.kind}"
        emit, escalation = self.debounce.should_emit(key, now, a.score)
        if not emit:
            return

        zone = ""
        if self.zones.zones:
            wh = int(frame.shape[0] * self.cfg.work_width / frame.shape[1])
            zone = self.zones.zone_at(a.box[0] + a.box[2] / 2,
                                      a.box[1] + a.box[3] / 2,
                                      self.cfg.work_width, wh)

        ev = Event(id=new_event_id(), camera=self.cfg.name, kind=a.kind,
                   score=round(a.score, 3), started=now, box=tuple(a.box),
                   reason=a.explain(), features={k: round(float(v), 4)
                                                 for k, v in a.features.items()},
                   zone=zone, escalated=escalation)

        # Snapshot at full source resolution: the responder wants detail, and the
        # box has to be scaled up from the detector's working resolution.
        snap = self._snapshot(frame, a)
        ev.snapshot = self.engine.store.media_path(ev.id, ".jpg")
        cv2.imwrite(ev.snapshot, snap)
        self.engine.store.add(ev)

        # Freeze the pre-roll NOW, and open a post-roll window, before the
        # follow-up thread goes off to spend a second or two on adjudication.
        pre = self.clip.snapshot_buffer()
        collector = {"frames": [], "until": now + self.postroll_s, "done": False}
        with self._coll_lock:
            self._collectors.append(collector)

        threading.Thread(target=self._followup, args=(ev, frame, a, pre, collector),
                         daemon=True, name=f"followup-{ev.id}").start()

    def _snapshot(self, frame: np.ndarray, a: Assessment) -> np.ndarray:
        img = frame.copy()
        s = frame.shape[1] / float(self.cfg.work_width)
        x, y, w, h = [int(v * s) for v in a.box]
        col = (0, 90, 255) if a.kind == "flame" else (190, 190, 190)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 3)
        label = f"{a.kind.upper()} {a.score:.0%}"
        cv2.rectangle(img, (x, max(0, y - 26)), (x + 9 * len(label), y), col, -1)
        cv2.putText(img, label, (x + 4, max(14, y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, f"{self.cfg.name}  {stamp}", (10, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return img

    DISMISS_MEMORY_S = 20 * 60
    DISMISS_IOU = 0.3
    DISMISS_MARGIN = 0.05     # a clearly stronger detection is judged afresh

    @staticmethod
    def _iou(b1, b2) -> float:
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2 = min(b1[0] + b1[2], b2[0] + b2[2])
        y2 = min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = b1[2] * b1[3] + b2[2] * b2[3] - inter
        return inter / union if union > 0 else 0.0

    def _filtered(self, kind: str) -> bool:
        t = self._filtered_at.get(kind)
        return t is not None and time.time() - t < self.DISMISS_MEMORY_S

    def _remember_dismissal(self, a: Assessment, score: float, what: str) -> None:
        now = time.time()
        self._dismissed = [d for d in self._dismissed if d[3] > now][-20:]
        self._dismissed.append((a.kind, tuple(a.box), score, now + self.DISMISS_MEMORY_S,
                                what or "nuisance"))

    def _recall_dismissal(self, a: Assessment) -> str | None:
        """Memory is bounded three ways so it cannot hide a real fire for long: it
        expires after 20 minutes, it only covers the same region, and anything
        scoring above the dismissed detection goes back to the adjudicator."""
        now = time.time()
        for kind, box, score, until, what in self._dismissed:
            if (until > now and kind == a.kind and self._iou(box, a.box) >= self.DISMISS_IOU
                    and a.score <= score + self.DISMISS_MARGIN):
                return what
        return None

    def _followup(self, ev: Event, frame: np.ndarray, a: Assessment,
                  pre: list | None = None, collector: dict | None = None) -> None:
        """Clip, adjudication, urgent escalation, delivery and cloud uplink.

        All of it runs off the detection path. The camera loop is already back to
        consuming frames while this thread spends a second or two talking to
        Claude and then to Pushover."""
        send, note = True, ev.reason
        adj = None
        remembered = self._recall_dismissal(a)
        if remembered is not None:
            # Same place, same kind, no stronger than what was already judged a
            # nuisance: reuse that verdict instead of paying for the same answer.
            send = False
            note = f"suppressed: same region dismissed earlier as {remembered}"
            ev.adjudication, ev.verdict = note, "dismissed"
        elif self.engine.adjudicator is not None:
            s = frame.shape[1] / float(self.cfg.work_width)
            x, y, w, h = [int(v * s) for v in a.box]
            pad = int(0.4 * max(w, h))
            H, W = frame.shape[:2]
            crop = frame[max(0, y - pad):min(H, y + h + pad),
                         max(0, x - pad):min(W, x + w + pad)]
            if crop.size == 0:
                crop = frame
            adj = self.engine.adjudicator.adjudicate(
                frame, crop, a.kind, ev.features, camera=self.cfg.name, zone=ev.zone)
            send, note = self.engine.adjudicator.apply(ev.score, adj)
            if not send and adj.ran:
                self._remember_dismissal(a, ev.score, adj.nuisance_type)
            ev.stage = "vlm" if adj.ran else ev.stage
            ev.adjudication = note
            ev.verdict = ("confirmed" if adj.confirms else
                          "dismissed" if (adj.dismisses and not send) else "pending")

        # Wait for the post-roll to fill, then write pre + post as one clip.
        # Adjudication has usually consumed part of this window already, so the
        # wait costs little wall time.
        if collector is not None:
            deadline = collector["until"] + 1.0
            while not collector.get("done") and time.time() < deadline:
                time.sleep(0.25)
            with self._coll_lock:
                if collector in self._collectors:
                    self._collectors.remove(collector)
        try:
            frames = list(pre or []) + list((collector or {}).get("frames") or [])
            if frames:
                ev.clip = RollingClip.write_frames(
                    frames, self.engine.store.media_path(ev.id, ".mp4"), fps=12)
                ev.clip_seconds = round(len(frames) / 12.0, 1)
        except Exception:
            pass

        ev.alerted = send
        if send:
            self._filtered_at.pop(ev.kind, None)
        else:
            self._filtered_at[ev.kind] = time.time()
        self.engine.store.update(ev)
        self.engine.notify(ev, sent=send, adjudication=adj)

        snap_bytes = b""
        try:
            if ev.snapshot and os.path.exists(ev.snapshot):
                with open(ev.snapshot, "rb") as f:
                    snap_bytes = f.read()
        except Exception:
            pass

        if send:
            # Ordinary channels (webhook / email / MQTT / log) fan out first...
            self.engine.dispatcher.dispatch(ev.to_json(), ev.snapshot)
            # ...then the ladder that is actually meant to wake somebody up.
            if self.engine.escalation is not None:
                self.engine.escalation.raise_incident(
                    event_id=ev.id, camera=self.cfg.name, kind=ev.kind,
                    score=ev.score, summary=(ev.adjudication or ev.reason)[:400],
                    url=self.engine.dashboard_url, image=snap_bytes or None)

        # The cloud mirror is told about suppressed events too - a dismissal is
        # exactly the evidence you want when tuning a site, and hiding it would
        # make the public log look better than reality.
        if self.engine.uplink is not None:
            import base64
            payload = ev.to_json()
            payload["sent"] = send
            if snap_bytes:
                payload["snapshot_b64"] = base64.b64encode(snap_bytes).decode()
            self.engine.uplink.send_event(payload)
            # The clip goes up separately: it is far larger than the event JSON,
            # and an event must reach the mirror even if the video never does.
            if ev.clip and os.path.exists(ev.clip):
                self.engine.uplink.send_clip(ev.id, ev.clip)

    # ------------------------------------------------------------- rendering
    def _annotate(self, res) -> np.ndarray | None:
        if res.frame is None:
            return None
        img = res.frame.copy()
        if self.cascade.zone_mask is not None:
            dim = (img * 0.45).astype(np.uint8)
            m = self.cascade.zone_mask > 0
            img = np.where(m[:, :, None], img, dim)
        for a in res.assessments:
            x, y, w, h = a.box
            if a.vetoed:
                col, tag = (110, 110, 110), "suppressed"
            elif a.score >= 0.72:
                col, tag = ((0, 60, 255) if a.kind == "flame" else (200, 200, 200),
                            f"{a.kind} {a.score:.0%}")
            elif a.score >= 0.35:
                col, tag = (0, 190, 245), f"{a.kind} {a.score:.0%}"
            else:
                col, tag = (90, 140, 90), f"{a.score:.0%}"
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            cv2.putText(img, tag, (x, max(11, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        return img

    def jpeg(self, annotated: bool = True, quality: int = 70) -> bytes | None:
        with self._lock:
            img = self.latest_annotated if annotated else self.latest
            if img is None:
                return None
            img = img.copy()
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    def status(self) -> dict:
        st = self.source.stats.to_json() if self.source else {"connected": False}
        with self._lock:
            hist = [(round(t, 1), round(f, 3), round(s, 3))
                    for t, f, s in self.history[-180:]]
        return dict(
            name=self.cfg.name, location=self.cfg.location, enabled=self.cfg.enabled,
            source=st, frames=self.frames, gated=self.gated,
            gate_rate=round(self.gated / max(1, self.frames), 3),
            proc_ms=round(self.proc_ms, 2),
            score=dict(self.score), error=self.err,
            armed=bool(self.cascade.zone_mask is None
                       or bool(self.cascade.zone_mask.any())),
            zones=[z.name for z in self.zones.zones],
            alarm={k: r.active and not self._filtered(k) for k, r in self.rules.items()},
            filtered={k: r.active and self._filtered(k) for k, r in self.rules.items()},
            history=hist,
        )

    def stop(self) -> None:
        self._stop.set()
        src = self.source
        if src is not None:
            # Kill ffmpeg now rather than waiting for the loop to notice, so a
            # removed camera's RTSP session is released immediately.
            try:
                src.stop()
            except Exception:
                pass


class Engine:
    def __init__(self, cameras: list[CameraConfig], alert_cfg: dict | None = None,
                 vlm_cfg: dict | None = None, neural_cfg=None,
                 store_root: str = "data/events", urgent_cfg: dict | None = None,
                 uplink=None, dashboard_url: str = ""):
        from ..alerts.channels import AlertDispatcher, build_channels
        from ..alerts.urgent import build_escalation
        from ..detect.neural import NeuralConfig

        self.store = EventStore(store_root)
        alert_cfg = alert_cfg or {}
        self.dispatcher = AlertDispatcher(build_channels(alert_cfg),
                                          dry_run=bool(alert_cfg.get("dry_run", True)))
        self.neural = NeuralVerifier(neural_cfg or NeuralConfig())
        self.dashboard_url = dashboard_url

        # Urgent escalation defaults to the same dry_run as ordinary alerting, so
        # nobody's phone gets rung by a first-run demo.
        urgent_cfg = dict(urgent_cfg or {})
        urgent_cfg.setdefault("dry_run", bool(alert_cfg.get("dry_run", True)))
        self.escalation = build_escalation(urgent_cfg)

        # The uplink is attached last and is entirely optional. Its ack callback
        # is what lets a button on the public dashboard cancel a ringing phone.
        self.uplink = uplink
        if self.uplink is not None:
            self.uplink.on_ack = self.acknowledge
            self.uplink.attach(self)

        vlm_cfg = vlm_cfg or {}
        self.adjudicator: Adjudicator | None = None
        if vlm_cfg.get("enabled", True):
            adj = Adjudicator(model=vlm_cfg.get("model", "claude-opus-5"),
                              effort=vlm_cfg.get("effort", "medium"),
                              max_calls_per_hour=int(vlm_cfg.get("max_calls_per_hour", 60)),
                              fail_open=bool(vlm_cfg.get("fail_open", True)),
                              max_calls_per_day=int(vlm_cfg.get("max_calls_per_day", 0)),
                              max_tokens=int(vlm_cfg.get("max_tokens", 2000)),
                              context_width=int(vlm_cfg.get("context_width", 900)),
                              crop_width=int(vlm_cfg.get("crop_width", 640)))
            self.adjudicator = adj if adj.available else None

        # Copy-on-write: add/remove swap in a new dict, so the uplink and status
        # threads iterating the old one never see it change size under them.
        self.workers: dict[str, CameraWorker] = {}
        self.camera_cfgs: dict[str, CameraConfig] = {}
        self._cam_lock = threading.Lock()
        for c in cameras:
            self.camera_cfgs[c.name] = c
            if c.enabled:
                self.workers[c.name] = CameraWorker(c, self)
        self.subscribers: list = []
        self.started = 0.0

    def start(self) -> "Engine":
        self.started = time.time()
        for w in self.workers.values():
            w.start()
        if self.uplink is not None:
            self.uplink.start()
        return self

    def stop(self) -> None:
        for w in self.workers.values():
            w.stop()
        self.dispatcher.stop()
        if self.escalation is not None:
            self.escalation.stop()
        if self.uplink is not None:
            self.uplink.stop()

    # ------------------------------------------------------- camera management
    def add_camera(self, cfg: CameraConfig) -> None:
        with self._cam_lock:
            if cfg.name in self.camera_cfgs:
                raise ValueError(f"camera {cfg.name} already exists")
            self.camera_cfgs = {**self.camera_cfgs, cfg.name: cfg}
            if cfg.enabled:
                self._start_worker(cfg)

    def replace_camera(self, cfg: CameraConfig) -> None:
        """Restart one camera with a new config. Other cameras never blink."""
        with self._cam_lock:
            self._stop_worker(cfg.name)
            self.camera_cfgs = {**self.camera_cfgs, cfg.name: cfg}
            if cfg.enabled:
                self._start_worker(cfg)

    def remove_camera(self, name: str) -> None:
        """Stops detection for the camera. Its recorded events and clips are kept."""
        with self._cam_lock:
            if name not in self.camera_cfgs:
                raise KeyError(name)
            self._stop_worker(name)
            self.camera_cfgs = {k: v for k, v in self.camera_cfgs.items() if k != name}

    def _start_worker(self, cfg: CameraConfig) -> None:
        w = CameraWorker(cfg, self)
        self.workers = {**self.workers, cfg.name: w}
        if self.started:
            w.start()

    def _stop_worker(self, name: str) -> None:
        w = self.workers.get(name)
        if w is None:
            return
        self.workers = {k: v for k, v in self.workers.items() if k != name}
        w.stop()

    def acknowledge(self, event_id: str, by: str = "operator") -> bool:
        """Stop the escalation ladder for one incident and record who did it.

        Reachable three ways: the local console, the public cloud dashboard (via
        the heartbeat downlink), and Pushover's own acknowledge button. All three
        land here, because there must be exactly one place where an alert is
        allowed to stop."""
        ok = False
        if self.escalation is not None:
            ok = self.escalation.acknowledge(event_id, by=by)
        for ev in self.store.events:
            if ev.id == event_id:
                ev.acknowledged = True
                self.store.update(ev)
                break
        return ok

    def notify(self, ev: Event, sent: bool, adjudication=None) -> None:
        payload = {"type": "event", "event": ev.to_json(), "sent": sent,
                   "adjudication": adjudication.to_json() if adjudication else None}
        for cb in list(self.subscribers):
            try:
                cb(payload)
            except Exception:
                pass

    def status(self) -> dict:
        return {
            "uptime_s": round(time.time() - self.started, 1) if self.started else 0,
            "cameras": [w.status() for w in self.workers.values()],
            "cameras_disabled": [n for n, c in self.camera_cfgs.items() if not c.enabled],
            "events_total": len(self.store.events),
            "alerts_total": sum(1 for e in self.store.events
                                if e.alerted and not e.adjudication.startswith("suppressed")),
            "adjudicator": {
                "enabled": self.adjudicator is not None,
                "model": self.adjudicator.model if self.adjudicator else None,
                "calls_last_hour": (sum(1 for t in self.adjudicator._calls
                                        if time.time() - t < 3600)
                                    if self.adjudicator else 0),
                "budget": self.adjudicator.max_calls_per_hour if self.adjudicator else 0,
            },
            "neural": self.neural.status(),
            "delivery": self.dispatcher.status(),
            "dry_run": self.dispatcher.dry_run,
            "alerting": self.escalation.status() if self.escalation else {},
            "uplink": self.uplink.stats.to_json() if self.uplink else None,
        }
