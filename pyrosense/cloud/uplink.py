"""
The local agent: pushes telemetry, frames and events OUT to the cloud mirror.

Everything here is best-effort by design. The uplink is a reporting path, not a
safety path - if it is broken, detection and the alert ladder carry on untouched
and the operator console on the LAN keeps working. Nothing in this file is
allowed to raise into the detection loop or block it.

Three behaviours worth calling out:

  Write-ahead events. Events are the one thing we must not lose to a dropped
  connection, so send_event() writes to disk FIRST and the sender only deletes
  what it has delivered. The spool directory is the queue. Frames and heartbeats
  are deliberately not durable - a stale frame has no value, and re-sending a
  heartbeat from 40 seconds ago actively misleads the staleness display.

  Adaptive frame rate. Idle cameras push a small frame every few seconds. A
  camera with a live candidate pushes immediately at higher quality. That keeps
  a domestic upload link free for the moment it actually matters.

  Downlink over the heartbeat. The cloud cannot dial in to a machine behind NAT,
  so the heartbeat *response* carries commands - currently acknowledgements
  raised from the public dashboard. The agent applies them locally, which is what
  lets a button on a webpage silence a phone that is being rung by this process.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2

from .protocol import MAX_CLIP_BYTES, PROTOCOL_VERSION, Telemetry, sign

AGENT_VERSION = "0.9.0"


@dataclass
class UplinkStats:
    connected: bool = False
    heartbeats: int = 0
    frames_sent: int = 0
    events_sent: int = 0
    clips_sent: int = 0
    failures: int = 0
    queued: int = 0
    last_ok: float = 0.0
    last_error: str = ""
    rtt_ms: float = 0.0

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["last_ok_age_s"] = round(time.time() - self.last_ok, 1) if self.last_ok else None
        d["last_error"] = self.last_error[:200]
        return d


class CloudUplink:
    def __init__(self, base_url: str, token: str, site: str = "site",
                 heartbeat_s: float = 5.0, frame_idle_s: float = 4.0,
                 frame_active_s: float = 1.0, frame_width: int = 640,
                 spool_dir: str = "data/spool", enabled: bool = True,
                 on_ack=None, timeout_s: float = 15.0):
        self.base = base_url.rstrip("/")
        self.token = token
        self.site = site
        self.heartbeat_s = heartbeat_s
        self.frame_idle_s = frame_idle_s
        self.frame_active_s = frame_active_s
        self.frame_width = frame_width
        self.enabled = enabled and bool(base_url and token)
        self.timeout_s = timeout_s
        self.on_ack = on_ack               # called with (event_id, who)
        self.stats = UplinkStats()

        self.spool_dir = spool_dir
        os.makedirs(spool_dir, exist_ok=True)

        self._engine = None
        self._stop = threading.Event()
        self._last_frame_at: dict[str, float] = {}
        self._wake = threading.Event()
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------- transport
    def _post(self, path: str, body: bytes, content_type: str,
              extra: dict | None = None) -> dict:
        ts = f"{time.time():.3f}"
        req = urllib.request.Request(self.base + path, data=body, method="POST")
        req.add_header("Content-Type", content_type)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("X-Pyro-Timestamp", ts)
        req.add_header("X-Pyro-Signature", sign(self.token, body, ts))
        req.add_header("X-Pyro-Site", self.site)
        req.add_header("X-Pyro-Protocol", str(PROTOCOL_VERSION))
        for k, v in (extra or {}).items():
            req.add_header(k, str(v))
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            raw = r.read()
        self.stats.rtt_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.stats.last_ok = time.time()
        self.stats.connected = True
        self.stats.last_error = ""
        try:
            return json.loads(raw.decode("utf-8", "ignore")) if raw else {}
        except json.JSONDecodeError:
            return {}

    def _fail(self, e: Exception) -> None:
        self.stats.failures += 1
        self.stats.connected = False
        detail = ""
        if isinstance(e, urllib.error.HTTPError):
            try:
                detail = e.read().decode("utf-8", "ignore")[:120]
            except Exception:
                pass
            self.stats.last_error = f"HTTP {e.code} {detail}"
        else:
            self.stats.last_error = f"{type(e).__name__}: {e}"[:200]

    # ---------------------------------------------------------------- start
    def attach(self, engine) -> "CloudUplink":
        self._engine = engine
        return self

    def start(self) -> "CloudUplink":
        if not self.enabled:
            return self
        for target, name in ((self._heartbeat_loop, "uplink-hb"),
                             (self._frame_loop, "uplink-frames"),
                             (self._event_loop, "uplink-events")):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)
        return self

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------ heartbeat
    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            if self._engine is None:
                continue
            try:
                st = self._engine.status()
                tel = Telemetry(
                    site=self.site, agent_version=AGENT_VERSION, sent_at=time.time(),
                    uptime_s=st.get("uptime_s", 0),
                    cameras=[{k: c.get(k) for k in
                              ("name", "location", "source", "score", "alarm",
                               "proc_ms", "gate_rate", "frames", "error", "armed",
                               "zones", "history")}
                             for c in st.get("cameras", [])],
                    adjudicator=st.get("adjudicator", {}),
                    neural=st.get("neural", {}),
                    alerting=st.get("alerting", {}),
                    events_total=st.get("events_total", 0),
                    dry_run=st.get("dry_run", True),
                )
                body = json.dumps(tel.to_json()).encode()
                resp = self._post("/api/ingest/heartbeat", body, "application/json")
                self.stats.heartbeats += 1
                self._apply_downlink(resp)
            except Exception as e:
                self._fail(e)

    def _apply_downlink(self, resp: dict) -> None:
        """Commands travelling back down the heartbeat. This is how a button on a
        public webpage reaches a machine that has no reachable address."""
        for ack in (resp or {}).get("acks", []) or []:
            eid = ack.get("event_id")
            who = ack.get("by", "dashboard")
            if eid and self.on_ack:
                try:
                    self.on_ack(eid, who)
                except Exception:
                    pass

    # --------------------------------------------------------------- frames
    def _frame_loop(self) -> None:
        while not self._stop.wait(0.4):
            if self._engine is None:
                continue
            now = time.time()
            for name, w in list(self._engine.workers.items()):
                try:
                    hot = max(w.score.get("flame", 0), w.score.get("smoke", 0))
                    interval = self.frame_active_s if hot > 0.35 else self.frame_idle_s
                    if now - self._last_frame_at.get(name, 0) < interval:
                        continue
                    img = w.latest_annotated if w.latest_annotated is not None else w.latest
                    if img is None:
                        continue
                    q = 78 if hot > 0.35 else 62
                    if img.shape[1] > self.frame_width:
                        s = self.frame_width / img.shape[1]
                        img = cv2.resize(img, (self.frame_width,
                                               int(img.shape[0] * s)),
                                         interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", img,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
                    if not ok:
                        continue
                    self._post("/api/ingest/frame", buf.tobytes(), "image/jpeg",
                               extra={"X-Pyro-Camera": name,
                                      "X-Pyro-Flame": f"{w.score.get('flame',0):.3f}",
                                      "X-Pyro-Smoke": f"{w.score.get('smoke',0):.3f}",
                                      "X-Pyro-Alarm": "1" if any(w.rules[k].active
                                                                 for k in w.rules) else "0"})
                    self._last_frame_at[name] = now
                    self.stats.frames_sent += 1
                except Exception as e:
                    self._fail(e)
                    self._last_frame_at[name] = now      # back off this camera too

    # --------------------------------------------------------------- events
    def send_event(self, payload: dict) -> None:
        """Write-ahead: the event hits the disk before anything tries to send it.

        The first version of this queued events in memory and only spooled them to
        disk *after* a send failed. A test pushing three events at a dead endpoint
        found the hole: the third was still sitting in RAM inside a socket timeout
        when the check ran, so a process death at that moment would have lost an
        event outright - and the window is as long as the HTTP timeout, which is
        exactly the window a crashing or power-cut machine occupies.

        Writing first inverts it. An event is durable the instant it exists, the
        spool directory *is* the queue, and the sender's only job is to delete
        what it has successfully delivered. Events are rare enough (single digits
        per day on a real site) that the disk write costs nothing."""
        try:
            os.makedirs(self.spool_dir, exist_ok=True)
            eid = str(payload.get("id") or f"ev-{time.time():.3f}")
            safe = "".join(c for c in eid if c.isalnum() or c in "-_.")[:80]
            path = os.path.join(self.spool_dir, f"{time.time():.3f}-{safe}.json")
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)            # atomic: never a half-written event
            self._wake.set()
        except Exception as e:
            # Disk full or read-only: say so loudly rather than pretend.
            self.stats.last_error = f"spool write failed: {type(e).__name__}: {e}"[:200]

    def send_clip(self, event_id: str, path: str) -> None:
        """Upload an event clip. Fire-and-forget, on its own thread.

        Not spooled the way events are, and that is deliberate: a clip is
        evidence, an event is the alarm. We must never delay or risk the alarm in
        order to carry a few hundred KB of video, and a clip that turns up an hour
        late is of little use anyway. The site keeps its own copy on disk
        regardless, and that copy is the authoritative one."""
        def work():
            try:
                size = os.path.getsize(path)
                if size > MAX_CLIP_BYTES:
                    self.stats.last_error = f"clip {size/1e6:.1f}MB over cap"
                    return
                with open(path, "rb") as f:
                    body = f.read()
                ext = os.path.splitext(path)[1] or ".mp4"
                self._post("/api/ingest/clip", body, "video/mp4",
                           extra={"X-Pyro-Event": event_id, "X-Pyro-Ext": ext})
                self.stats.clips_sent += 1
            except Exception as e:
                self._fail(e)
        threading.Thread(target=work, daemon=True, name="clip-" + event_id).start()

    def _drain(self) -> None:
        """Deliver spooled events oldest first, deleting each on success."""
        try:
            names = sorted(n for n in os.listdir(self.spool_dir)
                           if n.endswith(".json"))
        except FileNotFoundError:
            return
        self.stats.queued = len(names)
        for n in names[:25]:
            p = os.path.join(self.spool_dir, n)
            try:
                with open(p, encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                os.remove(p)                 # unreadable: drop it, don't spin
                continue
            try:
                self._post("/api/ingest/event", json.dumps(payload).encode(),
                           "application/json")
            except Exception as e:
                self._fail(e)
                return                       # still down; retry on the next tick
            try:
                os.remove(p)
            except OSError:
                pass
            self.stats.events_sent += 1
        try:
            self.stats.queued = len([n for n in os.listdir(self.spool_dir)
                                     if n.endswith(".json")])
        except Exception:
            pass

    def _event_loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=2.0)
            self._wake.clear()
            try:
                self._drain()
            except Exception as e:
                self._fail(e)
