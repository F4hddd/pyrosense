"""
The hosted mirror. This is what goes on Render.

It holds no camera credentials, opens no connection to the site, and runs no
computer vision. It accepts what the agent pushes, keeps a small window of it in
memory, and serves a public read-only page.

Two things here are load-bearing:

STALENESS IS THE PRIMARY STATE
    Everything this server knows is a memory of what the site last said. So the
    dashboard leads with how old that memory is. A monitoring page that renders a
    calm green "all clear" because it froze forty minutes ago is worse than no
    page at all, and the only way to make that impossible is to treat "last heard
    from" as the headline rather than a footnote.

THE WATCHDOG
    If the site stops reporting, that is itself an alert - a power cut, a dead
    machine, a severed link, or someone unplugging the box. Silence must never be
    mistaken for calm, so the cloud independently escalates a missing heartbeat
    through the same urgent channels the site uses. This is the one alerting job
    the cloud owns, precisely because it is the one the site cannot do for itself.

Configuration is entirely by environment variable, because that is how Render
(and every other PaaS) injects secrets:

    PYRO_INGEST_TOKEN     shared secret with the agent          (required)
    PYRO_SITE_NAME        display name                          (default "Site")
    PYRO_PUBLIC_TOKEN     if set, viewers need ?k=<token>       (default: open)
    PYRO_ACK_PIN          if set, acknowledging requires it     (recommended)
    PYRO_STALE_S          seconds before "stale"                (default 45)
    PYRO_OFFLINE_S        seconds before watchdog fires         (default 180)
    PUSHOVER_TOKEN / PUSHOVER_USER      watchdog push
    NTFY_TOPIC / NTFY_SERVER            watchdog push
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import json
import os
import threading
import time
from collections import deque

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .protocol import (MAX_CLIP_BYTES, clip_filename, MAX_EVENT_BYTES,
                       MAX_FRAME_BYTES, verify)

HERE = os.path.dirname(__file__)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _safe_id(v: str) -> bool:
    """Event ids become filenames, so refuse anything path-shaped."""
    return bool(v) and all(c.isalnum() or c in "-_." for c in v) \
        and ".." not in v


class Store:
    """Site state: in memory, with an optional append-only file behind it.

    Deliberately not a database. The whole dataset is one site's last few hundred
    events plus one JPEG per camera - a few megabytes. A database would add an
    operational dependency, a connection pool and a failure mode, to buy
    durability that a single appended JSONL file already provides at this size.

    Frames are never persisted: they are worthless a minute later, and writing
    them would turn a 1 GB disk into a liability. Events are persisted when
    PYRO_DATA_DIR points at a mounted disk, so the public incident history
    survives a redeploy. With no disk the mirror simply starts empty and refills
    as new events arrive - survivable, because the authoritative log always lives
    on the site machine, not here."""

    def __init__(self, max_events: int = 300, data_dir: str = ""):
        self.lock = threading.Lock()
        self.telemetry: dict = {}
        self.telemetry_at: float = 0.0
        self.frames: dict[str, tuple[bytes, float, dict]] = {}
        self.events: deque = deque(maxlen=max_events)
        self.acks: dict[str, dict] = {}          # event_id -> {by, at, delivered}
        self.first_seen: float = 0.0
        self.watchdog_fired_at: float = 0.0
        self.subscribers: list[asyncio.Queue] = []
        # Optional durability. Without a mounted disk the mirror simply starts
        # empty after a redeploy, which is survivable - the site holds the
        # authoritative event log and this is only a mirror - but a public
        # dashboard that forgets every incident on each deploy is a poor record,
        # so use it whenever a disk is attached.
        self.data_dir = data_dir
        self._events_path = os.path.join(data_dir, "events.jsonl") if data_dir else ""
        # Clips go to disk when one is mounted, otherwise into a small bounded
        # ring in memory so replay still works on a diskless free-tier deploy.
        self.clips: dict = {}
        self.clip_dir = os.path.join(data_dir, "clips") if data_dir else ""
        if self.clip_dir:
            os.makedirs(self.clip_dir, exist_ok=True)
        if self._events_path:
            self._load()

    def _load(self) -> None:
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            if not os.path.exists(self._events_path):
                return
            with open(self._events_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            print(f"  restored {len(self.events)} event(s) from {self._events_path}")
        except Exception as e:
            print(f"  could not restore events: {type(e).__name__}: {e}")

    def persist(self, ev: dict) -> None:
        if not self._events_path:
            return
        try:
            slim = {k: v for k, v in ev.items() if k != "snapshot_b64"}
            with open(self._events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(slim) + "\n")
        except Exception:
            pass

    def put_clip(self, event_id: str, body: bytes, ext: str = ".mp4") -> None:
        if self.clip_dir:
            try:
                with open(os.path.join(self.clip_dir, event_id + ext), "wb") as f:
                    f.write(body)
                self._prune_clips()
                return
            except Exception:
                pass
        self.clips[event_id] = (body, ext)
        while len(self.clips) > 12:            # bounded: RAM is not storage
            self.clips.pop(next(iter(self.clips)))

    def has_clip(self, event_id: str) -> bool:
        """Asked at read time, not stamped on arrival.

        The event and its clip travel independently - the event goes through the
        durable spool, the clip is a direct POST - so the clip frequently lands
        first and finds no event to mark. Deriving the flag when the dashboard
        asks makes the answer correct regardless of which arrived first."""
        if self.clip_dir:
            return any(os.path.exists(os.path.join(self.clip_dir, event_id + e))
                       for e in (".mp4", ".avi"))
        return event_id in self.clips

    def get_clip(self, event_id: str):
        if self.clip_dir:
            for ext in (".mp4", ".avi"):
                fp = os.path.join(self.clip_dir, event_id + ext)
                if os.path.exists(fp):
                    try:
                        with open(fp, "rb") as f:
                            return f.read(), ext
                    except Exception:
                        return None
        return self.clips.get(event_id)

    def _prune_clips(self, keep: int = 60) -> None:
        """A 1 GB disk holds a lot of clips, but not an unbounded number."""
        try:
            files = sorted(
                (os.path.join(self.clip_dir, n) for n in os.listdir(self.clip_dir)),
                key=os.path.getmtime)
            for fp in files[:-keep]:
                os.remove(fp)
        except Exception:
            pass

    def age(self) -> float | None:
        return (time.time() - self.telemetry_at) if self.telemetry_at else None

    def health(self, stale_s: float, offline_s: float) -> str:
        a = self.age()
        if a is None:
            return "never_connected"
        if a > offline_s:
            return "offline"
        if a > stale_s:
            return "stale"
        return "live"

    def publish(self, msg: dict) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(msg)
            except Exception:
                pass


def create_cloud_app() -> FastAPI:
    app = FastAPI(title="PyroSense Cloud", docs_url=None, redoc_url=None)
    store = Store(data_dir=_env("PYRO_DATA_DIR"))

    INGEST_TOKEN = _env("PYRO_INGEST_TOKEN")
    SITE_NAME = _env("PYRO_SITE_NAME", "Site")
    PUBLIC_TOKEN = _env("PYRO_PUBLIC_TOKEN")
    ACK_PIN = _env("PYRO_ACK_PIN")
    STALE_S = float(_env("PYRO_STALE_S", "45"))
    OFFLINE_S = float(_env("PYRO_OFFLINE_S", "180"))
    loop_holder: dict = {}

    if not INGEST_TOKEN:
        # Fail loudly at boot rather than silently accepting anonymous writes.
        print("  !! PYRO_INGEST_TOKEN is not set - ingest endpoints will reject "
              "everything. Set it on the service and on the local agent.")

    # ---------------------------------------------------------------- watchdog
    def _watchdog() -> None:
        """Silence from the site is an alert in its own right."""
        from ..alerts.urgent import Ntfy, Pushover
        po = Pushover(_env("PUSHOVER_TOKEN"), _env("PUSHOVER_USER")) \
            if _env("PUSHOVER_TOKEN") else None
        nt = Ntfy(_env("NTFY_TOPIC"), _env("NTFY_SERVER", "https://ntfy.sh")) \
            if _env("NTFY_TOPIC") else None
        while True:
            time.sleep(20)
            try:
                a = store.age()
                if a is None or a <= OFFLINE_S:
                    if a is not None and a <= STALE_S:
                        store.watchdog_fired_at = 0.0     # recovered; re-arm
                    continue
                if store.watchdog_fired_at and (time.time() - store.watchdog_fired_at) < 1800:
                    continue
                store.watchdog_fired_at = time.time()
                title = f"PyroSense OFFLINE - {SITE_NAME}"
                body = (f"No heartbeat from the site for {a/60:.0f} minutes. "
                        f"Fire detection may not be running. Check power, the "
                        f"machine, and the internet connection.")
                if po and po.configured:
                    po.send(title, body, emergency=True)
                if nt and nt.configured:
                    nt.send(title, body, tags="warning,electric_plug")
                print(f"  watchdog: site silent for {a:.0f}s - alerted")
            except Exception as e:
                print(f"  watchdog error: {type(e).__name__}: {e}")

    threading.Thread(target=_watchdog, daemon=True, name="cloud-watchdog").start()

    @app.on_event("startup")
    async def _startup():
        loop_holder["loop"] = asyncio.get_running_loop()

    # ------------------------------------------------------------------ auth
    async def _auth(request: Request) -> bytes:
        """Bearer token plus an HMAC over the body.

        TLS already protects the token in transit. The signature is belt and
        braces: it binds the token to this exact body and timestamp, so a leaked
        request cannot be replayed with different content."""
        if not INGEST_TOKEN:
            raise HTTPException(503, "server has no ingest token configured")
        auth = request.headers.get("authorization", "")
        supplied = auth[7:] if auth.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(supplied, INGEST_TOKEN):
            raise HTTPException(401, "bad token")
        body = await request.body()
        ok, why = verify(INGEST_TOKEN, body,
                         request.headers.get("x-pyro-timestamp", ""),
                         request.headers.get("x-pyro-signature", ""))
        if not ok:
            raise HTTPException(401, f"bad signature: {why}")
        return body

    def _public_ok(request: Request) -> None:
        if PUBLIC_TOKEN and request.query_params.get("k") != PUBLIC_TOKEN:
            raise HTTPException(403, "this dashboard is private")

    # ---------------------------------------------------------------- ingest
    @app.post("/api/ingest/heartbeat")
    async def ingest_heartbeat(request: Request):
        body = await _auth(request)
        try:
            tel = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(400, "bad json")
        with store.lock:
            store.telemetry = tel
            store.telemetry_at = time.time()
            if not store.first_seen:
                store.first_seen = store.telemetry_at
            pending = [{"event_id": eid, "by": a["by"]}
                       for eid, a in store.acks.items() if not a.get("delivered")]
            for eid in [p["event_id"] for p in pending]:
                store.acks[eid]["delivered"] = True
        store.publish({"type": "status"})
        return {"ok": True, "acks": pending, "server_time": time.time()}

    @app.post("/api/ingest/frame")
    async def ingest_frame(request: Request):
        body = await _auth(request)
        if len(body) > MAX_FRAME_BYTES:
            raise HTTPException(413, "frame too large")
        cam = request.headers.get("x-pyro-camera", "").strip()
        if not cam:
            raise HTTPException(400, "missing camera")
        meta = {
            "flame": float(request.headers.get("x-pyro-flame", 0) or 0),
            "smoke": float(request.headers.get("x-pyro-smoke", 0) or 0),
            "alarm": request.headers.get("x-pyro-alarm", "0") == "1",
        }
        with store.lock:
            store.frames[cam] = (body, time.time(), meta)
        return {"ok": True}

    @app.post("/api/ingest/event")
    async def ingest_event(request: Request):
        body = await _auth(request)
        if len(body) > MAX_EVENT_BYTES:
            raise HTTPException(413, "event too large")
        try:
            ev = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(400, "bad json")

        snap = ev.pop("snapshot_b64", "") or ""
        ev["has_snapshot"] = bool(snap)
        ev["received_at"] = time.time()
        with store.lock:
            if snap:
                try:
                    store.frames["event:" + ev["id"]] = (
                        base64.b64decode(snap), time.time(), {"event": True})
                except (binascii.Error, ValueError):
                    ev["has_snapshot"] = False
            if not any(e.get("id") == ev.get("id") for e in store.events):
                store.events.append(ev)
                store.persist(ev)
        store.publish({"type": "event", "event": ev})
        return {"ok": True}

    @app.post("/api/ingest/clip")
    async def ingest_clip(request: Request):
        body = await _auth(request)
        if len(body) > MAX_CLIP_BYTES:
            raise HTTPException(413, "clip too large")
        eid = request.headers.get("x-pyro-event", "").strip()
        if not eid or not _safe_id(eid):
            raise HTTPException(400, "bad event id")
        ext = request.headers.get("x-pyro-ext", ".mp4")
        ext = ext if ext in (".mp4", ".avi") else ".mp4"
        store.put_clip(eid, body, ext)
        store.publish({"type": "clip", "event_id": eid})
        return {"ok": True, "bytes": len(body)}

    # ---------------------------------------------------------------- public
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        _public_ok(request)
        with open(os.path.join(HERE, "static", "public.html"), encoding="utf-8") as f:
            return f.read()

    @app.get("/healthz")
    def healthz():
        """Liveness for the platform. Deliberately separate from site health -
        this says the web service is up, not that the fire detector is."""
        return {"ok": True, "site_health": store.health(STALE_S, OFFLINE_S)}

    @app.get("/api/public/status")
    def public_status(request: Request):
        _public_ok(request)
        with store.lock:
            tel = dict(store.telemetry)
            age = store.age()
            frames = {k: {"age_s": round(time.time() - v[1], 1), **v[2]}
                      for k, v in store.frames.items() if not k.startswith("event:")}
            acks = {k: v["by"] for k, v in store.acks.items()}
        return {
            "site": SITE_NAME,
            "health": store.health(STALE_S, OFFLINE_S),
            "last_heartbeat_age_s": round(age, 1) if age is not None else None,
            "stale_after_s": STALE_S, "offline_after_s": OFFLINE_S,
            "telemetry": tel, "frames": frames, "acks": acks,
            "ack_required_pin": bool(ACK_PIN),
            "persistent": bool(store.data_dir),
            "server_time": time.time(),
        }

    @app.get("/api/public/events")
    def public_events(request: Request, limit: int = 50):
        _public_ok(request)
        with store.lock:
            evs = [dict(e) for e in list(store.events)[-limit:][::-1]]
        for e in evs:
            e["has_clip"] = store.has_clip(e.get("id", ""))
        return evs

    @app.get("/api/public/frame/{camera}")
    def public_frame(camera: str, request: Request):
        _public_ok(request)
        with store.lock:
            item = store.frames.get(camera)
        if not item:
            raise HTTPException(404, "no frame yet")
        return Response(content=item[0], media_type="image/jpeg",
                        headers={"Cache-Control": "no-store",
                                 "X-Frame-Age": str(round(time.time() - item[1], 1))})

    @app.get("/api/public/clip/{event_id}")
    def public_clip(event_id: str, request: Request, download: int = 0):
        """Event replay. Served whole rather than range-streamed - these are a few
        hundred KB and a browser buffers one happily. ?download=1 sends it as an
        attachment named after the camera, kind and time instead of the id."""
        _public_ok(request)
        if not _safe_id(event_id):
            raise HTTPException(400, "bad event id")
        item = store.get_clip(event_id)
        if not item:
            raise HTTPException(404, "no clip for this event")
        body, ext = item
        mime = "video/mp4" if ext == ".mp4" else "video/x-msvideo"
        headers = {"Cache-Control": "public, max-age=3600"}
        if download:
            with store.lock:
                ev = next((e for e in store.events if e.get("id") == event_id), None)
            headers["Content-Disposition"] =                 f'attachment; filename="{clip_filename(ev, event_id, ext)}"'
        return Response(content=body, media_type=mime, headers=headers)

    @app.get("/api/public/stream/{camera}")
    def public_stream(camera: str, request: Request):
        """MJPEG assembled from pushed stills. Motion is only as smooth as the
        push interval, which is the honest representation of what we know."""
        _public_ok(request)

        def gen():
            last = 0.0
            for _ in range(6000):
                with store.lock:
                    item = store.frames.get(camera)
                if item and item[1] != last:
                    last = item[1]
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(item[0])).encode() +
                           b"\r\n\r\n" + item[0] + b"\r\n")
                time.sleep(0.5)

        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/public/ack/{event_id}")
    async def public_ack(event_id: str, request: Request):
        """Acknowledge from the public page. Travels back to the site on the next
        heartbeat, where it cancels the escalation ladder."""
        _public_ok(request)
        if ACK_PIN:
            try:
                supplied = (await request.json()).get("pin", "")
            except Exception:
                supplied = ""
            if not hmac.compare_digest(str(supplied), ACK_PIN):
                raise HTTPException(403, "wrong PIN")
        with store.lock:
            known = any(e.get("id") == event_id for e in store.events)
            if not known:
                raise HTTPException(404, "unknown event")
            store.acks[event_id] = {"by": "dashboard", "at": time.time(),
                                    "delivered": False}
            for e in store.events:
                if e.get("id") == event_id:
                    e["acknowledged"] = True
        store.publish({"type": "ack", "event_id": event_id})
        return {"ok": True, "note": "acknowledgement will reach the site on its "
                                    "next heartbeat"}

    @app.get("/api/public/sse")
    async def sse(request: Request):
        """Server-sent events. Chosen over websockets because it survives the
        proxies and idle timeouts of a PaaS far more predictably, reconnects by
        itself in the browser, and this feed is one-directional anyway."""
        _public_ok(request)
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        store.subscribers.append(q)

        async def gen():
            try:
                yield b": connected\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield b": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps(msg)}\n\n".encode()
            finally:
                if q in store.subscribers:
                    store.subscribers.remove(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # Bridge thread-side publishes into the asyncio loop.
    def _publish(msg: dict) -> None:
        loop = loop_holder.get("loop")
        if loop is None:
            return
        for q in list(store.subscribers):
            try:
                loop.call_soon_threadsafe(q.put_nowait, msg)
            except Exception:
                pass

    store.publish = _publish        # type: ignore[assignment]
    return app


app = create_cloud_app()
