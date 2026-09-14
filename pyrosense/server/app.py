"""
Operator console: REST + MJPEG + websocket push.

MJPEG rather than WebRTC or HLS on purpose. The console is for a supervisor on a
site PC or a phone on the same LAN, latency of a second is irrelevant to them, and
MJPEG works in every browser with no player, no codec negotiation, no TURN server
and no certificate. It is also trivially throttled - the preview runs at 6fps
regardless of what the detector is doing, so opening the dashboard cannot slow
down detection.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import base64
import hmac

from fastapi import (Body, FastAPI, HTTPException, Request, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)

from ..cloud.protocol import clip_filename
from ..core.engine import Engine

HERE = os.path.dirname(__file__)


def create_app(engine: Engine, site=None) -> FastAPI:
    """`site` is a core.site.SiteConfig. Without one (demo mode) the camera
    management endpoints report that there is no config file to change."""
    app = FastAPI(title="PyroSense", docs_url="/api/docs")
    loop_holder: dict = {}
    clients: set[WebSocket] = set()

    @app.on_event("startup")
    async def _startup():
        loop_holder["loop"] = asyncio.get_running_loop()

        def on_event(payload: dict) -> None:
            loop = loop_holder.get("loop")
            if loop is None:
                return
            for ws in list(clients):
                asyncio.run_coroutine_threadsafe(_safe_send(ws, payload), loop)

        engine.subscribers.append(on_event)

    async def _safe_send(ws: WebSocket, payload: dict) -> None:
        try:
            await ws.send_json(payload)
        except Exception:
            clients.discard(ws)

    # ------------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse)
    def index():
        p = os.path.join(HERE, "static", "index.html")
        with open(p, encoding="utf-8") as f:
            return f.read()

    # -------------------------------------------------------------------- api
    @app.get("/api/status")
    def status():
        return engine.status()

    @app.get("/api/events")
    def events(limit: int = 60):
        return engine.store.recent(limit)

    @app.post("/api/events/{event_id}/ack")
    def ack(event_id: str):
        """Acknowledging here also cancels the urgent escalation ladder - the
        repeat pushes and the pending phone call. There is exactly one place an
        alert can be stopped, and this routes into it."""
        stopped = engine.acknowledge(event_id, by="local console")
        for ev in engine.store.events:
            if ev.id == event_id:
                return {"ok": True, "escalation_stopped": stopped}
        raise HTTPException(404, "no such event")

    @app.get("/api/alerting")
    def alerting():
        return engine.escalation.status() if engine.escalation else {}

    @app.get("/api/uplink")
    def uplink():
        return engine.uplink.stats.to_json() if engine.uplink else {"enabled": False}

    @app.post("/api/events/{event_id}/verdict")
    def verdict(event_id: str, value: str):
        """Operator feedback. This is the loop that makes the system improve:
        every human correction becomes a labelled crop for a site-specific
        stage-2 model (see detect/neural.py:export_training_crops)."""
        if value not in ("confirmed", "dismissed", "pending"):
            raise HTTPException(400, "value must be confirmed|dismissed|pending")
        for ev in engine.store.events:
            if ev.id == event_id:
                ev.verdict = value
                engine.store.update(ev)
                return {"ok": True, "verdict": value}
        raise HTTPException(404, "no such event")

    @app.get("/api/media/{name}")
    def media(name: str, download: int = 0):
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(400, "bad name")
        p = os.path.join(engine.store.root, "media", name)
        if not os.path.exists(p):
            raise HTTPException(404, "not found")
        if download:
            ev = next((e for e in list(engine.store.events)
                       if e.clip and os.path.basename(e.clip) == name), None)
            stem, ext = os.path.splitext(name)
            fname = (clip_filename(ev.to_json(), stem, ext) if ext in (".mp4", ".avi")
                     else name)
            return FileResponse(p, filename=fname)
        return FileResponse(p)

    # ------------------------------------------------------ camera management
    def _admin(request: Request) -> None:
        """Changing cameras is allowed from this computer, or from elsewhere on
        the LAN only with PYRO_ADMIN_TOKEN. `pyrosense run` listens on 0.0.0.0 by
        default, and anyone on the Wi-Fi should not be able to delete a camera."""
        host = (request.client.host if request.client else "") or ""
        if host in ("127.0.0.1", "::1", "localhost", "testclient"):
            return
        want = os.environ.get("PYRO_ADMIN_TOKEN", "")
        got = request.headers.get("x-admin-token", "")
        if want and hmac.compare_digest(want, got):
            return
        raise HTTPException(403, "Camera changes are only allowed from the computer "
                                 "running PyroSense (or set PYRO_ADMIN_TOKEN).")

    def _site():
        if site is None:
            raise HTTPException(409, "Running without a config file (demo mode) - "
                                     "start with `pyrosense run -c <config>` to manage cameras.")
        return site

    def _camera_row(raw: dict) -> dict:
        from ..cameras.links import display_link, friendly_error
        name = raw.get("name", "")
        w = engine.workers.get(name)
        st = w.status() if w else None
        src = raw.get("source") or {}
        pw = str(src.get("password", ""))
        return {
            "name": name,
            "location": raw.get("location", ""),
            "enabled": bool(raw.get("enabled", True)),
            "running": w is not None,
            "kind": src.get("kind", "rtsp"),
            "link": display_link(src),
            "host": src.get("host") or "",
            "stream": src.get("stream", "as pasted" if src.get("url") else "sub"),
            "detect_smoke": bool(raw.get("detect_smoke", True)),
            "flame_threshold": float(raw.get("flame_threshold", 0.72)),
            "smoke_threshold": float(raw.get("smoke_threshold", 0.70)),
            "connected": bool(st and st["source"].get("connected")),
            "fps": st["source"].get("fps") if st else None,
            "error": (friendly_error(st["source"].get("error") or st.get("error") or "")
                      if st else ""),
            "password_from": ("saved" if "${secret:" in pw else
                              "environment" if "${" in pw else
                              "config" if pw else "none"),
        }

    @app.get("/api/cameras")
    def list_cameras():
        if site is None:
            return {"manageable": False,
                    "cameras": [dict(name=w.cfg.name, location=w.cfg.location,
                                     enabled=True, running=True, link="demo",
                                     connected=True) for w in engine.workers.values()]}
        return {"manageable": True,
                "cameras": [_camera_row(c) for c in site.raw_cameras()]}

    @app.post("/api/cameras/test")
    def test_camera(request: Request, body: dict = Body(...)):
        """Connect once and bring back one frame, so the operator sees the picture
        before committing. One attempt only - repeated bad logins lock cameras."""
        _admin(request)
        from ..cameras.links import (LinkError, grab_frame, source_from_link,
                                     with_credentials)
        from ..cameras.profiles import build_url
        try:
            src, pw = source_from_link(body.get("link", ""), body.get("stream", "sub"))
        except LinkError as e:
            raise HTTPException(400, str(e))
        url = (with_credentials(src["url"], src.get("user", ""), pw) if src.get("url")
               else build_url(src["brand"], src["host"], src.get("user", ""), pw,
                              src.get("channel", 1), src.get("stream", "sub"),
                              src.get("port")))
        r = grab_frame(url)
        out = {"ok": r["ok"], "error": r.get("error", ""), "detail": r.get("detail", "")}
        if r["ok"]:
            out["preview"] = "data:image/jpeg;base64," + base64.b64encode(r["jpeg"]).decode()
        return out

    @app.post("/api/cameras")
    def add_camera(request: Request, body: dict = Body(...)):
        _admin(request)
        sc = _site()
        from ..cameras.links import NAME_RE, LinkError, source_from_link
        name = (body.get("name") or "").strip().lower().replace(" ", "-")
        existing = {c.get("name") for c in sc.raw_cameras()}
        if not name:
            i = len(existing) + 1
            while f"cam-{i:02d}" in existing:
                i += 1
            name = f"cam-{i:02d}"
        if not NAME_RE.match(name):
            raise HTTPException(400, "Name can use lowercase letters, numbers, - and _ "
                                     "(up to 40 characters).")
        if name in existing:
            raise HTTPException(409, f"There is already a camera called {name}.")
        try:
            src, pw = source_from_link(body.get("link", ""), body.get("stream", "sub"))
        except LinkError as e:
            raise HTTPException(400, str(e))
        raw = {"name": name, "location": (body.get("location") or "").strip()[:60],
               "flame_threshold": float(body.get("flame_threshold", 0.72)),
               "smoke_threshold": float(body.get("smoke_threshold", 0.85)),
               "detect_smoke": bool(body.get("detect_smoke", False)),
               "enabled": True, "source": src}
        try:
            sc.add(raw, pw)
            engine.add_camera(sc.camera_config(name))
        except ValueError as e:
            raise HTTPException(409, str(e))
        row = next(c for c in sc.raw_cameras() if c.get("name") == name)
        return {"ok": True, "camera": _camera_row(row)}

    @app.patch("/api/cameras/{camera}")
    def edit_camera(camera: str, request: Request, body: dict = Body(...)):
        """Location, thresholds and smoke apply live. A new link or enabling /
        disabling restarts just this camera."""
        _admin(request)
        sc = _site()
        from ..cameras.links import LinkError, source_from_link
        fields = {}
        if "location" in body:
            fields["location"] = str(body["location"] or "").strip()[:60]
        for k in ("flame_threshold", "smoke_threshold"):
            if k in body:
                v = float(body[k])
                if not 0.3 <= v <= 0.99:
                    raise HTTPException(400, f"{k} must be between 0.30 and 0.99")
                fields[k] = round(v, 2)
        for k in ("enabled", "detect_smoke"):
            if k in body:
                fields[k] = bool(body[k])
        source = pw = None
        if body.get("link"):
            try:
                source, pw = source_from_link(body["link"], body.get("stream", "sub"))
            except LinkError as e:
                raise HTTPException(400, str(e))
        before = next((c for c in sc.raw_cameras() if c.get("name") == camera), None)
        if before is None:
            raise HTTPException(404, "no such camera")
        sc.update(camera, fields, source=source, password=pw)

        restart = source is not None or (
            "enabled" in fields and fields["enabled"] != bool(before.get("enabled", True)))
        w = engine.workers.get(camera)
        if restart or (w is None and fields.get("enabled", before.get("enabled", True))):
            engine.replace_camera(sc.camera_config(camera))
        elif w is not None:
            cfg = w.cfg
            if "location" in fields:
                cfg.location = fields["location"]
            if "detect_smoke" in fields:
                cfg.detect_smoke = fields["detect_smoke"]
                w.cascade.sig.detect_smoke = fields["detect_smoke"]
            if "flame_threshold" in fields:
                cfg.flame_threshold = fields["flame_threshold"]
                w.rules["flame"].threshold = fields["flame_threshold"]
            if "smoke_threshold" in fields:
                cfg.smoke_threshold = fields["smoke_threshold"]
                w.rules["smoke"].threshold = fields["smoke_threshold"]
        else:
            engine.camera_cfgs = {**engine.camera_cfgs, camera: sc.camera_config(camera)}
        row = next(c for c in sc.raw_cameras() if c.get("name") == camera)
        return {"ok": True, "restarted": bool(restart), "camera": _camera_row(row)}

    @app.delete("/api/cameras/{camera}")
    def delete_camera(camera: str, request: Request):
        """Removes the camera from detection and from the config file. Its past
        events and recordings stay in the log."""
        _admin(request)
        sc = _site()
        try:
            sc.remove(camera)
        except KeyError:
            raise HTTPException(404, "no such camera")
        try:
            engine.remove_camera(camera)
        except KeyError:
            pass
        return {"ok": True}

    @app.get("/api/camera/{camera}")
    def camera_detail(camera: str, limit: int = 12):
        """Everything the detail view needs for one camera, in one request."""
        w = engine.workers.get(camera)
        if not w:
            raise HTTPException(404, "no such camera")
        evs = [e.to_json() for e in engine.store.events
               if e.camera == camera][-limit:][::-1]
        return {"status": w.status(), "events": evs,
                "config": {"flame_threshold": w.cfg.flame_threshold,
                           "smoke_threshold": w.cfg.smoke_threshold,
                           "detect_smoke": w.cfg.detect_smoke,
                           "postroll_s": getattr(w.cfg, "postroll_s", 8.0),
                           "cooldown_s": w.cfg.cooldown_s}}

    @app.post("/api/camera/{camera}/smoke")
    def toggle_smoke(camera: str, enabled: bool):
        """Turn the smoke channel on or off live.

        On a site whose scene is pale sheeting or dust this is the switch you
        actually reach for, and making someone edit a config file and restart the
        detector is how one false alarm becomes three."""
        w = engine.workers.get(camera)
        if not w:
            raise HTTPException(404, "no such camera")
        w.cfg.detect_smoke = bool(enabled)
        w.cascade.sig.detect_smoke = bool(enabled)
        if site is not None:
            try:
                site.update(camera, {"detect_smoke": bool(enabled)})
            except Exception:
                pass
        return {"ok": True, "detect_smoke": w.cfg.detect_smoke}

    @app.get("/api/snapshot/{camera}")
    def snapshot(camera: str, annotated: bool = True):
        w = engine.workers.get(camera)
        if not w:
            raise HTTPException(404, "no such camera")
        jpg = w.jpeg(annotated=annotated)
        if jpg is None:
            raise HTTPException(503, "no frame yet")
        return StreamingResponse(iter([jpg]), media_type="image/jpeg")

    @app.get("/api/stream/{camera}")
    def stream(camera: str, annotated: bool = True, fps: int = 6):
        w = engine.workers.get(camera)
        if not w:
            raise HTTPException(404, "no such camera")

        def gen():
            interval = 1.0 / max(1, min(fps, 15))
            while engine.workers.get(camera) is w:
                jpg = w.jpeg(annotated=annotated)
                if jpg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpg)).encode() +
                           b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(interval)

        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/bench")
    def bench():
        p = "data/bench.json"
        if not os.path.exists(p):
            return JSONResponse({"error": "run: python -m pyrosense.bench "
                                          "--json data/bench.json"}, 404)
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        clients.add(ws)
        try:
            await ws.send_json({"type": "hello", "status": engine.status()})
            while True:
                await asyncio.sleep(2.0)
                await ws.send_json({"type": "status", "status": engine.status()})
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            clients.discard(ws)

    return app
