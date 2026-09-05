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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)

from ..core.engine import Engine

HERE = os.path.dirname(__file__)


def create_app(engine: Engine) -> FastAPI:
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
    def media(name: str):
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(400, "bad name")
        p = os.path.join(engine.store.root, "media", name)
        if not os.path.exists(p):
            raise HTTPException(404, "not found")
        return FileResponse(p)

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
            while True:
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
