"""
Alert delivery.

Design rules learned from things that go wrong at 3am:

  Never block the detector. Delivery runs on its own worker thread with a queue.
  A hung SMTP server must not stall the camera that is watching the fire.

  Every channel is independent. One failing channel must not prevent the others
  from firing, so each is wrapped and its outcome recorded per attempt.

  Retry, then give up loudly. Transient network failures are normal on a factory
  LAN. Silent permanent failure is not acceptable, so exhausted retries are
  written to the event record and surfaced in the dashboard.

  Always keep a local copy. The `file` and `console` sinks have no dependencies
  and cannot fail for network reasons. Whatever else is configured, the event is
  on disk.

  A dry-run mode exists and is the default for demos, so nothing is sent
  anywhere until an operator has explicitly configured a real destination.
"""
from __future__ import annotations

import json
import os
import queue
import smtplib
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.message import EmailMessage


@dataclass
class DeliveryResult:
    channel: str
    ok: bool
    detail: str = ""
    attempts: int = 1
    t: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        return dict(channel=self.channel, ok=self.ok, detail=self.detail[:300],
                    attempts=self.attempts)


class Channel:
    name = "base"

    def send(self, ev: dict, snapshot: str = "") -> str:
        raise NotImplementedError


class ConsoleChannel(Channel):
    name = "console"

    def send(self, ev: dict, snapshot: str = "") -> str:
        print(f"\n  *** {ev['kind'].upper()} ALERT  {ev['camera']}  "
              f"score {ev['score']:.2f}  {ev.get('started_iso','')}\n"
              f"      {ev.get('reason','')}\n"
              f"      {ev.get('adjudication','')}\n")
        return "printed"


class FileChannel(Channel):
    name = "file"

    def __init__(self, path: str = "data/events/alerts.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def send(self, ev: dict, snapshot: str = "") -> str:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
        return self.path


class WebhookChannel(Channel):
    """Generic JSON POST. This is the one that matters in practice - it is how the
    system reaches a BMS, a PLC gateway, PagerDuty, Slack, Teams, an SMS gateway,
    or whatever the site already uses."""
    name = "webhook"

    def __init__(self, url: str, headers: dict | None = None, timeout: float = 8.0):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout

    def send(self, ev: dict, snapshot: str = "") -> str:
        body = json.dumps(ev).encode()
        req = urllib.request.Request(self.url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        for k, v in self.headers.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return f"HTTP {r.status}"


class TelegramChannel(Channel):
    """Telegram is a genuinely good fit for small sites: free, instant, delivers a
    photo to a phone, and the group can include the site manager and the
    contracted keyholder without any per-seat licensing."""
    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str, timeout: float = 12.0):
        self.token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout

    def send(self, ev: dict, snapshot: str = "") -> str:
        cap = (f"{ev['kind'].upper()} on {ev['camera']}\n"
               f"confidence {ev['score']:.0%} - {ev.get('started_iso','')}\n"
               f"{ev.get('adjudication') or ev.get('reason','')}")[:1024]
        if snapshot and os.path.exists(snapshot):
            return self._photo(snapshot, cap)
        return self._text(cap)

    def _text(self, text: str) -> str:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = json.dumps({"chat_id": self.chat_id, "text": text}).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return f"HTTP {r.status}"

    def _photo(self, path: str, caption: str) -> str:
        boundary = "----pyrosense" + os.urandom(8).hex()
        with open(path, "rb") as f:
            img = f.read()
        parts = []
        for k, v in (("chat_id", self.chat_id), ("caption", caption)):
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                         f'name="{k}"\r\n\r\n{v}\r\n'.encode())
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f'name="photo"; filename="alert.jpg"\r\n'
                     f"Content-Type: image/jpeg\r\n\r\n".encode())
        parts.append(img)
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        body = b"".join(parts)
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/sendPhoto",
            data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return f"HTTP {r.status}"


class EmailChannel(Channel):
    name = "email"

    def __init__(self, host: str, port: int, user: str, password: str,
                 to: list[str], sender: str = "", use_tls: bool = True,
                 timeout: float = 20.0):
        self.host, self.port = host, port
        self.user, self.password = user, password
        self.to = to
        self.sender = sender or user
        self.use_tls = use_tls
        self.timeout = timeout

    def send(self, ev: dict, snapshot: str = "") -> str:
        msg = EmailMessage()
        msg["Subject"] = f"[{ev['kind'].upper()}] {ev['camera']} - PyroSense alert"
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.to)
        msg.set_content(
            f"{ev['kind'].upper()} detected on camera {ev['camera']}\n"
            f"Time:       {ev.get('started_iso','')}\n"
            f"Zone:       {ev.get('zone') or 'unspecified'}\n"
            f"Confidence: {ev['score']:.0%}\n"
            f"Stage:      {ev.get('stage','')}\n\n"
            f"Evidence:   {ev.get('reason','')}\n\n"
            f"Assessment: {ev.get('adjudication','(not adjudicated)')}\n\n"
            f"Event id:   {ev['id']}\n")
        if snapshot and os.path.exists(snapshot):
            with open(snapshot, "rb") as f:
                msg.add_attachment(f.read(), maintype="image", subtype="jpeg",
                                   filename=os.path.basename(snapshot))
        with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as s:
            if self.use_tls:
                s.starttls()
            if self.user:
                s.login(self.user, self.password)
            s.send_message(msg)
        return f"sent to {len(self.to)} recipient(s)"


class MQTTChannel(Channel):
    """For sites that already run an MQTT broker - most modern BMS and SCADA
    integrations land here, and it is the cleanest way to drive a physical
    sounder, beacon or relay."""
    name = "mqtt"

    def __init__(self, host: str, port: int = 1883, topic: str = "pyrosense/alerts",
                 user: str = "", password: str = ""):
        self.host, self.port, self.topic = host, port, topic
        self.user, self.password = user, password

    def send(self, ev: dict, snapshot: str = "") -> str:
        import paho.mqtt.publish as publish       # optional dependency
        auth = {"username": self.user, "password": self.password} if self.user else None
        publish.single(self.topic, json.dumps(ev), qos=1, hostname=self.host,
                       port=self.port, auth=auth)
        return f"published to {self.topic}"


# --------------------------------------------------------------------------
class AlertDispatcher:
    """Queued, retrying, non-blocking fan-out to every configured channel."""

    def __init__(self, channels: list[Channel] | None = None, retries: int = 3,
                 dry_run: bool = False):
        self.channels = channels or []
        self.retries = retries
        self.dry_run = dry_run
        self.log: list[DeliveryResult] = []
        self._q: queue.Queue = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="alert-dispatch")
        self._thread.start()

    def dispatch(self, ev: dict, snapshot: str = "") -> None:
        try:
            self._q.put_nowait((ev, snapshot))
        except queue.Full:
            self.log.append(DeliveryResult("dispatcher", False, "queue full - dropped"))

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                ev, snap = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            for ch in self.channels:
                self.log.append(self._deliver(ch, ev, snap))
            if len(self.log) > 500:
                del self.log[:-500]

    def _deliver(self, ch: Channel, ev: dict, snap: str) -> DeliveryResult:
        if self.dry_run and ch.name not in ("console", "file"):
            return DeliveryResult(ch.name, True, "dry-run: not actually sent")
        last = ""
        for attempt in range(1, self.retries + 1):
            try:
                return DeliveryResult(ch.name, True, ch.send(ev, snap), attempt)
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        return DeliveryResult(ch.name, False, last, self.retries)

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> list[dict]:
        return [r.to_json() for r in self.log[-40:]][::-1]


def build_channels(cfg: dict) -> list[Channel]:
    """Construct channels from config. Unknown or incomplete entries are skipped
    rather than raising - a typo in the Telegram token must not stop the whole
    detector from starting."""
    out: list[Channel] = [ConsoleChannel(),
                          FileChannel(cfg.get("file", "data/events/alerts.jsonl"))]
    try:
        if cfg.get("webhook", {}).get("url"):
            w = cfg["webhook"]
            out.append(WebhookChannel(w["url"], w.get("headers")))
        if cfg.get("telegram", {}).get("bot_token"):
            t = cfg["telegram"]
            out.append(TelegramChannel(t["bot_token"], t["chat_id"]))
        if cfg.get("email", {}).get("host"):
            e = cfg["email"]
            out.append(EmailChannel(e["host"], int(e.get("port", 587)),
                                    e.get("user", ""), e.get("password", ""),
                                    e.get("to", []), e.get("from", ""),
                                    bool(e.get("tls", True))))
        if cfg.get("mqtt", {}).get("host"):
            m = cfg["mqtt"]
            out.append(MQTTChannel(m["host"], int(m.get("port", 1883)),
                                   m.get("topic", "pyrosense/alerts"),
                                   m.get("user", ""), m.get("password", "")))
    except Exception:
        pass
    return out
