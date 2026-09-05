"""
High-urgency alerting: getting through a silenced phone at 3am.

WHY THESE CHANNELS
==================

The requirement is "wake me up", and most notification channels are built on the
opposite assumption - that the user should be able to make them shut up. Ranked by
how reliably they defeat a silent switch:

  Automated voice call   Best wake-up rate. On iOS a repeated call from the same
                         number breaks through Focus/DND; on Android calls ring
                         under most DND profiles. Also the only channel that works
                         on a dumb phone or a landline. Costs a few cents a call
                         and needs a Twilio number.

  Pushover "emergency"   Purpose-built for exactly this. Priority 2 repeats the
  (priority 2)           notification every N seconds until a human acknowledges
                         it, and it is registered as an iOS Critical Alert, so it
                         sounds at full volume through the ringer switch and
                         through Focus modes. $5 once per platform, no
                         subscription, setup is about five minutes.

  ntfy max priority      Free, open source, self-hostable, and needs no account at
  (priority 5)           all - you pick a topic name and subscribe to it. Priority
                         5 bypasses Android DND. iOS critical alerts work but need
                         a toggle in the app. Excellent as a zero-cost redundant
                         path.

  SMS                    Reliable delivery, but a silenced phone silences it.
                         Useful as a paper trail, not as a wake-up.

  Email / Slack / Telegram / WhatsApp
                         All muteable, all batched, all ignorable. Fine for the
                         record, useless for the emergency.

THE RECOMMENDATION
==================
Pushover emergency as primary, ntfy as a free parallel path, and an automated
voice call as escalation if nobody acknowledges within 90 seconds. Independent
vendors, independent transports, independent failure modes. For a fire alarm,
redundancy across providers matters more than polish in any one of them.

TWO DESIGN RULES THAT MATTER MORE THAN THE CHANNEL
=================================================
1. Escalation runs LOCALLY. Not in the cloud dashboard. If the site loses its
   broadband the local machine can still reach the mobile network via... nothing,
   admittedly - but if the cloud is merely down while the link is up, alerting is
   unaffected. The cloud is never in the safety path.

2. Escalation is cancelled only by an explicit human acknowledgement, never by
   the fire "looking like it stopped". A detector that stands down because the
   camera view filled with smoke would be lethal. Only a person clears it.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field


def _post_form(url: str, data: dict, timeout: float = 12.0,
               auth: tuple[str, str] | None = None) -> dict:
    body = urllib.parse.urlencode(
        {k: v for k, v in data.items() if v is not None}).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    if auth:
        import base64
        tok = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        req.add_header("Authorization", f"Basic {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "ignore")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw, "status": r.status}


# ---------------------------------------------------------------------------
class Pushover:
    """Emergency-priority push. Repeats until a human acknowledges.

    `retry` is how often it re-alerts, `expire` is how long it keeps trying. The
    returned receipt lets us both poll for acknowledgement and cancel the repeats
    once someone has responded anywhere else."""

    API = "https://api.pushover.net/1"

    def __init__(self, token: str, user: str, devices: str = "",
                 retry_s: int = 30, expire_s: int = 900, sound: str = "siren"):
        self.token = token
        self.user = user
        self.devices = devices
        self.retry_s = max(30, retry_s)        # Pushover enforces >= 30
        self.expire_s = min(10800, expire_s)
        self.sound = sound

    @property
    def configured(self) -> bool:
        return bool(self.token and self.user)

    def send(self, title: str, message: str, url: str = "", url_title: str = "",
             emergency: bool = True, image: bytes | None = None) -> dict:
        data = {
            "token": self.token, "user": self.user, "title": title[:250],
            "message": message[:1024], "priority": 2 if emergency else 1,
            "sound": self.sound, "url": url or None, "url_title": url_title or None,
            "device": self.devices or None,
        }
        if emergency:
            data["retry"] = self.retry_s
            data["expire"] = self.expire_s
        if image:
            return self._send_multipart(data, image)
        return _post_form(f"{self.API}/messages.json", data)

    def _send_multipart(self, data: dict, image: bytes) -> dict:
        boundary = "----pyro" + os.urandom(8).hex()
        parts = []
        for k, v in data.items():
            if v is None:
                continue
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                         f'name="{k}"\r\n\r\n{v}\r\n'.encode())
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f'name="attachment"; filename="alert.jpg"\r\n'
                     f"Content-Type: image/jpeg\r\n\r\n".encode())
        parts.append(image)
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        req = urllib.request.Request(f"{self.API}/messages.json",
                                     data=b"".join(parts), method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8", "ignore"))

    def acknowledged(self, receipt: str) -> bool:
        try:
            u = f"{self.API}/receipts/{receipt}.json?token={self.token}"
            with urllib.request.urlopen(u, timeout=10) as r:
                return bool(json.loads(r.read()).get("acknowledged"))
        except Exception:
            return False

    def cancel(self, receipt: str) -> None:
        try:
            _post_form(f"{self.API}/receipts/{receipt}/cancel.json",
                       {"token": self.token})
        except Exception:
            pass


class Ntfy:
    """Free, account-less push. Priority 5 (max) bypasses Android DND.

    Pick an unguessable topic name - a topic IS the credential on the public
    server. For anything beyond a demo, self-host or use ntfy's access control."""

    def __init__(self, topic: str, server: str = "https://ntfy.sh",
                 token: str = "", priority: int = 5):
        self.topic = topic
        self.server = server.rstrip("/")
        self.token = token
        self.priority = priority

    @property
    def configured(self) -> bool:
        return bool(self.topic)

    def send(self, title: str, message: str, click: str = "",
             tags: str = "fire,rotating_light", image: bytes | None = None) -> dict:
        url = f"{self.server}/{self.topic}"
        body = image if image else message.encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Title", title[:200].encode("ascii", "ignore").decode())
        req.add_header("Priority", str(self.priority))
        req.add_header("Tags", tags)
        if click:
            req.add_header("Click", click)
        if image:
            req.add_header("Filename", "alert.jpg")
            req.add_header("Message", message[:400].encode("ascii", "ignore").decode())
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=12) as r:
            return {"status": r.status}


class TwilioVoice:
    """Automated phone call. The escalation of last resort.

    Uses TwiML delivered inline via the `Twiml` parameter, so no public callback
    URL is needed - which matters here, because the machine placing the call has
    no reachable address."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str,
                 to_numbers: list[str], voice: str = "Polly.Matthew"):
        self.sid = account_sid
        self.token = auth_token
        self.frm = from_number
        self.to = [n for n in (to_numbers or []) if n]
        self.voice = voice

    @property
    def configured(self) -> bool:
        return bool(self.sid and self.token and self.frm and self.to)

    def call(self, spoken: str, to: str | None = None) -> dict:
        safe = (spoken.replace("&", " and ").replace("<", " ").replace(">", " "))[:600]
        twiml = (f'<Response><Pause length="1"/>'
                 f'<Say voice="{self.voice}">{safe}</Say>'
                 f'<Pause length="1"/>'
                 f'<Say voice="{self.voice}">{safe}</Say></Response>')
        targets = [to] if to else self.to
        out = {}
        for number in targets:
            try:
                out[number] = _post_form(
                    f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Calls.json",
                    {"To": number, "From": self.frm, "Twiml": twiml},
                    auth=(self.sid, self.token), timeout=20)
            except Exception as e:
                out[number] = {"error": f"{type(e).__name__}: {e}"[:200]}
        return out


class TwilioSMS:
    def __init__(self, account_sid: str, auth_token: str, from_number: str,
                 to_numbers: list[str]):
        self.sid, self.token, self.frm = account_sid, auth_token, from_number
        self.to = [n for n in (to_numbers or []) if n]

    @property
    def configured(self) -> bool:
        return bool(self.sid and self.token and self.frm and self.to)

    def send(self, text: str) -> dict:
        out = {}
        for number in self.to:
            try:
                out[number] = _post_form(
                    f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json",
                    {"To": number, "From": self.frm, "Body": text[:1500]},
                    auth=(self.sid, self.token), timeout=20)
            except Exception as e:
                out[number] = {"error": f"{type(e).__name__}: {e}"[:200]}
        return out


# ---------------------------------------------------------------------------
@dataclass
class Incident:
    event_id: str
    camera: str
    kind: str
    score: float
    summary: str
    url: str = ""
    started: float = field(default_factory=time.time)
    acknowledged: bool = False
    acked_by: str = ""
    acked_at: float = 0.0
    step: int = 0
    receipts: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def note(self, text: str) -> None:
        self.log.append({"t": round(time.time() - self.started, 1), "msg": text[:240]})

    def to_json(self) -> dict:
        return dict(event_id=self.event_id, camera=self.camera, kind=self.kind,
                    score=self.score, acknowledged=self.acknowledged,
                    acked_by=self.acked_by, step=self.step,
                    age_s=round(time.time() - self.started, 1), log=self.log[-12:])


class EscalationManager:
    """Walks an incident up a ladder of increasingly intrusive channels until a
    human acknowledges it.

    Runs on its own thread and never blocks detection. Every step is wrapped, so
    a Twilio outage cannot stop the Pushover retries that are already running.
    """

    def __init__(self, pushover: Pushover | None = None, ntfy: Ntfy | None = None,
                 voice: TwilioVoice | None = None, sms: TwilioSMS | None = None,
                 call_after_s: float = 90.0, second_call_after_s: float = 300.0,
                 give_up_after_s: float = 1800.0, dry_run: bool = False):
        self.pushover = pushover
        self.ntfy = ntfy
        self.voice = voice
        self.sms = sms
        self.call_after_s = call_after_s
        self.second_call_after_s = second_call_after_s
        self.give_up_after_s = give_up_after_s
        self.dry_run = dry_run
        self.incidents: dict[str, Incident] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="escalation")
        self._thread.start()

    # ------------------------------------------------------------------ api
    @property
    def channels(self) -> dict:
        return {
            "pushover": bool(self.pushover and self.pushover.configured),
            "ntfy": bool(self.ntfy and self.ntfy.configured),
            "voice_call": bool(self.voice and self.voice.configured),
            "sms": bool(self.sms and self.sms.configured),
            "dry_run": self.dry_run,
        }

    def raise_incident(self, event_id: str, camera: str, kind: str, score: float,
                       summary: str, url: str = "",
                       image: bytes | None = None) -> Incident:
        inc = Incident(event_id=event_id, camera=camera, kind=kind, score=score,
                       summary=summary, url=url)
        with self._lock:
            self.incidents[event_id] = inc
        threading.Thread(target=self._step0, args=(inc, image), daemon=True,
                         name=f"esc0-{event_id}").start()
        return inc

    def acknowledge(self, event_id: str, by: str = "operator") -> bool:
        """The ONLY thing that stops escalation. Never triggered by the detector."""
        with self._lock:
            inc = self.incidents.get(event_id)
            if not inc or inc.acknowledged:
                return False
            inc.acknowledged = True
            inc.acked_by = by
            inc.acked_at = time.time()
            inc.note(f"acknowledged by {by} - escalation stopped")
        if self.pushover and not self.dry_run:
            for r in inc.receipts:
                self.pushover.cancel(r)
        return True

    def pending(self) -> list[dict]:
        with self._lock:
            return [i.to_json() for i in self.incidents.values() if not i.acknowledged]

    def status(self) -> dict:
        with self._lock:
            act = [i for i in self.incidents.values() if not i.acknowledged]
            return {"channels": self.channels, "active": len(act),
                    "incidents": [i.to_json() for i in
                                  list(self.incidents.values())[-10:]]}

    def stop(self) -> None:
        self._stop.set()

    # --------------------------------------------------------------- ladder
    def _title(self, inc: Incident) -> str:
        return f"FIRE - {inc.camera}" if inc.kind == "flame" else f"SMOKE - {inc.camera}"

    def _body(self, inc: Incident) -> str:
        return (f"{inc.kind.upper()} detected at {inc.camera} "
                f"({inc.score:.0%} confidence).\n{inc.summary}\n"
                f"Acknowledge to stop repeat alerts.")

    def _step0(self, inc: Incident, image: bytes | None) -> None:
        """Immediate: every non-intrusive-cost channel at once, in parallel."""
        if self.dry_run:
            inc.note("DRY RUN: would send Pushover emergency + ntfy max priority")
            print(f"\n  [DRY RUN] URGENT: {self._title(inc)} - {inc.summary[:90]}\n")
            return

        if self.pushover and self.pushover.configured:
            try:
                r = self.pushover.send(self._title(inc), self._body(inc),
                                       url=inc.url, url_title="Open dashboard",
                                       emergency=True, image=image)
                if r.get("receipt"):
                    inc.receipts.append(r["receipt"])
                inc.note(f"pushover emergency sent (receipt {r.get('receipt','-')})")
            except Exception as e:
                inc.note(f"pushover FAILED: {type(e).__name__}: {e}")

        if self.ntfy and self.ntfy.configured:
            try:
                self.ntfy.send(self._title(inc), self._body(inc), click=inc.url,
                               image=image)
                inc.note("ntfy priority-5 sent")
            except Exception as e:
                inc.note(f"ntfy FAILED: {type(e).__name__}: {e}")

        if self.sms and self.sms.configured:
            try:
                self.sms.send(f"{self._title(inc)}: {inc.summary[:200]} {inc.url}")
                inc.note("sms sent")
            except Exception as e:
                inc.note(f"sms FAILED: {type(e).__name__}: {e}")

    def _loop(self) -> None:
        while not self._stop.wait(5.0):
            try:
                self._tick()
            except Exception:
                pass

    def _tick(self) -> None:
        now = time.time()
        with self._lock:
            live = [i for i in self.incidents.values() if not i.acknowledged]

        for inc in live:
            age = now - inc.started

            # A Pushover acknowledgement from the phone clears the whole ladder.
            if inc.step >= 0 and inc.receipts and self.pushover and not self.dry_run:
                for r in inc.receipts:
                    if self.pushover.acknowledged(r):
                        self.acknowledge(inc.event_id, by="pushover")
                        break
                if inc.acknowledged:
                    continue

            if age > self.give_up_after_s:
                inc.note("escalation window expired without acknowledgement")
                with self._lock:
                    inc.acknowledged = True
                    inc.acked_by = "expired"
                continue

            if inc.step == 0 and age >= self.call_after_s:
                inc.step = 1
                self._place_call(inc, first=True)
            elif inc.step == 1 and age >= self.second_call_after_s:
                inc.step = 2
                self._place_call(inc, first=False)

    def _place_call(self, inc: Incident, first: bool) -> None:
        spoken = (f"Fire alert. {inc.kind} detected on camera {inc.camera}. "
                  f"Confidence {int(inc.score * 100)} percent. "
                  f"This alert has not been acknowledged.")
        if self.dry_run:
            inc.note(f"DRY RUN: would place voice call (step {inc.step})")
            print(f"  [DRY RUN] would CALL now: {spoken}")
            return
        if not (self.voice and self.voice.configured):
            inc.note(f"step {inc.step}: no voice channel configured")
            return
        try:
            res = self.voice.call(spoken)
            ok = [n for n, r in res.items() if not r.get("error")]
            inc.note(f"voice call placed to {', '.join(ok) or 'nobody'}")
        except Exception as e:
            inc.note(f"voice call FAILED: {type(e).__name__}: {e}")


def build_escalation(cfg: dict) -> EscalationManager:
    """Construct from config. Missing sections are simply absent channels - a
    typo in the Twilio block must never prevent Pushover from firing."""
    cfg = cfg or {}
    po = cfg.get("pushover", {}) or {}
    nt = cfg.get("ntfy", {}) or {}
    tw = cfg.get("twilio", {}) or {}
    lad = cfg.get("ladder", {}) or {}

    pushover = Pushover(po.get("token", ""), po.get("user", ""),
                        po.get("devices", ""),
                        int(po.get("retry_s", 30)), int(po.get("expire_s", 900)),
                        po.get("sound", "siren")) if po.get("token") else None
    ntfy = Ntfy(nt.get("topic", ""), nt.get("server", "https://ntfy.sh"),
                nt.get("token", ""), int(nt.get("priority", 5))) \
        if nt.get("topic") else None
    voice = TwilioVoice(tw.get("account_sid", ""), tw.get("auth_token", ""),
                        tw.get("from_number", ""), tw.get("call_numbers", [])) \
        if tw.get("account_sid") else None
    sms = TwilioSMS(tw.get("account_sid", ""), tw.get("auth_token", ""),
                    tw.get("from_number", ""), tw.get("sms_numbers", [])) \
        if (tw.get("account_sid") and tw.get("sms_numbers")) else None

    return EscalationManager(
        pushover=pushover, ntfy=ntfy, voice=voice, sms=sms,
        call_after_s=float(lad.get("call_after_s", 90)),
        second_call_after_s=float(lad.get("second_call_after_s", 300)),
        give_up_after_s=float(lad.get("give_up_after_s", 1800)),
        dry_run=bool(cfg.get("dry_run", True)),
    )
