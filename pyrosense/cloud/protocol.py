"""
The edge <-> cloud contract.

ARCHITECTURE, AND WHY THIS ONE
==============================

The camera is on a private LAN behind a router. A cloud server cannot reach it.
There are four ways to bridge that, and only one of them is right for a fire
detector:

1. Tunnel the local console out (ngrok / Cloudflare Tunnel / Tailscale Funnel).
   Zero code. But it publishes the whole operator console to the internet, every
   viewer streams video off a domestic upload link, and the moment the house
   broadband hiccups the public dashboard dies *along with* the thing it is
   monitoring. A dashboard that goes dark exactly when the site loses power is
   the opposite of what a safety product needs.

2. Stream the video to the cloud and run detection there. A single 640x360
   substream is ~0.5 Mbit/s, which is ~5 GB per camera per day, forever, plus
   cloud GPU/CPU to process it. It also means the site's continuous video lives
   on someone else's disk. Expensive and worse in every dimension.

3. WebRTC relay for true live video. Signalling, STUN/TURN, a media server, and
   real bandwidth cost - to deliver something nobody watching a fire dashboard
   actually needs, which is 30fps.

4. >>> Run detection at the edge; push telemetry and stills OUT to the cloud. <<<

We do (4), and the properties that matter are:

  Outbound only.      The agent dials out over HTTPS. No port forwarding, no
                      inbound firewall rule, no dynamic DNS, and it works behind
                      CGNAT where options 1-3 need extra machinery.

  Cheap.              A 30 KB JPEG every 4 s plus a small heartbeat is roughly
                      20 KB/s, about 1.6 GB/month for one camera - two orders of
                      magnitude below streaming the video.

  The cloud is a MIRROR, never a dependency. Detection, adjudication and the
                      urgent alert ladder all run locally. If Render is down, or
                      the broadband is out, the system still detects fire and
                      still rings your phone. The public dashboard is how other
                      people watch; it is not in the safety path.

  Staleness is data.  Because the cloud only ever knows what was last pushed to
                      it, "I have not heard from the site in 90 seconds" is a
                      first-class state that the dashboard shows loudly, and that
                      the cloud watchdog escalates on. A monitoring page that
                      shows a cheerful green "all clear" because it is frozen is
                      actively dangerous, so this design makes that failure mode
                      impossible to render as healthy.

  Privacy.            Only low-rate stills leave the site, and only from cameras
                      you enable. The continuous video never leaves the LAN.

The one real tradeoff: the public page shows a frame every few seconds, not live
motion. On an event it pushes immediately at higher quality, so the thing you
actually care about arrives at full fidelity. For remote true-live video, keep a
Tailscale/WireGuard tunnel to the local console as a private admin path - that is
the correct place for option 1, alongside this rather than instead of it.

DOWNLINK
========
The heartbeat response carries commands back to the agent. That is how a cloud
dashboard button reaches a machine it cannot address: the agent asks every few
seconds, and the answer contains any pending acknowledgements. No inbound
connection is ever required.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field, asdict

PROTOCOL_VERSION = 1

# Wire limits, enforced on both sides so a bug at the edge cannot fill cloud RAM.
MAX_FRAME_BYTES = 1_500_000
MAX_EVENT_BYTES = 3_000_000
MAX_CLIP_BYTES = 12_000_000
MAX_CLOCK_SKEW_S = 300


@dataclass
class Telemetry:
    """One heartbeat from the site."""
    site: str
    agent_version: str
    sent_at: float
    uptime_s: float
    cameras: list = field(default_factory=list)
    adjudicator: dict = field(default_factory=dict)
    neural: dict = field(default_factory=dict)
    alerting: dict = field(default_factory=dict)
    events_total: int = 0
    dry_run: bool = True

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class EventPayload:
    id: str
    camera: str
    kind: str
    score: float
    started: float
    reason: str
    adjudication: str = ""
    verdict: str = "pending"
    zone: str = ""
    features: dict = field(default_factory=dict)
    snapshot_b64: str = ""
    sent: bool = True
    escalation: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def sign(secret: str, body: bytes, ts: str) -> str:
    """HMAC-SHA256 over the timestamp and body.

    TLS already protects this in transit; the signature is here so that the token
    itself is never replayable against a different body, and so a proxy log that
    captures a URL cannot be used to forge an event. `compare` is constant-time."""
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256)
    return mac.hexdigest()


def verify(secret: str, body: bytes, ts: str, signature: str,
           now: float | None = None) -> tuple[bool, str]:
    """Returns (ok, reason). Rejects stale timestamps to bound replay."""
    if not signature or not ts:
        return False, "missing signature"
    try:
        t = float(ts)
    except ValueError:
        return False, "bad timestamp"
    now = now if now is not None else time.time()
    if abs(now - t) > MAX_CLOCK_SKEW_S:
        return False, f"timestamp skew {abs(now - t):.0f}s exceeds {MAX_CLOCK_SKEW_S}s"
    if not hmac.compare_digest(sign(secret, body, ts), signature):
        return False, "signature mismatch"
    return True, ""
