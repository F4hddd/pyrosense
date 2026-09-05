# Deploying PyroSense

Two pieces, and the split between them is the whole design:

```
    YOUR PREMISES                                  THE INTERNET
    ---------------------------------              --------------------------
    camera (RTSP, private IP)
        |  LAN only, never leaves
        v
    site machine  ── detection ──> ALERTS ─────────> your phone
    (mini-PC / laptop)                               (Pushover / ntfy / call)
        |
        |  outbound HTTPS only, ~20 KB/s
        v
    ................................................> Render: public mirror
                                                      (read-only dashboard)
```

**Detection and alerting run on the site machine.** The cloud is a mirror. If
Render is down, or your broadband is out, the system still detects fire and still
rings your phone. Nothing in the cloud is in the safety path — except the
watchdog, which alerts you when the *site* goes quiet.

---

## Why push, not a tunnel

Your camera is on `192.168.100.x` behind a router. A cloud server cannot reach
it. Four ways to bridge that, and one right answer:

| Approach | Verdict |
|---|---|
| Tunnel the local console out (ngrok, Cloudflare Tunnel) | Publishes your whole operator console; every viewer streams video off your domestic upload; and the public page dies exactly when the site loses power — precisely when someone would check it |
| Stream video to the cloud, detect there | ~5 GB/camera/day forever, plus cloud compute, plus your continuous video on someone else's disk |
| WebRTC relay | Signalling, STUN/TURN, a media server, real bandwidth — to deliver 30fps that nobody watching a fire dashboard needs |
| **Push telemetry + stills out** | **Outbound-only (works behind CGNAT, no port forwarding), ~1.6 GB/month, video never leaves the LAN, and the cloud is not a dependency** |

The one tradeoff: the public page shows a frame every few seconds, not live
motion. On an event it pushes immediately at higher quality. If you want true
live video remotely, add a Tailscale/WireGuard tunnel to the local console as a
*private admin path* — alongside this, not instead of it.

---

## Part 1 — the cloud mirror (Render)

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"   # your ingest token
```

1. Push this repo to GitHub.
2. Render → **New → Blueprint** → select the repo. It reads `render.yaml`.
3. Set these in the Render dashboard (they are `sync: false`, so they never enter git):

| Variable | Value |
|---|---|
| `PYRO_INGEST_TOKEN` | the token you just generated |
| `PYRO_ACK_PIN` | a short PIN, so a passer-by cannot silence your fire alarm |
| `PYRO_PUBLIC_TOKEN` | optional — set it to make the dashboard private (`?k=...`) |
| `PUSHOVER_TOKEN` / `PUSHOVER_USER` | so the **watchdog** can alert you when the site goes silent |

You get `https://pyrosense-mirror-xxxx.onrender.com`. That is the public URL.

**On Render's free tier**, services spin down after ~15 minutes idle and cold-start
in ~30s. The agent's 5-second heartbeat keeps it awake in practice, but a cold
start still costs you the first request after any gap. The `starter` plan removes
the behaviour entirely; for a safety dashboard that is worth the few dollars.

**Privacy:** by default anyone with the URL sees stills of your premises. Set
`PYRO_PUBLIC_TOKEN` unless you genuinely want it open.

---

## Part 2 — the site machine

```bash
pip install -r requirements-local.txt
python -m pyrosense init                    # writes pyrosense.config.json
```

### Find the camera — do not hardcode the IP

Your camera was at `192.168.100.8`; when I checked, that address had been
reassigned (its ARP entry showed a randomised, phone-like MAC) and the RTSP
server had moved. DHCP leases move. So:

```bash
python -m pyrosense discover --cidr 192.168.100.0/24
python -m pyrosense probe --host <ip> --user admin --password '<password>'
```

`probe` tries the brand's documented path first, then sweeps common ones, and
verifies each by actually decoding a frame. It prints a config snippet.

> **Careful:** Dahua/Imou firmware locks the account after a handful of failed
> logins. If `probe` returns 401, stop and confirm the password in the Imou app
> rather than guessing — a lockout takes 5–30 minutes to clear.

Best fix long-term: give the camera a **DHCP reservation** in your router so the
address stops moving.

### Configure

```jsonc
{
  "site": "Home Workshop",
  "dashboard_url": "https://pyrosense-mirror-xxxx.onrender.com",
  "cloud": {
    "enabled": true,
    "url": "https://pyrosense-mirror-xxxx.onrender.com",
    "token": "${PYRO_INGEST_TOKEN}",
    "heartbeat_s": 5, "frame_idle_s": 4, "frame_active_s": 1
  },
  "urgent": {
    "dry_run": false,
    "pushover": { "token": "${PUSHOVER_TOKEN}", "user": "${PUSHOVER_USER}" },
    "ntfy":     { "topic": "${NTFY_TOPIC}", "priority": 5 },
    "twilio":   { "account_sid": "${TWILIO_ACCOUNT_SID}",
                  "auth_token": "${TWILIO_AUTH_TOKEN}",
                  "from_number": "${TWILIO_FROM_NUMBER}",
                  "call_numbers": ["${ALERT_PHONE_PRIMARY}"] },
    "ladder":   { "call_after_s": 90, "second_call_after_s": 300 }
  },
  "cameras": [{
    "name": "workshop-01", "location": "Bench",
    "source": { "kind": "rtsp", "brand": "dahua", "host": "192.168.100.6",
                "user": "admin", "password": "${CAM_PASSWORD}", "stream": "sub" }
  }]
}
```

`${VAR}` reads from the environment, so no password is ever in the file.

```bash
export CAM_PASSWORD='...'      PYRO_INGEST_TOKEN='...'
export PUSHOVER_TOKEN='...'    PUSHOVER_USER='...'
python -m pyrosense run -c pyrosense.config.json
```

Local console on `http://localhost:8080`, public mirror updates within seconds.

### Keep it running

`systemd` on Linux, or Task Scheduler on Windows ("run whether user is logged on
or not", "restart on failure"). The cloud watchdog covers you if it dies anyway.

---

## Part 3 — the alert ladder

| When | What happens |
|---|---|
| t=0 | Pushover **emergency** (repeats every 30s until acknowledged, sounds through silent/DND as an iOS Critical Alert) + ntfy priority 5 + webhook/email/MQTT |
| t+90s | Automated **phone call** if still unacknowledged |
| t+300s | Second call |
| t+30min | Gives up, records that nobody responded |

Acknowledge from the Pushover notification, the public dashboard, or the local
console — all three land in one place and stop the whole ladder.

**Escalation is only ever cancelled by a human.** It is never stood down because
the fire "looks like it stopped" — a detector that cleared itself because the
camera filled with smoke would be lethal.

### Setting up the channels

**Pushover** (~5 min, $5 once per platform — the primary channel):
1. Install the app, sign up at pushover.net, copy your **User Key**.
2. Create an Application → copy the **API Token**.
3. Set `PUSHOVER_USER` and `PUSHOVER_TOKEN`.
4. On iOS: allow **Critical Alerts** for Pushover in notification settings — this
   is what makes it sound through the ringer switch.

**ntfy** (free, ~2 min — the redundant path): install the app, subscribe to an
**unguessable** topic name, set `NTFY_TOPIC`. On the public server the topic name
*is* the credential, so treat it like a password.

**Twilio voice** (~15 min, a few cents a call — the escalation): sign up, buy a
number, set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`,
`ALERT_PHONE_PRIMARY`. On a trial account you must verify the destination number
first.

### Test it before you rely on it

```bash
python -m pyrosense test-alert -c pyrosense.config.json            # dry run
python -m pyrosense test-alert -c pyrosense.config.json --live     # really send
```

It raises a fake incident, walks the ladder, and prints each step. Acknowledge on
your phone and watch escalation stop. **Do this at night, with the phone on
silent, in your pocket.** An alert path you have not tested under the exact
conditions it exists for is not an alert path.

---

## Findings from this site (Imou, 192.168.100.8)

**The substream needs the ONVIF path.** The bare Dahua substream path completes
the RTSP handshake and then never delivers parseable data - ffprobe and ffmpeg
both sit until they time out. This works immediately:

```
rtsp://admin:<SC>@192.168.100.8:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif
```

Measured: substream `h264 640x352 @15fps`, main stream `hevc 2880x1620 @25fps`.
There is now an `imou` brand profile that uses this path by default, and the
probe sweep tries it first. The RTSP password is the **SC / security code** on
the device label, not the Imou cloud password.

**The motion gate does not help on this camera.** Scene noise gives a median
`motion_ratio` of 0.073 - about 90x the 0.0008 gate threshold - so the gate never
closes and saves nothing. At one camera and ~13 ms/frame that is irrelevant.
At twenty cameras it matters, so raise `motion_threshold` toward 0.10 for this
camera and re-measure before scaling out.

**Smoke detection is OFF on this camera, deliberately.** The view is pale
sheeting, dust and bare surfaces: low saturation, mid brightness, soft edges -
which is precisely the smoke mask's signature. A 120-second baseline looked fine
(smoke peaked at 0.562, never reaching 0.70), but within minutes of going live a
scene change produced a **0.98 false positive** with `growth=1.00,
internal_updraft=0.60, persistence=5.1s`. That is not fixable by nudging a
threshold.

Flame detection on the same camera measured **0.000 at every percentile over 530
frames** and is left on.

To re-enable smoke here, in order of value:
1. Set `ANTHROPIC_API_KEY` so stage 3 adjudicates. This is exactly the case it
   exists for - a vision model asked "is this smoke or pale sheeting?" answers
   correctly, and this alert would have been suppressed with a reason.
2. Draw an `exclude` zone over the static sheeting.
3. Run a week of dry-run recording and re-measure the updraft distribution the
   way `detect/cascade.py` documents, then set the veto line from *your* numbers.

**Stage 3 is currently failing open, correctly.** With no API key the adjudicator
errors and the alert is sent anyway, labelled `sent without adjudication
(TypeError: Could not resolve authentication method) - stage-1 confidence 0.98`.
That is the designed behaviour: a fire detector must never be silenced by a
missing credential. But it does mean every alert is currently unfiltered.

---

## What is verified, and what is not

Verified end to end on the bench:

- Ingest auth rejects no token, a wrong token, and a valid token with no signature
- Telemetry, frames and events reach the mirror; frame bytes are valid JPEG
- **Ack round-trip through NAT**: public dashboard → heartbeat downlink → local
  escalation cancelled, in ~2 seconds
- **Store-and-forward**: with the cloud dead, events are on disk within 1s and all
  are delivered after recovery
- Agent reconnects by itself when the mirror returns
- Health degrades LIVE → STALE (21s) → OFFLINE (62s), and the offline page refuses
  to show stale scores
- Event history survives a mirror restart when a disk is mounted

Verified against the real Imou camera at 192.168.100.8:

- 640x352 h264 substream pulled through the ffmpeg reader: 1966 frames, 18 fps,
  zero reconnects
- Detector cost 12.9 ms/frame with smoke on, 5.1 ms/frame with smoke off
- Flame channel clean: 0.000 at p50/p90/p99/max across 530 frames
- Telemetry and stills reaching the mirror with 0 uplink failures
- Fail-open confirmed in production: stage 3 unavailable, alert still delivered
  with the reason stated

Not yet verified, because it needs your accounts:

- Pushover / ntfy / Twilio actually firing — `test-alert --live` is the check
- Behaviour of your specific phone under Do Not Disturb
- Stage 3 adjudication (needs ANTHROPIC_API_KEY)
- Detection of an actual fire. Nothing here has seen one on your hardware.
