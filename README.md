# PyroSense

AI fire and smoke detection that runs on the CCTV a site already owns.

No new sensors, no vendor SDKs, no cloud upload of video. Point it at the existing
cameras — Imou, Hikvision, Dahua, Reolink, Tapo, Uniview, Axis, or anything else
that speaks RTSP or ONVIF — and it watches them for uncontrolled fire and smoke.

```bash
python -m pyrosense demo          # six virtual cameras, no hardware required
```

---

## The two problems this actually solves

**1. Every site has a different camera stack.**

A workshop typically has three brands installed over a decade by two contractors,
each with its own phone app and its own abandoned Windows SDK. Integrating with
those apps is a treadmill.

Underneath, though, essentially every IP camera sold in the last decade speaks
**RTSP** for video and **ONVIF** for discovery. So PyroSense speaks those, and the
only brand-specific knowledge in the whole system is a table of URL paths
([`cameras/profiles.py`](pyrosense/cameras/profiles.py)) — about forty lines,
instead of forty SDKs. Anything not in the table is handled by asking the camera
for its own stream URL over ONVIF, and failing that, by sweeping the common paths.

**2. Workshops are full of things that look exactly like fire.**

Welding arcs, grinder sparks, hot metal, rotating beacons, halogen lamps and their
reflections, sunlight through a roller door, hi-vis vests. A colour-based detector
alarms on all of them, gets muted within a week, and is then worth nothing.

So detection is built around the fact that **fire is defined by its behaviour over
time, not by its colour**:

| | flicker | periodic | grows | ragged edge | travels | blinks off | internal updraft |
|---|---|---|---|---|---|---|---|
| **fire** | yes | no | yes | yes | no | no | — |
| **smoke** | — | no | yes | yes | no | no | **yes** |
| welding arc | yes | hard on/off | no | yes | no | **yes** | — |
| grinder sparks | yes | no | no | yes | no | yes | — |
| rotating beacon | **yes** | **yes** | no | **no** | no | no | — |
| work lamp | **no** | no | **no** | no | no | no | — |
| sunlight patch | **no** | no | slow | yes | slow | no | **no** |
| hi-vis worker | some | no | apparent | yes | **yes** | no | — |
| forklift (grey) | — | no | apparent | yes | **yes** | no | **no** |

Bold cells are the ones that do the discriminating.

---

## Measured results

The nuisance gauntlet (`python -m pyrosense bench`) renders eleven procedural
workshop scenarios and runs the full pipeline over them. Every nuisance is
deliberately given fire-like colour, so colour alone cannot pass.

```
  scenario                  truth     alarm   t+      peak    result
  ------------------------------------------------------------------
  Pallet stack fire         fire      YES     7.0s    1.00    DETECT
  Early ignition (small)    fire      YES     9.0s    1.00    DETECT
  Smouldering smoke plume   smoke     YES    12.5s    1.00    DETECT
  MIG welding               nuisance  no        -     0.99    clean
  Angle grinder sparks      nuisance  no        -     1.00    clean
  Sunlight through door     nuisance  no        -     0.79    clean
  Forklift crossing         nuisance  no        -     0.71    clean
  Worker in hi-vis          nuisance  no        -     0.61    clean
  Rotating amber beacon     nuisance  no        -     0.36    clean
  Halogen work lamp         nuisance  no        -     0.00    clean
  Quiet workshop (empty)    nuisance  no        -     0.00    clean
  ------------------------------------------------------------------
  detection     3/3        false alarms  0/8       mean time to alarm 9.5s
  stage-1 cost  8.0 ms/frame on one core
```

Note that six of the eight nuisances peak *above* the 0.72 threshold at some
instant. None of them alarms. Peak confidence is one frame; an alarm additionally
requires surviving the vetoes and holding for seconds. That gap is where the work
went.

### Run it continuously, not for twenty seconds

```bash
python -m pyrosense bench --soak     # 60s per scenario, looped
```

The fixed-length run above gave a clean sheet while a real bug was still present.
Looping each scenario the way a live camera is actually watched exposed a sunlit
patch raising a false smoke alarm at t=47s — a failure that needs a track to
outlive the point where the short run simply stops. The soak now passes 11/11, and
it is the run to trust.

**These are synthetic scenes and they are a regression harness, not a field
accuracy claim.** They exist to stop a change that fixes smoke from quietly
breaking welding rejection. Real numbers require real site footage, which is
step one of any deployment (see *Deploying* below).

Measured throughput on a 12-core desktop, detector only, frames pre-rendered:

| threads | aggregate fps | cameras @ 12fps |
|---|---|---|
| 1 | 106 | 9 |
| 2 | 169 | 14 |
| 4 | 239 | 20 |
| 8 | 210 | 18 |

Scaling flattens after about four threads — there is residual GIL contention, so
this is not linear. Roughly **18–20 cameras at 12 fps on one mini-PC**, or about
35 at 6 fps, which is ample for a phenomenon that develops over seconds.

---

## How detection works

A five-stage cascade. Each stage is more expensive and sees fewer frames.

```
Stage 0   motion gate              ~0.1 ms   skips ~70-99% of frames
Stage 1   physics + temporal       ~10 ms    proposes candidates, vetoes nuisances
Stage 2   neural verifier          ~20 ms    only on candidates, only if weights exist
Stage 3   Claude vision            ~1-2 s    only on would-be alerts: a few a day
Stage 4   persistence + hysteresis  free     must hold for seconds before anyone is told
```

**Stage 1** is a plain additive log-odds model, deliberately. Every alert can be
explained term by term, tuned on site by a human, and defended in a safety review.
The features are in [`detect/tracker.py`](pyrosense/detect/tracker.py); the
weights and the hard vetoes are in [`detect/cascade.py`](pyrosense/detect/cascade.py).

An alert reads like this, because the model is transparent:

> `flicker=1.00 (turbulent area oscillation); growth=1.00 (expanding over time);
> continuity=1.00 (burning continuously, never blinks out);
> edge_roughness=1.00 (ragged flame perimeter)`

and a suppression reads like this:

> `suppressed: source blinks fully off 0.25 times/s — an intermittent tool, arc or
> indicator lamp; a fire does not stop`

**Stage 3** sends two crops and the stage-1 measurements to Claude and asks
directly whether this is an uncontrolled fire or one of the dozen workshop things
that look like one. It returns prose an operator can act on rather than a number,
which is the difference between an alarm people respond to and an alarm people
mute. It runs only on candidates that survived everything else — a handful of
calls a day — so it costs far less than a single false call-out.

Critically, **stage 3 can only ever cancel an alert when it is both confident and
specific about what the thing actually is.** An API error, a timeout, a rate
limit, or an "unclear" all result in the alert being sent anyway, with a note that
adjudication did not run. A detector that can be silenced by an unreachable API is
not a fire detector.

---

## Getting it onto a real site

```bash
python -m pyrosense discover
```

Runs an ONVIF multicast probe, sweeps the local /24 for RTSP ports, reads the ARP
table, and fingerprints each device by MAC OUI and HTTP banner. Prints what it
found, the likely brand, and the brand-specific gotcha (Imou needs RTSP enabled in
the app; Tapo needs a *camera account*, not the cloud login; EZVIZ wants the
verification code off the label).

```bash
python -m pyrosense probe --host 192.168.1.64 --user admin --password 'secret'
```

Finds the working RTSP URL by trying the brand's documented path first, then
sweeping the common ones, verifying each by actually decoding a frame with
ffprobe. Prints a config snippet.

```bash
python -m pyrosense init                       # write an example config
python -m pyrosense run -c pyrosense.config.json
```

The console is at `http://<host>:8080`.

### Deployment order that actually works

1. **Record first, alert never.** Run with `alerts.dry_run: true` for a week. It
   logs everything and sends nothing.
2. **Review the queue.** Every event carries a snapshot, a 10-second clip that
   starts *before* detection, the feature vector, and the adjudication. Mark each
   one real or false in the console.
3. **Draw zones.** The single highest-leverage false-alarm control is a polygon
   around the welding bay, plus a schedule that arms it only outside shift hours.
   No algorithm competes with that.
4. **Go live**, and keep the review queue. Every correction is a labelled training
   crop; after a few weeks you have a site-specific stage-2 model
   (`detect/neural.py:export_training_crops`) worth far more than anything
   pre-trained on outdoor bonfires.

---

## Deploying it for real

Two pieces: detection runs on a machine at the site, and a **public read-only
mirror** runs in the cloud. The site pushes telemetry and stills outbound over
HTTPS, so no port forwarding, no dynamic DNS, and it works behind CGNAT. The
cloud is never in the safety path - if it is down, detection and the alert ladder
carry on.

```bash
python -m pyrosense cloud            # the hosted mirror, locally
python -m pyrosense run -c cfg.json  # the site machine
python -m pyrosense test-alert --live  # prove the alert path before trusting it
```

When a fire is confirmed the alert ladder runs: Pushover **emergency** (repeats
until acknowledged, sounds through silent/DND) and ntfy priority 5 immediately,
then an automated **phone call** at 90 seconds if nobody has acknowledged, and a
second at five minutes. Only a human acknowledgement stops it. A cloud-side
watchdog separately alerts you if the site itself goes silent.

Every alert records a clip that spans **before and after** detection - a 10s
pre-roll held in memory plus 8s captured after the trigger - so reviewing an alert
answers both "what led up to this" and "did it spread". Clips replay inline in
both dashboards and upload to the mirror alongside the event.

Clicking any camera opens a detail view: live feed, per-camera health, score
history, every recording for that camera, and a live smoke on/off switch for
scenes where the smoke channel is a nuisance.

- **[SETUP.md](SETUP.md)** - notification accounts, GitHub, and Render hosting
- **[DEPLOY.md](DEPLOY.md)** - architecture, tradeoffs and the site findings

---

## Configuration

```jsonc
{
  "alerts": {
    "dry_run": true,                        // log locally, send nothing
    "webhook": { "url": "https://...", "headers": {"Authorization": "Bearer ${TOKEN}"} },
    "telegram": { "bot_token": "${TELEGRAM_BOT_TOKEN}", "chat_id": "-100..." },
    "email":    { "host": "smtp...", "to": ["duty@example.com"], "password": "${SMTP_PASSWORD}" },
    "mqtt":     { "host": "192.168.10.5", "topic": "pyrosense/alerts" }
  },
  "vlm": { "enabled": true, "model": "claude-opus-5", "max_calls_per_hour": 60,
           "fail_open": true },
  "cameras": [{
    "name": "weldbay-02",
    "source": { "kind": "rtsp", "brand": "hikvision", "host": "192.168.10.65",
                "user": "admin", "password": "${CAM_PASSWORD}", "stream": "sub" },
    "zones": [
      { "name": "weld-booth", "mode": "exclude",
        "points": [[0.05,0.3],[0.45,0.28],[0.47,0.95],[0.03,0.95]] },
      { "name": "store-area", "mode": "include", "threshold_delta": 0.0,
        "points": [[0.5,0.2],[0.98,0.2],[0.98,0.95],[0.5,0.95]],
        "schedule": { "days": ["mon","tue","wed","thu","fri"],
                      "start": "18:00", "end": "06:00" } }
    ]
  }]
}
```

`${VAR}` is read from the environment, so the config can live in version control
without carrying camera passwords. Credentials are redacted everywhere they could
reach a log line or the UI.

Substreams (`"stream": "sub"`) are the default on purpose: 640×360 at 12 fps is
ample for fire detection and costs roughly 10× less than the main stream. The main
stream is only pulled for the alert snapshot.

---

## Layout

```
pyrosense/
  cameras/profiles.py    brand RTSP path table + fingerprinting + credential redaction
  cameras/discovery.py   ONVIF WS-Discovery, subnet sweep, ARP, path probing
  cameras/source.py      ffmpeg subprocess reader, virtual + file sources
  detect/signals.py      stage 0-1: YCbCr flame rules, smoke rules, region proposal
  detect/tracker.py      temporal features, vectorised (flicker, periodicity, updraft...)
  detect/cascade.py      the log-odds scorer and the hard vetoes
  detect/neural.py       stage 2 ONNX slot (optional) + training-crop export
  detect/vlm.py          stage 3 Claude adjudication + the fail-open policy
  core/zones.py          polygons, arming schedules, per-zone sensitivity
  core/events.py         persistence rule, hysteresis, debounce, rolling pre-roll clip
  core/engine.py         per-camera worker threads + supervisor
  alerts/channels.py     webhook / Telegram / email / MQTT / file, queued and retried
  server/                FastAPI console: REST, MJPEG, websocket push
  synth/scenarios.py     the procedural nuisance gauntlet
  bench.py               regression harness
```

## Requirements

Python 3.10+, `ffmpeg` on PATH, and:

```bash
pip install opencv-python-headless numpy fastapi "uvicorn[standard]" anthropic
```

`onnxruntime` only if you add a stage-2 model. `paho-mqtt` only for MQTT.
Set `ANTHROPIC_API_KEY` to enable stage 3; without it the system runs on stages
0, 1 and 4 and says so in the console.

---

## Honest limitations

- The accuracy numbers above are from synthetic scenes. They are a regression
  harness. Field accuracy needs field footage.
- Smoke is the weaker channel. It is detected here reliably, but with less margin
  than flame, and it is the one most worth routing through stage 3.
- No stage-2 model ships. Public fire datasets are dominated by outdoor bonfires
  and transfer poorly to a ceiling camera in a dim aisle; shipping one would add a
  confident number without adding accuracy. The slot is there and the path to
  filling it from your own review queue is built in.
- A camera that cannot see a fire cannot detect it. Coverage, framing and lens
  cleanliness matter more than any algorithm here.
- **This is a detection aid, not a certified fire alarm system.** It does not
  replace smoke detectors, sprinklers, or a compliant fire alarm panel, and it
  should not be presented to an insurer or a fire officer as if it did.
