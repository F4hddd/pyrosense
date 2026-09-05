# Setup: notifications, GitHub, and hosting

Two things you asked for, plus the one blocker on your camera.

---

## Part 1 — What I need from you for notifications

Nothing is wired to your phone yet. `urgent.dry_run` is `true`, so the ladder runs
and logs but sends nothing. Here is exactly what to collect.

### Pushover — the primary channel (do this one first)

Best wake-up rate of any push channel: emergency priority repeats until you
acknowledge, and it registers as an iOS **Critical Alert**, so it sounds through
the ringer switch and Focus modes. $5 once per platform, no subscription.

1. Install **Pushover** on your phone, create an account at
   [pushover.net](https://pushover.net).
2. Your **User Key** is on the dashboard home page — 30 characters.
3. Click **Create an Application/API Token**, name it `PyroSense` → copy the
   **API Token**.
4. **iOS only, and this is the step people miss:** Settings → Notifications →
   Pushover → enable **Critical Alerts**. Without it, silent mode still silences it.

**Send me / set:** `PUSHOVER_USER`, `PUSHOVER_TOKEN`

### ntfy — free redundant path (2 minutes)

Different vendor, different transport, zero cost. Worth having purely so a
Pushover outage is not a silent failure.

1. Install **ntfy**, subscribe to a topic.
2. Pick something **unguessable** — on the public server the topic name *is* the
   password. `pyrosense-a7f3k9x2qm`, not `fire`.
3. Android: long-press the topic → set priority to max so it bypasses Do Not Disturb.

**Send me / set:** `NTFY_TOPIC`

### Twilio voice — the escalation (optional, ~15 min)

Fires at 90 seconds if nobody has acknowledged. A ringing phone beats a
notification when you are properly asleep.

1. Sign up at twilio.com, buy a number (a few dollars/month).
2. Console home shows **Account SID** and **Auth Token**.
3. On a trial account you must **verify** the destination number first.

**Send me / set:** `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`,
`TWILIO_FROM_NUMBER`, `ALERT_PHONE_PRIMARY` (E.164, e.g. `+441234567890`)

### Claude adjudication — highest value for *your* site specifically

Right now every alert is unfiltered, and you have already seen why that matters:
your camera produced a 0.98 false smoke alarm on pale sheeting. Stage 3 is exactly
the thing that catches that — asked "smoke, or sheeting?", it answers correctly
and suppresses with a written reason.

**Send me / set:** `ANTHROPIC_API_KEY` from console.anthropic.com

### Then turn it on

```bash
setx PUSHOVER_USER   "your-user-key"
setx PUSHOVER_TOKEN  "your-app-token"
setx NTFY_TOPIC      "pyrosense-something-unguessable"
setx ANTHROPIC_API_KEY "sk-ant-..."
```

Set `"dry_run": false` under `urgent` in `pyrosense.config.json`, then:

```bash
python -m pyrosense test-alert -c pyrosense.config.json --live
```

It raises a fake incident and walks the ladder, printing each step. Acknowledge on
your phone and watch escalation stop.

> **Do this test at night, phone on silent, in your pocket.** An alert path you
> have not tested under the exact conditions it exists for is not an alert path.
> Everything above is about defeating a silenced phone; the only proof is a
> silenced phone.

---

## Part 2 — GitHub and hosting

### Push to GitHub

`.gitignore` already excludes `data/`, `pyrosense.config.json` and `*.env`, so no
camera password or token goes up. Verify that before your first push.

```bash
cd C:\Users\USER\Documents\pyrosense
git init
git add -A
git status                  # <- CHECK: no pyrosense.config.json, no data/
git commit -m "PyroSense: fire and smoke detection for existing CCTV"
```

Then create an empty repo on github.com (no README, no .gitignore) and:

```bash
git remote add origin https://github.com/<you>/pyrosense.git
git branch -M main
git push -u origin main
```

**Private or public?** The code contains no secrets and nothing site-specific.
Public is fine. Your `pyrosense.config.json` — camera IP, credentials — stays
local either way.

### Deploy the public dashboard on Render

The repo already contains `render.yaml`, so Render configures itself.

1. **render.com** → sign up with GitHub.
2. Either path works:

   **Blueprint** (reads `render.yaml`): New → Blueprint → select repo → Apply.

   **Manual** (New → Web Service) ignores `render.yaml`, so fill in:

   | Field | Value |
   |---|---|
   | Language | `Python 3` |
   | Branch | `main` |
   | Root Directory | *(leave blank)* |
   | Build Command | `pip install -r requirements-cloud.txt` |
   | Start Command | `uvicorn pyrosense.cloud.server:app --host 0.0.0.0 --port $PORT` |

   On the **Free** plan there is no persistent disk, so leave `PYRO_DATA_DIR`
   unset: the mirror keeps ~300 events and 12 clips in memory and starts empty
   after a restart. The site machine holds the authoritative log either way.
   One free service running continuously uses about 730 of the 750 free
   instance-hours a month, so it just fits - a second free service will not.
3. Generate an ingest token:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
4. In the Render dashboard → your service → **Environment**, set:

   | Variable | Value |
   |---|---|
   | `PYRO_INGEST_TOKEN` | the token you just generated |
   | `PYRO_ACK_PIN` | a short PIN — otherwise anyone can silence your fire alarm |
   | `PYRO_SITE_NAME` | `Home Workshop` |
   | `PUSHOVER_TOKEN` / `PUSHOVER_USER` | so the **watchdog** alerts you when the site goes silent |
   | `PYRO_PUBLIC_TOKEN` | *optional* — set it and the page needs `?k=<token>` |

5. You get `https://pyrosense-mirror-xxxx.onrender.com`. That is the public URL.

6. Point the site machine at it:
   ```bash
   setx PYRO_CLOUD_URL     "https://pyrosense-mirror-xxxx.onrender.com"
   setx PYRO_INGEST_TOKEN  "the-same-token"
   ```
   Set `"dashboard_url"` in `pyrosense.config.json` to the same URL so alert
   notifications carry a working link, then restart:
   ```bash
   python -m pyrosense run -c pyrosense.config.json
   ```

Within about five seconds the dashboard goes from "waiting for the site" to live.

### Two things to decide before sharing the link

**Anyone with the URL sees stills of your premises.** That is what you asked for,
but set `PYRO_PUBLIC_TOKEN` if you want it link-only-private.

**Free tier spins down after ~15 minutes idle.** The 5-second heartbeat keeps it
awake in practice, but the first request after any gap cold-starts in ~30s. The
`starter` plan removes that; for a safety dashboard it is worth the few dollars.

### Keeping the detector running

Windows Task Scheduler → Create Task → *Run whether user is logged on or not*,
trigger *At startup*, action `python -m pyrosense run -c <full path to config>`,
and on the Settings tab enable *Restart if the task fails*. The cloud watchdog
covers you if it dies anyway.

---

## Part 3 — The blocker on your camera

**Your camera keeps dropping off the network, and a fire detector cannot work on
a camera that sleeps.**

Measured: it streamed fine for several minutes while you had VLC open (1966
frames, 18 fps, zero reconnects). Once VLC closed, a real TCP connect to
`192.168.100.8:554` timed out **12 times out of 12 over 60 seconds**, and the
detector logged 7 reconnect attempts.

That pattern — reachable only while something is actively streaming — is Wi-Fi
power saving or a device sleep mode. Things to check, in order:

1. **Imou Life app → Device Settings → Power Management / Sleep** — disable any
   sleep, standby or low-power mode.
2. If it is battery powered, run it on mains. A battery camera cannot do
   continuous monitoring.
3. **Router → Wi-Fi settings** — disable any client power-save or "green mode"
   for that device, and give it a **DHCP reservation** so the address stops moving.
4. Confirm with:
   ```bash
   python -m pyrosense discover --cidr 192.168.100.0/24
   ```
   Run it twice, ten minutes apart, with nothing streaming. The camera should
   appear both times.

Until that is fixed, the dashboard will correctly show the camera as offline —
which is the honest answer, but not a useful fire detector.
