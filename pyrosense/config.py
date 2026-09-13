"""
Configuration.

JSON by default, YAML if PyYAML happens to be installed. Passwords are read from
the environment when a value looks like ${ENV_VAR}, so the config file itself can
live in version control without carrying camera credentials.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict

from .core.engine import CameraConfig

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(obj):
    if isinstance(obj, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), ""), obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.endswith((".yml", ".yaml")):
        try:
            import yaml
            return _expand(yaml.safe_load(text))
        except ImportError:
            raise SystemExit("PyYAML is not installed - use a .json config, "
                             "or pip install pyyaml")
    return _expand(json.loads(text))


def cameras_from(cfg: dict) -> list[CameraConfig]:
    out = []
    d = cfg.get("defaults", {})
    for c in cfg.get("cameras", []):
        merged = {**d, **c}
        out.append(CameraConfig(
            name=merged["name"],
            source=merged.get("source", {}),
            zones=merged.get("zones", []),
            flame_threshold=float(merged.get("flame_threshold", 0.72)),
            smoke_threshold=float(merged.get("smoke_threshold", 0.70)),
            flame_sustain=float(merged.get("flame_sustain", 2.5)),
            smoke_sustain=float(merged.get("smoke_sustain", 1.0)),
            detect_smoke=bool(merged.get("detect_smoke", True)),
            work_width=int(merged.get("work_width", 480)),
            motion_threshold=float(merged.get("motion_threshold", 0.0008)),
            cooldown_s=float(merged.get("cooldown_s", 180)),
            enabled=bool(merged.get("enabled", True)),
            location=merged.get("location", ""),
            postroll_s=float(merged.get("postroll_s", 8.0)),
        ))
    return out


def uplink_from(cfg: dict, engine_site: str = ""):
    """Build the cloud uplink from config, or None when it is not configured."""
    c = (cfg.get("cloud") or {})
    if not c.get("url") or not c.get("token"):
        return None
    from .cloud.uplink import CloudUplink
    return CloudUplink(
        base_url=c["url"], token=c["token"],
        site=c.get("site") or engine_site or cfg.get("site", "site"),
        heartbeat_s=float(c.get("heartbeat_s", 5)),
        frame_idle_s=float(c.get("frame_idle_s", 4)),
        frame_active_s=float(c.get("frame_active_s", 1)),
        frame_width=int(c.get("frame_width", 640)),
        spool_dir=c.get("spool_dir", "data/spool"),
        enabled=bool(c.get("enabled", True)),
    )


def demo_config() -> dict:
    """Six virtual cameras: two that will catch fire, four that will try very hard
    to look like they have. This is the configuration the demo runs on."""
    return {
        "site": "Demo Site — Unit 4 Engineering",
        "alerts": {"dry_run": True, "file": "data/events/alerts.jsonl"},
        "urgent": {"dry_run": True,
                   "pushover": {"token": "${PUSHOVER_TOKEN}", "user": "${PUSHOVER_USER}"},
                   "ntfy": {"topic": "${NTFY_TOPIC}"},
                   "ladder": {"call_after_s": 90, "second_call_after_s": 300}},
        "cloud": {"enabled": False, "url": "${PYRO_CLOUD_URL}",
                  "token": "${PYRO_INGEST_TOKEN}", "site": "Demo Site"},
        "vlm": {"enabled": True, "model": "claude-haiku-4-5-20251001", "max_tokens": 400, "context_width": 512, "crop_width": 384, "max_calls_per_day": 30,
                "max_calls_per_hour": 6, "fail_open": True},
        "defaults": {"work_width": 480, "cooldown_s": 60},
        "cameras": [
            {"name": "cam-01-warehouse", "location": "Pallet racking, bay A",
             "source": {"kind": "scenario", "scenario": "pallet_fire", "fps": 12}},
            {"name": "cam-02-store", "location": "Component store, far aisle",
             "source": {"kind": "scenario", "scenario": "early_ignition", "fps": 12}},
            {"name": "cam-03-machine", "location": "CNC bay",
             "source": {"kind": "scenario", "scenario": "smoke_plume", "fps": 12}},
            {"name": "cam-04-weldbay", "location": "Welding bay",
             "source": {"kind": "scenario", "scenario": "welding", "fps": 12},
             "zones": [{
                 "name": "welding-bay", "mode": "include",
                 "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                 "threshold_delta": 0.08,
                 "schedule": {"days": ["mon", "tue", "wed", "thu", "fri",
                                       "sat", "sun"],
                              "start": "00:00", "end": "24:00"},
                 "note": "Armed 24/7 for the demo so the physics veto is visible. "
                         "On a real site this zone would be scheduled 18:00-06:00 "
                         "so the arc is not even assessed during shift hours.",
             }]},
            {"name": "cam-05-yard", "location": "Loading door, west",
             "source": {"kind": "scenario", "scenario": "sun_patch", "fps": 12}},
            {"name": "cam-06-aisle", "location": "Main aisle",
             "source": {"kind": "scenario", "scenario": "forklift", "fps": 12}},
        ],
    }


EXAMPLE = {
    "site": "Acme Engineering, Unit 4",
    "dashboard_url": "https://pyrosense-yoursite.onrender.com",
    "cloud": {
        "enabled": True,
        "url": "https://pyrosense-yoursite.onrender.com",
        "token": "${PYRO_INGEST_TOKEN}",
        "site": "Acme Engineering, Unit 4",
        "heartbeat_s": 5, "frame_idle_s": 4, "frame_active_s": 1, "frame_width": 640
    },
    "urgent": {
        "dry_run": False,
        "pushover": {"token": "${PUSHOVER_TOKEN}", "user": "${PUSHOVER_USER}",
                     "retry_s": 30, "expire_s": 900, "sound": "siren"},
        "ntfy": {"topic": "${NTFY_TOPIC}", "server": "https://ntfy.sh", "priority": 5},
        "twilio": {"account_sid": "${TWILIO_ACCOUNT_SID}",
                   "auth_token": "${TWILIO_AUTH_TOKEN}",
                   "from_number": "${TWILIO_FROM_NUMBER}",
                   "call_numbers": ["${ALERT_PHONE_PRIMARY}"],
                   "sms_numbers": []},
        "ladder": {"call_after_s": 90, "second_call_after_s": 300,
                   "give_up_after_s": 1800}
    },
    "alerts": {
        "dry_run": True,
        "webhook": {"url": "https://example.com/hooks/fire",
                    "headers": {"Authorization": "Bearer ${PYROSENSE_WEBHOOK_TOKEN}"}},
        "telegram": {"bot_token": "${TELEGRAM_BOT_TOKEN}", "chat_id": "-1001234567890"},
        "email": {"host": "smtp.example.com", "port": 587,
                  "user": "alerts@example.com", "password": "${SMTP_PASSWORD}",
                  "to": ["duty@example.com"], "tls": True},
        "mqtt": {"host": "192.168.10.5", "topic": "pyrosense/alerts"},
    },
    "vlm": {"enabled": True, "model": "claude-haiku-4-5-20251001", "max_tokens": 400, "context_width": 512, "crop_width": 384, "max_calls_per_day": 30,
            "max_calls_per_hour": 6, "fail_open": True},
    "neural": {"flame_model": "data/models/flame.onnx", "smoke_model": ""},
    "defaults": {"work_width": 480, "flame_threshold": 0.72, "smoke_threshold": 0.70,
                 "cooldown_s": 180},
    "cameras": [
        {"name": "warehouse-01", "location": "Racking bay A",
         "source": {"kind": "rtsp", "brand": "dahua", "host": "192.168.10.64",
                    "user": "admin", "password": "${CAM_PASSWORD}",
                    "stream": "sub", "channel": 1, "fps": 12}},
        {"name": "weldbay-02", "location": "Welding bay",
         "source": {"kind": "rtsp", "brand": "hikvision", "host": "192.168.10.65",
                    "user": "admin", "password": "${CAM_PASSWORD}", "stream": "sub"},
         "zones": [
             {"name": "weld-booth", "mode": "exclude",
              "points": [[0.05, 0.3], [0.45, 0.28], [0.47, 0.95], [0.03, 0.95]],
              "note": "Permanent arc source - never assessed"},
             {"name": "store-area", "mode": "include",
              "points": [[0.5, 0.2], [0.98, 0.2], [0.98, 0.95], [0.5, 0.95]],
              "schedule": {"days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                           "start": "00:00", "end": "24:00"}},
         ]},
        {"name": "yard-03", "location": "External, north",
         "source": {"kind": "rtsp", "brand": "reolink", "host": "192.168.10.66",
                    "user": "admin", "password": "${CAM_PASSWORD}"},
         "detect_smoke": False},
    ],
}


def write_example(path: str = "pyrosense.config.json") -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(EXAMPLE, f, indent=2)
    return path
