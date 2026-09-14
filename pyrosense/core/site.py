"""
The site config file, edited from the console.

Adding a camera in the UI has to survive a restart, so it is written back to the
same JSON file `pyrosense run -c` reads. Two rules keep that safe:

  * Write whole files atomically (temp file + os.replace) and keep one .bak, so a
    crash mid-save cannot leave the site with an unreadable config.
  * Never write a password into the config. Console-added passwords go to
    data/secrets.json and the config holds a ${secret:cam.<name>} reference;
    passwords already supplied as ${ENV_VAR} are left exactly as they are.
"""
from __future__ import annotations

import json
import os
import shutil
import threading

from ..config import SECRETS_PATH, _expand, cameras_from, read_secrets

EDITABLE = ("location", "enabled", "detect_smoke", "flame_threshold",
            "smoke_threshold", "cooldown_s")


def _atomic_write(path: str, data: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class SiteConfig:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- raw io
    def _read(self) -> dict:
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _write(self, cfg: dict) -> None:
        if os.path.exists(self.path):
            shutil.copyfile(self.path, self.path + ".bak")
        _atomic_write(self.path, cfg)

    @staticmethod
    def secret_key(name: str) -> str:
        return f"cam.{name}"

    def _set_secret(self, key: str, value: str | None) -> None:
        s = read_secrets()
        if value is None:
            s.pop(key, None)
        else:
            s[key] = value
        _atomic_write(SECRETS_PATH, s)

    # ------------------------------------------------------------------ api
    def raw_cameras(self) -> list[dict]:
        return list(self._read().get("cameras", []))

    def camera_config(self, name: str):
        """The runnable CameraConfig for one camera, with defaults and secrets applied."""
        cfg = self._read()
        cam = next((c for c in cfg.get("cameras", []) if c.get("name") == name), None)
        if cam is None:
            return None
        return cameras_from(_expand({"defaults": cfg.get("defaults", {}),
                                     "cameras": [cam]}))[0]

    def add(self, cam: dict, password: str) -> None:
        with self._lock:
            cfg = self._read()
            cams = cfg.setdefault("cameras", [])
            if any(c.get("name") == cam["name"] for c in cams):
                raise ValueError(f"a camera called {cam['name']} already exists")
            cam = dict(cam)
            src = dict(cam.get("source") or {})
            if password:
                self._set_secret(self.secret_key(cam["name"]), password)
                src["password"] = "${secret:" + self.secret_key(cam["name"]) + "}"
            cam["source"] = src
            cams.append(cam)
            self._write(cfg)

    def update(self, name: str, fields: dict, source: dict | None = None,
               password: str | None = None) -> dict:
        with self._lock:
            cfg = self._read()
            cam = next((c for c in cfg.get("cameras", []) if c.get("name") == name), None)
            if cam is None:
                raise KeyError(name)
            for k in EDITABLE:
                if k in fields:
                    cam[k] = fields[k]
            if source is not None:
                src = dict(source)
                if password:
                    self._set_secret(self.secret_key(name), password)
                    src["password"] = "${secret:" + self.secret_key(name) + "}"
                cam["source"] = src
            self._write(cfg)
            return cam

    def remove(self, name: str) -> None:
        with self._lock:
            cfg = self._read()
            before = len(cfg.get("cameras", []))
            cfg["cameras"] = [c for c in cfg.get("cameras", []) if c.get("name") != name]
            if len(cfg["cameras"]) == before:
                raise KeyError(name)
            self._write(cfg)
            self._set_secret(self.secret_key(name), None)
