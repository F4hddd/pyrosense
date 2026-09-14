"""
Turn a pasted stream link into a camera source.

Installers and camera apps hand people a full URL, credentials and all:

    rtsp://admin:password@192.168.1.20:554/cam/realmonitor?channel=1&subtype=0

That is the most natural thing to paste into an "add camera" box, so this module
accepts it, splits the password out (it is stored separately, never in the site
config), and recognises the common vendor path shapes so detection can run on
the substream even when the link given was the main stream.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from urllib.parse import parse_qs, quote, unquote, urlsplit

SCHEMES = ("rtsp", "rtsps", "http", "https")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class LinkError(ValueError):
    pass


def parse_link(link: str) -> dict:
    """Split a stream URL into parts. Raises LinkError with a human message."""
    link = (link or "").strip().strip("⁠​﻿")   # pasted zero-width chars
    if not link:
        raise LinkError("Paste the camera's stream link.")
    try:
        u = urlsplit(link)
        port = u.port
    except ValueError:
        raise LinkError("That link is not a valid URL.")
    if u.scheme.lower() not in SCHEMES:
        raise LinkError("The link should start with rtsp:// (or http:// for MJPEG cameras).")
    if not u.hostname:
        raise LinkError("The link has no camera address in it.")
    path = u.path or "/"
    return {
        "scheme": u.scheme.lower(),
        "host": u.hostname,
        "port": port,
        "user": unquote(u.username or ""),
        "password": unquote(u.password or ""),
        "path": path + (f"?{u.query}" if u.query else ""),
        "query": parse_qs(u.query),
    }


def source_from_link(link: str, stream: str = "sub") -> tuple[dict, str]:
    """Build a source spec from a link.

    Returns (source, password). The source never contains the password; the
    caller stores it and puts a reference in `password`.

    stream="sub" moves a recognised main-stream link onto its substream, which is
    about a tenth of the CPU for no loss in fire detection. stream="as_is" keeps
    exactly what was pasted.
    """
    p = parse_link(link)
    q = {k: v[0] for k, v in p["query"].items()}
    path_l = p["path"].lower()

    # Dahua / Imou / Amcrest / Lorex
    if p["scheme"] == "rtsp" and path_l.startswith("/cam/realmonitor"):
        ch = int(q.get("channel", "1") or 1)
        want = stream if stream in ("sub", "main") else (
            "main" if q.get("subtype", "0") == "0" else "sub")
        return ({"kind": "rtsp", "brand": "imou", "host": p["host"],
                 "port": p["port"] or 554, "user": p["user"], "channel": ch,
                 "stream": want, "fps": 12}, p["password"])

    path = p["path"]
    if stream == "sub":
        # Hikvision / EZVIZ: /Streaming/Channels/101 is main, 102 is sub
        m = re.match(r"(?i)^(/Streaming/Channels/)(\d+?)01(\b.*)$", path)
        if m:
            path = f"{m.group(1)}{m.group(2)}02{m.group(3)}"
        # Reolink: h264Preview_01_main -> _sub
        path = re.sub(r"(?i)(Preview_\d+_)main", r"\1sub", path)

    port = f":{p['port']}" if p["port"] else ""
    url = f"{p['scheme']}://{p['host']}{port}{path}"
    return ({"kind": "rtsp", "url": url, "user": p["user"], "fps": 12}, p["password"])


def with_credentials(url: str, user: str, password: str) -> str:
    """Insert credentials into a credential-less URL."""
    if not user or "@" in url.split("://", 1)[-1].split("/", 1)[0]:
        return url
    scheme, rest = url.split("://", 1)
    cred = quote(user, safe="") + (":" + quote(password, safe="") if password else "")
    return f"{scheme}://{cred}@{rest}"


def display_link(source: dict) -> str:
    """What the UI shows: the effective URL with the password masked."""
    from .profiles import build_url, redact
    if source.get("kind", "rtsp") != "rtsp":
        return source.get("kind", "")
    if source.get("url"):
        return redact(with_credentials(source["url"], source.get("user", ""), "x"))
    try:
        return redact(build_url(source.get("brand", "generic"), source.get("host", ""),
                                source.get("user", ""), "x", source.get("channel", 1),
                                source.get("stream", "sub"), source.get("port")))
    except Exception:
        return source.get("host", "")


def friendly_error(err: str) -> str:
    """ffmpeg's stderr, translated for someone who is not reading ffmpeg's stderr."""
    low = (err or "").lower()
    if not low:
        return ""
    if "401" in low or "unauthorized" in low:
        return "The camera rejected the username or password."
    if "404" in low or "not found" in low:
        return "The camera is reachable but that stream path does not exist."
    if any(s in low for s in ("error number -138", "timed out", "10060", "no route",
                              "unreachable", "connection refused", "10061")):
        return ("Can't reach the camera at this address - it may be asleep, off, "
                "or have a new IP address.")
    if "no frames" in low or "stalled" in low or "stale" in low:
        return "Connected, but the camera stopped sending video."
    return "The camera is not sending video."


def grab_frame(url: str, timeout: float = 12.0) -> dict:
    """One connection, one decoded frame. Deliberately a single attempt: several
    failed logins in a row lock the account on Dahua and Hikvision firmware."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return {"ok": False, "error": "ffmpeg is not installed on this computer."}
    args = [exe, "-v", "error", "-nostdin"]
    if url.startswith("rtsp"):
        args += ["-rtsp_transport", "tcp", "-timeout", str(int(timeout * 1_000_000))]
    args += ["-i", url, "-frames:v", "1", "-vf", "scale=640:-2",
             "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5", "pipe:1"]
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout + 6)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "The camera did not answer in time. Check the address "
                                      "and that the camera is powered on and awake."}
    if r.returncode == 0 and r.stdout[:2] == b"\xff\xd8":
        return {"ok": True, "jpeg": r.stdout}
    err = r.stderr.decode("utf-8", "ignore")
    msg = friendly_error(err) or "The camera did not return video."
    from .profiles import redact
    detail = re.sub(r"\w+://[^\s]+", lambda m: redact(m.group(0)), err.strip())[-240:]
    return {"ok": False, "error": msg, "detail": detail}
