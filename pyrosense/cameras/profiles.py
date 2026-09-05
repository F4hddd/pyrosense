"""
Camera brand profiles.

The single most important design decision in this system:
    WE NEVER INTEGRATE WITH A VENDOR APP, CLOUD OR SDK.

Imou, Hikvision, Dahua, Reolink, Tapo, Uniview, Axis... every one of them ships a
different phone app, a different desktop client, a different cloud, and a different
(often paid, often Windows-only, often abandoned) SDK. Integrating with those is a
treadmill you can never get off.

But underneath, ~99% of IP cameras sold in the last decade speak two open protocols:

    RTSP   (RFC 2326)  - the actual video stream
    ONVIF  (Profile S) - discovery, "tell me your stream URL", PTZ

So we speak RTSP. The only brand-specific knowledge we need is the URL *path*,
because vendors gratuitously differ there. That knowledge is one table, below,
instead of thirty vendor SDKs.

For anything not in the table, ONVIF discovery asks the camera for its own URL,
and if even that fails we fall back to a path-probe sweep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    # {user} {pw} {host} {port} {ch}  - ch is the 1-based channel (matters on NVRs)
    main: str
    sub: str
    port: int = 554
    onvif_port: int = 80
    # HTTP still-image URL: cheap health checks, and cameras with no usable RTSP
    snapshot: Optional[str] = None
    oui: tuple = ()          # MAC prefixes seen on this brand (heuristic fingerprint)
    http_hints: tuple = ()   # strings that show up in the web UI or headers
    notes: str = ""


# ---------------------------------------------------------------------------
# The table. Every path here is the documented or field-observed RTSP path.
# ---------------------------------------------------------------------------
PROFILES: dict[str, Profile] = {p.key: p for p in [
    Profile(
        key="dahua", label="Dahua / Imou / Amcrest / Lorex (Dahua OEM)",
        main="rtsp://{user}:{pw}@{host}:{port}/cam/realmonitor?channel={ch}&subtype=0",
        sub="rtsp://{user}:{pw}@{host}:{port}/cam/realmonitor?channel={ch}&subtype=1",
        snapshot="http://{host}/cgi-bin/snapshot.cgi?channel={ch}",
        oui=("3c:ef:8c", "4c:11:bf", "90:02:a9", "e0:50:8b", "14:a7:8b", "bc:32:5f", "38:af:29"),
        http_hints=("Dahua", "DH-", "IMOU", "Amcrest", "webplugin"),
        notes="Imou is Dahua consumer branding and is the same RTSP stack byte for byte. "
              "Imou cameras need RTSP switched on once in the Imou Life app: "
              "Device Settings -> Advanced -> RTSP/ONVIF, then create a local account. "
              "If the substream connects but never delivers parseable data, append "
              "'&unicast=true&proto=Onvif' - observed on Imou firmware where the bare "
              "subtype=1 path negotiates RTSP and then stalls. The password is the "
              "SC / security code printed on the device label.",
    ),
    Profile(
        key="imou", label="Imou (Dahua consumer) - ONVIF substream variant",
        # Same stack as Dahua, but observed Imou firmware completes the RTSP
        # handshake on the bare subtype=1 substream and then never delivers
        # parseable data - ffprobe and ffmpeg both sit there until they time out.
        # Adding the ONVIF unicast hint returns a clean substream immediately.
        # Verified on a 5MP unit: substream h264 640x352 @15fps, main hevc
        # 2880x1620 @25fps.
        main="rtsp://{user}:{pw}@{host}:{port}/cam/realmonitor?channel={ch}&subtype=0",
        sub="rtsp://{user}:{pw}@{host}:{port}/cam/realmonitor?channel={ch}"
            "&subtype=1&unicast=true&proto=Onvif",
        snapshot="http://{host}/cgi-bin/snapshot.cgi?channel={ch}",
        oui=("3c:ef:8c", "4c:11:bf", "90:02:a9", "e0:50:8b", "14:a7:8b", "bc:32:5f"),
        http_hints=("IMOU", "Imou"),
        notes="Enable RTSP once in the Imou Life app (Device Settings -> Advanced "
              "-> RTSP/ONVIF). The RTSP password is the SC / security code printed "
              "on the device label, not the Imou cloud account password. If the "
              "substream stalls, this profile's ONVIF unicast variant is the fix.",
    ),
    Profile(
        key="hikvision", label="Hikvision / Annke / LTS / HiWatch / Safire",
        # 101 = channel 1 main stream, 102 = channel 1 sub stream
        main="rtsp://{user}:{pw}@{host}:{port}/Streaming/Channels/{ch}01",
        sub="rtsp://{user}:{pw}@{host}:{port}/Streaming/Channels/{ch}02",
        snapshot="http://{host}/ISAPI/Streaming/channels/{ch}01/picture",
        oui=("44:19:b6", "4c:bd:8f", "c0:56:e3", "bc:ad:28", "28:57:be", "54:c4:15", "e0:ba:ad"),
        http_hints=("Hikvision", "DS-", "App-webs", "ANNKE", "ISAPI"),
        notes="Older firmware uses /h264/ch{ch}/main/av_stream - both are probed.",
    ),
    Profile(
        key="reolink", label="Reolink",
        main="rtsp://{user}:{pw}@{host}:{port}/h264Preview_{ch:02d}_main",
        sub="rtsp://{user}:{pw}@{host}:{port}/h264Preview_{ch:02d}_sub",
        snapshot="http://{host}/cgi-bin/api.cgi?cmd=Snap&channel={ch0}",
        oui=("ec:71:db",),
        http_hints=("Reolink", "rs.js"),
        notes="H.265 models drop the h264 prefix (Preview_01_main). Both are probed.",
    ),
    Profile(
        key="tplink", label="TP-Link Tapo / VIGI",
        main="rtsp://{user}:{pw}@{host}:{port}/stream1",
        sub="rtsp://{user}:{pw}@{host}:{port}/stream2",
        oui=("50:c7:bf", "60:a4:b7", "a4:2b:b0", "9c:53:22", "5c:e9:31"),
        http_hints=("TP-LINK", "Tapo", "VIGI"),
        notes="Tapo needs a Camera Account created in the Tapo app "
              "(Advanced Settings -> Camera Account). The Tapo cloud login will NOT work.",
    ),
    Profile(
        key="uniview", label="Uniview (UNV)",
        main="rtsp://{user}:{pw}@{host}:{port}/media/video{ch}",
        sub="rtsp://{user}:{pw}@{host}:{port}/media/video{ch}_sub",
        oui=("48:ea:63",),
        http_hints=("Uniview", "UNV", "IPC2"),
        notes="Alternate path /unicast/c{ch}/s0/live on some firmware.",
    ),
    Profile(
        key="axis", label="Axis Communications",
        main="rtsp://{user}:{pw}@{host}:{port}/axis-media/media.amp?camera={ch}",
        sub="rtsp://{user}:{pw}@{host}:{port}/axis-media/media.amp?camera={ch}&resolution=640x360",
        snapshot="http://{host}/axis-cgi/jpg/image.cgi?camera={ch}",
        oui=("00:40:8c", "ac:cc:8e", "b8:a4:4f"),
        http_hints=("AXIS", "axis-cgi"),
    ),
    Profile(
        key="hanwha", label="Hanwha Vision / Samsung Techwin / Wisenet",
        main="rtsp://{user}:{pw}@{host}:{port}/profile{ch}/media.smp",
        sub="rtsp://{user}:{pw}@{host}:{port}/profile{ch}s/media.smp",
        oui=("00:16:6c", "00:09:18", "00:00:f0"),
        http_hints=("Wisenet", "Hanwha", "SNB-", "XNP-"),
    ),
    Profile(
        key="vivotek", label="Vivotek",
        main="rtsp://{user}:{pw}@{host}:{port}/live.sdp",
        sub="rtsp://{user}:{pw}@{host}:{port}/live2.sdp",
        oui=("00:02:d1",),
        http_hints=("VIVOTEK",),
    ),
    Profile(
        key="foscam", label="Foscam",
        main="rtsp://{user}:{pw}@{host}:{port}/videoMain",
        sub="rtsp://{user}:{pw}@{host}:{port}/videoSub",
        port=88,
        http_hints=("Foscam",),
    ),
    Profile(
        key="ezviz", label="EZVIZ (Hikvision consumer)",
        main="rtsp://{user}:{pw}@{host}:{port}/h264/ch{ch}/main/av_stream",
        sub="rtsp://{user}:{pw}@{host}:{port}/h264/ch{ch}/sub/av_stream",
        http_hints=("EZVIZ",),
        notes="Password is the 6-character VERIFICATION CODE printed on the camera "
              "label, not the EZVIZ cloud password.",
    ),
    Profile(
        key="unifi", label="Ubiquiti UniFi Protect",
        main="rtsps://{host}:{port}/{ch}?enableSrtp",
        sub="rtsps://{host}:{port}/{ch}?enableSrtp",
        port=7441,
        oui=("74:83:c2", "fc:ec:da", "78:8a:20"),
        http_hints=("UniFi", "Ubiquiti"),
        notes="Here {ch} is the per-stream token from Protect -> Settings -> RTSP. No user/pass.",
    ),
    Profile(
        key="onvif", label="Generic ONVIF (auto-discovered URL)",
        main="{onvif_uri}", sub="{onvif_uri}",
        notes="URL is fetched from the camera itself via ONVIF Profile S GetStreamUri.",
    ),
    Profile(
        key="generic", label="Generic / unknown - probe common paths",
        main="rtsp://{user}:{pw}@{host}:{port}/", sub="rtsp://{user}:{pw}@{host}:{port}/",
    ),
]}

# Paths swept when the brand is unknown, ordered by how common they are in the field.
PROBE_PATHS: tuple[str, ...] = (
    # The ONVIF-flavoured Dahua substream goes first on purpose. On Imou firmware
    # the bare subtype=1 path completes the RTSP handshake and then never sends
    # parseable data - ffprobe sits there until it times out - while this variant
    # returns a clean 640x352 h264 substream immediately. Confirmed on an Imou
    # 5MP unit whose main stream is H.265 2880x1620.
    "/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif",
    "/cam/realmonitor?channel=1&subtype=1",      # Dahua / Imou sub
    "/Streaming/Channels/102",                   # Hikvision sub
    "/h264Preview_01_sub",                       # Reolink sub
    "/stream2",                                  # Tapo / VIGI sub
    "/media/video1_sub",                         # Uniview sub
    "/cam/realmonitor?channel=1&subtype=0",
    "/Streaming/Channels/101",
    "/h264/ch1/sub/av_stream",
    "/h264/ch1/main/av_stream",
    "/h264Preview_01_main",
    "/stream1",
    "/live.sdp", "/live/ch0", "/live", "/live0", "/11", "/12",
    "/media/video1", "/videoMain", "/onvif1", "/onvif2",
    "/axis-media/media.amp",
    "/profile1/media.smp",
    "/ch01/1", "/1/h264major", "/mpeg4", "/video1", "/av0_1", "/0",
)


def build_url(profile_key: str, host: str, user: str = "", pw: str = "",
              channel: int = 1, stream: str = "sub", port: int | None = None,
              onvif_uri: str | None = None) -> str:
    """Render a concrete RTSP URL for a brand.

    stream='sub' is the default on purpose. Substreams are typically 640x360 to
    704x576 at 10-15fps, which is ample for fire detection and costs roughly 10x
    less CPU and bandwidth than the main stream. We only reach for the main stream
    when an alert fires and we want a high-resolution still for the responder.
    """
    p = PROFILES.get(profile_key, PROFILES["generic"])
    tmpl = p.main if stream == "main" else p.sub
    return tmpl.format(
        user=quote(user, safe=""), pw=quote(pw, safe=""),
        host=host, port=port or p.port, ch=channel, ch0=channel - 1,
        onvif_uri=onvif_uri or "",
    )


def fingerprint(mac: str = "", http_banner: str = "", onvif_vendor: str = "") -> tuple[str, float]:
    """Guess the brand from whatever weak signals we scraped. Returns (key, confidence)."""
    hay = f"{http_banner} {onvif_vendor}".lower()
    for p in PROFILES.values():
        for hint in p.http_hints:
            if hint.lower() in hay:
                return p.key, 0.95
    m = (mac or "").lower().replace("-", ":")[:8]
    if m:
        for p in PROFILES.values():
            if m in p.oui:
                return p.key, 0.75
    return "generic", 0.0


def redact(url: str) -> str:
    """Never log credentials. Applied everywhere a URL can reach a log line or the UI."""
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" in rest:
        creds, tail = rest.rsplit("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:****@{tail}"
    return url
