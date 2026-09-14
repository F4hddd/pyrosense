"""
Frame sources.

RTSP is read through an ffmpeg subprocess rather than cv2.VideoCapture. That is a
deliberate choice and it is worth defending, because it is the difference between
a demo and something that survives a month on a factory LAN:

  Transport control.  `-rtsp_transport tcp` forces TCP. The default is UDP, and on
    a busy site network with PoE switches and a welder on the same circuit, UDP
    packet loss produces torn, green-smeared frames. A fire detector fed torn
    frames raises false alarms at exactly the moments the network is worst.

  Timeouts.  A camera that is powered but wedged will accept a TCP connection and
    then send nothing, forever. cv2.VideoCapture will block on that with no way
    out. ffmpeg takes `-timeout` and we can also just kill the process.

  Codec coverage.  Half the cameras shipping today default to H.265. OpenCV's
    bundled FFmpeg support for it varies by wheel and platform; the system ffmpeg
    handles it consistently.

  Blast radius.  A malformed stream crashes a subprocess, not the detector. The
    supervisor restarts it and the other 29 cameras never notice.

  Server-side thinning.  `-r 12` makes ffmpeg drop frames before they are decoded
    into our address space, so a 25fps camera costs us 12fps of work.

Every source presents the same tiny interface: `.read()` returns a BGR frame or
None, and `.stats` describes health. Virtual scenario cameras implement it too,
so the whole product can be demonstrated with no hardware present.
"""
from __future__ import annotations

from collections import deque
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .profiles import redact

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


@dataclass
class SourceStats:
    connected: bool = False
    frames: int = 0
    dropped: int = 0
    reconnects: int = 0
    last_frame_t: float = 0.0
    fps: float = 0.0
    error: str = ""
    codec: str = ""
    stale_after_s: float = 10.0

    @property
    def live(self) -> bool:
        """Connected AND actually delivering.

        `connected` used to be set the moment the ffmpeg process spawned, which
        meant a camera that accepted the RTSP session and then sent nothing showed
        a green light and "live" on the dashboard while delivering zero frames.
        That is the same lie as a monitoring page frozen on "all clear", and it is
        worse here because it is the camera you would be relying on."""
        if not self.connected or not self.last_frame_t:
            return False
        return (time.time() - self.last_frame_t) < self.stale_after_s

    def to_json(self) -> dict:
        return dict(connected=self.live, spawned=self.connected,
                    frames=self.frames,
                    reconnects=self.reconnects, fps=round(self.fps, 1),
                    error=self.error[:200], stale_s=round(
                        max(0.0, time.time() - self.last_frame_t), 1)
                    if self.last_frame_t else None)


class BaseSource:
    name: str = "source"

    def start(self) -> "BaseSource":
        return self

    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def stop(self) -> None:
        pass

    @property
    def stats(self) -> SourceStats:
        raise NotImplementedError


class FFmpegSource(BaseSource):
    """RTSP/RTMP/HTTP stream via an ffmpeg subprocess, with supervised restart."""

    def __init__(self, url: str, name: str = "cam", width: int = 640,
                 fps: int = 12, transport: str = "tcp", timeout_s: float = 8.0,
                 reconnect_backoff: tuple[float, ...] = (1, 2, 5, 10, 20, 30)):
        self.url = url
        self.name = name
        self.width = width
        self.fps = fps
        self.transport = transport
        self.timeout_s = timeout_s
        self.backoff = reconnect_backoff
        self._proc: subprocess.Popen | None = None
        self._h = 0
        self._stats = SourceStats()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._t_last = 0.0
        self._recent: deque = deque(maxlen=300)

    # ---------------------------------------------------------------- ffmpeg
    def _transport_args(self) -> list[str]:
        """-rtsp_transport is an RTSP demuxer option; ffmpeg rejects it for an
        HTTP (MJPEG) camera, so it is only passed for rtsp:// links."""
        return ["-rtsp_transport", self.transport] if self.url.startswith("rtsp") else []

    def _probe_size(self) -> tuple[int, int]:
        """Ask ffprobe for the real frame size so we know how many bytes make a
        frame. Falls back to a 16:9 guess, which is right for almost everything."""
        exe = shutil.which("ffprobe")
        if exe:
            try:
                out = subprocess.run(
                    [exe, "-v", "error", *self._transport_args(),
                     "-select_streams", "v:0", "-show_entries",
                     "stream=width,height,codec_name", "-of", "csv=p=0",
                     "-timeout", str(int(self.timeout_s * 1_000_000)), self.url],
                    capture_output=True, text=True, timeout=self.timeout_s + 4).stdout.strip()
                parts = out.split(",")
                if len(parts) >= 2 and parts[0].isdigit():
                    w, h = int(parts[0]), int(parts[1])
                    if len(parts) > 2:
                        self._stats.codec = parts[2]
                    scale = self.width / float(w)
                    return self.width, int(round(h * scale / 2) * 2)
            except Exception:
                pass
        return self.width, int(self.width * 9 / 16 / 2) * 2

    def _spawn(self) -> None:
        w, h = self._probe_size()
        self._h = h
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            *self._transport_args(),
            "-timeout", str(int(self.timeout_s * 1_000_000)),
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-i", self.url,
            "-an", "-sn",
            "-vf", f"scale={w}:{h}",
            "-r", str(self.fps),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=w * h * 3 * 2)

    def _pump(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                self._spawn()
                nbytes = self.width * self._h * 3
                self._stats.error = ""
                attempt = 0
                while not self._stop.is_set():
                    buf = self._proc.stdout.read(nbytes)      # type: ignore[union-attr]
                    if not buf or len(buf) < nbytes:
                        raise IOError("stream ended or short read")
                    frame = np.frombuffer(buf, np.uint8).reshape(self._h, self.width, 3)
                    now = time.time()
                    with self._lock:
                        self._latest = frame
                    self._stats.frames += 1
                    self._stats.connected = True    # earned by a real frame

                    # Frame rate over a sliding window, not a smoothed 1/dt.
                    # ffmpeg hands us frames in bursts as its buffer flushes, so
                    # consecutive arrivals can be microseconds apart; an EMA of
                    # the instantaneous rate turned that into a dashboard proudly
                    # reporting 634 fps from a 15 fps camera. Counting arrivals in
                    # a real time window cannot do that.
                    self._recent.append(now)
                    cutoff = now - 5.0
                    while self._recent and self._recent[0] < cutoff:
                        self._recent.popleft()
                    span = (self._recent[-1] - self._recent[0]) if len(self._recent) > 1 else 0.0
                    self._stats.fps = (len(self._recent) - 1) / span if span > 0.5 else 0.0
                    self._t_last = now
                    self._stats.last_frame_t = now
            except Exception as e:
                self._stats.connected = False
                err = ""
                if self._proc and self._proc.stderr:
                    try:
                        err = self._proc.stderr.read(400).decode("utf-8", "ignore")
                    except Exception:
                        pass
                self._stats.error = (err or str(e)).strip()[:300]
            finally:
                self._kill()

            if self._stop.is_set():
                break
            self._stats.reconnects += 1
            delay = self.backoff[min(attempt, len(self.backoff) - 1)]
            attempt += 1
            self._stop.wait(delay)

    def _kill(self) -> None:
        p, self._proc = self._proc, None
        if p is None:
            return
        try:
            p.kill()
            p.wait(timeout=2)
        except Exception:
            pass

    # ------------------------------------------------------------ interface
    def start(self) -> "FFmpegSource":
        self._thread = threading.Thread(target=self._pump, daemon=True,
                                        name=f"src-{self.name}")
        self._thread.start()
        return self

    def read(self) -> np.ndarray | None:
        with self._lock:
            f, self._latest = self._latest, None
        return f

    def stop(self) -> None:
        self._stop.set()
        self._kill()

    @property
    def stats(self) -> SourceStats:
        return self._stats

    def __repr__(self) -> str:
        return f"<FFmpegSource {self.name} {redact(self.url)}>"


class ScenarioSource(BaseSource):
    """A virtual camera backed by a procedural scenario. Lets the entire product
    be demonstrated - dashboard, alerts, adjudication, event log - with no
    hardware, no network and no fire."""

    def __init__(self, scenario, name: str = "virtual", fps: int = 12,
                 loop: bool = True):
        self.sc = scenario
        self.name = name
        self.fps = fps
        self.loop = loop
        self._t0 = time.time()
        self._stats = SourceStats(connected=True, fps=float(fps))
        self._last_emit = 0.0

    def read(self) -> np.ndarray | None:
        now = time.time()
        if now - self._last_emit < 1.0 / self.fps:
            return None
        self._last_emit = now
        t = now - self._t0
        if self.loop and t > self.sc.duration:
            self._t0 = now
            t = 0.0
        self._stats.frames += 1
        self._stats.last_frame_t = now
        return self.sc.frame(t)

    @property
    def stats(self) -> SourceStats:
        return self._stats


class FileSource(BaseSource):
    """Video file or image folder. Used for replaying real site footage, which is
    how a deployment should actually be tuned before it goes live."""

    def __init__(self, path: str, name: str = "file", fps: int | None = None,
                 loop: bool = False):
        self.path = path
        self.name = name
        self.loop = loop
        self._cap = cv2.VideoCapture(path)
        self.fps = fps or (self._cap.get(cv2.CAP_PROP_FPS) or 12)
        self._stats = SourceStats(connected=self._cap.isOpened())
        self._last_emit = 0.0

    def read(self) -> np.ndarray | None:
        now = time.time()
        if now - self._last_emit < 1.0 / max(1.0, self.fps):
            return None
        ok, frame = self._cap.read()
        if not ok:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return None
            self._stats.connected = False
            return None
        self._last_emit = now
        self._stats.frames += 1
        self._stats.last_frame_t = now
        return frame

    def stop(self) -> None:
        self._cap.release()

    @property
    def stats(self) -> SourceStats:
        return self._stats


def make_source(spec: dict) -> BaseSource:
    """Build a source from a config dict. `kind` is rtsp | file | scenario."""
    kind = spec.get("kind", "rtsp")
    name = spec.get("name", "cam")
    if kind == "scenario":
        from ..synth.scenarios import SCENARIOS
        return ScenarioSource(SCENARIOS[spec["scenario"]], name=name,
                              fps=spec.get("fps", 12))
    if kind == "file":
        return FileSource(spec["path"], name=name, loop=spec.get("loop", True))
    from .profiles import build_url
    if spec.get("url"):
        from .links import with_credentials
        url = with_credentials(spec["url"], spec.get("user", ""), spec.get("password", ""))
    else:
        url = build_url(
            spec.get("brand", "generic"), spec["host"], spec.get("user", ""),
            spec.get("password", ""), spec.get("channel", 1),
            spec.get("stream", "sub"), spec.get("port"))
    return FFmpegSource(url, name=name, width=spec.get("width", 640),
                        fps=spec.get("fps", 12),
                        transport=spec.get("transport", "tcp"))
