"""
Procedural workshop scenarios - the evaluation harness and the demo feed.

Two jobs:

  1. Prove the detector on the cases that actually matter. Any fire detector can
     find a bonfire. The question a client should ask is "what does it do when my
     welder strikes an arc at 2am". So this module renders a *nuisance gauntlet*:
     welding, grinder sparks, a rotating beacon, a halogen work lamp, a sunlight
     patch, a hi-vis vest, a grey forklift. Every one of them is deliberately
     given fire-like colour, so colour alone cannot pass the test.

  2. Drive the live demo without hardware. Each scenario is a `frame(t)` function,
     so the dashboard can run "virtual cameras" at 12 fps and you can demo the
     whole product on a laptop on a plane.

These are renderings, not real footage, and the numbers they produce are a
regression harness rather than a field accuracy claim - real validation needs
real site footage, which is step one of any deployment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

W, H = 640, 360
rng = np.random.default_rng(7)


# --------------------------------------------------------------------------
# Background: a plausible workshop. Concrete floor, block wall, steel racking,
# a couple of crates and a machine silhouette.
# --------------------------------------------------------------------------
def _workshop_plate() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = (58, 60, 63)
    cv2.rectangle(img, (0, 235), (W, H), (78, 80, 84), -1)          # floor
    for y in range(0, 235, 26):                                      # block wall
        cv2.line(img, (0, y), (W, y), (52, 54, 57), 1)
    for x in range(0, W, 52):
        cv2.line(img, (x, 0), (x, 235), (52, 54, 57), 1)
    # steel racking, right third
    for y in (96, 152, 208):
        cv2.rectangle(img, (405, y), (628, y + 11), (96, 92, 84), -1)
    for x in (405, 512, 628):
        cv2.rectangle(img, (x - 4, 70), (x + 4, 235), (88, 85, 78), -1)
    # pallets / crates, left
    for (x, y, w, h, c) in ((44, 186, 96, 50, (74, 96, 122)),
                            (150, 200, 74, 36, (70, 90, 114)),
                            (58, 150, 70, 34, (78, 100, 128))):
        cv2.rectangle(img, (x, y), (x + w, y + h), c, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (46, 58, 74), 1)
    # machine
    cv2.rectangle(img, (250, 140), (360, 236), (66, 68, 72), -1)
    cv2.rectangle(img, (268, 156), (342, 190), (44, 46, 50), -1)
    cv2.line(img, (0, 235), (W, 235), (44, 46, 48), 2)
    return img


_PLATE = _workshop_plate()


def _sensor(img: np.ndarray, t: float, noise: float = 3.2) -> np.ndarray:
    """Cheap-camera realism: read noise, mild auto-exposure hunting, JPEG-ish
    softening. Without this the detector gets an unrealistically clean signal."""
    out = img.astype(np.float32)
    out += rng.normal(0, noise, out.shape)
    out *= 1.0 + 0.012 * math.sin(t * 0.9) + 0.006 * math.sin(t * 4.3)
    out = cv2.GaussianBlur(out, (3, 3), 0.6)
    return np.clip(out, 0, 255).astype(np.uint8)


def _hash01(i: int, seed: float) -> float:
    v = math.sin(i * 12.9898 + seed * 78.233) * 43758.5453
    return v - math.floor(v)


def _vnoise(t: float, freq: float, seed: float) -> float:
    """Smooth value noise: deterministic in t, but genuinely aperiodic."""
    x = t * freq
    i = math.floor(x)
    f = x - i
    a, b = _hash01(int(i), seed), _hash01(int(i) + 1, seed)
    return (a + (b - a) * (f * f * (3 - 2 * f))) * 2 - 1


def _chaotic(t: float, seed: float = 0.0) -> float:
    """Turbulent flicker.

    An earlier version was a sum of four incommensurate sinusoids. That is not
    good enough as a test signal: it still has enough self-similarity that the
    autocorrelation climbs, and the detector's own strobe veto was firing on it.
    Fixing the detector against a fake fire that behaves like a strobe would have
    been tuning to the test rather than to reality, so the *scenario* was made
    honest instead - octaves of value noise over a slow sinusoidal carrier, which
    is much closer to how a real flame front actually moves."""
    return (0.34 * math.sin(2 * math.pi * 1.73 * t + seed)
            + 0.30 * _vnoise(t, 4.7, seed + 1.0)
            + 0.22 * _vnoise(t, 9.3, seed + 2.0)
            + 0.14 * _vnoise(t, 17.1, seed + 3.0))


def _blob(img: np.ndarray, cx: float, cy: float, rx: float, ry: float,
          colour: tuple[int, int, int], alpha: float = 1.0, ragged: float = 0.0,
          phase: float = 0.0) -> None:
    """Draw a soft, optionally ragged ellipse. Raggedness is what gives flames
    their high perimeter-to-area ratio."""
    layer = np.zeros_like(img)
    if ragged <= 0:
        cv2.ellipse(layer, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, colour, -1)
    else:
        pts = []
        for i in range(28):
            a = 2 * math.pi * i / 28
            k = 1.0 + ragged * (0.5 * math.sin(5 * a + phase) + 0.5 * math.sin(9 * a - phase * 1.7))
            pts.append([int(cx + rx * k * math.cos(a)), int(cy + ry * k * math.sin(a))])
        cv2.fillPoly(layer, [np.array(pts, np.int32)], colour)
    layer = cv2.GaussianBlur(layer, (0, 0), max(1.0, rx * 0.16))
    m = (layer.astype(np.float32) / 255.0) * alpha
    img[:] = np.clip(img.astype(np.float32) * (1 - m) + layer.astype(np.float32) * m,
                     0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
@dataclass
class Scenario:
    key: str
    label: str
    truth: str          # "fire" | "smoke" | "nuisance"
    detail: str
    duration: float = 20.0

    def frame(self, t: float) -> np.ndarray:      # pragma: no cover - overridden
        return _sensor(_PLATE.copy(), t)


# ----------------------------- TRUE POSITIVES -----------------------------
class PalletFire(Scenario):
    def frame(self, t):
        img = _PLATE.copy()
        if t > 2.0:
            g = min(1.0, (t - 2.0) / 12.0)                 # ignition -> established
            base_r = 12 + 44 * g
            f = 1.0 + 0.34 * _chaotic(t)
            cx, cy = 92, 196
            glow = int(52 * g)
            _blob(img, cx, cy + 10, base_r * 2.4, base_r * 1.0,
                  (glow // 3, glow // 2, glow), 0.45)       # floor glow
            for i, (dy, rs, col) in enumerate((
                    (0, 1.00, (18, 92, 240)),               # BGR: deep orange
                    (-base_r * 0.55, 0.72, (36, 150, 250)), # orange
                    (-base_r * 0.95, 0.42, (96, 210, 252)), # yellow core
                    (-base_r * 1.25, 0.20, (170, 238, 254)))):
                rr = base_r * rs * f * (1.0 + 0.10 * _chaotic(t * 1.3, i))
                _blob(img, cx + 5 * _chaotic(t, i * 2), cy + dy - base_r * 0.5,
                      rr * 0.8, rr * 1.35, col, 0.92, ragged=0.30, phase=t * 6 + i)
            if t > 7:                                       # smoke follows fire
                for i in range(4):
                    ph = (t * 0.42 + i * 0.25) % 1.0
                    _blob(img, cx + 22 * math.sin(t * 0.7 + i), cy - 70 - 150 * ph,
                          22 + 60 * ph, 18 + 50 * ph, (128, 130, 132),
                          0.30 * (1 - ph) * min(1, (t - 7) / 5), ragged=0.4, phase=t + i)
        return _sensor(img, t)


class EarlyIgnition(Scenario):
    """The one that matters commercially: a small flame, early, far from camera.
    Catching this at 30 seconds instead of 5 minutes is the whole product."""
    def frame(self, t):
        img = _PLATE.copy()
        if t > 3.0:
            g = min(1.0, (t - 3.0) / 15.0)
            r = 5 + 16 * g
            f = 1.0 + 0.38 * _chaotic(t, 1.4)
            cx, cy = 468, 214
            for i, (dy, rs, col) in enumerate((
                    (0, 1.0, (22, 104, 238)),
                    (-r * 0.6, 0.62, (60, 176, 250)),
                    (-r * 1.0, 0.28, (150, 230, 253)))):
                rr = r * rs * f
                _blob(img, cx + 3 * _chaotic(t, i), cy + dy - r * 0.4,
                      rr * 0.75, rr * 1.4, col, 0.9, ragged=0.34, phase=t * 7 + i)
        return _sensor(img, t)


class SmokePlume(Scenario):
    def frame(self, t):
        img = _PLATE.copy()
        if t > 2.0:
            g = min(1.0, (t - 2.0) / 10.0)
            src = (300, 150)
            for i in range(11):
                # Puff emission is jittered with value noise. A strictly periodic
                # puff train gives the plume a rhythm that real smoke does not
                # have, and the detector was right to be suspicious of it.
                ph = ((t * 0.30) + i / 11.0 + 0.08 * _vnoise(t, 0.7, i)) % 1.0
                rise = 210 * ph
                rad = (14 + 74 * ph) * g * (1.0 + 0.18 * _vnoise(t, 1.9, i + 20))
                a = 0.42 * (1 - ph * 0.85) * g * (1.0 + 0.15 * _vnoise(t, 2.6, i + 40))
                _blob(img, src[0] + 34 * math.sin(t * 0.55 + i * 1.9) * ph
                      + 12 * _vnoise(t, 1.1, i + 60),
                      src[1] - rise, rad * 1.15, rad,
                      (140, 142, 145), a, ragged=0.45, phase=t * 1.5 + i)
            # smoke scatters light: local contrast in the plume drops
            x0, y0, x1, y1 = 200, max(0, int(150 - 210)), 420, 175
            roi = img[max(0, y0):y1, x0:x1]
            if roi.size:
                img[max(0, y0):y1, x0:x1] = cv2.GaussianBlur(roi, (0, 0), 1.6 * g)
        return _sensor(img, t)


# ------------------------------- NUISANCES --------------------------------
class Welding(Scenario):
    """Blue-white saturated core, hard on/off duty cycle, orange spatter.
    The single most common false alarm in any metal workshop."""
    def frame(self, t):
        img = _PLATE.copy()
        duty = (t * 1.0) % 3.4
        on = duty < 2.0
        cx, cy = 305, 172
        if on:
            j = 1.0 + 0.5 * math.sin(t * 61.0)             # arc hiss
            _blob(img, cx, cy, 46 * j, 40 * j, (150, 160, 170), 0.30)
            _blob(img, cx, cy, 15 * j, 13 * j, (255, 250, 240), 0.95)
            _blob(img, cx, cy, 7 * j, 6 * j, (255, 255, 255), 1.0)
            _blob(img, cx, cy, 25 * j, 21 * j, (255, 216, 170), 0.55)  # blue-white
            for _ in range(26):                                        # spatter
                a = rng.uniform(0, 2 * math.pi)
                d = rng.uniform(8, 70)
                p = (int(cx + d * math.cos(a)), int(cy + abs(d * math.sin(a)) * 0.7))
                cv2.circle(img, p, rng.integers(1, 3), (40, 170, 255), -1)
        return _sensor(img, t)


class RotatingBeacon(Scenario):
    """Forklift / machine-guard beacon. Amber, and it flickers hard - but on a
    fixed 1.6 Hz period. Autocorrelation is what kills it."""
    def frame(self, t):
        img = _PLATE.copy()
        cx, cy = 556, 88
        k = abs(math.sin(2 * math.pi * 1.6 * t)) ** 2
        cv2.rectangle(img, (546, 96), (566, 130), (60, 62, 66), -1)
        _blob(img, cx, cy, 30 + 16 * k, 26 + 14 * k, (20, 120, 245), 0.30 + 0.45 * k)
        _blob(img, cx, cy, 12 + 8 * k, 11 + 7 * k, (80, 200, 255), 0.55 + 0.42 * k)
        return _sensor(img, t)


class WorkLamp(Scenario):
    """Halogen lamp: exactly the right colour, and completely static."""
    def frame(self, t):
        img = _PLATE.copy()
        cx, cy = 180, 74
        n = 1.0 + 0.012 * math.sin(t * 3.0)
        _blob(img, cx, cy, 40 * n, 36 * n, (60, 150, 235), 0.34)
        _blob(img, cx, cy, 19 * n, 17 * n, (140, 220, 252), 0.85)
        _blob(img, cx, cy, 9 * n, 8 * n, (215, 245, 255), 1.0)
        return _sensor(img, t)


class SunPatch(Scenario):
    """Late-afternoon sun through a roller door, tracking slowly across the floor."""
    def frame(self, t):
        img = _PLATE.copy()
        x = 120 + 9 * t
        pts = np.array([[x, 250], [x + 150, 244], [x + 186, 330], [x + 28, 340]], np.int32)
        layer = np.zeros_like(img)
        cv2.fillPoly(layer, [pts], (95, 175, 232))
        layer = cv2.GaussianBlur(layer, (0, 0), 11)
        m = layer.astype(np.float32) / 255.0 * 0.55
        img[:] = np.clip(img * (1 - m) + layer * m, 0, 255).astype(np.uint8)
        return _sensor(img, t)


class HiVisWorker(Scenario):
    """Orange vest crossing the frame. Right hue, wrong everything else."""
    def frame(self, t):
        img = _PLATE.copy()
        x = int(40 + 46 * t) % (W + 120) - 60
        y = 190 + int(3 * math.sin(t * 5))
        cv2.rectangle(img, (x, y), (x + 34, y + 52), (30, 130, 245), -1)   # vest
        cv2.rectangle(img, (x + 2, y + 18), (x + 32, y + 25), (210, 235, 245), -1)
        cv2.circle(img, (x + 17, y - 12), 12, (95, 125, 160), -1)          # head
        cv2.rectangle(img, (x + 4, y + 52), (x + 14, y + 92), (55, 58, 70), -1)
        cv2.rectangle(img, (x + 20, y + 52), (x + 30, y + 92), (55, 58, 70), -1)
        return _sensor(img, t)


class GrinderSparks(Scenario):
    """Angle grinder: a shower of tiny incandescent particles on ballistic arcs."""
    def frame(self, t):
        img = _PLATE.copy()
        ox, oy = 262, 178
        burst = (t % 4.0) < 2.6
        if burst:
            _blob(img, ox, oy, 16, 14, (120, 200, 250), 0.55)
            for _ in range(150):
                age = rng.uniform(0, 1)
                a = rng.uniform(-2.7, -0.5)
                v = rng.uniform(60, 210)
                px = ox + math.cos(a) * v * age
                py = oy + math.sin(a) * v * age + 110 * age * age
                c = (60, 190, 255) if age < 0.6 else (30, 110, 200)
                cv2.circle(img, (int(px), int(py)), 1, c, -1)
        return _sensor(img, t)


class Forklift(Scenario):
    """Grey vehicle crossing: the classic smoke-channel false positive."""
    def frame(self, t):
        img = _PLATE.copy()
        x = int(-140 + 74 * t)
        cv2.rectangle(img, (x, 168), (x + 118, 236), (120, 122, 124), -1)
        cv2.rectangle(img, (x + 16, 130), (x + 26, 170), (108, 110, 112), -1)
        cv2.circle(img, (x + 24, 236), 15, (40, 42, 44), -1)
        cv2.circle(img, (x + 96, 236), 15, (40, 42, 44), -1)
        return _sensor(img, t)


class QuietWorkshop(Scenario):
    """Nothing happening. Verifies the motion gate and the false-alarm floor."""
    def frame(self, t):
        return _sensor(_PLATE.copy(), t, noise=3.6)


SCENARIOS: dict[str, Scenario] = {s.key: s for s in [
    PalletFire("pallet_fire", "Pallet stack fire", "fire",
               "Established flame at the base of a pallet stack, growing, with smoke after 7s"),
    EarlyIgnition("early_ignition", "Early ignition (small, distant)", "fire",
                  "Small flame on racking at the far side of the bay - the commercially important case"),
    SmokePlume("smoke_plume", "Smouldering smoke plume", "smoke",
               "Grey plume rising from a machine, no visible flame"),
    Welding("welding", "MIG welding", "nuisance",
            "Blue-white arc with orange spatter, 2s on / 1.4s off"),
    RotatingBeacon("beacon", "Rotating amber beacon", "nuisance",
                   "Machine-guard beacon flashing at 1.6 Hz"),
    WorkLamp("work_lamp", "Halogen work lamp", "nuisance",
             "Static warm lamp, fire-coloured and completely steady"),
    SunPatch("sun_patch", "Sunlight through roller door", "nuisance",
             "Large warm patch drifting across the floor"),
    HiVisWorker("hi_vis", "Worker in hi-vis", "nuisance",
                "Orange vest crossing the frame at walking pace"),
    GrinderSparks("grinder", "Angle grinder sparks", "nuisance",
                  "Shower of incandescent particles, intermittent"),
    Forklift("forklift", "Forklift crossing", "nuisance",
             "Grey vehicle - the classic smoke-channel false positive"),
    QuietWorkshop("quiet", "Quiet workshop (empty)", "nuisance",
                  "Idle scene, sensor noise only - measures the false-alarm floor"),
]}


def write_video(sc: Scenario, path: str, fps: int = 12, seconds: float | None = None) -> str:
    seconds = seconds or sc.duration
    for fourcc, ext in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        p = path if path.endswith(ext) else path.rsplit(".", 1)[0] + ext
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))
        if not vw.isOpened():
            continue
        for i in range(int(seconds * fps)):
            vw.write(sc.frame(i / fps))
        vw.release()
        return p
    raise RuntimeError("no usable OpenCV video encoder found")
