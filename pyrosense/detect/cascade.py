"""
The detection cascade.

    Stage 0  motion gate            ~0.1 ms   rejects ~99% of frames on a calm site
    Stage 1  physics + temporal     ~2-4 ms   proposes candidates, kills known nuisances
    Stage 2  neural verifier        ~15-30 ms only on candidates, only if weights present
    Stage 3  VLM adjudication       ~1-2 s    only on would-be alerts, a few times a day
    Stage 4  persistence / N-of-M   free      an alert must survive several seconds

Why a cascade rather than "run a fire model on every frame of every camera":

  Cost.     A 30-camera site on one mini-PC has roughly 300 frame-slots/second to
            spend. A neural model per frame per camera does not fit. A motion gate
            plus 3 ms of numpy does, with headroom.
  Accuracy. The expensive stages only ever see hard cases, so you can afford a
            genuinely good model (or a VLM) at that point.
  Trust.    Stage 3 returns *prose*. The operator gets "growing orange flame at the
            base of the pallet stack, no blue arc core, not consistent with
            welding" instead of "0.87". That is the difference between an alarm
            people act on and an alarm people mute.

The stage-1 score is a plain additive log-odds model on purpose. Every alert can
therefore be explained term by term, tuned by a human on site, and defended in a
safety review. A black box cannot do any of those things.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .signals import SignalExtractor, FrameSignals
from .tracker import TrackPool, Track


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


@dataclass
class Term:
    name: str
    value: float
    weight: float
    note: str = ""

    @property
    def contribution(self) -> float:
        return self.value * self.weight


@dataclass
class Assessment:
    kind: str                       # "flame" | "smoke"
    score: float
    box: tuple[int, int, int, int]
    track_id: int
    terms: list[Term] = field(default_factory=list)
    vetoes: list[str] = field(default_factory=list)
    age: float = 0.0
    features: dict = field(default_factory=dict)

    @property
    def vetoed(self) -> bool:
        return bool(self.vetoes)

    def explain(self) -> str:
        if self.vetoes:
            return "suppressed: " + "; ".join(self.vetoes)
        top = sorted(self.terms, key=lambda t: -abs(t.contribution))[:4]
        return "; ".join(f"{t.name}={t.value:.2f}{(' (' + t.note + ')') if t.note else ''}"
                         for t in top)


# ---------------------------------------------------------------------------
# Stage 1 scoring. Weights are log-odds contributions, hand-calibrated against
# the synthetic nuisance suite in pyrosense/synth and intended to be re-tuned per
# site from the review queue. `bias` is deliberately strongly negative: the prior
# for "this blob is a fire" on a random workshop camera is very low, and the
# evidence has to work to overcome it.
# ---------------------------------------------------------------------------
FLAME_BIAS = -4.6
SMOKE_BIAS = -5.0

# Evidence shrinkage. Every feature in this model is temporal - flicker,
# periodicity, growth, intermittency - and none of them means anything after
# 400 ms of observation. Without this, a brand-new track full of not-yet-measured
# features scored 0.99 within half a second, and a rotating beacon could raise an
# alarm *before* its own "you are a beacon" veto had enough history to fire.
#
# So the evidence sum is scaled toward the prior by how much of the observation
# window we actually have. No evidence -> the score is just the (very low) prior.
FLAME_MATURITY_S = 4.0
SMOKE_MATURITY_S = 5.0

# Minimum observation before a track is scored AT ALL.
#
# This closes a structural hole rather than tuning a threshold. Several vetoes
# carry their own age gate, because the feature they read is meaningless on a
# very young track - "this blinks off 0.25 times/s" needs to have watched it
# blink. But scoring had no such gate, so a track in the window between "old
# enough to score" and "old enough to veto" was judged on its positive evidence
# with all of its disqualifying evidence still switched off.
#
# It showed up as a drifting sunlit patch raising one smoke alarm 47 seconds into
# a soak run. Every frame around it was correctly vetoed; the alarm came from a
# 3.7-second-old track that the updraft veto (gated at 4.0s) could not yet see.
# Raising that one gate would have hidden this instance and left the shape of the
# bug in place, so instead nothing is scored until every veto that applies to it
# can be evaluated.
MIN_EVIDENCE_S = {"flame": 2.0, "smoke": 4.0}


def _band(v: float, lo: float, hi: float) -> float:
    """1.0 inside [lo,hi], tapering to 0 outside. Used where 'too much' is as
    suspicious as 'too little' - e.g. a blob flickering at CV 3.0 is sensor noise,
    not a flame."""
    if v <= 0:
        return 0.0
    if lo <= v <= hi:
        return 1.0
    if v < lo:
        return max(0.0, v / lo)
    return max(0.0, 1.0 - (v - hi) / (hi * 2.0))


def score_flame(tr: Track, frame_px: int) -> Assessment:
    terms: list[Term] = []
    vetoes: list[str] = []

    flicker = tr.flicker
    period = tr.periodicity
    growth = tr.growth
    rough = tr.roughness
    speed = tr.speed
    density = tr.density
    intermittency = tr.intermittency
    area_frac = (sum(tr.areas) / len(tr.areas)) / max(1, frame_px) if tr.areas else 0.0

    # ---- positive evidence -------------------------------------------------
    terms.append(Term("flicker", _band(flicker, 0.10, 0.75), 3.1,
                      "turbulent area oscillation"))
    terms.append(Term("edge_roughness", _band(rough, 0.30, 4.0), 1.5,
                      "ragged flame perimeter"))
    terms.append(Term("growth", max(0.0, min(1.0, growth / 0.25)), 2.3,
                      "expanding over time"))
    terms.append(Term("persistence", min(1.0, tr.age / 3.0), 1.4,
                      f"{tr.age:.1f}s on camera"))
    terms.append(Term("size_plausible", _band(area_frac, 0.0008, 0.25), 0.9))
    terms.append(Term("body_density", _band(density, 0.30, 0.95), 1.2,
                      "contiguous flame body"))
    terms.append(Term("anchored", 1.0 - min(1.0, speed / 0.5), 1.6,
                      "fixed to its fuel, not travelling"))
    terms.append(Term("continuity", tr.presence, 1.8,
                      "burning continuously, never blinks out"))

    # ---- learned nuisance penalties (soft) ---------------------------------
    terms.append(Term("periodicity", -min(1.0, period / 0.7), 2.6,
                      "regular = machine, not combustion"))
    terms.append(Term("blue_core", -min(1.0, tr.blue_core / 0.25), 3.0,
                      "welding arcs have a blue-white core"))

    # ---- hard vetoes (physics says this cannot be a fire) ------------------
    if tr.blue_core > 0.18:
        vetoes.append(f"welding/arc signature (blue-white saturated core "
                      f"{tr.blue_core:.0%} of blob)")
    if period > 0.78 and flicker > 0.08:
        vetoes.append(f"periodic source, autocorrelation peak {period:.2f} "
                      f"- rotating beacon or strobe, not combustion")
    if flicker < 0.06 and abs(growth) < 0.03 and tr.age > 2.0:
        vetoes.append(f"no flicker (area CV {flicker:.3f}) and not growing - a "
                      f"fixed lamp, a sunlit patch or a painted surface. Combustion "
                      f"always modulates its own area")
    if rough < 0.22 and tr.age > 2.0:
        vetoes.append(f"smooth near-circular boundary (roughness {rough:.2f}) - a "
                      f"lens, bulb or beacon. Flame fronts are always ragged")
    if speed > 0.45 and tr.age > 1.5:
        vetoes.append(f"travelling at {speed:.1f} blob-widths/s - a moving object "
                      f"(hi-vis clothing, vehicle light), not an anchored fire")
    if intermittency > 0.10 and tr.age > 4.0:
        vetoes.append(f"source blinks fully off {intermittency:.2f} times/s - an "
                      f"intermittent tool, arc or indicator lamp; a fire does not stop")
    if density < 0.20 and tr.age > 1.5:
        vetoes.append(f"sparse particle cloud ({density:.0%} of its own bounding "
                      f"box) - grinder or spark shower, not a flame body")
    if area_frac > 0.55:
        vetoes.append("covers most of the frame - exposure shift or IR-cut "
                      "switch, not a localised fire")

    maturity = min(1.0, tr.age / FLAME_MATURITY_S)
    logit = FLAME_BIAS + maturity * sum(t.contribution for t in terms)
    return Assessment(
        kind="flame", score=0.0 if vetoes else _sigmoid(logit), box=tr.box,
        track_id=tr.tid, terms=terms, vetoes=vetoes, age=tr.age,
        features=dict(flicker=flicker, periodicity=period, growth=growth,
                      roughness=rough, area_frac=area_frac, speed=speed,
                      density=density, blue_core=tr.blue_core, mean_v=tr.mean_v,
                      intermittency=intermittency, presence=tr.presence,
                      maturity=maturity),
    )


def score_smoke(tr: Track, frame_px: int) -> Assessment:
    terms: list[Term] = []
    vetoes: list[str] = []

    growth = tr.growth
    rise = tr.rise
    updraft = tr.updraft
    period = tr.periodicity
    speed = tr.speed
    area_frac = (sum(tr.areas) / len(tr.areas)) / max(1, frame_px) if tr.areas else 0.0

    # Centroid rise is a weak proxy and it is demonstrably spoofable: a sunlit
    # patch drifting across a floor produced rise=0.11 and raised a false smoke
    # alarm on a long run, while its internal transport stayed at ~0.001. So the
    # centroid term is kept only as corroboration, and the veto below leans
    # entirely on the flow measurement.
    terms.append(Term("upward_drift", max(0.0, min(1.0, rise / 0.12)), 1.0,
                      "plume outline climbing"))
    # Internal upward transport is the strongest discriminator in the smoke
    # channel. Measured over 60s of continuous running per scenario, on
    # smoke-channel tracks older than 4s (blob-scales per second):
    #
    #   smouldering plume    p50 0.0651   p90 0.1145   max 0.1250
    #   sunlit patch         p50 0.0000   p90 0.0096   max 0.0172
    #   forklift crossing    p50 0.0000   p90 0.0000   max 0.0000
    #   welding bay          p50 0.0000   p90 0.0000   max 0.0000
    #
    # The veto line below sits at 0.030 - roughly 1.7x above the worst nuisance
    # observed and half the plume median. An earlier value of 0.010 was set from
    # a single 20-second run and turned out to sit inside the sunlit patch's own
    # tail, which is what let that patch raise a smoke alarm 47 seconds into a
    # soak. These are synthetic distributions; the same measurement must be
    # repeated on site footage before the line is trusted in the field.
    terms.append(Term("internal_updraft", max(0.0, min(1.0, updraft / 0.060)), 3.4,
                      "texture inside the region transporting upward"))
    terms.append(Term("growth", max(0.0, min(1.0, growth / 0.15)), 2.8,
                      "volume expanding"))
    terms.append(Term("persistence", min(1.0, tr.age / 5.0), 1.8,
                      f"{tr.age:.1f}s on camera"))
    terms.append(Term("diffuse_edge", _band(tr.roughness, 0.4, 6.0), 1.2,
                      "soft irregular boundary"))
    terms.append(Term("size_plausible", _band(area_frac, 0.004, 0.4), 0.9))
    terms.append(Term("periodicity", -min(1.0, period / 0.7), 2.0))
    terms.append(Term("continuity", tr.presence, 1.4, "plume persists"))

    if period > 0.80:
        vetoes.append(f"periodic grey motion (autocorr {period:.2f}) - machinery, "
                      f"conveyor or fan, not a plume")
    if rise < -0.10 and tr.age > 2.0:
        vetoes.append("region is descending - dust kicked up by traffic or a "
                      "vehicle crossing, not smoke")
    if speed > 0.55 and rise < 0.10 and tr.age > 2.0:
        vetoes.append(f"translating sideways at {speed:.1f} widths/s without "
                      f"rising - a vehicle or pallet, not a plume")
    if updraft < 0.030 and tr.age > 4.0:
        vetoes.append(f"grey region with no internal upward transport (updraft "
                      f"{updraft:+.4f}/s) - a sunlit patch, a shadow, a shifting "
                      f"light or a passing vehicle. Smoke always carries material "
                      f"upward through itself; this does not")
    if tr.age < MIN_EVIDENCE_S["smoke"]:
        vetoes.append("too brief to be a developing plume")

    maturity = min(1.0, tr.age / SMOKE_MATURITY_S)
    logit = SMOKE_BIAS + maturity * sum(t.contribution for t in terms)
    return Assessment(
        kind="smoke", score=0.0 if vetoes else _sigmoid(logit), box=tr.box,
        track_id=tr.tid, terms=terms, vetoes=vetoes, age=tr.age,
        features=dict(rise=rise, updraft=updraft, growth=growth,
                      periodicity=period, speed=speed,
                      roughness=tr.roughness, area_frac=area_frac,
                      presence=tr.presence, maturity=maturity),
    )


# ---------------------------------------------------------------------------


@dataclass
class CascadeResult:
    t: float
    assessments: list[Assessment] = field(default_factory=list)
    motion_ratio: float = 0.0
    gated: bool = False                 # True when stage 0 skipped the frame
    scene_cut: bool = False             # True while recovering from a discontinuity
    stage1_ms: float = 0.0
    frame: np.ndarray | None = None
    signals: FrameSignals | None = None

    @property
    def best(self) -> Assessment | None:
        live = [a for a in self.assessments if not a.vetoed]
        return max(live, key=lambda a: a.score) if live else None


class Cascade:
    """One per camera. Stateful: owns the background model and the track pool."""

    def __init__(self, work_width: int = 480, detect_smoke: bool = True,
                 motion_threshold: float = 0.0008, neural=None,
                 zone_mask: np.ndarray | None = None,
                 scene_cut_ratio: float = 0.5, scene_cut_settle_s: float = 2.5):
        self.sig = SignalExtractor(work_width=work_width, detect_smoke=detect_smoke)
        self.pool = TrackPool()
        self.motion_threshold = motion_threshold
        self.neural = neural            # optional stage 2, see neural.py
        self.zone_mask = zone_mask      # uint8 mask in work resolution, 255 = armed
        self.scene_cut_ratio = scene_cut_ratio
        self.scene_cut_settle_s = scene_cut_settle_s
        self._settle_until = 0.0
        self.scene_cuts = 0

    def __call__(self, frame_bgr: np.ndarray, now: float | None = None) -> CascadeResult:
        now = now if now is not None else time.time()
        t0 = time.perf_counter()

        s = self.sig.process(frame_bgr, self.motion_threshold)
        res = CascadeResult(t=now, motion_ratio=s.motion_ratio,
                            frame=s.frame_bgr, signals=s)

        # ---- scene-discontinuity guard --------------------------------------
        # When half the frame changes between two consecutive frames, nothing
        # continuous happened - the scene was replaced. In the field that means
        # the IR-cut filter switching at dusk, the lights going on or off, a PTZ
        # preset recalling, a camera rebooting, or an exposure step. Every
        # temporal feature this detector relies on is meaningless across such a
        # cut: tracks get associated to unrelated regions, and the optical flow
        # reports the whole scene sliding, which reads as a plume with a strong
        # updraft.
        #
        # A looping test scenario is exactly this discontinuity, which is how it
        # was found - a drifting sunlit patch raised a smoke alarm seven seconds
        # after the loop wrapped, and only on the long soak run. Rather than
        # patch the symptom, we detect the cut and start again: drop the tracks,
        # forget the change memory, and assess nothing until the scene has
        # settled. Losing 2.5 s after a lighting change is a good trade for not
        # alarming on every sunset.
        if s.motion_ratio >= self.scene_cut_ratio:
            self.pool.tracks.clear()
            self.sig.reset_change_memory()
            self._settle_until = now + self.scene_cut_settle_s
            self.scene_cuts += 1
        if now < self._settle_until:
            res.scene_cut = True
            res.stage1_ms = (time.perf_counter() - t0) * 1000
            return res

        if not s.regions and s.motion_ratio < self.motion_threshold:
            res.gated = True
            res.stage1_ms = (time.perf_counter() - t0) * 1000
            return res

        regions = s.regions
        if self.zone_mask is not None and regions:
            regions = [r for r in regions if self._in_zone(r)]

        frame_px = (s.frame_bgr.shape[0] * s.frame_bgr.shape[1]) if s.frame_bgr is not None else 1
        tracks = self.pool.update(regions, now)

        for tr in tracks:
            # Skip tracks that are currently absent (last sample is a gap), too
            # short to hold a feature, or younger than the point at which their
            # own vetoes become evaluable.
            if len(tr.areas) < 4 or (tr.areas and tr.areas[-1] <= 0):
                continue
            if tr.age < MIN_EVIDENCE_S.get(tr.kind, 2.0):
                continue
            a = score_flame(tr, frame_px) if tr.kind == "flame" else score_smoke(tr, frame_px)

            # ---- Stage 2: neural verifier on the candidate crop -------------
            if self.neural is not None and not a.vetoed and a.score > 0.25 and s.frame_bgr is not None:
                x, y, w, h = a.box
                pad = int(0.25 * max(w, h))
                H, W = s.frame_bgr.shape[:2]
                crop = s.frame_bgr[max(0, y - pad):min(H, y + h + pad),
                                   max(0, x - pad):min(W, x + w + pad)]
                if crop.size:
                    p = self.neural.predict(crop, a.kind)
                    if p is not None:
                        a.terms.append(Term("neural", p * 2 - 1, 2.8, "CNN verifier"))
                        logit = math.log(a.score / max(1e-6, 1 - a.score)) + (p * 2 - 1) * 2.8
                        a.score = _sigmoid(logit)
                        a.features["neural"] = p

            res.assessments.append(a)

        res.stage1_ms = (time.perf_counter() - t0) * 1000
        return res

    def _in_zone(self, r) -> bool:
        m = self.zone_mask
        cx, cy = int(r.centroid[0]), int(r.centroid[1])
        if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1]:
            return bool(m[cy, cx])
        return False
