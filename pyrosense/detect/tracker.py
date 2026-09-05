"""
Temporal evidence accumulation.

A single frame cannot tell you whether an orange blob is a fire. Five seconds of
that blob's behaviour can. This module keeps candidate regions identified across
frames and derives the features that actually discriminate:

  flicker        coefficient of variation of the blob area
  periodicity    peak autocorrelation of that signal, *after detrending*
  growth         fractional area change per second (least-squares fit)
  rise           upward centroid drift (smoke goes up; a forklift does not)
  updraft        upward optical flow *inside* the region
  speed          net translation in blob-widths/s (a fire is anchored to fuel)
  density        filled fraction of the bounding box (flame body vs spark cloud)
  intermittency  how often the source blinks fully off (tools blink; fires do not)
  roughness      perimeter irregularity, median over the window

Three of these earned their place by catching bugs rather than by theory:

  Detrending before autocorrelation. A growing plume correlates with itself simply
  because it is growing, and that trend alone pushed the autocorrelation to 0.82 -
  over the "this is a machine" veto line. The detector was rejecting real smoke
  *because* it was spreading. Removing the linear trend first measures what the
  veto was always meant to measure: genuine repetition.

  Intermittency. An angle grinder's glow point is small, dense, ragged, orange and
  anchored - it passes every other test a flame passes. What it does that a fire
  never does is stop. Recording "absent" as a zero sample, rather than as no
  sample at all, is what makes that visible.

  Updraft. A plume that has reached steady state stops moving its own centroid -
  it is a standing column, continuously replenished from below - so `rise` decays
  to nothing exactly when the plume is most obvious to a human. Measured against
  the nuisance suite, internal upward transport runs ~0.035 for a real plume,
  ~0.0009 for a drifting sunlit patch, and slightly negative for a passing
  forklift: a 30x margin, on the one signal that never switches off.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from collections import deque

import numpy as np

from .signals import Region

WINDOW = 72          # samples retained per track (~6 s at 12 fps)
MAX_GAP_S = 3.0      # a track survives this long unmatched before it is dropped


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / float(aw * ah + bw * bh - inter)


def _centre_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Fallback association: centres close relative to blob size. Flames change
    shape fast enough that IoU alone drops tracks mid-fire."""
    acx, acy = a[0] + a[2] / 2, a[1] + a[3] / 2
    bcx, bcy = b[0] + b[2] / 2, b[1] + b[3] / 2
    reach = 0.7 * max(a[2], a[3], b[2], b[3])
    return math.hypot(acx - bcx, acy - bcy) <= reach


@dataclass
class Track:
    tid: int
    kind: str
    box: tuple[int, int, int, int]
    first_t: float
    last_t: float                 # last time it was actually SEEN
    last_update: float = 0.0      # last time it was touched (seen or gap)
    hits: int = 1
    areas: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    times: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    cx: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    cy: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    rough: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    dens: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    flow: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    blue_core: float = 0.0
    sat_ratio: float = 0.0
    mean_v: float = 0.0
    last_region: Region | None = None
    _ckey: tuple = ()
    _cache: dict = field(default_factory=dict)

    # --------------------------------------------------------------- helpers
    def _compute(self) -> dict:
        """Every temporal feature in ONE vectorised pass, cached per frame.

        This began as a set of tidy little @property methods, each looping over
        the sample deque in pure Python. Read one at a time they were fine. Read
        by the scorer they were ruinous: nine properties, each rebuilding the same
        filtered lists, over a 72-sample window, per track, per frame - all of it
        holding the GIL. Single-camera cost was 9.8 ms/frame; with six cameras
        running it became 40 ms, because the camera threads were serialising on
        Python bytecode instead of overlapping inside numpy and OpenCV.

        Numpy releases the GIL, so doing the work once per frame in array form
        both cuts the absolute cost and lets the threads genuinely run in
        parallel. The properties below are kept as thin readers over this cache,
        because `tr.flicker` reads better at the call site than a dict lookup.

        Gap samples (area == 0) are excluded from every shape statistic, so an
        absence never masquerades as a dramatic change in size - but they are kept
        for `presence` and `intermittency`, which are precisely about absence.
        """
        key = (self.last_update, len(self.areas))
        if self._ckey == key:
            return self._cache

        A = np.fromiter(self.areas, np.float64, len(self.areas))
        T = np.fromiter(self.times, np.float64, len(self.times))
        live = A > 0
        n_all = A.size
        a, t = A[live], T[live]
        n = a.size

        f = dict(flicker=0.0, periodicity=0.0, growth=0.0, rise=0.0, speed=0.0,
                 density=0.0, roughness=0.0, updraft=0.0, presence=1.0,
                 intermittency=0.0, scale=8.0)

        if n:
            f["scale"] = max(8.0, float(np.sqrt(np.median(a))))
            f["density"] = float(np.median(
                np.fromiter(self.dens, np.float64, len(self.dens))[live]))
            f["roughness"] = float(np.median(
                np.fromiter(self.rough, np.float64, len(self.rough))[live]))

        if n >= 6:
            m = float(a.mean())
            if m > 0:
                f["flicker"] = float(a.std() / m)

        if n >= 8:
            ts = t - t[0]
            mt, ma = float(ts.mean()), float(a.mean())
            den = float(((ts - mt) ** 2).sum())
            if den > 1e-9 and ma > 0:
                f["growth"] = float(((ts - mt) * (a - ma)).sum() / den / ma)

            cy = np.fromiter(self.cy, np.float64, len(self.cy))[live]
            cx = np.fromiter(self.cx, np.float64, len(self.cx))[live]
            span = float(t[-1] - t[0])
            if span > 0.2:
                f["rise"] = float((cy[0] - cy[-1]) / f["scale"] / span)
            if span > 0.3:
                d = float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))
                f["speed"] = d / max(6.0, float(self.box[2])) / span
            fl = np.fromiter(self.flow, np.float64, len(self.flow))[live]
            if fl.size >= 6 and span > 0.2:
                f["updraft"] = float(-np.median(fl) * ((n - 1) / span) / f["scale"])

        if n >= 16:
            # Detrended autocorrelation via FFT: O(n log n) and entirely inside
            # numpy, so the GIL stays free while it runs.
            x = t - t.mean()
            y = a - a.mean()
            dx = float(x @ x)
            if dx > 1e-9:
                y = y - (float(x @ y) / dx) * x        # remove the linear trend
            y = y - y.mean()
            denom = float(y @ y)
            if denom > 1e-9:
                nfft = 1 << int(np.ceil(np.log2(2 * n)))
                F = np.fft.rfft(y, nfft)
                ac = np.fft.irfft(F * np.conjugate(F), nfft)[:n]
                hi = max(3, n // 3)
                f["periodicity"] = float(np.clip(ac[2:hi].max() / denom, 0.0, 1.0))

        if n_all >= 8:
            f["presence"] = float(live.mean())
        if n_all >= 12:
            span_all = float(T[-1] - T[0])
            if span_all > 1.0:
                f["intermittency"] = float(
                    np.count_nonzero(live[1:] != live[:-1]) / 2.0 / span_all)

        self._ckey = key
        self._cache = f
        return f

    # -------------------------------------------------------------- features
    @property
    def age(self) -> float:
        return self.last_update - self.first_t

    @property
    def seen_age(self) -> float:
        return self.last_t - self.first_t

    @property
    def flicker(self) -> float:
        """Coefficient of variation of area. Steady lamp ~0.02, fire ~0.15-0.5."""
        return self._compute()["flicker"]

    @property
    def periodicity(self) -> float:
        """Peak autocorrelation of the detrended area signal over lags 2..N/3.
        High (>0.7) means a machine is doing it: strobe, beacon, blinking
        indicator, a fan chopping a light beam. Turbulent combustion stays low."""
        return self._compute()["periodicity"]

    @property
    def growth(self) -> float:
        """Fractional area growth per second. Fire grows; a work lamp has been
        exactly the same size for six years."""
        return self._compute()["growth"]

    @property
    def rise(self) -> float:
        """Upward centroid drift in blob-scales/s (image y grows downward)."""
        return self._compute()["rise"]

    @property
    def updraft(self) -> float:
        """Median upward texture drift inside the region, in blob-scales/s."""
        return self._compute()["updraft"]

    @property
    def speed(self) -> float:
        """Net centroid translation in blob-widths per second. This is what
        separates a walking worker in a hi-vis vest from a fire: the vest is
        exactly the right colour and its apparent area varies as it crosses the
        frame, but it travels, and a fire stays with its fuel."""
        return self._compute()["speed"]

    @property
    def density(self) -> float:
        """Median filled fraction of the bounding box. A flame is a contiguous
        body; a shower of grinder sparks is a sparse cloud in a large box."""
        return self._compute()["density"]

    @property
    def roughness(self) -> float:
        return self._compute()["roughness"]

    @property
    def scale(self) -> float:
        """Characteristic size in px: sqrt of the median mask area. Used to
        normalise rise and updraft instead of the bounding box, which a sprawling
        ragged plume inflates until genuine motion rounds to zero."""
        return self._compute()["scale"]

    @property
    def presence(self) -> float:
        """Fraction of the observation window in which the source was visible."""
        return self._compute()["presence"]

    @property
    def intermittency(self) -> float:
        """Full on/off blink cycles per second. Roughly 0 for combustion, 0.25/s
        for a grinder on a 4-second duty cycle."""
        return self._compute()["intermittency"]


class TrackPool:
    def __init__(self, iou_thresh: float = 0.10):
        self.tracks: dict[int, Track] = {}
        self._next = 1
        self.iou_thresh = iou_thresh

    def update(self, regions: list[Region], now: float | None = None) -> list[Track]:
        now = now if now is not None else time.time()
        unmatched = list(regions)

        pairs = []
        for tid, tr in self.tracks.items():
            for ri, r in enumerate(unmatched):
                if r.kind != tr.kind:
                    continue
                s = _iou(tr.box, r.box)
                if s < self.iou_thresh and _centre_ok(tr.box, r.box):
                    s = self.iou_thresh          # proximity rescue, ranked last
                if s >= self.iou_thresh:
                    pairs.append((s, tid, ri))
        pairs.sort(key=lambda p: (-p[0], p[1]))

        used_t: set[int] = set()
        used_r: set[int] = set()
        for s, tid, ri in pairs:
            if tid in used_t or ri in used_r:
                continue
            used_t.add(tid)
            used_r.add(ri)
            self._absorb(self.tracks[tid], unmatched[ri], now)

        for ri, r in enumerate(unmatched):
            if ri in used_r:
                continue
            t = Track(tid=self._next, kind=r.kind, box=r.box,
                      first_t=now, last_t=now, last_update=now)
            self._next += 1
            self._absorb(t, r, now)
            self.tracks[t.tid] = t

        # Unmatched tracks record an explicit absence rather than simply pausing.
        # That zero sample is what the intermittency feature reads.
        for tid, tr in list(self.tracks.items()):
            if tid in used_t:
                continue
            if (now - tr.last_t) > MAX_GAP_S:
                del self.tracks[tid]
                continue
            if now > tr.last_update:
                tr.areas.append(0.0)
                tr.times.append(now)
                tr.cx.append(tr.cx[-1] if tr.cx else 0.0)
                tr.cy.append(tr.cy[-1] if tr.cy else 0.0)
                tr.rough.append(0.0)
                tr.dens.append(0.0)
                tr.flow.append(0.0)
                tr.last_update = now

        return list(self.tracks.values())

    @staticmethod
    def _absorb(tr: Track, r: Region, now: float) -> None:
        tr.box = r.box
        tr.last_t = now
        tr.last_update = now
        tr.hits += 1
        tr.areas.append(float(r.mask_area))
        tr.times.append(now)
        tr.cx.append(r.centroid[0])
        tr.cy.append(r.centroid[1])
        tr.rough.append(r.roughness)
        tr.dens.append(r.density)
        tr.flow.append(r.flow_dy)
        tr.blue_core = max(tr.blue_core, r.blue_core)
        tr.sat_ratio = r.sat_ratio
        tr.mean_v = r.mean_v
        tr.last_region = r
