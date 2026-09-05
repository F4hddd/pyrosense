"""
Stage 1: physics-based candidate finding.

Anyone can threshold on "orange pixels". That detector is useless in a real
workshop, because a workshop is *full* of orange: welding arcs, grinder sparks,
hot metal, forklift beacons, sodium lamps, hi-vis vests, sunset through a roller
door, and the reflection of all of the above in every polished surface.

What actually separates fire from those is not colour, it is *behaviour over time*:

  fire            chaotic flicker, grows, jagged perimeter, warm-only spectrum
  welding arc     blue-white saturated core, hard on/off, tiny + intensely bright
  strobe beacon   flickers, but *periodically* - a sharp autocorrelation peak
  work lamp       does not flicker, does not grow, near-circular
  sunlight patch  huge, smooth, drifts slowly, does not flicker
  hi-vis vest     right hue, but low value + no flicker + moves like a person

So this module extracts colour masks per frame, and `tracker.py` accumulates the
temporal evidence that does the real discrimination.

Everything here is deliberately cheap - plain numpy and OpenCV primitives on a
downscaled frame - because it runs on every camera, every frame, forever.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Region:
    """A candidate blob in one frame."""
    x: int
    y: int
    w: int
    h: int
    area: int
    kind: str                 # "flame" | "smoke"
    mask_area: int
    perimeter: float
    mean_v: float             # mean HSV value inside the blob
    sat_ratio: float          # fraction of pixels at/near sensor saturation
    blue_core: float          # fraction that is blue-white hot (welding signature)
    centroid: tuple[float, float] = (0.0, 0.0)
    flow_dy: float = 0.0      # vertical texture drift, px/frame (negative = up)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def roughness(self) -> float:
        """Perimeter^2 / (4*pi*area) - 1. Zero for a perfect circle, large for a
        ragged flame edge. Lamps and reflections are round; fire is not."""
        if self.mask_area <= 0:
            return 0.0
        return max(0.0, (self.perimeter ** 2) / (4.0 * np.pi * self.mask_area) - 1.0)

    @property
    def density(self) -> float:
        """Filled fraction of the bounding box. A flame body is dense; a shower
        of grinder sparks is a sparse cloud of specks inside a large box."""
        return self.mask_area / float(max(1, self.w * self.h))


@dataclass
class FrameSignals:
    regions: list[Region] = field(default_factory=list)
    flame_mask: np.ndarray | None = None
    smoke_mask: np.ndarray | None = None
    motion_ratio: float = 0.0
    frame_bgr: np.ndarray | None = None


class SignalExtractor:
    """Per-camera stateful extractor. Not thread-safe; one per camera worker."""

    def __init__(self, work_width: int = 480, min_area_frac: float = 0.00035,
                 detect_smoke: bool = True):
        self.work_width = work_width
        self.min_area_frac = min_area_frac
        self.detect_smoke = detect_smoke
        # MOG2 gives us "what changed vs the learned scene", which both gates the
        # expensive work and tells smoke apart from permanently grey machinery.
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=24, detectShadows=False)
        self._prev_gray: np.ndarray | None = None
        self._k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._bg_lap: float = 0.0     # running baseline of scene edge energy
        self._change_mem: np.ndarray | None = None   # decaying 'changed recently' map
        self._prev_resid: np.ndarray | None = None

    # ------------------------------------------------------------------ utils
    def _resize(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        if w <= self.work_width:
            return bgr
        s = self.work_width / float(w)
        return cv2.resize(bgr, (self.work_width, int(round(h * s))),
                          interpolation=cv2.INTER_AREA)

    def reset_change_memory(self) -> None:
        """Forget where the scene has recently changed. Called after a scene cut:
        the memory is a claim about continuity, and a cut invalidates it."""
        self._change_mem = None
        self._prev_resid = None

    # ------------------------------------------------------- Stage 0: motion
    def motion_gate(self, gray: np.ndarray, fg: np.ndarray | None = None) -> float:
        """Activity ratio: how much of the frame is 'doing something'.

        Two signals, and we take the larger. Frame differencing catches fast
        change; the background model catches slow change. Smoke needs the second
        one - a translucent plume drifting across grey racking moves individual
        pixels by only a few grey levels per frame, so pure frame differencing
        scores it as a static scene and gates it away. That bug cost this detector
        every smoke scenario until the background term was added.

        On an idle night shift both are ~0 and everything below is skipped, which
        is what makes 30+ cameras on one mini-PC affordable."""
        diff_ratio = 1.0
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            d = cv2.absdiff(gray, self._prev_gray)
            diff_ratio = float((d > 6).mean())
        self._prev_gray = gray
        fg_ratio = float((fg > 0).mean()) if fg is not None else 0.0
        return max(diff_ratio, fg_ratio)

    # --------------------------------------------------- Stage 1a: flame mask
    @staticmethod
    def flame_mask(bgr: np.ndarray) -> np.ndarray:
        """Chrominance rule set in YCbCr (Celik & Demirel), plus an HSV guard.

        YCbCr is used rather than raw RGB because the luma/chroma split makes the
        rules survive auto-exposure and white-balance swings, which every cheap
        camera does constantly. The core observation is that in fire, the red
        chroma exceeds the blue chroma by a wide margin, and both exceed the
        frame's own averages - a self-calibrating test that does not need a fixed
        brightness threshold."""
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycc)
        yf, crf, cbf = y.astype(np.int16), cr.astype(np.int16), cb.astype(np.int16)

        ym, crm, cbm = float(yf.mean()), float(crf.mean()), float(cbf.mean())

        r1 = yf > cbf                       # luma above blue chroma
        r2 = crf > cbf                      # red chroma dominates
        r3 = np.abs(cbf - crf) >= 42        # by a wide margin
        r4 = (yf > ym) & (cbf < cbm) & (crf > crm)   # brighter+warmer than scene

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # OpenCV hue is 0-179. Fire lives in red->yellow, i.e. 0-35 and 160-179.
        hue_ok = ((h <= 35) | (h >= 160))
        r5 = hue_ok & (v >= 120) & (s >= 55)

        m = (r1 & r2 & r3 & r4 & r5).astype(np.uint8) * 255
        return m

    # --------------------------------------------------- Stage 1b: smoke mask
    def smoke_mask(self, bgr: np.ndarray, fg: np.ndarray) -> np.ndarray:
        """Smoke is grey, translucent and - critically - it *blurs the scene
        behind it*. So we look for regions that (a) are desaturated and mid-bright,
        (b) are colour-neutral, (c) the background model says have changed, and
        (d) have lost local edge energy relative to the scene baseline.

        (d) is the one that matters: a grey forklift driving past keeps its own
        sharp edges. Smoke drapes over the racking behind it and softens it."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        b, g, r = cv2.split(bgr.astype(np.int16))
        spread = np.maximum.reduce([b, g, r]) - np.minimum.reduce([b, g, r])

        grey = (s < 46) & (v > 55) & (v < 235) & (spread < 34)

        # Local edge energy, compared against a running baseline of the scene.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_16S, ksize=3))
        lap_blur = cv2.blur(lap, (15, 15)).astype(np.float32)

        # The baseline is a MEDIAN and it adapts slowly, both deliberately.
        # An earlier version used a fast-moving mean, and a real plume dragged the
        # baseline down toward itself within a few seconds - the detector went
        # blind to smoke precisely because smoke was present. The median is robust
        # while the plume covers a minority of the frame, and once it covers a lot
        # of the frame we stop updating altogether rather than learn it as normal.
        cur = float(np.median(lap_blur)) + 1e-6
        grey_frac = float(grey.mean())
        if self._bg_lap == 0:
            self._bg_lap = cur
        elif grey_frac < 0.25:
            self._bg_lap = 0.998 * self._bg_lap + 0.002 * cur

        softened = lap_blur < (self._bg_lap * 0.75)

        # Change memory, rather than instantaneous foreground.
        #
        # Requiring fg>0 outright made the detector forget about smoke that stayed
        # put: MOG2 absorbs a hanging plume into the background within ~20s, which
        # is exactly backwards for the smouldering case we most want to catch.
        # Dropping the test entirely is worse though - a permanently out-of-focus
        # corner is grey and soft forever and would flag on every frame.
        #
        # So we keep a decaying memory of where the scene has changed recently.
        # It stays warm for roughly half a minute after the last motion there,
        # which covers a settled plume, and never lights up for scenery that has
        # simply always been blurry.
        if self._change_mem is None or self._change_mem.shape != fg.shape:
            self._change_mem = np.zeros(fg.shape, np.float32)
        self._change_mem *= 0.995
        np.maximum(self._change_mem, (fg > 0).astype(np.float32), out=self._change_mem)

        m = (grey & softened & (self._change_mem > 0.2)).astype(np.uint8) * 255
        return m

    # ----------------------------------------------------------- region props
    def _regions(self, mask: np.ndarray, bgr: np.ndarray, kind: str,
                 min_area: int, close_iters: int = 2) -> list[Region]:
        # Open first to drop speckle, then close hard. Closing matters more than
        # it looks: a flame is a stack of differently-coloured layers (deep red
        # base, orange body, yellow core) and the colour rules cut it into
        # separate components. Without an aggressive close the tracker sees three
        # short-lived blobs instead of one persistent flame, and the temporal
        # features - which need track continuity - never accumulate.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k5, iterations=close_iters)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        out: list[Region] = []
        hsv_v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        bch, _, rch = cv2.split(bgr)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < min_area:
                continue
            sub = (labels[y:y + h, x:x + w] == i)
            if sub.sum() == 0:
                continue
            cnts, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            per = max((cv2.arcLength(c, True) for c in cnts), default=0.0)

            v_sub = hsv_v[y:y + h, x:x + w][sub]
            b_sub = bch[y:y + h, x:x + w][sub].astype(np.int16)
            r_sub = rch[y:y + h, x:x + w][sub].astype(np.int16)
            # Welding / arc signature: sensor-saturated pixels whose blue channel
            # keeps up with red. Fire never does that - it has no blue-white core.
            blue_core = float(((v_sub >= 250) & (b_sub >= r_sub * 0.85)).mean())

            out.append(Region(
                x=int(x), y=int(y), w=int(w), h=int(h), area=int(w * h), kind=kind,
                mask_area=int(a), perimeter=float(per),
                mean_v=float(v_sub.mean()), sat_ratio=float((v_sub >= 250).mean()),
                blue_core=blue_core,
                centroid=(float(cents[i][0]), float(cents[i][1])),
            ))
        return out

    @staticmethod
    def _cluster(regions: list[Region], reach: float = 0.6) -> list[list[Region]]:
        """Group blobs that are close enough to be one physical thing.

        Unioning *every* smoke component in the frame was a mistake: a few stray
        flecks in opposite corners produced one region spanning a fifth of the
        image, and every size-normalised feature (rise, updraft) was divided by
        that bogus extent and went to nearly zero. Proximity clustering keeps a
        plume together without gluing unrelated specks onto it."""
        n = len(regions)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                a, b = regions[i], regions[j]
                pad_a = reach * max(a.w, a.h)
                pad_b = reach * max(b.w, b.h)
                pad = max(pad_a, pad_b)
                if (a.x - pad < b.x + b.w and b.x - pad < a.x + a.w and
                        a.y - pad < b.y + b.h and b.y - pad < a.y + a.h):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj

        groups: dict[int, list[Region]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(regions[i])
        return list(groups.values())

    @staticmethod
    def _aggregate(regions: list[Region]) -> list[Region]:
        """Collapse many blobs into one.

        Used for smoke, and it fixed the worst bug in the detector. Smoke is not
        an object, it is a field: the mask breaks into a handful of puffs that
        merge, split and swap identity every frame. Tracking them individually
        produced tracks that were present only a third of the time, so the
        temporal features - which are the whole point - never accumulated, and a
        clearly-visible plume peaked at 0.98 without ever sustaining an alarm.

        Treating the whole plume as one region makes the track continuous and the
        features meaningful. The cost is that two separate plumes on one camera
        report as one, which for an alerting product is the right trade."""
        if len(regions) <= 1:
            return regions
        x0 = min(r.x for r in regions)
        y0 = min(r.y for r in regions)
        x1 = max(r.x + r.w for r in regions)
        y1 = max(r.y + r.h for r in regions)
        tot = sum(r.mask_area for r in regions) or 1
        return [Region(
            x=x0, y=y0, w=x1 - x0, h=y1 - y0, area=(x1 - x0) * (y1 - y0),
            kind=regions[0].kind, mask_area=tot,
            # Perimeter comes from the dominant component, not the sum. Summing
            # perimeters over summed area produced roughness values near 30,
            # which is not "very ragged", it is meaningless.
            perimeter=max(r.perimeter for r in regions),
            mean_v=sum(r.mean_v * r.mask_area for r in regions) / tot,
            sat_ratio=sum(r.sat_ratio * r.mask_area for r in regions) / tot,
            blue_core=max(r.blue_core for r in regions),
            centroid=(sum(r.centroid[0] * r.mask_area for r in regions) / tot,
                      sum(r.centroid[1] * r.mask_area for r in regions) / tot),
        )]

    def _updraft(self, resid: np.ndarray, r: Region) -> float:
        """Vertical drift of the texture inside a region, in px/frame.

        This is the feature that finally made steady smoke detectable. A plume
        that has reached its steady state stops moving its own centroid - it is a
        standing column, continuously replenished from below - so the centroid
        "rise" feature decays to nothing exactly when the plume is most obvious to
        a human. What never stops is the upward transport *inside* it.

        Phase correlation over the region between consecutive frames measures that
        directly, and costs about 0.3 ms because it only runs where smoke was
        already proposed. Negative dy means upward in image coordinates.

        Crucially this runs on the background-subtracted RESIDUAL, not on the raw
        frame. Smoke is translucent, so a raw ROI is dominated by the stationary
        racking showing through the plume, and phase correlation dutifully reports
        the motion of the dominant texture: zero. Differencing against the learned
        background removes the scenery and leaves the plume, which is the thing we
        actually wanted to measure."""
        if self._prev_resid is None or self._prev_resid.shape != resid.shape:
            return 0.0
        x, y, w, h = r.box
        if w < 24 or h < 24:
            return 0.0
        a = self._prev_resid[y:y + h, x:x + w].astype(np.float32)
        b = resid[y:y + h, x:x + w].astype(np.float32)
        if a.shape != b.shape or a.size == 0:
            return 0.0
        try:
            (_, dy), response = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(
                (a.shape[1], a.shape[0]), cv2.CV_32F))
        except cv2.error:
            return 0.0
        # A low response means the correlation found nothing trustworthy.
        return float(dy) if response > 0.02 else 0.0

    # ------------------------------------------------------------------- main
    def process(self, bgr_full: np.ndarray, motion_threshold: float = 0.0008) -> FrameSignals:
        bgr = self._resize(bgr_full)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sig = FrameSignals(frame_bgr=bgr)

        # The background model is fed on every frame, including gated ones, so
        # the scene stays learned while the expensive stages are skipped.
        fg = self._bg.apply(bgr, learningRate=0.0015)
        sig.motion_ratio = self.motion_gate(gray, fg)

        if sig.motion_ratio < motion_threshold:
            return sig

        px = bgr.shape[0] * bgr.shape[1]
        min_area = max(12, int(px * self.min_area_frac))

        fm = self.flame_mask(bgr)
        sig.flame_mask = fm
        sig.regions.extend(self._regions(fm, bgr, "flame", min_area, close_iters=3))

        if self.detect_smoke:
            sm = self.smoke_mask(bgr, fg)
            sig.smoke_mask = sm
            # Smoke is diffuse and its own boundary is soft, so it needs both a
            # larger minimum footprint and a heavier close to become one region.
            raw = self._regions(sm, bgr, "smoke", min_area * 2, close_iters=4)
            smoke_regions = []
            for grp in self._cluster(raw):
                smoke_regions.extend(self._aggregate(grp))
            # Keep only the two largest plumes; the rest is speckle.
            smoke_regions.sort(key=lambda r: -r.mask_area)
            smoke_regions = smoke_regions[:2]
            bgimg = self._bg.getBackgroundImage()
            if bgimg is not None and bgimg.shape[:2] == bgr.shape[:2]:
                resid = cv2.absdiff(gray, cv2.cvtColor(bgimg, cv2.COLOR_BGR2GRAY))
                for r in smoke_regions:
                    r.flow_dy = self._updraft(resid, r)
                self._prev_resid = resid
            sig.regions.extend(smoke_regions)

        return sig
