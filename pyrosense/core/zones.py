"""
Zones and arming schedules.

The highest-leverage false-alarm control in the entire product is not an
algorithm - it is a polygon. A welding bay generates arcs all day; drawing a
polygon around it and arming it only outside shift hours removes an entire class
of alarm without weakening detection anywhere else in the building.

Three controls, all per-camera:

  mask       polygons that are watched (or explicitly excluded)
  schedule   when each zone is armed, in local time
  sensitivity a per-zone threshold offset, so the paint store can be twitchy and
             the welding bay can be conservative

A zone with `mode: exclude` is a hard mask: nothing inside it is ever assessed.
That is the right tool for a permanently-visible furnace door or a monitor.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


@dataclass
class Schedule:
    """e.g. days=['mon'..'fri'], start='18:00', end='06:00' (wraps midnight)."""
    days: list[str] = field(default_factory=lambda: list(_DAYS))
    start: str = "00:00"
    end: str = "24:00"

    def armed_at(self, ts: float | None = None) -> bool:
        lt = time.localtime(ts if ts is not None else time.time())
        if _DAYS[lt.tm_wday] not in self.days:
            return False
        mins = lt.tm_hour * 60 + lt.tm_min
        s = self._m(self.start)
        e = self._m(self.end)
        return (s <= mins < e) if s <= e else (mins >= s or mins < e)

    @staticmethod
    def _m(hhmm: str) -> int:
        h, m = (hhmm.split(":") + ["0"])[:2]
        return int(h) * 60 + int(m)


@dataclass
class Zone:
    name: str
    points: list[tuple[float, float]]      # normalised 0..1, so resolution-independent
    mode: str = "include"                  # include | exclude
    schedule: Schedule = field(default_factory=Schedule)
    threshold_delta: float = 0.0           # + = less sensitive here
    kinds: list[str] = field(default_factory=lambda: ["flame", "smoke"])
    note: str = ""

    def polygon(self, w: int, h: int) -> np.ndarray:
        return np.array([[int(x * w), int(y * h)] for x, y in self.points], np.int32)


class ZoneSet:
    """Compiles zones into a mask at the detector's working resolution."""

    def __init__(self, zones: list[Zone] | None = None):
        self.zones = zones or []
        self._cache: dict[tuple, np.ndarray] = {}

    def mask(self, w: int, h: int, ts: float | None = None) -> np.ndarray | None:
        """255 where detection is armed. None means 'watch everything'.

        The distinction below is a safety-relevant bug fix. There are two
        different reasons the armed-include list can come back empty:

          no zones are configured at all   -> watch everything (None)
          zones exist but none are on duty -> watch nothing (a zero mask)

        The first version collapsed both into None. The effect was that a camera
        whose only zone was scheduled for 18:00-06:00 on weekdays did not go quiet
        at the weekend - it silently reverted to watching the entire frame,
        including the welding bay the schedule existed to exclude. It was caught
        on a Saturday morning: the weld-bay camera was reporting 100% flame
        confidence on an arc it was supposed to be ignoring.

        Failing open on missing configuration is right. Failing open on an
        explicit schedule is just ignoring the operator."""
        if not self.zones:
            return None                                # nothing configured

        includes = [z for z in self.zones if z.mode == "include" and z.schedule.armed_at(ts)]
        excludes = [z for z in self.zones if z.mode == "exclude"]
        any_include_defined = any(z.mode == "include" for z in self.zones)

        key = (w, h, tuple(z.name for z in includes), tuple(z.name for z in excludes))
        if key in self._cache:
            return self._cache[key]

        m = np.zeros((h, w), np.uint8)
        if includes:
            for z in includes:
                cv2.fillPoly(m, [z.polygon(w, h)], 255)
        elif any_include_defined:
            pass                             # include zones exist but are off duty
        else:
            m[:] = 255                       # only exclusions defined: watch the rest
        for z in excludes:
            cv2.fillPoly(m, [z.polygon(w, h)], 0)

        if len(self._cache) > 32:
            self._cache.clear()
        self._cache[key] = m
        return m

    def zone_at(self, x: float, y: float, w: int, h: int) -> str:
        """Which zone a point falls in - used to label the alert."""
        for z in self.zones:
            if z.mode != "include":
                continue
            if cv2.pointPolygonTest(z.polygon(w, h), (float(x), float(y)), False) >= 0:
                return z.name
        return ""

    def threshold_delta(self, x: float, y: float, w: int, h: int) -> float:
        for z in self.zones:
            if z.mode != "include":
                continue
            if cv2.pointPolygonTest(z.polygon(w, h), (float(x), float(y)), False) >= 0:
                return z.threshold_delta
        return 0.0

    @staticmethod
    def from_config(items: list[dict]) -> "ZoneSet":
        zones = []
        for it in items or []:
            sch = it.get("schedule") or {}
            zones.append(Zone(
                name=it.get("name", "zone"),
                points=[tuple(p) for p in it.get("points", [])],
                mode=it.get("mode", "include"),
                schedule=Schedule(days=sch.get("days", list(_DAYS)),
                                  start=sch.get("start", "00:00"),
                                  end=sch.get("end", "24:00")),
                threshold_delta=float(it.get("threshold_delta", 0.0)),
                kinds=it.get("kinds", ["flame", "smoke"]),
                note=it.get("note", ""),
            ))
        return ZoneSet(zones)

    def to_config(self) -> list[dict]:
        return [dict(name=z.name, points=[list(p) for p in z.points], mode=z.mode,
                     schedule=dict(days=z.schedule.days, start=z.schedule.start,
                                   end=z.schedule.end),
                     threshold_delta=z.threshold_delta, kinds=z.kinds, note=z.note)
                for z in self.zones]
