"""
Stage 3: adjudication by a vision model.

This stage is what makes the product deployable, and it is worth being precise
about why.

Stages 0-2 produce a number. A number cannot be argued with, audited, or trusted
by a night-shift supervisor who has been woken up twice this month for a welder.
Once an alarm has been muted, the system is worth nothing - and mute is exactly
what happens to any detector that cries wolf.

So before an alert leaves the building, we send Claude two crops - a wide
establishing frame and a tight crop of the candidate - together with the physical
measurements stage 1 already made, and ask a direct question: is this an
uncontrolled fire, or is it one of the dozen workshop things that look like one?

Three properties make this affordable and safe:

  It runs rarely.  Only on candidates that already survived every earlier stage.
  On a real site that is a handful of calls a day, not thousands, so the cost is
  rounding-error next to one truck roll for a false alarm.

  It returns prose.  The operator gets "flame at the base of the pallet stack,
  growing, no blue arc core and no periodic strobe - not consistent with welding"
  rather than "0.87". That is an alert someone acts on.

  It can only ever *reduce* what leaves the building on its own initiative.
  A `fail_open` policy means an API outage, a rate limit or a refusal results in
  the alert being sent with stage-1 evidence and a clear note that adjudication
  did not run. A fire alarm must never be silenced by a network problem.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

SYSTEM = """You are the verification stage of an industrial fire-detection system \
watching CCTV in workshops, factories and warehouses.

An upstream computer-vision stage has already flagged a candidate region using \
physical measurements: colour chrominance, area flicker, autocorrelation \
(periodicity), growth rate, perimeter roughness, translation speed and, for smoke, \
internal upward optical flow. You are the final check before a human is alerted.

Your job is to prevent false alarms without ever suppressing a real fire.

Industrial scenes routinely contain things that look like fire to a colour-based \
detector. The most common are:
- welding and cutting arcs (blue-white saturated core, orange spatter)
- angle grinder and cutting spark showers
- hot or glowing metal, furnaces, kilns, molten material
- rotating amber beacons, strobes, warning and indicator lamps
- halogen and sodium work lamps, and their reflections in polished metal or wet floors
- sunlight or sunset through doors and skylights, moving across surfaces
- hi-vis clothing, orange machinery, safety paint, plastic barriers
- steam, condensation, exhaust plumes, dust from traffic (which mimic smoke)

Judge whether the region shows UNCONTROLLED COMBUSTION - a fire that should not \
be there - or SMOKE from such a fire.

Deliberate industrial processes are NOT fires: a welder welding, a furnace \
burning, a gas torch in use. Flag those as nuisance and name the process. But if \
a deliberate process appears to have ignited surrounding material - flame spread \
to a pallet, a rag, packaging, insulation - that IS a fire.

Weigh the measurements you are given. They are real physical evidence, not a \
guess. If they conflict with your visual read, say so in your reasoning.

When the image is too dark, too small, too compressed or too ambiguous to judge, \
return "unclear" rather than guessing. "unclear" is escalated to a human, so it \
is a safe answer; a wrong "nuisance" is not."""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string",
                    "enum": ["fire", "smoke", "nuisance", "unclear"]},
        "confidence": {"type": "number"},
        "nuisance_type": {
            "type": "string",
            "description": "If verdict is nuisance, the specific source: welding, "
                           "grinding, hot_metal, beacon, work_lamp, reflection, "
                           "sunlight, hi_vis_clothing, steam, dust, exhaust, "
                           "screen_or_display, other. Empty string otherwise.",
        },
        "reasoning": {"type": "string",
                      "description": "Two or three sentences an operator can act "
                                     "on, citing what you actually see."},
        "spread_risk": {"type": "string", "enum": ["none", "low", "medium", "high"]},
        "recommended_action": {"type": "string",
                               "enum": ["ignore", "log", "notify", "evacuate_and_call"]},
    },
    "required": ["verdict", "confidence", "nuisance_type", "reasoning",
                 "spread_risk", "recommended_action"],
    "additionalProperties": False,
}


@dataclass
class Adjudication:
    verdict: str = "unclear"
    confidence: float = 0.0
    nuisance_type: str = ""
    reasoning: str = ""
    spread_risk: str = "none"
    recommended_action: str = "notify"
    ran: bool = False
    error: str = ""
    latency_ms: int = 0
    usage: dict = field(default_factory=dict)

    @property
    def confirms(self) -> bool:
        return self.verdict in ("fire", "smoke")

    @property
    def dismisses(self) -> bool:
        return self.verdict == "nuisance"

    def to_json(self) -> dict:
        return dict(self.__dict__)


def _jpeg_b64(img: np.ndarray, quality: int = 80, max_w: int = 900) -> str:
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


class Adjudicator:
    """Thread-safe, rate-limited wrapper around the Claude vision call."""

    def __init__(self, model: str = "claude-opus-5", api_key: str | None = None,
                 effort: str = "medium", max_calls_per_hour: int = 60,
                 fail_open: bool = True, timeout_s: float = 25.0):
        self.model = model
        self.effort = effort
        self.max_calls_per_hour = max_calls_per_hour
        self.fail_open = fail_open
        self.timeout_s = timeout_s
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None
        self._lock = threading.Lock()
        self._calls: list[float] = []

    # ------------------------------------------------------------------ setup
    @property
    def available(self) -> bool:
        """The SDK resolves credentials from more places than the env var (an
        `ant auth login` profile, for one), so an unset key is not proof that we
        cannot call. We only report unavailable if the import itself fails."""
        try:
            import anthropic  # noqa: F401
        except Exception:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            import anthropic
            kwargs = {"timeout": self.timeout_s, "max_retries": 1}
            if self._key:
                kwargs["api_key"] = self._key
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _budget_ok(self) -> bool:
        """A crude circuit breaker. If something upstream starts flapping, this
        stops it from turning into an unbounded API bill overnight."""
        now = time.time()
        with self._lock:
            self._calls = [t for t in self._calls if now - t < 3600]
            if len(self._calls) >= self.max_calls_per_hour:
                return False
            self._calls.append(now)
            return True

    # ------------------------------------------------------------------- call
    def adjudicate(self, context_bgr: np.ndarray, crop_bgr: np.ndarray,
                   kind: str, features: dict, camera: str = "",
                   zone: str = "") -> Adjudication:
        a = Adjudication()
        if not self.available:
            a.error = "anthropic SDK not installed"
            return a
        if not self._budget_ok():
            a.error = f"rate limit: {self.max_calls_per_hour} adjudications/hour reached"
            return a

        t0 = time.perf_counter()
        try:
            client = self._get_client()
            feat_lines = "\n".join(
                f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}"
                for k, v in sorted(features.items()))
            prompt = (
                f"Camera: {camera or 'unknown'}\n"
                f"Zone: {zone or 'unspecified'}\n"
                f"Upstream channel: {kind}\n\n"
                f"Physical measurements from the vision stage:\n{feat_lines}\n\n"
                f"Image 1 is the full camera view. Image 2 is a zoomed crop of the "
                f"flagged region.\n\n"
                f"Is this an uncontrolled fire (or smoke from one), or a workshop "
                f"nuisance? Answer with the required JSON only."
            )

            resp = client.beta.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM,
                # Server-side fallback: if a safety classifier declines this
                # request, the platform routes it to another model rather than
                # handing us a refusal. A fire alert must not be lost to one.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": SCHEMA}},
                # The system prompt above is byte-stable across every call, so it
                # is worth a cache breakpoint. Whether it actually caches depends
                # on the model's minimum cacheable prefix; on a low-volume alert
                # path this is a nice-to-have, not the main cost lever.
                cache_control={"type": "ephemeral"},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg",
                            "data": _jpeg_b64(context_bgr)}},
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg",
                            "data": _jpeg_b64(crop_bgr, quality=88, max_w=640)}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )

            a.latency_ms = int((time.perf_counter() - t0) * 1000)

            if getattr(resp, "stop_reason", "") == "refusal":
                # Fail open: report it and let stage-1 evidence stand.
                det = getattr(resp, "stop_details", None)
                a.error = f"model declined ({getattr(det, 'category', 'unknown')})"
                return a

            text = next((b.text for b in resp.content if b.type == "text"), "")
            data = json.loads(text)
            a.verdict = data["verdict"]
            a.confidence = float(data["confidence"])
            a.nuisance_type = data.get("nuisance_type", "")
            a.reasoning = data.get("reasoning", "")
            a.spread_risk = data.get("spread_risk", "none")
            a.recommended_action = data.get("recommended_action", "notify")
            a.ran = True
            u = getattr(resp, "usage", None)
            if u is not None:
                a.usage = {"input": getattr(u, "input_tokens", 0),
                           "output": getattr(u, "output_tokens", 0),
                           "cache_read": getattr(u, "cache_read_input_tokens", 0)}
        except Exception as e:
            a.error = f"{type(e).__name__}: {e}"[:300]
            a.latency_ms = int((time.perf_counter() - t0) * 1000)
        return a

    # --------------------------------------------------------------- policy
    def apply(self, stage1_score: float, adj: Adjudication) -> tuple[bool, str]:
        """Combine stage-1 evidence with the adjudication into a send/hold call.

        The asymmetry here is deliberate and is the safety-critical part of the
        whole system: adjudication is allowed to CANCEL an alert only when it is
        both confident and specific about what the thing actually is. Anything
        else - an error, a timeout, a rate limit, an "unclear", a low-confidence
        dismissal - results in the alert going out anyway.

        A detector that can be silenced by an unreachable API is not a fire
        detector."""
        if not adj.ran:
            if self.fail_open:
                return True, (f"sent without adjudication ({adj.error or 'not run'}) "
                              f"- stage-1 confidence {stage1_score:.2f}")
            return False, f"held: adjudication unavailable ({adj.error})"

        if adj.dismisses and adj.confidence >= 0.75 and adj.nuisance_type:
            return False, (f"suppressed: identified as {adj.nuisance_type} "
                           f"({adj.confidence:.0%} confident). {adj.reasoning}")
        if adj.verdict == "unclear":
            return True, f"sent for human review - ambiguous. {adj.reasoning}"
        if adj.confirms:
            return True, adj.reasoning
        return True, (f"sent despite low-confidence dismissal "
                      f"({adj.confidence:.0%}). {adj.reasoning}")
