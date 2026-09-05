"""
Regression harness over the synthetic scenario suite.

Reports, per scenario: whether the alarm rule fired, how long it took, the peak
stage-1 score, and the reason. Then a confusion matrix and the headline numbers
a client will ask about - detection rate, false-alarm rate, mean time to alarm.

Run:  python -m pyrosense.bench
"""
from __future__ import annotations

import argparse
import json
import time

from .synth.scenarios import SCENARIOS, Scenario
from .detect.cascade import Cascade
from .core.events import AlarmRule


def run_scenario(sc: Scenario, fps: int = 12, seconds: float | None = None,
                 threshold: float = 0.72, sustain: float = 2.5,
                 loop: bool = False) -> dict:
    """Run one scenario through the full pipeline.

    `loop` repeats the scenario to fill `seconds`, which is how the live system
    actually sees a camera: continuously, not for one tidy 20-second window. That
    mode earns its keep - it caught a false alarm at t=47s that the fixed-length
    run could not produce, because the failure needed a track to survive past the
    point where the short run simply stopped."""
    seconds = seconds or sc.duration
    cas = Cascade()
    # Flame and smoke get separate rules. Smoke is the lower-contrast signal and
    # the earlier warning in a real fire, so it runs at a shorter persistence
    # requirement; the trade is that smoke alerts are the ones most worth routing
    # through stage-3 adjudication before they wake anyone up.
    rules = {"flame": AlarmRule(threshold=threshold, sustain_s=sustain),
             "smoke": AlarmRule(threshold=0.70, sustain_s=1.0)}
    peak, peak_reason, peak_kind = 0.0, "", ""
    fired_at = None
    veto_reasons: dict[str, dict[str, int]] = {}
    t_proc = 0.0
    n = int(seconds * fps)

    for i in range(n):
        t = i / fps
        frame = sc.frame(t % sc.duration if loop else t)   # rendering not measured
        t0 = time.perf_counter()
        res = cas(frame, now=t)
        t_proc += time.perf_counter() - t0

        # Attribute vetoes per channel. Counting them together was misleading:
        # the smoke channel vetoes something on almost every frame of every
        # camera, so it drowned out the reason a *flame* candidate was actually
        # rejected - the welding scenario was reporting "no upward transport"
        # when what really stopped it was its blue-white arc core.
        for a in res.assessments:
            for v in a.vetoes:
                key = v.split(" - ")[0].split(" (")[0].strip()
                veto_reasons.setdefault(a.kind, {})
                veto_reasons[a.kind][key] = veto_reasons[a.kind].get(key, 0) + 1

        b = res.best
        if b and b.score > peak:
            peak, peak_reason, peak_kind = b.score, b.explain(), b.kind
        for kind, rule in rules.items():
            live = [a.score for a in res.assessments
                    if a.kind == kind and not a.vetoed]
            if rule.update(t, max(live) if live else 0.0) and fired_at is None:
                fired_at = t

    truth_positive = sc.truth in ("fire", "smoke")
    return dict(
        key=sc.key, label=sc.label, truth=sc.truth, detail=sc.detail,
        alarmed=fired_at is not None, time_to_alarm=fired_at,
        peak=round(peak, 3), peak_kind=peak_kind, reason=peak_reason,
        vetoes={k: sorted(v.items(), key=lambda kv: -kv[1])[:2]
                for k, v in veto_reasons.items()},
        decisive_veto=(sorted(veto_reasons.get(peak_kind, {}).items(),
                              key=lambda kv: -kv[1])[:1] or [("", 0)])[0][0],
        ms_per_frame=round(t_proc / n * 1000, 2),
        correct=(fired_at is not None) == truth_positive,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.72)
    ap.add_argument("--sustain", type=float, default=2.5)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--json", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--soak", action="store_true",
                    help="loop each scenario continuously (default 60s) - catches "
                         "failures that need a track to outlive a short run")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="override run length per scenario")
    args = ap.parse_args()
    if args.soak and not args.seconds:
        args.seconds = 60.0

    keys = args.only.split(",") if args.only else list(SCENARIOS)
    rows = [run_scenario(SCENARIOS[k], fps=args.fps, threshold=args.threshold,
                         sustain=args.sustain, seconds=args.seconds or None,
                         loop=args.soak) for k in keys]

    print()
    mode = f"SOAK {args.seconds:.0f}s looped" if args.soak else "fixed-length"
    print(f"  PyroSense benchmark [{mode}]   threshold={args.threshold} "
          f"sustain={args.sustain}s  fps={args.fps}")
    print("  " + "-" * 100)
    print(f"  {'scenario':<26}{'truth':<10}{'alarm':<8}{'t+':<8}{'peak':<8}{'ms/f':<7}result")
    print("  " + "-" * 100)
    for r in rows:
        tta = f"{r['time_to_alarm']:.1f}s" if r["time_to_alarm"] is not None else "-"
        mark = "OK " if r["correct"] else "XX "
        verdict = ("DETECT" if r["alarmed"] else "miss") if r["truth"] != "nuisance" \
                  else ("FALSE+" if r["alarmed"] else "clean")
        print(f"  {r['label'][:25]:<26}{r['truth']:<10}{'YES' if r['alarmed'] else 'no':<8}"
              f"{tta:<8}{r['peak']:<8.2f}{r['ms_per_frame']:<7}{mark}{verdict}")

    pos = [r for r in rows if r["truth"] != "nuisance"]
    neg = [r for r in rows if r["truth"] == "nuisance"]
    tp = sum(r["alarmed"] for r in pos)
    fp = sum(r["alarmed"] for r in neg)
    ttas = [r["time_to_alarm"] for r in pos if r["time_to_alarm"] is not None]
    mspf = sum(r["ms_per_frame"] for r in rows) / len(rows)

    print("  " + "-" * 100)
    print(f"  detection      {tp}/{len(pos)} fire+smoke scenarios")
    print(f"  false alarms   {fp}/{len(neg)} nuisance scenarios")
    if ttas:
        print(f"  mean time to alarm  {sum(ttas)/len(ttas):.1f}s "
              f"(worst {max(ttas):.1f}s)")
    print(f"  stage-1 cost   {mspf:.2f} ms/frame  ->  ~{int(1000/max(mspf,0.01))} "
          f"frames/s on one core")
    print()
    for r in rows:
        if not r["correct"]:
            print(f"  !! {r['label']}: peak {r['peak']:.2f} | {r['reason'][:90]}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
