"""
Command line.

    python -m pyrosense demo                    virtual cameras, no hardware needed
    python -m pyrosense run  -c config.json     a real site
    python -m pyrosense discover                find cameras on the LAN
    python -m pyrosense probe  --host ... --user ... --password ...
    python -m pyrosense bench                   the nuisance gauntlet
    python -m pyrosense init                    write an example config
    python -m pyrosense scenarios --write       render the test videos to disk
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _serve(engine, host: str, port: int, open_browser: bool = False) -> None:
    import uvicorn
    from .server.app import create_app
    app = create_app(engine)
    engine.start()
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    print(f"\n  PyroSense console -> {url}\n")
    if open_browser:
        import threading, webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        engine.stop()


def cmd_demo(a) -> None:
    from .config import cameras_from, demo_config
    from .core.engine import Engine
    cfg = demo_config()
    print(f"  site: {cfg['site']}")
    print("  6 virtual cameras: 3 genuine incidents, 3 nuisances that mimic fire")
    engine = Engine(cameras_from(cfg), alert_cfg=cfg.get("alerts"),
                    vlm_cfg={**cfg.get("vlm", {}), "enabled": not a.no_vlm})
    if engine.adjudicator is None:
        print("  stage 3 (Claude adjudication) is OFF - set ANTHROPIC_API_KEY to enable")
    else:
        print(f"  stage 3 adjudication: {engine.adjudicator.model}")
    _serve(engine, a.host, a.port, open_browser=not a.no_browser)


def cmd_run(a) -> None:
    from .config import cameras_from, load, uplink_from
    from .core.engine import Engine
    from .detect.neural import NeuralConfig
    cfg = load(a.config)
    cams = cameras_from(cfg)
    if not cams:
        raise SystemExit("no cameras in config")
    n = cfg.get("neural", {})
    uplink = None if a.no_cloud else uplink_from(cfg)
    engine = Engine(cams, alert_cfg=cfg.get("alerts"), vlm_cfg=cfg.get("vlm"),
                    neural_cfg=NeuralConfig(flame_model=n.get("flame_model", ""),
                                            smoke_model=n.get("smoke_model", "")),
                    urgent_cfg=cfg.get("urgent"), uplink=uplink,
                    dashboard_url=cfg.get("dashboard_url", ""))
    print(f"  site: {cfg.get('site','(unnamed)')}   cameras: {len(cams)}")
    ch = engine.escalation.channels if engine.escalation else {}
    live = [k for k, v in ch.items() if v and k != "dry_run"]
    print(f"  urgent alerting: {', '.join(live) or 'none configured'}"
          f"{'  (DRY RUN)' if ch.get('dry_run') else ''}")
    print(f"  cloud uplink: {'-> ' + cfg['cloud']['url'] if uplink else 'disabled'}")
    _serve(engine, a.host, a.port)


def cmd_cloud(a) -> None:
    """Run the hosted mirror locally - same app Render serves."""
    import os
    import uvicorn
    os.environ.setdefault("PYRO_INGEST_TOKEN", a.token)
    os.environ.setdefault("PYRO_SITE_NAME", a.site)
    from .cloud.server import create_cloud_app
    print(f"\n  PyroSense cloud mirror -> http://127.0.0.1:{a.port}\n")
    uvicorn.run(create_cloud_app(), host=a.host, port=a.port, log_level="warning")


def cmd_test_alert(a) -> None:
    """Fire the urgent ladder end to end without waiting for a real fire."""
    from .config import load
    from .alerts.urgent import build_escalation
    cfg = load(a.config) if a.config else {}
    urgent = dict(cfg.get("urgent") or {})
    if a.live:
        urgent["dry_run"] = False
    esc = build_escalation(urgent)
    ch = esc.channels
    print(f"  channels: {ch}")
    if ch.get("dry_run"):
        print("  DRY RUN - nothing will actually be sent. Add --live to really send.")
    inc = esc.raise_incident("test-" + str(int(time.time())), a.camera, "flame",
                             0.97, "TEST ALERT - this is a PyroSense self-test, "
                                   "there is no fire.",
                             url=cfg.get("dashboard_url", ""))
    print(f"  incident {inc.event_id} raised; watching the ladder for {a.watch}s")
    print("  (acknowledge on your phone to see escalation stop)")
    t0 = time.time()
    while time.time() - t0 < a.watch:
        time.sleep(3)
        st = [i for i in esc.status()["incidents"] if i["event_id"] == inc.event_id]
        if st:
            i = st[0]
            print(f"    t+{i['age_s']:>5}s step={i['step']} "
                  f"acked={i['acknowledged']} {i['log'][-1]['msg'] if i['log'] else ''}")
            if i["acknowledged"]:
                print("  acknowledged - ladder stopped. This is the behaviour you want.")
                break
    esc.stop()


def cmd_discover(a) -> None:
    from .cameras.discovery import discover
    print("  probing ONVIF multicast and sweeping the local subnet...")
    t0 = time.time()
    found = discover(cidr=a.cidr, do_sweep=not a.no_sweep, timeout=a.timeout)
    print(f"  {len(found)} candidate camera(s) in {time.time()-t0:.1f}s\n")
    if not found:
        print("  Nothing found. Common causes: the cameras are on another VLAN,")
        print("  multicast is blocked by the switch, or a host firewall is dropping")
        print("  the replies. Try: python -m pyrosense discover --cidr 192.168.1.0/24")
        return
    print(f"  {'host':<17}{'ports':<16}{'brand':<34}mac")
    print("  " + "-" * 92)
    for f in found:
        from .cameras.profiles import PROFILES
        label = PROFILES.get(f.brand, PROFILES["generic"]).label
        conf = f" ({f.confidence:.0%})" if f.confidence else ""
        print(f"  {f.host:<17}{','.join(map(str,f.ports)) or '-':<16}"
              f"{(label+conf)[:33]:<34}{f.mac or '-'}")
        if f.note:
            print(f"        note: {f.note}")
    print("\n  Next: python -m pyrosense probe --host <ip> --user admin --password '<pw>'")


def cmd_probe(a) -> None:
    from .cameras.discovery import probe_paths
    from .cameras.profiles import redact
    print(f"  probing {a.host} ...")
    url = probe_paths(a.host, a.user, a.password, brand=a.brand)
    if not url:
        print("\n  No working RTSP path found. Check that:")
        print("   - RTSP is enabled on the camera (Imou/Tapo need it switched on in the app)")
        print("   - the account is a LOCAL camera account, not the vendor cloud login")
        print("   - the password has no characters that need URL-escaping, or pass --brand")
        return
    print(f"\n  WORKS: {redact(url)}\n")
    print("  Config snippet:\n")
    print(json.dumps({"name": f"cam-{a.host.split('.')[-1]}",
                      "source": {"kind": "rtsp", "url": url, "fps": 12}}, indent=2))


def cmd_bench(a) -> None:
    from .bench import main as bench_main
    sys.argv = ["bench"] + (["--json", a.json] if a.json else [])
    bench_main()


def cmd_init(a) -> None:
    from .config import write_example
    p = write_example(a.out)
    print(f"  wrote {p} - edit it, then: python -m pyrosense run -c {p}")


def cmd_scenarios(a) -> None:
    from .synth.scenarios import SCENARIOS, write_video
    print(f"  {len(SCENARIOS)} scenarios\n")
    for k, sc in SCENARIOS.items():
        print(f"  {k:<16}{sc.truth:<10}{sc.label}")
        print(f"  {'':16}{sc.detail}")
    if a.write:
        os.makedirs(a.out, exist_ok=True)
        print()
        for k, sc in SCENARIOS.items():
            p = write_video(sc, os.path.join(a.out, f"{k}.mp4"))
            print(f"  wrote {p}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser("pyrosense", description="AI fire & smoke detection "
                                                          "for existing CCTV")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run with virtual cameras - no hardware needed")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8080)
    d.add_argument("--no-vlm", action="store_true", help="skip Claude adjudication")
    d.add_argument("--no-browser", action="store_true")
    d.set_defaults(fn=cmd_demo)

    r = sub.add_parser("run", help="run against a real site config")
    r.add_argument("-c", "--config", required=True)
    r.add_argument("--host", default="0.0.0.0")
    r.add_argument("--port", type=int, default=8080)
    r.add_argument("--no-cloud", action="store_true",
                   help="run detection but do not push to the cloud mirror")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("cloud", help="run the hosted public mirror (what Render runs)")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=8090)
    c.add_argument("--token", default="dev-token")
    c.add_argument("--site", default="Dev Site")
    c.set_defaults(fn=cmd_cloud)

    ta = sub.add_parser("test-alert", help="fire the urgent alert ladder as a self-test")
    ta.add_argument("-c", "--config", default="")
    ta.add_argument("--camera", default="self-test")
    ta.add_argument("--live", action="store_true", help="actually send (not dry-run)")
    ta.add_argument("--watch", type=float, default=120)
    ta.set_defaults(fn=cmd_test_alert)

    s = sub.add_parser("discover", help="find cameras on the network")
    s.add_argument("--cidr", default="", help="e.g. 192.168.1.0/24")
    s.add_argument("--no-sweep", action="store_true", help="ONVIF multicast only")
    s.add_argument("--timeout", type=float, default=4.0)
    s.set_defaults(fn=cmd_discover)

    p = sub.add_parser("probe", help="find the working RTSP URL for one camera")
    p.add_argument("--host", required=True)
    p.add_argument("--user", default="admin")
    p.add_argument("--password", default="")
    p.add_argument("--brand", default="")
    p.set_defaults(fn=cmd_probe)

    b = sub.add_parser("bench", help="run the nuisance gauntlet")
    b.add_argument("--json", default="data/bench.json")
    b.set_defaults(fn=cmd_bench)

    i = sub.add_parser("init", help="write an example config")
    i.add_argument("--out", default="pyrosense.config.json")
    i.set_defaults(fn=cmd_init)

    sc = sub.add_parser("scenarios", help="list or render the test scenarios")
    sc.add_argument("--write", action="store_true")
    sc.add_argument("--out", default="data/scenarios")
    sc.set_defaults(fn=cmd_scenarios)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
