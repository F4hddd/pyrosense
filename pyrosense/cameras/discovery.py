"""
Find the cameras that are already on the site network.

The realistic install story is: someone hands you a factory with 24 cameras from
three vendors, installed over nine years by two different contractors, and nobody
knows the passwords or the IP plan. So discovery runs three ways and merges:

  1. ONVIF WS-Discovery. A UDP multicast probe that conformant cameras answer with
     their own service address. Costs one packet and finds most modern devices.
  2. TCP sweep of the local /24 for the RTSP and ONVIF ports, for the older or
     mis-configured devices that ignore multicast (and for networks where the
     switch drops multicast between VLANs, which is common).
  3. ARP table lookup for MAC addresses, then an OUI match to guess the vendor.

Then, given a credential, we verify by actually pulling a frame - because a port
being open proves nothing, and the only test that matters is whether we can
decode video.

No vendor SDK is involved at any point.
"""
from __future__ import annotations

import concurrent.futures as cf
import ipaddress
import re
import shutil
import socket
import struct
import subprocess
import time
import uuid
from dataclasses import dataclass, field

from .profiles import PROFILES, PROBE_PATHS, build_url, fingerprint, redact

WS_DISCOVERY_ADDR = ("239.255.255.250", 3702)

_PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
  xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
  xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
  xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
 <e:Header>
  <w:MessageID>uuid:{mid}</w:MessageID>
  <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
  <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
 </e:Header>
 <e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>"""


@dataclass
class Found:
    host: str
    ports: list[int] = field(default_factory=list)
    onvif_xaddr: str = ""
    mac: str = ""
    vendor_hint: str = ""
    brand: str = "generic"
    confidence: float = 0.0
    verified_url: str = ""
    note: str = ""

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["verified_url"] = redact(self.verified_url)
        d["brand_label"] = PROFILES.get(self.brand, PROFILES["generic"]).label
        return d


# --------------------------------------------------------------------------
def ws_discover(timeout: float = 4.0) -> dict[str, Found]:
    """Multicast ONVIF probe. Returns {host: Found}."""
    out: dict[str, Found] = {}
    msg = _PROBE.format(mid=uuid.uuid4()).encode()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        s.settimeout(0.6)
        s.bind(("", 0))
        for _ in range(2):                      # multicast is lossy; ask twice
            s.sendto(msg, WS_DISCOVERY_ADDR)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data, addr = s.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            host = addr[0]
            text = data.decode("utf-8", "ignore")
            xaddr = ""
            m = re.search(r"<[^>]*XAddrs[^>]*>(.*?)</[^>]*XAddrs>", text, re.S | re.I)
            if m:
                xaddr = m.group(1).strip().split()[0] if m.group(1).strip() else ""
            scopes = " ".join(re.findall(r"onvif://www\.onvif\.org/\S+", text))
            f = out.setdefault(host, Found(host=host))
            f.onvif_xaddr = xaddr or f.onvif_xaddr
            f.vendor_hint = (f.vendor_hint + " " + scopes).strip()[:400]
        s.close()
    except Exception:
        pass
    for f in out.values():
        f.brand, f.confidence = fingerprint(onvif_vendor=f.vendor_hint)
    return out


def local_subnets() -> list[str]:
    nets = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip.startswith("127."):
                continue
            nets.append(str(ipaddress.ip_network(ip + "/24", strict=False)))
    except Exception:
        pass
    return sorted(set(nets))


def _port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def sweep(cidr: str, ports: tuple[int, ...] = (554, 8554, 80, 8000),
          workers: int = 256) -> dict[str, Found]:
    """TCP sweep for camera-ish ports. Catches devices that ignore multicast."""
    net = ipaddress.ip_network(cidr, strict=False)
    hosts = [str(h) for h in net.hosts()]
    out: dict[str, Found] = {}

    def probe(h: str) -> tuple[str, list[int]]:
        return h, [p for p in ports if _port_open(h, p)]

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for host, open_ports in ex.map(probe, hosts):
            if any(p in (554, 8554) for p in open_ports):
                out[host] = Found(host=host, ports=open_ports)
    return out


def arp_table() -> dict[str, str]:
    """ip -> mac, from the OS ARP cache. Works on Windows and Linux."""
    table: dict[str, str] = {}
    exe = shutil.which("arp")
    if not exe:
        return table
    try:
        txt = subprocess.run([exe, "-a"], capture_output=True, text=True,
                             timeout=8).stdout
    except Exception:
        return table
    for line in txt.splitlines():
        m = re.search(r"(\d+\.\d+\.\d+\.\d+)\D+([0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}"
                      r"[:-][0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}"
                      r"[:-][0-9a-fA-F]{2})", line)
        if m:
            table[m.group(1)] = m.group(2).replace("-", ":").lower()
    return table


def http_banner(host: str, timeout: float = 1.5) -> str:
    """Grab whatever the device says about itself over plain HTTP."""
    try:
        with socket.create_connection((host, 80), timeout=timeout) as s:
            s.sendall(b"GET / HTTP/1.0\r\nHost: " + host.encode() + b"\r\n\r\n")
            return s.recv(4096).decode("utf-8", "ignore")
    except Exception:
        return ""


def verify_stream(url: str, timeout: float = 8.0) -> bool:
    """The only test that counts: can ffprobe actually decode video here."""
    exe = shutil.which("ffprobe")
    if not exe:
        return False
    try:
        r = subprocess.run(
            [exe, "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width", "-of", "csv=p=0",
             "-timeout", str(int(timeout * 1_000_000)), url],
            capture_output=True, text=True, timeout=timeout + 4)
        return bool(r.stdout.strip()) and r.returncode == 0
    except Exception:
        return False


def probe_paths(host: str, user: str, pw: str, brand: str = "",
                timeout: float = 6.0, max_tries: int = 10) -> str:
    """Try the brand's documented path first, then sweep the common ones.

    This is the function that turns 'my installer left and I don't know the URL'
    into a working camera. It is intentionally sequential and capped - hammering
    a camera with parallel RTSP DESCRIBEs makes some firmware lock the account."""
    candidates: list[str] = []
    if brand and brand in PROFILES:
        for stream in ("sub", "main"):
            candidates.append(build_url(brand, host, user, pw, stream=stream))
    from urllib.parse import quote
    u, p = quote(user, safe=""), quote(pw, safe="")
    for path in PROBE_PATHS:
        candidates.append(f"rtsp://{u}:{p}@{host}:554{path}")

    seen: set[str] = set()
    tries = 0
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        tries += 1
        if tries > max_tries:
            break
        if verify_stream(url, timeout):
            return url
    return ""


def discover(cidr: str = "", do_sweep: bool = True, timeout: float = 4.0) -> list[Found]:
    """Full discovery pass. Merges all three methods and fingerprints the result."""
    found = ws_discover(timeout=timeout)

    if do_sweep:
        nets = [cidr] if cidr else local_subnets()
        for net in nets:
            for host, f in sweep(net).items():
                if host in found:
                    found[host].ports = sorted(set(found[host].ports) | set(f.ports))
                else:
                    found[host] = f

    arp = arp_table()
    for host, f in found.items():
        f.mac = arp.get(host, "")
        if f.confidence < 0.9:
            banner = http_banner(host)
            brand, conf = fingerprint(mac=f.mac, http_banner=banner,
                                      onvif_vendor=f.vendor_hint)
            if conf > f.confidence:
                f.brand, f.confidence = brand, conf
        prof = PROFILES.get(f.brand, PROFILES["generic"])
        f.note = prof.notes

    return sorted(found.values(), key=lambda f: tuple(int(o) for o in f.host.split(".")))
