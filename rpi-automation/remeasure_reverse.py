#!/usr/bin/env python3
"""Backfill reverse-direction bandwidth and RTT in an existing pair_links CSV.

The existing CSV has both directions for each pair but with identical values
(the reverse was mirrored). This script re-measures bandwidth and RTT in the
reverse direction via iperf3/ping and updates those rows in place.
Unreachable nodes are skipped gracefully.

Usage:
    python3 rpi-automation/remeasure_reverse.py [--csv <file>] [--ips <file>]
                                                 [--skip-measured]

--skip-measured  Skip any reverse row whose BW already differs from the
                 forward row (i.e. it was measured individually before).
"""
import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time

SSH_USER = "pi"
IPERF_PORT = 5201
IPERF_DURATION = 5
PING_COUNT = 20

# Will be set in main() so the signal handler can flush the file.
_rows = []
_csv_path = ""
_updated = 0


def _flush(signum=None, frame=None):
    """Write current rows to CSV and exit cleanly on SIGINT/SIGTERM."""
    if _csv_path and _rows:
        tmp = _csv_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            csv.writer(f).writerows(_rows)
        os.replace(tmp, _csv_path)
        print(f"\n\nInterrupted — saved {_updated} updates so far to {_csv_path}.")
    sys.exit(0)


def load_ips(path):
    m = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                m[parts[0]] = parts[1]
    return m


def my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def ssh(host, cmd, timeout=30):
    try:
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", f"ConnectTimeout=5",
             f"{SSH_USER}@{host}", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout, r.returncode
    except subprocess.TimeoutExpired:
        return "", 1
    except Exception:
        return "", 1


def extract_bw(s):
    try:
        d = json.loads(s)
        return round(d["end"]["sum_received"]["bits_per_second"] / 1e6, 2)
    except Exception:
        return None


def measure_bw_only(src, dst, client_ip):
    """Measure BW (Mbps) from src→dst without running ping. src/dst are IPs or 'client'."""
    if src == "client" and dst == "client":
        return None

    if src == "client":
        ssh(dst, f"pkill iperf3 2>/dev/null || true; iperf3 -s -p {IPERF_PORT} -D --one-off")
        time.sleep(1)
        r = subprocess.run(
            ["iperf3", "-c", dst, "-p", str(IPERF_PORT), "-t", str(IPERF_DURATION), "-J"],
            capture_output=True, text=True, timeout=IPERF_DURATION + 10,
        )
        return extract_bw(r.stdout)

    elif dst == "client":
        srv = subprocess.Popen(
            ["iperf3", "-s", "-p", str(IPERF_PORT), "--one-off"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
        out, rc = ssh(src, f"iperf3 -c {client_ip} -p {IPERF_PORT} -t {IPERF_DURATION} -J 2>/dev/null",
                      timeout=IPERF_DURATION + 15)
        srv.wait(timeout=IPERF_DURATION + 5)
        return extract_bw(out)

    else:
        ssh(dst, f"pkill iperf3 2>/dev/null || true; iperf3 -s -p {IPERF_PORT} -D --one-off")
        time.sleep(1)
        out, rc = ssh(src, f"iperf3 -c {dst} -p {IPERF_PORT} -t {IPERF_DURATION} -J 2>/dev/null",
                      timeout=IPERF_DURATION + 15)
        return extract_bw(out)


def is_reachable(ip):
    r = subprocess.run(["ping", "-c", "1", "-W", "2", ip],
                       capture_output=True, timeout=5)
    return r.returncode == 0


def main():
    global _rows, _csv_path, _updated

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="graph_partitioning/tests/pair_links_measured.csv")
    parser.add_argument("--ips", default="rpi-automation/dprgnet_ips.txt")
    parser.add_argument(
        "--skip-measured", action="store_true",
        help="Skip reverse rows whose BW already differs from the forward row",
    )
    args = parser.parse_args()

    _csv_path = args.csv
    signal.signal(signal.SIGINT,  _flush)
    signal.signal(signal.SIGTERM, _flush)

    name_to_ip = load_ips(args.ips)
    client = my_ip()
    print(f"Client IP: {client}")

    with open(args.csv, newline="") as f:
        _rows = list(csv.reader(f))

    # Second occurrence of each unordered pair is the mirrored reverse direction.
    seen = {}
    to_remeasure = []

    for idx, row in enumerate(_rows):
        if len(row) < 3 or row[0] == "device1":
            continue
        d1, d2 = row[0], row[1]
        key = frozenset({d1, d2})
        if key not in seen:
            seen[key] = idx
        else:
            src_ip = client if d1 == "client" else name_to_ip.get(d1)
            dst_ip = client if d2 == "client" else name_to_ip.get(d2)
            if src_ip is None or dst_ip is None:
                print(f"SKIP {d1}→{d2}: no IP found")
                continue

            if args.skip_measured:
                fwd_bw = _rows[seen[key]][2] if len(_rows[seen[key]]) > 2 else ""
                rev_bw = row[2] if len(row) > 2 else ""
                try:
                    already_done = float(fwd_bw) != float(rev_bw)
                except (ValueError, TypeError):
                    already_done = False
                if already_done:
                    print(f"SKIP {d1}→{d2}: already measured ({rev_bw} Mbps)")
                    continue

            fwd_rtt = _rows[seen[key]][3] if len(_rows[seen[key]]) > 3 else ""
            to_remeasure.append((idx, d1, d2, src_ip, dst_ip, fwd_rtt))

    n = len(to_remeasure)
    secs_per_pair = IPERF_DURATION + 3  # iperf3 run + SSH overhead, no ping
    total_s = n * secs_per_pair
    m, s = divmod(total_s, 60)
    print(f"Found {n} reverse pairs to re-measure.")
    print(f"Estimated time: {m}m {s}s  ({n} pairs × ~{secs_per_pair}s each)\n")

    for row_idx, src_name, dst_name, src_ip, dst_ip, fwd_rtt in to_remeasure:
        check_ip = src_ip if src_ip != client else dst_ip
        if check_ip != client and not is_reachable(check_ip):
            print(f"[{time.strftime('%H:%M:%S')}] SKIP {src_name}→{dst_name}: unreachable")
            continue

        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] Measuring {src_name} → {dst_name} ...", end=" ", flush=True)
        src_arg = "client" if src_ip == client else src_ip
        dst_arg = "client" if dst_ip == client else dst_ip
        try:
            bw = measure_bw_only(src_arg, dst_arg, client)
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        if bw is not None:
            _rows[row_idx][2] = str(bw)
            # RTT is symmetric — reuse value from the forward row
            _rows[row_idx][3] = fwd_rtt
            print(f"{bw} Mbps, RTT={fwd_rtt} ms (from forward row)")
            _updated += 1
        else:
            print("no data")

    tmp = args.csv + ".tmp"
    with open(tmp, "w", newline="") as f:
        csv.writer(f).writerows(_rows)
    os.replace(tmp, args.csv)
    print(f"\nUpdated {_updated}/{len(to_remeasure)} rows in {args.csv}.")


if __name__ == "__main__":
    main()