#!/usr/bin/env python3
"""Parse measure_gflops.sh output into a tidy CSV + per-device profile.

Usage:
    python3 parse_gflops.py <path> [<path> ...] [--quant q4_K] [--out <dir>]

Each <path> is a results dir produced by measure_gflops.sh (it reads <path>/raw/*.txt
and, if present, <path>/devices.csv), OR a directory of raw .txt files, OR a single
raw .txt file. Multiple paths are MERGED — so you can fold older runs into a new one:

    python3 parse_gflops.py results/gflops/<new_ts> results/gflops/legacy_rpi12

Device name comes from each raw file's basename (rpi1.txt -> rpi1). Duplicate
(device, dtype, shape) rows across paths are de-duplicated (last one wins).

Outputs (into --out, default = the first directory given):
    measurements.csv     one row per (device, type, n): shape, us, gflops, eff_GBps
    device_profiles.csv  one row per device: R_bw + decode/peak/prefill GFLOPS

eff_GBps = total bytes moved / time, using ggml block sizes for the weight bytes.
At n=1 (decode) it ~equals the weight-read bandwidth and is nearly dtype-invariant
— the device's memory-bandwidth ceiling, the key decode constant for the simulator.
(At large n the op is compute-bound, so eff_GBps is low and not meaningful there.)
"""
import argparse
import csv
import glob
import os
import re
import sys
from statistics import median

# bytes per weight per ggml type = (block type size) / (weights per block)
BYTES_PER_WEIGHT = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q4_0": 18 / 32, "q4_1": 20 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32, "q8_0": 34 / 32,
    "q2_K": 84 / 256, "q3_K": 110 / 256, "q4_K": 144 / 256, "q5_K": 176 / 256, "q6_K": 210 / 256,
    "iq2_xxs": 66 / 256, "iq2_xs": 74 / 256, "iq2_s": 82 / 256,
    "iq3_xxs": 98 / 256, "iq3_s": 110 / 256, "iq1_s": 50 / 256, "iq1_m": 56 / 256,
    "iq4_nl": 18 / 32, "iq4_xs": 136 / 256,
}
# "simple" (~pure-bandwidth) dtypes used to estimate the bandwidth ceiling R_bw
BW_TYPES = {"f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f32"}

LINE = re.compile(
    r"(?P<op>\w+)\(type_a=(?P<ta>\w+),type_b=(?P<tb>\w+),"
    r"m=(?P<m>\d+),n=(?P<n>\d+),k=(?P<k>\d+).*?"
    r"(?P<runs>\d+)\s+runs\s*-\s*(?P<us>[\d.]+)\s*us/run\s*-\s*"
    r"[\d.]+\s*[MG]FLOP/run\s*-\s*(?P<gflops>[\d.]+)\s*GFLOPS"
)
# test-backend-ops colorizes the GFLOPS number; strip ANSI escapes before parsing.
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_raw(path, device, rows):
    with open(path) as f:
        for line in f:
            line = ANSI.sub("", line)
            mt = LINE.search(line)
            if not mt:
                continue
            d = mt.groupdict()
            ta = d["ta"]
            m, n, k = int(d["m"]), int(d["n"]), int(d["k"])
            us, gflops = float(d["us"]), float(d["gflops"])
            bpw = BYTES_PER_WEIGHT.get(ta)
            eff = ""
            if bpw is not None:
                tot_bytes = m * k * bpw + n * k * 4 + m * n * 4  # weight + f32 act + f32 out
                eff = round(tot_bytes / (us * 1e-6) / 1e9, 3)
            rows.append(dict(
                device=device, op=d["op"], type_a=ta, type_b=d["tb"],
                m=m, n=n, k=k, runs=int(d["runs"]), us_per_run=us,
                gflops=gflops, eff_GBps=eff,
                bytes_per_weight=round(bpw, 4) if bpw is not None else "",
            ))


def collect_raws(path):
    if os.path.isfile(path):
        return [path]
    return (sorted(glob.glob(os.path.join(path, "raw", "*.txt")))
            or sorted(glob.glob(os.path.join(path, "*.txt"))))


def load_meta(path, meta):
    if os.path.isdir(path):
        dcsv = os.path.join(path, "devices.csv")
        if os.path.isfile(dcsv):
            with open(dcsv) as f:
                for d in csv.DictReader(f):
                    meta[d["name"]] = d


def natkey(s):  # natural sort: rpi2 before rpi10
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="+", help="one or more results dirs / raw files to merge")
    ap.add_argument("--quant", default="q4_K", help="dtype to summarize in device_profiles.csv")
    ap.add_argument("--out", default=None, help="output dir (default: first dir given, else .)")
    args = ap.parse_args()

    raws, meta = [], {}
    for p in args.path:
        rs = collect_raws(p)
        if not rs:
            print(f"  warning: no raw .txt found under {p}", file=sys.stderr)
        raws += rs
        load_meta(p, meta)
    if not raws:
        sys.exit("No raw .txt files found in any input path")

    rows = []
    for r in raws:
        parse_raw(r, os.path.splitext(os.path.basename(r))[0], rows)
    if not rows:
        sys.exit("Parsed 0 measurement rows — check the raw files / op name.")

    # de-duplicate (device, op, dtype, shape) across paths; last wins
    seen = {}
    for r in rows:
        seen[(r["device"], r["op"], r["type_a"], r["type_b"], r["m"], r["n"], r["k"])] = r
    dups = len(rows) - len(seen)
    rows = list(seen.values())
    if dups:
        print(f"  note: dropped {dups} duplicate (device,dtype,shape) rows (kept last)")

    out = args.out or next((p for p in args.path if os.path.isdir(p)), ".")
    os.makedirs(out, exist_ok=True)

    # tidy measurements.csv
    mcsv = os.path.join(out, "measurements.csv")
    cols = ["device", "op", "type_a", "type_b", "m", "n", "k",
            "runs", "us_per_run", "gflops", "eff_GBps", "bytes_per_weight"]
    rows.sort(key=lambda r: (natkey(r["device"]), r["type_a"], r["n"]))
    with open(mcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # per-device profile = the heterogeneous-device dataset
    q = args.quant
    devs = sorted(set(r["device"] for r in rows), key=natkey)
    pcsv = os.path.join(out, "device_profiles.csv")
    pcols = ["device", "model", "cores", "mem_gb", "R_bw_GBps",
             f"{q}_decode_gflops", f"{q}_peak_gflops", f"{q}_prefill512_gflops",
             "temp_before_c", "temp_after_c", "throttled_after"]
    with open(pcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(pcols)
        for dev in devs:
            dr = [r for r in rows if r["device"] == dev]
            bw = [r["eff_GBps"] for r in dr
                  if r["n"] == 1 and r["type_a"] in BW_TYPES and r["eff_GBps"] != ""]
            r_bw = round(median(bw), 3) if bw else ""
            qr = [r for r in dr if r["type_a"] == q]
            decode = next((r["gflops"] for r in qr if r["n"] == 1), "")
            peak = max((r["gflops"] for r in qr), default="")
            prefill = next((r["gflops"] for r in qr if r["n"] == 512), "")
            md = meta.get(dev, {})
            mem_gb = ""
            try:
                # mem_kb is KiB (binary, from /proc/meminfo); emit decimal GB so the
                # dataset matches the simulator's decimal-GB convention.
                mem_gb = round(int(md.get("mem_kb", "")) * 1024 / 1e9, 2)
            except (ValueError, TypeError):
                pass
            w.writerow([dev, md.get("model", ""), md.get("cores", ""), mem_gb, r_bw,
                        decode, peak, prefill,
                        md.get("temp_before_c", ""), md.get("temp_after_c", ""),
                        md.get("throttled_after", "")])

    print(f"Wrote {mcsv} ({len(rows)} rows)")
    print(f"Wrote {pcsv} ({len(devs)} devices: {', '.join(devs)})")
    print(f"  R_bw = median eff_GBps over n=1 {sorted(BW_TYPES)};  GFLOPS cols for --quant {q}")


if __name__ == "__main__":
    main()
