#!/usr/bin/env python3
"""Calibrate the shared-medium airtime-contention exponent
(``Network.airtime_contention_exp``) against the measured ``-sm row``
(tensor-parallel) N-sweep.

Ground truth (bench-results.md, 2026-06-16; Mac coordinator over the AP;
``llama-bench -ngl 23 -sm row -p 128 -n 8``, rpi{1,2,4}):

    N |  pp128 t/s | tg8 t/s
    1 |    2.71    |  0.22
    2 |    1.11    |  0.03
    4 |    0.42    |  0.02

Over infrastructure Wi-Fi every transfer — coordinator-relayed OR peer-to-peer —
funnels through the one AP, so the collective is airtime-bound and P2P costs the
same as centralized on hardware (peer-p2p-status memory: HW N=2 P2P = 0.03 t/s =
parallel's centralized 0.03 t/s). This sweep is therefore the airtime signal.

The airtime term scales the PEER-TO-PEER ring/all-to-all collective, whose base
model assumes the g concurrent transfers run on independent parallel links (per-step
``max``) and so UNDER-predicts the growth. We fit the exponent to the clean
N=2->N=4 all-reduce scaling (both have a collective; the shared regime cancels the
N=1 baseline, which carries Mac-coordination overhead the P2P path does not model).

Run from repo root:  PYTHONPATH=. python3 graph_partitioning/tests/root/calibrate_airtime.py
"""
import math
import os

import numpy as np

# --- silence policy-side plotting / file output --------------------------------
import matplotlib
matplotlib.use("Agg")
import graph_partitioning.graph_plots as gp
gp.set_save_figures(False)
import graph_partitioning.policies.tensor as TP
import graph_partitioning.policies.single_node as SN

_noop = lambda *a, **k: None
os.makedirs("/tmp/_calib_air", exist_ok=True)
for _mod in (TP, SN):
    for _attr in ("draw_partitions", "draw_device_computation_breakdown",
                  "draw_tensor_partitions", "plot_device_computation_breakdown"):
        if hasattr(_mod, _attr):
            setattr(_mod, _attr, _noop)
    if hasattr(_mod, "setup_output_dir"):
        _mod.setup_output_dir = lambda *a, **k: "/tmp/_calib_air"

from graph_partitioning import Network, ModelSpec
from graph_partitioning.common import (CommunicationModel, CostModel,
                                        PeerToPeerPolicy, devices_from_profiles)
from graph_partitioning.policies.tensor import simulate_tensor_parallel
from graph_partitioning.policies.single_node import simulate_single_node

TINY = ModelSpec(name="TinyLlama-1.1B", num_layers=23, num_heads=32, num_kv_heads=4,
                 d_model=2048, d_k=64, d_ff=5632, vocab_size=32000)
PROFILE = "graph_partitioning/tests/device_profiles.csv"
LINKS = "graph_partitioning/tests/pair_links_measured.csv"
RPIS = ["rpi1", "rpi2", "rpi3", "rpi4"]

_pool = {d.name: d for d in devices_from_profiles(
    PROFILE, quant="q4_K", compute="peak", cost_model=CostModel.ROOFLINE,
    default_mem_gb=2.0, swap_bandwidth_mbps=320.0)}
SERVERS = [_pool[n] for n in RPIS]

# measured -sm row sweep: N -> (pp128 t/s, tg8 t/s)
MEAS = {1: (2.71, 0.22), 2: (1.11, 0.03), 4: (0.42, 0.02)}
PP_SEQ = 128


def make_net(n, mode, exp, eta):
    net = Network(servers=SERVERS[:n], coordinator_name=None,
                  default_bw_mbps=42.0, default_rtt_ms=20.0)
    net.set_pairwise_links(filename=LINKS)
    if mode == "cs":
        net.set_comm_config(communication_model=CommunicationModel.CLIENT_SERVER)
    else:  # peer-to-peer ring
        net.set_comm_config(communication_model=CommunicationModel.PEER_TO_PEER,
                            peer2peer_policy=PeerToPeerPolicy.RING)
    net.rpc_efficiency = eta
    net.airtime_contention_exp = exp
    return net


def tps(n, typ, mode, exp, eta):
    net = make_net(n, mode, exp, eta)
    seq = PP_SEQ if typ == "pp" else 1
    runner = simulate_single_node if n == 1 else simulate_tensor_parallel
    lat = runner(TINY, net, seq_len=seq, gen_tokens=1)["total_latency_s"]
    return (PP_SEQ / lat) if typ == "pp" else (1.0 / lat)


def scaling_42(typ, mode, exp, eta):
    """Latency ratio T(4)/T(2) = tps(2)/tps(4) — the all-reduce growth from N=2->4."""
    return tps(2, typ, mode, exp, eta) / tps(4, typ, mode, exp, eta)


def meas_scaling_42(typ):
    i = 0 if typ == "pp" else 1
    return MEAS[2][i] / MEAS[4][i]


def rms_logerr_42(exp, eta, typs=("pp", "tg")):
    se = [(math.log(scaling_42(t, "ring", exp, eta)) - math.log(meas_scaling_42(t))) ** 2
          for t in typs]
    return math.sqrt(float(np.mean(se)))


def fit(exps, etas, typs):
    best = (None, None, 1e9)
    for exp in exps:
        for eta in etas:
            e = rms_logerr_42(exp, eta, typs)
            if e < best[2]:
                best = (exp, eta, e)
    return best


if __name__ == "__main__":
    exps = [round(x, 2) for x in np.arange(0.0, 3.01, 0.1)]
    etas = [0.3, 0.5, 0.7, 1.0]

    print("Measured N=2->4 latency scaling T(4)/T(2):  "
          f"prefill={meas_scaling_42('pp'):.2f}x  decode={meas_scaling_42('tg'):.2f}x")
    print()
    print("RING base (exp=0) predicts T(4)/T(2):")
    for t in ("pp", "tg"):
        print(f"  {t}: {scaling_42(t, 'ring', 0.0, 0.5):.2f}x   (measured {meas_scaling_42(t):.2f}x)")
    print()

    # fit on prefill (bandwidth/airtime-dominated; decode 2->4 is RTT-saturated)
    fp = fit(exps, etas, ("pp",))
    print(f"[airtime fit, prefill N2->4]  exp={fp[0]}  eta={fp[1]}  RMS log-err={fp[2]:.3f}")
    fb = fit(exps, etas, ("pp", "tg"))
    print(f"[airtime fit, pp+tg  N2->4]   exp={fb[0]}  eta={fb[1]}  RMS log-err={fb[2]:.3f}")
    print()

    exp, eta = fp[0], fp[1]
    print(f"With exp={exp}, eta={eta} (ring / peer-to-peer):")
    print(f"{'N':>2} {'phase':>5} {'meas t/s':>9} {'ring t/s':>9}")
    for n in (1, 2, 4):
        for j, t in enumerate(("pp", "tg")):
            print(f"{n:>2} {t:>5} {MEAS[n][j]:9.3f} {tps(n, t, 'ring', exp, eta):9.3f}")
    print()
    print("N=2->4 scaling with fitted exponent:")
    for t in ("pp", "tg"):
        print(f"  {t}: ring={scaling_42(t, 'ring', exp, eta):.2f}x   measured={meas_scaling_42(t):.2f}x"
              f"   (exp=0 ring was {scaling_42(t, 'ring', 0.0, eta):.2f}x)")
