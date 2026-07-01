"""Calibrate the RPC hop model (msg_overhead_s C, rpc_efficiency η) against the
measured llama.cpp sweeps, minimizing RMS log-error of tok/s (prefill + decode).

Replicates hw_calibration.ipynb §10's predictor: n=1 -> single_node, n>=2 -> pipeline,
coordinator = off-cluster (None) for 'mac*' scenarios else on-mesh rpi1.
"""
import glob, os, math
import numpy as np
import pandas as pd

import graph_partitioning.graph_plots as gp
gp.set_save_figures(False)
import graph_partitioning.policies.pipeline as PL
import graph_partitioning.policies.single_node as SN
_noop = lambda *a, **k: None
PL.draw_partitions = _noop
PL.draw_device_computation_breakdown = _noop
os.makedirs("/tmp/_calib", exist_ok=True)
PL.setup_output_dir = lambda *a, **k: "/tmp/_calib"
SN.setup_output_dir = lambda *a, **k: "/tmp/_calib"

from graph_partitioning import Network, ModelSpec
from graph_partitioning.common import CommunicationModel, CostModel, devices_from_profiles
from graph_partitioning.policies.single_node import simulate_single_node
from graph_partitioning.policies.pipeline import simulate_pipeline

TINY = ModelSpec(name="TinyLlama-1.1B", num_layers=23, num_heads=32, num_kv_heads=4,
                 d_model=2048, d_k=64, d_ff=5632, vocab_size=32000)
PROFILE = "results/gflops/20260611_163608/device_profiles.csv"
RPI_NAMES = ["rpi1","rpi2","rpi3","rpi4","rpi5","rpi6","rpi7","rpi8","rpi9","rpi13",
             "rpi15","rpi16","rpi17","rpi19","rpi20","rpi22","rpi24","rpi25"]
_pool = {d.name: d for d in devices_from_profiles(
    PROFILE, quant="q4_K", compute="peak", cost_model=CostModel.ROOFLINE,
    default_mem_gb=2.0, swap_bandwidth_mbps=320.0)}
SERVERS = [_pool[n] for n in RPI_NAMES if n in _pool]


def load_measured():
    rows = []
    for path in sorted(glob.glob("results/sweep_*.csv")):
        name = os.path.basename(path).replace("sweep_", "").replace(".csv", "")
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            typ = "pp" if r["n_gen"] == 0 else "tg"
            seq = int(r["n_prompt"]) if typ == "pp" else int(r["n_gen"])
            rows.append(dict(scenario=name, n=int(r["n_devices"]), typ=typ,
                             seq=seq, meas=float(r["avg_ts"])))
    m = pd.DataFrame(rows)
    return (m.groupby(["scenario", "n", "typ", "seq"], as_index=False)
             .agg(meas=("meas", "mean"), spread=("meas", lambda x: x.max()/x.min())))


def _net(scenario, n, C, eta, hub_share=True):
    coord = None if scenario.startswith("mac") else "rpi1"
    net = Network(servers=SERVERS[:max(n, 1)], coordinator_name=coord,
                  default_bw_mbps=50.0, default_rtt_ms=20.0)
    net.set_pairwise_links(filename="graph_partitioning/tests/pair_links_measured.csv")
    net.set_comm_config(communication_model=CommunicationModel.CLIENT_SERVER)
    net.msg_overhead_s = C
    net.rpc_efficiency = eta
    net.hub_bw_share = hub_share
    return net


def sim_tps(scenario, n, typ, seq, C, eta, hub_share=True):
    runner = simulate_single_node if n == 1 else simulate_pipeline
    sl = seq if typ == "pp" else 1
    r = runner(TINY, _net(scenario, n, C, eta, hub_share), seq_len=sl, gen_tokens=1)
    lat = r["total_latency_s"]
    return (seq / lat) if typ == "pp" else (1.0 / lat)


def rms_logerr(meas, C, eta, mask=None, hub_share=True):
    sub = meas if mask is None else meas[mask]
    errs = [math.log(sim_tps(r.scenario, r.n, r.typ, r.seq, C, eta, hub_share))
            - math.log(r.meas) for r in sub.itertuples()]
    return math.sqrt(np.mean(np.square(errs))), errs


def grid_fit(meas, Cs, etas, mask=None, hub_share=True):
    best = (None, None, 1e9)
    for C in Cs:
        for eta in etas:
            e, _ = rms_logerr(meas, C, eta, mask, hub_share)
            if e < best[2]:
                best = (C, eta, e)
    return best


def resid(meas, C, eta, mask=None, hub_share=True):
    sub = meas if mask is None else meas[mask]
    for r in sub.sort_values(["scenario", "typ", "seq", "n"]).itertuples():
        s = sim_tps(r.scenario, r.n, r.typ, r.seq, C, eta, hub_share)
        flag = "  <<" if abs(math.log(s / r.meas)) > 0.4 else ""
        print(f"  {r.scenario:22s} n{r.n:<2d} {r.typ}{r.seq:<4d} "
              f"meas={r.meas:6.2f} sim={s:6.2f} ratio={s/r.meas:4.2f}{flag}")


if __name__ == "__main__":
    meas = load_measured()
    is_mac = meas.scenario.str.startswith("mac")
    not16 = meas.n != 16                     # the n=16 mac contention cliff (footnoted)
    print(f"{len(meas)} points, N={sorted(meas.n.unique())}, "
          f"median spread={meas.spread.median():.2f}")
    base, _ = rms_logerr(meas, 0.0, 1.0)
    print(f"BASELINE C=0 eta=1: RMS={base:.3f} ({100*(math.exp(base)-1):.0f}%)")

    Cs = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    etas = [0.05, 0.08, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35, 0.5, 0.7, 1.0]

    print("\n========== MESH (rpi coord) ==========")
    Cr, etar, er = grid_fit(meas, Cs, etas, mask=~is_mac)
    print(f"best: C={Cr}, eta={etar} -> RMS={er:.3f} ({100*(math.exp(er)-1):.0f}%)")
    resid(meas, Cr, etar, mask=~is_mac)

    print("\n========== MAC (off-cluster), hub_bw_share ON, excl n16 ==========")
    Cm, etam, em = grid_fit(meas, Cs, etas, mask=is_mac & not16, hub_share=True)
    print(f"best: C={Cm}, eta={etam} -> RMS={em:.3f} ({100*(math.exp(em)-1):.0f}%)")

    print("\n========== MAC, hub_bw_share OFF, excl n16 ==========")
    Cm2, etam2, em2 = grid_fit(meas, Cs, etas, mask=is_mac & not16, hub_share=False)
    print(f"best: C={Cm2}, eta={etam2} -> RMS={em2:.3f} ({100*(math.exp(em2)-1):.0f}%)")
    resid(meas, Cm2, etam2, mask=is_mac & not16, hub_share=False)

    print("\n========== FINE universal eta, per-scenario C (hub OFF), excl n16 ==========")
    etas_f = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]
    Cmac = [0.0, 0.03, 0.05, 0.08, 0.1, 0.12]
    best = (1e9, None, None, None)
    for eta in etas_f:
        bm = min((rms_logerr(meas, C, eta, is_mac & not16, False)[0], C) for C in Cmac)
        br = min((rms_logerr(meas, C, eta, ~is_mac, False)[0], C) for C in Cmac)
        errs = (rms_logerr(meas, bm[1], eta, is_mac & not16, False)[1]
                + rms_logerr(meas, br[1], eta, ~is_mac, False)[1])
        comb = math.sqrt(np.mean(np.square(errs)))
        if comb < best[0]:
            best = (comb, eta, bm[1], br[1])
        print(f"  eta={eta:.2f}: mac C={bm[1]} ({100*(math.exp(bm[0])-1):.0f}%) "
              f"mesh C={br[1]} ({100*(math.exp(br[0])-1):.0f}%) "
              f"COMBINED={comb:.3f} ({100*(math.exp(comb)-1):.0f}%)")
    _, ETA, CMAC, CMESH = best
    print(f"\nWINNER: eta={ETA}, C_mac={CMAC}, C_mesh={CMESH}, hub_share=OFF")

    # ── probe: does a global prefill compute-scale fix the n=1 baseline? ──
    # (prefill is compute-bound -> sim too slow at n1 means R_compute too low there)
    print("\n========== compute-scale probe (multiply servers' gflops & curve) ==========")
    orig = [(d.gflops, dict(d.gflops_curve) if d.gflops_curve else None) for d in SERVERS]
    for cscale in [1.0, 1.1, 1.15, 1.2, 1.3]:
        for d, (g, cv) in zip(SERVERS, orig):
            d.gflops = g * cscale
            d.gflops_curve = {k: v * cscale for k, v in cv.items()} if cv else None
        errs = (rms_logerr(meas, CMAC, ETA, is_mac & not16, False)[1]
                + rms_logerr(meas, CMESH, ETA, ~is_mac, False)[1])
        comb = math.sqrt(np.mean(np.square(errs)))
        # split pp vs tg
        pp = meas[(meas.typ == "pp") & (meas.n != 16)]
        tg = meas[(meas.typ == "tg") & (meas.n != 16)]
        ep = math.sqrt(np.mean(np.square([math.log(sim_tps(r.scenario, r.n, r.typ, r.seq,
                CMAC if r.scenario.startswith("mac") else CMESH, ETA, False) / r.meas)
                for r in pp.itertuples()])))
        et = math.sqrt(np.mean(np.square([math.log(sim_tps(r.scenario, r.n, r.typ, r.seq,
                CMAC if r.scenario.startswith("mac") else CMESH, ETA, False) / r.meas)
                for r in tg.itertuples()])))
        print(f"  cscale={cscale}: COMBINED={comb:.3f} ({100*(math.exp(comb)-1):.0f}%)  "
              f"pp={100*(math.exp(ep)-1):.0f}%  tg={100*(math.exp(et)-1):.0f}%")
    for d, (g, cv) in zip(SERVERS, orig):  # restore
        d.gflops, d.gflops_curve = g, cv
