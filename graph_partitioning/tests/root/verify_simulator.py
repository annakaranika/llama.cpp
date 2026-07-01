"""Analytical sanity checks for the partitioning simulator.

Each check asserts a physical invariant that should hold *regardless of how
the policies are implemented internally*. When a check fails, the simulator
is either modelling something it shouldn't, or there's a unit / scaling bug
of the kind we've been hitting (tensor swap accounting, alpa missing
compute_time_s, dataclass-default-frozen KV bytes, etc).

Run:
    source graph_partitioning/.venv/bin/activate
    python verify_simulator.py
"""

from __future__ import annotations

import math
import sys
from typing import Callable, List, Tuple

from graph_partitioning import Device, ModelSpec, Network
from graph_partitioning.common import CommunicationModel, PeerToPeerPolicy
from graph_partitioning.sweep import POLICY_RUNNERS, quiet_policies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _tinyllama() -> ModelSpec:
    return ModelSpec(
        name="TinyLlama-1.1B",
        num_layers=23, num_heads=32, num_kv_heads=4,
        d_model=2048, d_k=64, d_ff=5632, vocab_size=32000,
    )


def _make_net(servers: List[Device], bw_mbps=200.0, rtt_ms=18.0) -> Network:
    coordinator = Device("coordinator", gflops=100.0, memory_gb=16.0)
    net = Network(coordinator, servers, default_bw_mbps=bw_mbps, default_rtt_ms=rtt_ms)
    net.set_comm_config(
        communication_model=CommunicationModel.PEER_TO_PEER,
        peer2peer_policy=PeerToPeerPolicy.ALL_TO_ALL,
    )
    return net


def _approx(a: float, b: float, rel: float = 0.05, abs_: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


# ---------------------------------------------------------------------------
# Check definitions
# ---------------------------------------------------------------------------

Check = Callable[[], Tuple[bool, str]]


def check_single_node_floor() -> Tuple[bool, str]:
    """single_node on a device that *fits* the model: latency = pure compute time.

    For a device with enough DRAM to host the entire model with KV cache,
    compute_time_s shouldn't charge a swap penalty, so the per-token cost
    must equal flops / flops_per_sec exactly.
    """
    model = _tinyllama()
    # Big DRAM => no swap.
    big = Device("big", gflops=10.0, memory_gb=64.0, swap_bandwidth_mbps=50.0)
    net = _make_net([big])
    seq_len = 64
    gen = 4

    # single_node is seq_len-aware: per-forward FLOPs = layer_compute_flops(seq_len)
    # * num_layers, NOT total_compute_flops() (which hardcodes seq_len=1). Net is
    # PEER_TO_PEER here, so coordinator I/O is 0.
    forward_flops = model.layer_compute_flops(seq_len) * model.num_layers
    expected = (forward_flops / big.flops_per_sec()) * gen
    from graph_partitioning.policies.single_node import simulate_single_node
    with quiet_policies():
        r = simulate_single_node(model, net, seq_len, gen,
                                 kv_cache_seq_len=seq_len)
    got = r["total_latency_s"]
    ok = _approx(got, expected, rel=0.001)
    return ok, (
        f"single_node fits-in-DRAM: expected pure compute "
        f"{expected:.6f}s, got {got:.6f}s "
        f"(diff {abs(got-expected):.2e})"
    )


def check_single_node_swap() -> Tuple[bool, str]:
    """single_node where model overflows DRAM: latency ≈ compute + swap I/O.

    Swap time = (model_mem - dram) / swap_bandwidth (GB / GB-per-second).
    """
    model = _tinyllama()
    dev = Device("rpi", gflops=5.0, memory_gb=1.0, swap_bandwidth_mbps=50.0)
    net = _make_net([dev])
    seq_len = 32
    gen = 1

    model_mem = model.total_memory_usage(seq_len=seq_len)
    overflow_gb = max(0.0, model_mem - dev.memory_gb)
    swap_bw_gbs = dev.swap_bandwidth_mbps / 8000.0
    expected_swap = overflow_gb / swap_bw_gbs
    # seq_len-aware FLOPs (see check_single_node_floor); PEER_TO_PEER => no coord I/O.
    forward_flops = model.layer_compute_flops(seq_len) * model.num_layers
    expected_compute = forward_flops / dev.flops_per_sec()
    expected = (expected_compute + expected_swap) * gen

    from graph_partitioning.policies.single_node import simulate_single_node
    with quiet_policies():
        r = simulate_single_node(model, net, seq_len, gen,
                                 kv_cache_seq_len=seq_len)
    got = r["total_latency_s"]
    ok = _approx(got, expected, rel=0.001)
    return ok, (
        f"single_node swap-bound: expected compute({expected_compute:.3f}s)+"
        f"swap({expected_swap:.3f}s)*{gen} = {expected:.3f}s; got {got:.3f}s"
    )


def check_latency_linear_in_gen_tokens() -> Tuple[bool, str]:
    """For any policy, doubling gen_tokens doubles total latency.

    The simulator's model is `per_token_steady × gen_tokens`. We check the
    ratio, allowing 1% tolerance to absorb integer rounding (e.g. SCT layer
    rounding can shift fractionally).
    """
    model = _tinyllama()
    servers = [Device(f"d{i}", gflops=3.0, memory_gb=4.0,
                      swap_bandwidth_mbps=1e9)  # effectively no swap
               for i in range(4)]
    net = _make_net(servers, bw_mbps=10000)
    seq_len = 16

    failures = []
    with quiet_policies():
        for name, runner in POLICY_RUNNERS.items():
            try:
                r1 = runner(model, net, seq_len, 4)
                r2 = runner(model, net, seq_len, 8)
            except Exception:
                continue
            t1 = r1.get("total_latency_s", float("nan"))
            t2 = r2.get("total_latency_s", float("nan"))
            if not (t1 > 0 and t2 > 0):
                continue
            ratio = t2 / t1
            if not _approx(ratio, 2.0, rel=0.02):
                failures.append(f"{name}: {t1:.4f}s → {t2:.4f}s (ratio {ratio:.3f})")
    ok = not failures
    msg = ("all policies linear in gen_tokens" if ok
           else "non-linear policies: " + "; ".join(failures))
    return ok, msg


def check_pipeline_bandwidth_monotonic() -> Tuple[bool, str]:
    """Pipeline latency must not increase as link bandwidth increases."""
    model = _tinyllama()
    servers = [Device(f"d{i}", gflops=2.0, memory_gb=8.0,
                      swap_bandwidth_mbps=1e9) for i in range(4)]
    from graph_partitioning.policies.pipeline import simulate_pipeline

    last = None
    bws = [10.0, 100.0, 1_000.0, 10_000.0]
    rows = []
    with quiet_policies():
        for bw in bws:
            net = _make_net(servers, bw_mbps=bw)
            r = simulate_pipeline(model, net, 32, 4)
            rows.append((bw, r["total_latency_s"]))
            if last is not None and r["total_latency_s"] > last * 1.0001:
                return False, (
                    f"pipeline latency went UP from {last:.3f}s to "
                    f"{r['total_latency_s']:.3f}s as bw went up. "
                    f"rows={rows}"
                )
            last = r["total_latency_s"]
    return True, f"pipeline monotone: " + "  ".join(
        f"{bw:>6g}Mbps→{t:.3f}s" for bw, t in rows
    )


def check_tensor_scales_with_k() -> Tuple[bool, str]:
    """With no swap and very fast network, tensor compute should drop ~1/k.

    Run with k=1, 2, 4 (=num_kv_heads); check the compute portion of the
    result roughly halves each time. We allow generous tolerance because of
    rounding in head/FFN splitting.
    """
    model = _tinyllama()  # num_kv_heads = 4
    # 4 identical, no-swap devices on infinite bandwidth.
    base = [Device(f"d{i}", gflops=5.0, memory_gb=64.0,
                   swap_bandwidth_mbps=1e9) for i in range(4)]
    from graph_partitioning.policies.tensor import simulate_tensor_parallel

    compute_times = {}
    with quiet_policies():
        for k in (1, 2, 4):
            servers = base[:k]
            net = _make_net(servers, bw_mbps=10_000_000.0)  # near-zero comm
            r = simulate_tensor_parallel(model, net, 16, 1)
            compute_times[k] = r["component_times_s"]["compute"]

    # Expect compute_time(k) ≈ compute_time(1) / k
    ratio_2 = compute_times[1] / compute_times[2] if compute_times[2] else 0
    ratio_4 = compute_times[1] / compute_times[4] if compute_times[4] else 0
    ok = _approx(ratio_2, 2.0, rel=0.20) and _approx(ratio_4, 4.0, rel=0.30)
    return ok, (
        f"tensor compute scaling: k=1 {compute_times[1]:.4f}s, "
        f"k=2 {compute_times[2]:.4f}s (ratio×2={ratio_2:.2f}), "
        f"k=4 {compute_times[4]:.4f}s (ratio×4={ratio_4:.2f})"
    )


def check_memory_accounts_for_kv_growth() -> Tuple[bool, str]:
    """``memory_usage_per_device`` must grow when KV-cache length grows."""
    model = _tinyllama()
    m_short = model.total_memory_usage(seq_len=1)
    m_long  = model.total_memory_usage(seq_len=4096)
    grew = m_long > m_short + 1e-9
    return grew, (
        f"total_memory_usage(1) = {m_short:.4f} GB, "
        f"total_memory_usage(4096) = {m_long:.4f} GB "
        f"(delta {m_long - m_short:.4f} GB)"
    )


def check_all_policies_run() -> Tuple[bool, str]:
    """Every policy must produce a finite latency on a basic config."""
    model = _tinyllama()
    servers = [Device(f"d{i}", gflops=2.0 + i, memory_gb=2.0 + i % 2,
                      swap_bandwidth_mbps=50.0) for i in range(4)]
    net = _make_net(servers)
    nan_or_fail = []
    with quiet_policies():
        for name, runner in POLICY_RUNNERS.items():
            try:
                r = runner(model, net, 32, 4)
                t = r.get("total_latency_s", float("nan"))
                if not (math.isfinite(t) and t > 0):
                    nan_or_fail.append(f"{name}={t}")
            except Exception as exc:
                nan_or_fail.append(f"{name} raised {type(exc).__name__}: {exc}")
    ok = not nan_or_fail
    return ok, ("all policies returned finite > 0 latency" if ok
                else "failed: " + "; ".join(nan_or_fail))


def check_ap_routing_through_client() -> Tuple[bool, str]:
    """In CLIENT_SERVER mode, server-server link uses the coordinator as relay.

    With only coordinator↔server links loaded (AP topology), link(rpi_a, rpi_b)
    should return bw = min(bw_a_coord, bw_coord_b) and
    rtt = rtt_a_coord + rtt_coord_b, not the 300 Mbps default.
    """
    coordinator = Device("coordinator", gflops=15.0, memory_gb=4.0)
    rpi1 = Device("rpi1", gflops=2.0, memory_gb=4.0)
    rpi2 = Device("rpi2", gflops=2.0, memory_gb=4.0)

    net = Network(coordinator, [rpi1, rpi2])
    net.set_comm_config(CommunicationModel.CLIENT_SERVER)
    net.set_pairwise_links(links=[
        ("coordinator", "rpi1", 50.0, 20.0),
        ("coordinator", "rpi2", 40.0, 25.0),
    ])

    bw, rtt = net.link("rpi1", "rpi2")
    expected_bw = 40.0   # min(50, 40)
    expected_rtt = 45.0  # 20 + 25

    ok = _approx(bw, expected_bw) and _approx(rtt, expected_rtt)
    return ok, (
        f"AP link(rpi1, rpi2): bw={bw:.1f} Mbps (expected {expected_bw}), "
        f"rtt={rtt:.1f} ms (expected {expected_rtt})"
    )


def check_pipeline_lower_bound() -> Tuple[bool, str]:
    """Pipeline latency cannot beat the per-stage compute lower bound.

    For k *identical* devices and a model that fits in DRAM (no swap), the
    minimum per-token cost is num_layers/k × per_layer_flops / gflops. The
    simulator's pipeline result must be >= that floor.
    """
    model = _tinyllama()
    k = 4
    gflops_each = 4.0
    servers = [Device(f"d{i}", gflops=gflops_each, memory_gb=8.0,
                      swap_bandwidth_mbps=1e9) for i in range(k)]
    net = _make_net(servers, bw_mbps=10_000)
    from graph_partitioning.policies.pipeline import simulate_pipeline

    seq_len, gen = 16, 1
    per_layer_flops = model.layer_compute_flops(seq_len)
    lower_bound = math.ceil(model.num_layers / k) * per_layer_flops / (
        gflops_each * 1e9
    ) * gen

    with quiet_policies():
        r = simulate_pipeline(model, net, seq_len, gen)
    got = r["total_latency_s"]
    ok = got >= lower_bound * 0.999  # tiny tolerance for fp
    return ok, (
        f"pipeline lower bound: floor={lower_bound:.6f}s, "
        f"sim={got:.6f}s {'(OK)' if ok else '(VIOLATES LOWER BOUND)'}"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


CHECKS: List[Tuple[str, Check]] = [
    ("single_node fits in DRAM equals pure compute",     check_single_node_floor),
    ("single_node overflowing DRAM equals compute+swap", check_single_node_swap),
    ("memory_usage_per_device grows with KV seq_len",    check_memory_accounts_for_kv_growth),
    ("every policy returns a finite latency",            check_all_policies_run),
    ("latency scales linearly with gen_tokens",          check_latency_linear_in_gen_tokens),
    ("pipeline latency monotone-decreasing in bandwidth", check_pipeline_bandwidth_monotonic),
    ("tensor compute roughly halves when k doubles",     check_tensor_scales_with_k),
    ("pipeline never beats per-stage compute lower bound", check_pipeline_lower_bound),
    ("AP mode routes server-server through client",        check_ap_routing_through_client),
]


def main() -> int:
    print(f"\n{'=' * 80}")
    print(f"Simulator invariant checks — {len(CHECKS)} total\n")
    n_pass = 0
    n_fail = 0
    for label, check in CHECKS:
        try:
            ok, msg = check()
        except Exception as exc:
            ok = False
            msg = f"check raised {type(exc).__name__}: {exc}"
        tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"[{tag}] {label}")
        print(f"        {msg}\n")
        n_pass += int(ok)
        n_fail += int(not ok)
    print(f"{'=' * 80}")
    print(f"{n_pass} passed, {n_fail} failed of {len(CHECKS)}")
    if n_fail:
        print(f"{YELLOW}\nReminder: real-hardware llama.cpp RPC comparison is "
              f"the next-step validation.{RESET}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
