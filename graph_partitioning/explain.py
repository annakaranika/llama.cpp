"""Human-readable rationale for why a policy chose its device allocation.

Centralized so it works for every policy without per-policy churn: it reads the
decision signals already present in a policy's result dict (device count /
placement / alpa mesh_size) plus a couple of derived facts (does the model fit on
one device, is the fabric slow) and renders a one-line explanation.
"""
from typing import Dict

from .model import ModelSpec
from .network import Network


def _devices_used(policy: str, result: Dict, net: Network) -> int:
    """How many devices the chosen allocation actually uses."""
    if policy == "single_node":
        return 1
    if policy == "alpa":
        sp = result.get("selected_plan") or {}
        return int(sp.get("mesh_size") or len(sp.get("devices") or []) or 1)
    placement = result.get("placement")
    if placement:
        devs = {p.get("device") for p in placement if isinstance(p, dict)}
        devs.discard(None)
        if devs:
            return len(devs)
    # pipeline / tensor split across every server they are handed
    return len(net.servers)


def _slow_fabric(net: Network) -> bool:
    """True if a representative server<->server hop is bandwidth-poor (RPi mesh/AP)."""
    servers = net.servers
    if len(servers) < 2:
        return False
    bw, _ = net.link(servers[0].name, servers[1].name)
    return bw < 100.0  # Mbps; RPi mesh/AP links are ~20-50


def explain_allocation(
    policy: str, result: Dict, model: ModelSpec, net: Network, seq_len: int = 1
) -> str:
    """One-line rationale for the policy's device allocation."""
    servers = net.servers
    fit_gb = model.total_memory_usage(seq_len)
    max_mem = max((d.memory_gb for d in servers), default=0.0)
    fits_one = fit_gb <= max_mem
    n_used = _devices_used(policy, result, net)
    n_avail = len(servers)
    hop = "slow " if _slow_fabric(net) else ""

    if n_used <= 1:
        if fits_one:
            return (
                f"1 device — the model fits in one node's DRAM "
                f"({fit_gb:.2f}/{max_mem:.0f} GB), and single-stream decode gains "
                f"nothing from splitting, so any extra device only adds a {hop}hop."
            )
        return (
            f"1 device — no smaller split helps (model {fit_gb:.2f} GB vs "
            f"{max_mem:.0f} GB DRAM)."
        )
    if fits_one:
        return (
            f"{n_used}/{n_avail} devices — splits a model that already fits on one "
            f"node, so each boundary adds a {hop}hop with no single-stream "
            f"parallelism benefit (slower than 1 node here; shown for contrast)."
        )
    return (
        f"{n_used}/{n_avail} devices — required: the model ({fit_gb:.2f} GB) "
        f"exceeds one node's {max_mem:.0f} GB DRAM, so it must be partitioned."
    )
