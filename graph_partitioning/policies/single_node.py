"""Single-node no-graph-partitioning policy for single-request inference."""

import time
from typing import Dict, Optional

from ..logging_utils import setup_output_dir
from ..model import ModelSpec
from ..network import Network


def simulate_single_node(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    kv_cache_seq_len: Optional[int] = None,
) -> Dict:
    """All layers on one device; simple latency estimate.

    ``kv_cache_seq_len`` sets the context length used for KV-cache memory
    pressure (and therefore the swap penalty). Defaults to ``seq_len``.
    Set to ``prompt_len + decoded_so_far`` to model decode-phase pressure.
    """
    _t0 = time.perf_counter()
    devices = net.servers
    if not devices:
        raise ValueError("No devices available in the network.")
    mem_seq_len = kv_cache_seq_len if kv_cache_seq_len is not None else seq_len
    # Compute must scale with the forward's token count: prefill processes the
    # whole prompt in one batched forward (seq_len tokens), decode is one token
    # (seq_len=1). total_compute_flops() hardcodes seq_len=1, so use the
    # seq_len-aware per-layer cost here to stay consistent with simulate_pipeline.
    forward_flops = model.layer_compute_flops(seq_len) * model.num_layers
    working_mem = model.total_memory_usage(seq_len=mem_seq_len)
    in_bytes = model.activation_bytes * seq_len
    # Per-token result returned to the driver is the SMALL result (sampled token /
    # final hidden state), not the full vocab logits — consistent with pipeline.py.
    out_bytes = model.activation_bytes

    # Choose the host device by comparing the TOTAL single-node cost on EACH
    # device: on-device compute + the coordinator round-trip (input inject +
    # result extract) every token pays. In CLIENT_SERVER that round-trip is a
    # real hop to/from the chosen device, so selection becomes proximity-aware —
    # a fast device far from the driver loses to a slightly slower one nearby,
    # and the driver's own node (a self-link, zero coordinator I/O) wins when it
    # can host the model. In PEER_TO_PEER coordinator I/O is 0, so this reduces
    # to fastest-compute. (compute_time_s already adds a swap penalty for devices
    # the model overflows, so those are naturally avoided.)
    def _single_node_cost(d) -> float:
        return d.compute_time_s(
            forward_flops, working_mem, seq_len=seq_len
        ) + net.coordinator_io_time_s(in_bytes, out_bytes, d.name, d.name)

    chosen = min(net.servers, key=_single_node_cost)
    compute_s = chosen.compute_time_s(forward_flops, working_mem, seq_len=seq_len)
    coord_io = net.coordinator_io_time_s(in_bytes, out_bytes, chosen.name, chosen.name)
    t_token = compute_s + coord_io
    total = t_token * gen_tokens
    planning_time_s = time.perf_counter() - _t0

    output_dir = setup_output_dir("sim_output")

    with open(f"{output_dir}/setup_info.yaml", "w", encoding="utf-8") as f:
        f.write("policy: single_node\n")
        f.write(f"model: {model.name}\n")
        f.write(f"num_layers: {model.num_layers}\n")
        f.write(f"devices: {[chosen]}\n")
        f.write(f"total_gflops: {model.total_compute_flops() / 1e9:.2f}\n")
        f.write(f"total_mem_gb: {model.total_memory_usage():.2f}\n")
        f.write(f"communication_model: {net.communication_model.name}\n")
        f.write(f"total_latency_s: {total}\n")

    return {
        "policy": "single_node",
        "device": chosen.name,
        "gflops": f"{model.total_compute_flops() / 1e9:.2f}",
        "mem_gb": float(model.total_memory_usage(seq_len=mem_seq_len)),
        "per_token_s": t_token,
        "total_latency_s": total,
        "planning_time_s": planning_time_s,
    }
