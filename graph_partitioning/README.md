# graph_partitioning

A simulator for distributed LLM inference across heterogeneous, network-connected
devices. It exists to explore placement and partitioning strategies that are too
expensive to measure directly on hardware, and to explain *why* the measured
results on the Raspberry Pi cluster come out the way they do.

It is calibrated against measurements from this repository. See
[`../HANDOFF.md`](../HANDOFF.md) for the hardware work and
[`../OPTIMIZATIONS.md`](../OPTIMIZATIONS.md) Part 2 for the design analysis this
simulator supports.

---

## Install

```bash
pip install -e ".[dev]"          # from this directory
pip install -e ".[metis,dev]"    # add PyMetis if you need the METIS policy
```

Python 3.11+. METIS is optional because it is a compiled dependency; the other six
policies work without it.

---

## Core concepts

Four objects compose into a simulation:

- **`Device`** (`common.py`) — one machine: `gflops`, `memory_gb`,
  `mem_bandwidth_gbps`, `swap_bandwidth_mbps`, `cores`.
- **`ModelSpec`** (`model.py`) — the model and its per-op cost model. Handles
  quantization (`Quantization` enum, bytes-per-parameter), weight and KV memory,
  and FLOP counts.
- **`Network`** (`network.py`) — the topology and communication cost model,
  including all-reduce cost and the airtime contention term.
- **A policy** (`policies/`) — how the model is partitioned across the devices.

### Policies

| Key | Module | What |
|---|---|---|
| `single_node` | `single_node.py` | Everything on one device; the baseline. |
| `pipeline` | `pipeline.py` | Layer-wise pipeline parallelism. |
| `tensor` | `tensor.py` | Tensor parallelism with all-reduce. |
| `metis` | `metis.py` | METIS k-way partitioning of the layer graph. |
| `alpa` | `alpa.py` | Alpa-style 2D partitioning, ILP-based (PuLP + HiGHS). |
| `hybrid_pp_tp` | `hybrid_pp_tp.py` | Memory-aware greedy hybrid. Devices that can hold a whole layer become solo pipeline stages; only devices too small are bundled into tensor groups. On a homogeneous cluster where every device fits, this collapses to pure pipeline. |
| `msct` | `msct.py` | Baechi memory-constrained SCT: operator DAG, favorite-child LP, then memory-constrained ETF list scheduling. |

Note on naming: `hybrid_pp_tp.py` was previously called `sct.py`, which was a
misnomer — it is a greedy communication-aware heuristic, not the SCT algorithm.
`msct.py` is the faithful SCT implementation.

### Cost models

`CostModel` selects how `Device.compute_time_s` works:

- **`GFLOPS`** — `time = flops / (1e9 * gflops)`. One effective-compute number per
  device.
- **`ROOFLINE`** — `time = max(flops / R_compute, bytes / R_bw)`, where `R_bw` is
  measured memory bandwidth (about 3.5 GB/s on a Pi 4B). This makes
  bandwidth-bound decode realistic without hand-tuning `gflops` downward, while
  compute-bound prefill stays governed by the FLOPs term. The regime is selected
  automatically by the `max`.

### Coordinator placement matters

`Network.coordinator_name` controls where the coordinator sits: `None` means an
off-cluster client (the Mac), or set it to a server name (`'rpi1'`) for an
on-cluster coordinator.

This is not cosmetic. In `CLIENT_SERVER` mode, `Network.link()` relays every
server-to-server transfer through the coordinator, because llama.cpp's RPC ferries
all tensors through the client and backends never talk directly. A Mac
coordinator therefore pays a wireless AP crossing on every leg and scales steeply,
while an on-cluster Pi coordinator keeps traffic local and stays flat. This
reproduces the measured 1.6-2.9x difference between the two scenarios. Set
`coordinator_name` per scenario, or the two curves collapse into one.

`PEER_TO_PEER` mode has no coordinator in the loop.

---

## Entry points

There is no CLI. The simulator is driven from notebooks or imported directly.

```python
from graph_partitioning.simulator import Simulator

sim = Simulator(...)
sim.compare_sched_policies(...)         # policies against each other
sim.compare_dev_policies(...)           # device-count sweeps
sim.compare_tensor_parallel_strategies(...)
sim.compare_comm_configs_for_strategy(...)
sim.plot_all_partitions(...)
sim.evaluate_ttft_tpt(...)              # time-to-first-token / time-per-token
```

Other useful modules:

- **`compare_policies.plot_all_policies(model, net, seq_len, gen_tokens)`** —
  runs every policy and plots each as a `(num_devices, num_layers)` matrix where
  `M[d, l]` is the fraction of layer `l` assigned to device `d`, drawn as stacked
  bars so policies are visually comparable.
- **`elastic.py`** — KV-growth-driven elastic pipeline scheduling, the simulation
  counterpart to the elastic rebalancing implemented in the C++ code. Runnable
  directly (`python -m graph_partitioning.elastic`).
- **`sweep.py`** — parameter sweeps across policies and device counts.
- **`gflops_measure.py`** — device compute calibration (runnable directly).

### Notebooks

- **`hw_calibration.ipynb`** — compares simulator output against measured cluster
  data. This is the entry point for understanding calibration.
- **`design_exploration.ipynb`** — the design-space analysis.

### Calibration data

`tests/` holds the measured inputs: `device_profiles.csv`, and pairwise link
measurements for each network scenario (`pair_links_ap.csv`,
`pair_links_mesh.csv`, `pair_links_measured.csv`). These come from
`measure_links.sh` in the `rpi-automation` repository.

---

## Calibration status

Structural work is complete; constant fitting is not. **Treat absolute numbers as
indicative and relative comparisons between policies as the useful output.**

Current accuracy against measured TinyLlama-1.1B Q5_K_M on the Pi cluster:

| Scenario | Simulator vs measured |
|---|---|
| On-cluster Pi coordinator | within 1.2-1.4x across device counts |
| Mac coordinator | 0.5-0.85x; under-predicts, and the gap grows with N |

### What was fixed structurally

- Quantization is modelled properly (`Quantization` enum) rather than assuming
  FP32. The weight count itself was also incomplete: it now includes the attention
  output projection, treats FFN as SwiGLU (gate, up, down — three matrices, not
  two), and adds the tied embedding/output term. TinyLlama Q5_K_M now computes to
  about 0.713 GB against a 745 MiB file.
- Pipeline simulation is a serial traversal (sum of stage compute plus inter-stage
  hops plus coordinator inject/extract), not `max(stages)`. This corrects the
  direction of decode scaling: single-stream autoregressive decode gets *slower*
  with more devices, because stages serialize. The decode ratio against measured
  went from 50-10000x to about 1.2-1.4x.
- KV memory validated exactly: simulated 88.0 MiB equals measured 88.0 MiB at
  `n_ctx=4096` FP16.
- `num_layers` is 22 for TinyLlama, not 23. There are 22 transformer blocks; the
  `-ngl 23` "+1" is the output head, counted separately via
  `embedding_weights_gb()`.

### Remaining work

- **Constant fitting.** Effective decode GFLOPS wants roughly 2.5 to 3.0. Prefill
  needs about 2x the decode GFLOPS, reflecting batched-matmul efficiency (about
  7.5 versus 3.9 tok/s on the same device).
- **Per-hop RPC overhead is under-modelled.** Measured is about 81 ms per AP hop
  against roughly 28 ms simulated. Raw bandwidth plus RTT misses llama.cpp's
  per-message serialization and handshake cost. A candidate fix is a fixed
  per-message constant, or an `rtt_coeff` knob on `Network`.
- **m-SCT under-prices its all-reduce.** It models the gather as a cheap
  point-to-point transfer while the tensor policy uses the full centralized
  all-reduce cost, so m-SCT looks far cheaper than it should at high device counts.
  The fix is to make m-SCT's all-reduce nodes cost a real all-reduce over their
  participant devices, which preserves SCT's benefit (favorite-child co-location
  means fewer participants, hence genuinely cheaper) without the artificial
  advantage.
- **Storage profiles are unmeasured.** `Device.swap_bandwidth_mbps` gates the
  DRAM-overflow paging penalty and defaults to `STORAGE_PROFILES["microsd_pi"]`
  (320 Mbps, about 40 MB/s — the Pi 4 microSD sequential-read ceiling). The
  `STORAGE_PROFILES` table in `common.py` covers microSD through NVMe Gen4 for
  heterogeneous simulation, but the values are datasheet sequential-read ceilings,
  not measurements. Real mmap paging is slower because of page-fault and random-access
  overhead, so they are optimistic. Replacing them with measured mmap paging rates
  is an open calibration task. This only matters when a model overflows DRAM, which
  never happens for TinyLlama Q5 on a 2 GB Pi.
