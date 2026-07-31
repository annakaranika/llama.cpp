# Handoff notes

Orientation for someone picking up this work. It covers what the repository is,
where things live, how to run it, and what is known to be broken. It does not
repeat the technical detail in [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md), which
remains the reference for what was implemented and why.

Last updated 2026-07-31.

---

## What this repository is

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that distributes LLM
inference across a cluster of Raspberry Pi 4B devices connected over WiFi, using
llama.cpp's RPC backend. The research question is how far single-stream LLM
inference can be pushed on commodity edge hardware where the interconnect, not
compute, is the bottleneck.

All work is on the **`peer`** branch, roughly 206 commits ahead of `master`
(which tracks upstream). Two parallelism modes are supported:

- **Pipeline parallelism** (`-sm layer`): each device holds a contiguous range of
  layers and passes activations to the next. Cheap peer-to-peer handoffs.
- **Tensor parallelism** (`-sm row`): every device holds a slice of each layer and
  the devices all-reduce after each one.

The headline empirical result is that **pipeline parallelism beats tensor
parallelism in both phases** on this hardware, by 3.5x for prefill and 2.4x for
decode at N=4 on Llama-2-7B. Tensor parallelism's all-reduce is all-to-all and
scales roughly as N^2.1 in airtime on a shared wireless medium, so it is a
memory-relief mechanism rather than a speed mechanism here. The same tensor-parallel
code runs at 6.43 t/s on loopback versus 0.30 t/s on the WiFi cluster: a 21x gap
attributable entirely to the interconnect.

Beyond the parallelism work, the branch adds a resident serving process,
cross-process resident weights, a growable KV cache, and elastic pipeline
rebalancing that shifts layers between devices as the KV cache grows.

---

## Start here

Read in this order:

1. **This file**, for orientation and the known-broken list.
2. **[`OPTIMIZATIONS.md`](OPTIMIZATIONS.md)** — the main technical reference.
   Part 1 covers implemented optimizations with A/B measurements; Part 2 covers
   the design analysis and open research questions. Start with the "Measured
   impact" summary table.
3. **[`rpi-automation/ENV_VARS.md`](rpi-automation/ENV_VARS.md)** — every
   environment variable the RPC backend reads. Needed before running anything,
   because most optimizations are gated behind these and **a mismatch between the
   coordinator and the servers corrupts decode**.
4. **[`bench-results.md`](bench-results.md)** — raw measurements.

---

## Code map

The work is concentrated in a small number of files.

| File | Lines changed | What it holds |
|---|---|---|
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | ~8500 total | The bulk of the work. RPC wire protocol, tensor splitting and reconstruction, peer-to-peer all-reduce, the diff cache for graph sends, buffer pooling, the weight cache, and resident-weight rebinding. |
| `src/llama-rebalance.cpp` / `.h` | 742 | Elastic rebalancing **policy**, behind a per-context callback with a `llama_rebalance_policy` enum. Deciding *when* and *what* to move lives here. |
| `src/llama-model.cpp` | 542 | Model loading, layer placement, and the **mechanism** for moving a layer's weights between devices. |
| `src/llama-kv-cache.cpp` / `.h` | 344 | Growable KV cache; allocates in blocks as context grows rather than committing `n_ctx` up front. |
| `src/llama-context.h`, `src/llama.cpp` | 93 | Wiring for the rebalance callback and context-level plumbing. |

The mechanism/policy split is deliberate: mechanism lives in `llama-model`,
`llama-kv-cache`, and `ggml-rpc`; policy lives in `llama-rebalance.cpp`. A new
rebalancing strategy should be a new policy in the enum, not changes to the
mechanism.

---

## Development workflow

**Reproduce bugs locally before touching the cluster.** The split, serialize, and
deserialize logic is device-count-agnostic and behaves identically with two
`rpc-server` processes on localhost as with four Pis. A find-fix iteration takes
seconds locally and 10-15 minutes on the cluster. Use the cluster for final
validation and performance numbers only.

```bash
# Build both targets
cmake --build build --target rpc-server llama-cli -j8

# Two local servers (pkill -x rpc-server between runs; always start fresh)
build/bin/rpc-server -H 127.0.0.1 -p 50052 -m 6000 &
build/bin/rpc-server -H 127.0.0.1 -p 50053 -m 6000 &

# Tensor parallel
build/bin/llama-cli -m <model.gguf> --rpc 127.0.0.1:50052,127.0.0.1:50053 \
    -ngl 23 -sm row -no-cnv

# Pipeline parallel: same, without -sm row, and -ngl 99
```

**Get a ground-truth reference with `-ngl 0`** (pure CPU, no `--rpc`) and diff
against it. This is how the permute and empty-node bugs were localized to the RPC
reconstruction rather than the model graph.

To force multiple ubatches without a long prompt, use `-ub 64` with a ~70-token
prompt. Note that `-ub 64` makes `n_tokens == head_dim == 64`, which introduces an
ambiguity the default `-ub 512` does not, so cross-check both. For long decode that
crosses KV padding boundaries, use `-n 150 --ignore-eos`.

### Running on the cluster

`make.sh` builds, restarts the peer servers cleanly, and runs from the
coordinator. Run it **on rpi1**, which is on the ad-hoc cell and reaches the peers
directly.

```bash
./make.sh                      # clean restart of peers, then run
./make.sh compile deploy       # rebuild locally, push ggml-rpc.cpp to peers, rebuild there
./make.sh -n 4 -N 64 -p "..."  # 4 peers, 64 tokens, custom prompt
```

Cluster access, network switching, and fleet operations are **no longer in this
repository**. They live in a separate repo, `rpi-automation`, which is the
operational manual for the hardware: how to reach the Pis, how to move them
between the lab WiFi and the ad-hoc cell, and how to recover an unreachable node.
The copy under `rpi-automation/` here is retained only because `make.sh` and the
benchmark scripts reference `peer_ips.txt` and `peer_servers.sh`. Treat the
standalone repo as authoritative for anything network-related.

---

## Known broken, and open bugs

This is the most important section for anyone continuing the work.

### Continuous batching does not work

**Multi-sequence batching (`-np > 1`) is broken on the peer backend.** In a batched
decode only one slot produces tokens; the others release at `n_past = prompt_len`
having generated nothing, because they receive garbage logits. Worse, after a
batched forward the backend is left in a corrupted state, so even subsequent
single-sequence requests return empty until `llama-server` is restarted.

Serialized multi-request serving (`-np 1`) works correctly.

This matters beyond serving convenience. The throughput-serving argument in
`OPTIMIZATIONS.md` Part 2 — that batching amortizes the per-forward weight read, so
aggregate throughput scales with concurrency and larger models regain value under
concurrent load — **does not currently hold**, because batching itself is broken.
Multi-request is serialized, so aggregate throughput equals single-stream
throughput. Anyone wanting to make that argument must fix this first.

The likely cause, unverified, is the same family as the earlier tensor-parallel
N>1 bug: the peer `graph_compute` and KV handling probably only set up `seq_id 0`
correctly, leaving other sequences' KV caches uninitialized. A fix would be in the
per-sequence KV and graph handling on the peer path. This was never attempted.

An earlier note claiming "localhost np=2 batching validated" was wrong. Localhost
was fast enough that the two requests completed sequentially and never actually
batched into one forward, so the bug was not exercised. Only a slow interconnect
makes requests genuinely overlap.

### Other known limits

- **Cross-process resident weights cover `-sm layer` only.** `-sm row` falls back
  to the disk weight cache.
- **One coordinator at a time.** The compute lock serializes backend compute, but
  peer topology and all-reduce state are not isolated per connection. Concurrent
  multi-coordinator serving is not supported.
- **`RPC_DIFF_DATA_TRIM` is unsafe and must stay off.** It suppresses data-only
  patches and corrupts decode, because ggml-alloc reuses buffer slots.
- **7B does not fit at N=2.** Per-device weights are roughly 1.9 GB against 1.64 GB
  available, and loading dies partway through the blocks. This is a weight floor;
  the activation pool shrinks activations, not weights, so it cannot help. N=2
  needs a smaller quantization.
- **`llama-bench` teardown race with `-sm row`.** The run completes and prints
  valid rows, then asserts during socket teardown.

---

## Operational traps

These cost significant time to diagnose and are easy to hit again.

**Rebuild the coordinator, not just the servers.** `ggml-rpc.cpp` compiles into
both `rpc-server` and `llama-cli`. Much of the logic — graph-send computation,
tensor splitting, buffer caching — runs **client-side on the coordinator**. A fix
deployed only to the peers does nothing. This cost hours: three genuine
client-side fixes appeared not to work because they were never in the running
client. `make.sh deploy` handles both; an ad-hoc scp loop to the peers does not.

**All nodes must run the same commit.** `ggml-rpc.cpp` differs materially across
commits, and version skew produces `GGML_ASSERT` aborts during
`get_device_memory` that look like a connection-count scale limit but are not.

**Use exact-name process matching.** `pgrep -f rpc-server` and `pkill -f
rpc-server` **match the ssh shell running them**, because the string appears in
its argv. `pkill -9 -f rpc-server` will kill its own shell, so cleanup silently
half-completes. Use `pgrep -x rpc-server` and `pkill -9 -x rpc-server`.

**Orphaned servers poison later runs.** Servers started detached survive the ssh
session that launched them. Stale servers accumulate across ports, and a new
client then talks to a mix of fresh and stale servers with mismatched buffer
pointers, producing intermittent `GGML_ASSERT(tensor->data >= buffer_start)`
failures in `deserialize_tensor` that look like a code bug. Before every run:
`pkill -9 -x rpc-server` on each peer, confirm zero remain, start one, confirm one.

**rpi12 is 32-bit `armhf`.** Its `rpc-server` replies with a mismatched struct ABI
that a 64-bit client cannot parse, aborting in `get_device_memory`. Exclude it.

**`llama-bench` needs small batches on 7B.** Its defaults (`-b 2048 -ub 512`)
allocate buffers large enough to OOM a peer during load, which surfaces as a
`SET_TENSOR_CACHE` send failure. Use `-b 128 -ub 64`, or use `llama-cli` and read
the `prompt eval` line.

**Environment-variable gates must match on every node.** Optimization flags change
the wire and compute path; setting one on the coordinator but not the servers
corrupts decode. Debug flags are safe to set per-node.
`rpi-automation/ablate.sh` and `rpc_profiles.conf` apply a profile fleet-wide.

---

## Research artifacts

- **`graph_partitioning/`** — a Python simulator for placement and partitioning
  policies, used to explore configurations too expensive to measure directly. It
  models device compute, network airtime including the contention term that
  explains the tensor-parallel anti-scaling, and elastic policies.
  `simulator.py`, `model.py`, `network.py`, and `policies/` are the core;
  `hw_calibration.ipynb` and `design_exploration.ipynb` are the analysis
  notebooks. **This directory has no documentation** — reading `simulator.py` and
  the notebooks is currently the only way in, and writing a short README for it
  would be a genuine contribution.
- **`bench-results.md`** — raw measurements by configuration.
- **`sim_output/`** — simulator run outputs (~2900 files).
- **`paper/`** — a symlink to a separate paper repository, which will not resolve
  on another machine.

---

## Where to pick up

From `OPTIMIZATIONS.md` Part 2, the open research questions are:

- Does incremental boundary-shifting stay ahead of KV growth, or does migration
  latency become the bottleneck at long context?
- Can KV migration be made lazy or streamed, moving cold KV in the background
  before saturation, to hide the ~13 s boundary-shift cost? Is there a useful
  device *prefetch* — recruiting device N+1 a few hundred tokens before the
  capacity limit, so migration overlaps decode?
- Where is the practical ceiling with measured overhead? Hops and RPC add roughly
  0.4 s to the 7B floor of 1.09 s, which suggests an interactive ceiling nearer
  5-6B.

Two concrete engineering tasks stand out:

1. **Fix multi-sequence batching.** It is the single blocking item for the
   throughput-serving argument, and the suspected cause is identified above.
2. **Document the simulator.** It is a substantial body of work with no entry
   point, and it underpins the design analysis.
