# RPC backend environment variables (`peer` branch)

Every knob the RPC tensor-parallel backend reads, with defaults and rationale. All live in
[`ggml/src/ggml-rpc/ggml-rpc.cpp`](../ggml/src/ggml-rpc/ggml-rpc.cpp).

**Scope rule:** optimization gates must be set **identically on the client (llama-cli on the
coordinator) AND every rpc-server** — they change the wire/compute path and a mismatch corrupts
decode. Debug flags only affect logging and can be set per-node (the "Set on" column says where the
output appears). `ablate.sh` + `rpc_profiles.conf` apply a profile to all nodes for you.

## Master switch

| Var | Default | Effect | Set on |
|---|---|---|---|
| `RPC_NO_OPT` | unset (**opts ON**) | Set=1 → disables **every** optimization below = faithful stock-ish baseline (same binary). `rpc_opt_enabled()` is the gate most opts hang off. | both |

## All-reduce (the dominant decode cost)

| Var | Default | Effect | Set on |
|---|---|---|---|
| `RPC_AR_PARTIAL` | unset (**f32**) | `=f32\|fp16\|int8\|int8b\|fp8` — precision of the ~44 all-reduce partials/token. **fp16** exact-enough; **int8b** (per-block, ~1.06 B/elem) the usable 1-byte format; int8/fp8 too lossy. Cuts airtime ∝ bytes. Quality tradeoff (not bit-identical). | both |
| `RPC_AR_FP16` | unset | Back-compat alias for `RPC_AR_PARTIAL=fp16`. Prefer `RPC_AR_PARTIAL`. | both |
| `RPC_AR_TREE` | unset (**all-to-all**) | Set=1 → reduce-to-root + broadcast (2(N−1) transmissions vs N(N−1)); ~5% on the contention-bound cell. Result bit-identical to all-to-all. | both |
| `RPC_ALLREDUCE_TIMEOUT` | **30** (seconds) | Per-reduce wait before declaring a peer dead. | server |

## Diff cache / graph-send / prefetch

| Var | Default | Effect | Set on |
|---|---|---|---|
| `RPC_NO_PREFETCH` | **ON** by default (opt-out) | Default-on win: predict the next-token patch + ship a 1-byte `GRAPH_ADVANCE` when it holds → ~0-byte graph-send. Byte-identical. Set `=1` to disable (ablation). Prediction holds at 100% because `RPC_PERSIST_BUFFERS` is also default-on (stabilizes `data` ptrs). | both |
| `RPC_NO_GRAPH_ONEWAY` | **ON** by default (opt-out) | Default-on win: fire-and-forget the graph-send (server skips the ack), pipelined before DO_COMPUTATION → removes one ack RTT/device/token. Byte-identical (TCP guarantees delivery). Set `=1` to disable. | both |
| `RPC_PREFETCH_SKIP_BUILD` | unset (**off**) | Skip the per-token client REBUILD (advance prev/pred arithmetically, send blind ADVANCE). Requires prefetch. **Moot** — once persist kills the per-token RPC storm the rebuild is already ~5 ms, so there's little left to skip. | client |
| `RPC_SKIP_VERIFY_PERIOD` | **32** | Skip-build: do a full rebuild every N tokens to re-verify (0 = never until a MISS). | client |
| `RPC_DIFF_DATA_TRIM` | unset (**off**) | ⚠️ **EXPERIMENTAL + UNSAFE — do not enable.** Suppresses data-only patches; corrupts decode because ggml-alloc reuses buffer slots. | both |

## Buffers / load

| Var | Default | Effect | Set on |
|---|---|---|---|
| `RPC_NO_PERSIST_BUFFERS` | **ON** by default (opt-out) | Default-on win: reuse a split tensor's server buffer **across tokens** instead of re-allocating per token. Kills ~1500 `ALLOC_BUFFER` + ~1500 `get_base` RPCs/token (the dominant ~2.2 s build cost). **HW: build 3.3 s → 0.005 s, decode 0.13 → 0.56 t/s (4.3×), byte-identical**; also stabilizes `data` ptrs so prefetch prediction recovers 15% → 100%. Stacks with int8b → 0.79 t/s (6.1×). Set `=1` to disable. Client-side. | client |
| `RPC_NO_WEIGHT_CACHE` | unset (**cache ON**) | Set=1 → disable the local weight cache (re-upload weights each load). Cache is on by default when opts on. | server |
| `RPC_WEIGHT_CACHE_DIR` | `$HOME/.cache/llama-rpc-weights` | Where the weight cache lives (auto-mkdir). | server |
| `RPC_SERIAL_UPLOAD` | unset (**concurrent**) | Set=1 → serial (not threaded) buffer alloc/upload/download. Diagnostic; concurrent is the default when opts on. | both |

## Debug / instrumentation (all default OFF, logging only)

| Var | Logs | Set on |
|---|---|---|
| `RPC_DBG_TIMING` | per-token `[rpc-timing]` (graph_send / execute+allreduce) + `[rpc-phase]` (build / send / compute_wall / gather / build_seg_tensors breakdown) | **client** |
| `RPC_DBG_DIFFCACHE` | `[rpc-diffcache]` HIT/MISS + `[rpc-skip]` decisions (client) + per-token/per-MISS server lifecycle | both |
| `RPC_DBG_PATCH` | `[PATCH]` per-token changed-field classification (pos_only / data_only / mixed) + data-ptr samples | client |
| `RPC_DBG_GETBASE` | `[GETBASE]` one line per actual get_base RPC (cache miss) | client |
| `RPC_DBG_ALLOC` | `[ALLOC]` one line per init_tensor `ALLOC_BUFFER` (dev0) | client |
| `RPC_DBG_IDSTAB` | `[IDSTAB]` per-segment cgraph pointer-stability counts (id_same/id_diff across tokens) | client |
| `RPC_PREFETCH_DBG` | `[PREFETCH]` prediction match-rate (pred_ok/pred_total over changed tensors) | client |
| `RPC_DBG_GHASH` | graph structure-hash debug | client |
| `RPC_DBG_DIFF` | per-node diff debug | client |
| `RPC_DBG_AR` | all-reduce internals (server-side) | server |
| `RPC_DBG_COMPARE` | graph-compare debug | server |
| `RPC_DBG_WCACHE` | weight-cache hit/miss debug | server |

## Common combos (see `rpc_profiles.conf`)

- **baseline**: `RPC_NO_OPT=1` (disables everything below, incl. persist/prefetch/oneway)
- **optimized** (default path, no vars): now **includes persist + prefetch + oneway** (the byte-identical
  6.1×-stack wins) — they graduated from opt-in to default-on. Just run with no env vars.
- **+ reduced-precision all-reduce** (opt-in tradeoff): add `RPC_AR_PARTIAL=int8b` (or `fp16` exact-enough)
- **ablation** (turn one default-on win OFF): `RPC_NO_PERSIST_BUFFERS=1` / `RPC_NO_PREFETCH=1` / `RPC_NO_GRAPH_ONEWAY=1`
- **profiling a run**: `RPC_DBG_TIMING=1` on the client
