# Peer-branch: optimizations & design (distributed LLM inference over RPC/WiFi)

Everything built on the `peer` branch for tensor-parallel / pipeline LLM inference across
RPC-connected Raspberry Pi 4B devices on a shared (IBSS / AP) WiFi cell. Models:
TinyLlama-1.1B-chat Q5_K_M and Llama-2-7B Q4_K_M, N=2–4. This is the single reference for
what was implemented, how to toggle it, and the design that motivates it. Raw measurements
live in `bench-results.md`.

**Headlines:** decode ~7.9 → ~6.5 s/tok (~18%) and prefill ~16% (all-reduce work); warm
model load 4.2×; a per-server-memory fix that turned N=4 long prefills from OOM-killed into
fitting; a **resident serving process** (load once, serve many, no per-request reload); and
**cross-process resident weights** (a restarted coordinator rebinds to weights still on the
servers — cluster 7B reload 283 s → 153 s).

---

# Part 1 — Implemented optimizations

## Measured impact (A/B summary)

Consolidated before→after for each optimization; details + setup in the sections below. All
bit-identical / output-preserving unless noted. 7B Q4_K_M on 4× Pi 4B over WiFi unless stated.

| Optimization | Metric | Before → After | Gate |
|---|---|---|---|
| P2P all-reduce + deterministic fold | decode | 7.9 → 6.5 s/tok (~18%) | on |
| Diff cache | graph-send / token | ~halved | `RPC_PP_DIFF` (pipeline) |
| Direct pipeline handoff | prefill / decode | −0.8% / −0.7% | on |
| Activation pool | per-server RAM | 1205 → 475 MB (long prefill OOM → fits) | on |
| Per-token buffer reuse | graph-build; decode | 3.3 s → 5 ms; 4.3× | on |
| On-disk weight cache | load | 194 → 46 s (4.2×) | on |
| Cross-process persist | reload / restart | 283 → 153 s (~130 s saved) | `RPC_PERSIST` |
| Resident serving process | load / request | cold 1015 / warm 283 / **resident 0 s** | — |
| Growable KV cache | per-Pi KV footprint | commit full `n_ctx` → grow in blocks (~500 MB/Pi saved) | `LLAMA_KV_GROW_BLOCK` |
| Elastic boundary shift | per-shift stall | 44 s → **0.2 s** (prefetch) / **3.4 s** (cache-aware) | `LLAMA_REBALANCE_*` |

The elastic boundary-shift row is broken out per weight-move strategy in *Elastic rebalancing* below.

## Communication (all-reduce is ≈74% of decode — the primary target)

- **P2P all-reduce** — servers exchange tensor-parallel partials *directly* peer-to-peer
  instead of relaying every partial through the client/coordinator. Auto-enabled whenever
  `--rpc` is passed (the client broadcasts a create-peer-connection command; each server
  dials all peers and holds the sockets open).
- **All-reduce latency + determinism** — parallel broadcast (one thread per peer) +
  fire-and-forget (no acknowledgement round-trip; TCP guarantees delivery+order, dead links
  caught by timeout) + a deterministic **source-ordered fold** (each partial carries its
  source device id; the receiver folds in ascending id order = single-device contiguous-K
  grouping). **decode ~7.9 → ~6.5 s/tok (~18%)**, and now bit-reproducible run-to-run.
- **Diff cache** — per decode token, ship only the tensors that *changed* since the last
  token, not the full ~1000-tensor graph. **graph-send roughly halved.** Gate: on by default
  for the split path; opt-in on the pipeline path via `RPC_PP_DIFF`.
- **Prefetch (graph-advance)** — predict the next token's patch and send a *payload-free*
  advance command when the prediction holds. Byte-identical to a full send.
- **Graph one-way** — fire-and-forget the graph-send (server skips the ack), removing one
  round-trip per device per token. Opt out: `RPC_NO_GRAPH_ONEWAY`.
- **Direct pipeline handoff** — on the pipeline (`-sm layer`) path, a producer stage pushes
  its output tensor straight into the next stage's server buffer (a dedicated send-to-peer
  command) instead of relaying it through the coordinator. HW-validated correct; ~0.8%
  prefill / ~0.7% decode at 7B N=4 (small by design; larger at higher N / throughput).
  Opt out: `RPC_NO_DIRECT_HANDOFF`; debug: `RPC_DBG_HANDOFF`.
- **Opt-in accuracy/latency tradeoffs** — reduced-precision all-reduce partials
  (int8-per-block / fp16) and a tree all-reduce (reduce-to-root + broadcast, `RPC_AR_TREE`).

## Memory

- **Activation pool** — cross-layer sub-allocation: each device uses one pool buffer per
  compute buffer at the allocator's reuse offsets, instead of one server buffer per
  replicated activation (which held every layer's activations at once).
  **per-server 1205 → 475 MB; N=4 long prefill: OOM-killed → fits.** On by default; opt out
  `RPC_NO_POOL`.
- **Per-token buffer reuse** — reuse a split tensor's server buffer across tokens instead of
  re-allocating per token. Removes ~1500 alloc + ~1500 get-base RPCs per token.
  **graph-build 3.3 s → ~5 ms; decode 4.3×.** On by default; opt out `RPC_NO_PERSIST_BUFFERS`.
  (Note: this is per-token *activation* reuse within one process — distinct from the
  cross-process *weight* persistence below.)

## Load time

- **On-disk weight cache** — servers persist weight slices to local disk and skip the network
  re-upload on subsequent runs. **cold load 194 s → warm 46 s (4.2×)** at 7B N=4. Details:
  - Covers **both** the TP split path and the pipeline (`-sm layer`) whole-tensor path; the
    pipeline coverage cut the ~14-min per-run cold re-upload that pipeline runs used to pay.
  - Content-hash keys are **namespaced** by `{kind (whole vs split), device id, split_dim}`
    so a pipeline whole tensor and a TP slice with identical bytes never collide, and TP's
    replicated non-split tensors stay off the cache path (routing them through it corrupted
    `-sm row` output — the namespacing + a `!split` gate fixed it).
  - **Atomic writes** (temp file + rename) so a reader never sees a partial file and two
    writers of the same content race harmlessly.
  - **Size-cap LRU eviction** (`RPC_WEIGHT_CACHE_MAX_GB`, default 8) prunes oldest-mtime
    files; mtime is refreshed on hit so active models aren't evicted.
  - Cache dir: `$RPC_WEIGHT_CACHE_DIR` else `$HOME/.cache/llama-rpc-weights`. Disable:
    `RPC_NO_WEIGHT_CACHE`; debug: `RPC_DBG_WCACHE`.
- **Concurrent per-device weight upload/download** — one thread per device (sockets pre-fetched
  and held on the main thread so workers don't churn connections). Opt out `RPC_SERIAL_UPLOAD`.
- **Note — the client GGUF-read floor.** The coordinator reads the whole GGUF (~3.9 GB,
  ~95 s on the rpi1 SD) on **every** process start regardless of caching. Only a persistent
  *client* (the serving process below) removes it; the weight cache and cross-process persist
  remove the *server-side* upload/alloc, not this client read.

## Serving process & cross-process persistence

- **Resident serving process** — `llama-server` runs as-is over the peer RPC backend: it loads
  the model **once** and serves many requests with **no per-request reload** (remote weight
  buffers stay resident for the process lifetime; compute is serialized by the single decode
  loop, so concurrent requests queue or continuous-batch rather than racing). Validated
  localhost + 4-Pi 7B cluster. Load amortization: cold 1015 s / warm 283 s / **resident 0 s
  per request**. (`-sm layer` is safe for unbounded serving; `-sm row`'s graph-number cache is
  bounded — see Known limits.)
- **Cross-process resident weights** (`RPC_PERSIST`, default off) — a *fresh* coordinator
  process rebinds to weight buffers still resident on the servers from a previous process,
  skipping the server alloc **and** the WiFi upload. Survives coordinator restarts (and
  crashes). **Cluster 7B: run-2 reload 153 s vs 283 s warm (~130 s saved/restart); every
  weight tensor's upload skipped.** Mechanism:
  - llama.cpp computes a `model_key` = hash(model bytes/tensors/elements + split mode +
    n_gpu_layers + tensor split) and announces it to the RPC backend around the weight-load
    loop (no-op unless `RPC_PERSIST` is set).
  - Each server holds a registry keyed by `(model_key, size)` (each server is one device). On
    load, the client asks to **bind**; a hit returns the buffer's *live* pointer (pointers
    change every process, so the buffer is rebound, never assumed) and the client skips the
    upload; a miss allocates + uploads and then **registers** the buffer so it's retained past
    teardown. On teardown the client **detaches** (keeps it resident, marks it free to rebind).
  - **Release-on-disconnect** — each resident buffer records the connection id that holds it;
    when a client's serve loop exits for *any* reason (clean exit, error return, or crash /
    kill), the server releases that connection's claims. So a coordinator that dies without a
    clean detach doesn't leave its weights locked — the next process still rebinds.
  - **RAM-cap eviction** (`RPC_PERSIST_MAX_GB`, default 8; set ~1.2 on the 2 GB Pis) evicts the
    least-recently-used *fully-unclaimed* model when a new registration would exceed the cap,
    so switching models can't OOM the memory-tight peers (2 GB forces reuse-in-place — no
    server-local copies). Debug: `RPC_DBG_PERSIST`.
  - Scope: covers the pipeline (`-sm layer`) load path only. The TP (`-sm row`) split load path
    was attempted and reverted — its per-slice server buffers can't be reliably reused across
    processes (a correctly-rebound slice still computed garbage), and TP has no usable WiFi regime
    anyway (Part 2); `-sm row` gets warm loads from the on-disk weight cache instead.
- **Concurrent-forward serialization** — a per-server compute lock serializes backend compute across
  connections, so two coordinator processes sharing resident weights can't run on the shared compute
  backend at once. Held only around the local compute, never across the peer all-reduce wait (the
  all-reduce fold is a separate, unlocked backend call), so it can't deadlock a forward. This fully
  serializes concurrent `-sm layer` (pipeline, no all-reduce) forwards; concurrent `-sm row` is not
  fully supported (its all-reduce state is keyed only by tensor name + seq, so two concurrent TP
  forwards would cross-talk). The persist use case is sequential processes, where the lock is
  uncontended.

## Elastic rebalancing (growable KV + adaptive boundary shift)

The pipeline is sized to fit, then rebalanced as the KV cache grows (Part 2 is the analysis). Two
mechanisms, both opt-in and bit-identical to baseline:

- **Growable KV cache** (`LLAMA_KV_GROW_BLOCK=N`) — grow the cache in blocks of N tokens as context
  grows, instead of committing the full `n_ctx` up front. Frees ~500 MB/Pi on the 7B cluster and is the
  prerequisite for elastic sizing. Ring-buffer-aware; block ≥ `n_ubatch`.
- **Gated boundary shift** (`LLAMA_REBALANCE_BUDGET_MB`) — when a device's KV footprint crosses the
  budget, move one of its layers to another device (weights + KV), gated by a cooldown and a
  converge-not-thrash rule (shift only when the source has ≥2 more layers than the target). The KV part
  is negligible (1–1.6 MB, ~0.2–0.3 s); the **weight** part (~130 MB/layer) dominates and is what the
  strategies below optimize. Policy selected by `LLAMA_REBALANCE_POLICY`.

**Per-shift cost, 7B / 4-Pi / layer ≈130 MB (measured, A/B):**

| Weight-move strategy | weight move | total shift | WiFi | how |
|---|---|---|---|---|
| Host relay (baseline) | 43.9 s | 44.3 s | 2× | src → coordinator → dst |
| Direct peer copy | 20.2 s | 20.4 s | 1× | src pushes straight to dst (`SEND_TO_PEER`) |
| **Prefetch** (adjacent policy) | ~0 visible | **0.21 s** | 1× overlapped | staged in background, hidden behind decode |
| **Cache-aware**, disk hit | 3.2 s | **3.4 s** | **0×** | dst loads the layer from its local weight cache |
| Cache-aware, page-cached | 0.3 s | 0.6 s | 0× | re-read from the OS page cache |

- **Direct peer copy + adjacent-neighbor** (default `adjacent` policy): shift the overloaded device's
  edge layer to its lower-loaded pipeline neighbor (stays contiguous, no extra handoffs), pushing the
  weights straight to the destination — halves the transfer (44 → 20 s) by removing the host round-trip.
- **Prefetch**: predict the budget crossing from the linear KV growth and pre-stage the layer's weights
  in the background (server-side async push on a dedicated socket), so the ~20 s transfer overlaps the
  compute-bound decode where the WiFi is idle; the shift then commits in ~0.2 s. Needs a small enough
  grow-block that the device crosses budget gradually (so there's decode-time lead).
- **Cache-aware** (`LLAMA_REBALANCE_POLICY=cache`): shift to whichever device already holds the layer in
  its on-disk weight cache and load it **locally** — 0× WiFi. WHOLE weight-cache keys are
  device-independent content hashes, so a layer's key is identical on every server; a cold destination
  transfers once and lazily caches, so subsequent moves hit. Robust exactly where prefetch degrades
  (busy channel / throughput serving, where the idle-WiFi assumption breaks). On the well-used cluster
  every shift already hit (`cache=9/9`); a true miss costs the direct-peer transfer (20.4 s) measured
  above, so the SD read (~3.2 s, or ~0.3 s from page cache) is the steady-state cost.

**Payoff — reaching contexts static provisioning can't (7B, 4× 2 GB Pi, `-ts 3,2,2,2`, measured):**
Static provisioning of `n_ctx = 8192` (no growable KV) commits the full KV at context creation: dev0
(~11 layers) needs ~1.34 GB weights + ~1.4 GB KV ≈ 2.7 GB > 2 GB, so a server is OOM-killed during
`kv_cache_init` (`kv_size = 8192`) and the client aborts (`Connection closed by peer`). **It cannot
run.** The *same* provisioning under elastic (`LLAMA_KV_GROW_BLOCK=64` + `LLAMA_REBALANCE_BUDGET_MB=20`)
commits the KV on demand and sheds a layer off dev0 as it crosses budget: **4 grows, 1 shift (layer 10
dev0→dev1 at 64 cells), 242 coherent tokens, 0 errors**, and every Pi stays under 2 GB (rpi20=1395,
rpi22=1140, rpi24=874, rpi25=905 MB at ctx ≈160). Elastic runs a provisioning static crashes on.

- *Caveat (honest):* shedding relieves KV *growth* (fewer active layers on the pressured device → a
  smaller KV slope), but a shed layer's **weights** stay resident on the source (not reclaimed), so it
  lowers the slope, not the ~1.34 GB weight floor. Reclaiming source weights (re-partition the shared
  buffer) is future work; until then the reachable-context gain is bounded by the retained weights.

## Environment-variable reference

| Var | Default | Effect |
|---|---|---|
| `RPC_NO_WEIGHT_CACHE` | off | disable the on-disk weight cache |
| `RPC_WEIGHT_CACHE_DIR` | `$HOME/.cache/llama-rpc-weights` | cache location |
| `RPC_WEIGHT_CACHE_MAX_GB` | 8 | weight-cache size cap (LRU prune) |
| `RPC_DBG_WCACHE` | off | log cache HIT/STORE/EVICT |
| `RPC_PP_DIFF` | off | enable the diff cache on the pipeline path |
| `RPC_AR_TREE` | off | tree all-reduce instead of all-to-all |
| `RPC_NO_GRAPH_ONEWAY` | off | re-enable the graph-send ack |
| `RPC_NO_POOL` | off | disable the activation pool |
| `RPC_NO_PERSIST_BUFFERS` | off | disable per-token activation buffer reuse |
| `RPC_SERIAL_UPLOAD` | off | serialize per-device upload/download |
| `RPC_NO_DIRECT_HANDOFF` | off | relay pipeline handoff through the coordinator |
| `RPC_DBG_HANDOFF` | off | log direct handoff |
| `RPC_PERSIST` | off | cross-process resident weights (bind/register/detach) |
| `RPC_PERSIST_MAX_GB` | 8 | resident-model RAM cap per server (≤0 = unlimited) |
| `RPC_DBG_PERSIST` | off | log BIND/REGISTER/DETACH/RELEASE/EVICT |
| `RPC_GRAPH_WRAP_AT` | 256 | lower the diff-cache graph-number wrap point for testing the reuse path |
| `LLAMA_KV_GROW_BLOCK` | 0 (off) | grow the KV cache in blocks of N tokens instead of committing full `n_ctx` |
| `LLAMA_REBALANCE_BUDGET_MB` | 0 (off) | per-device KV budget; a device over it sheds a layer (enables elastic rebalance) |
| `LLAMA_REBALANCE_COOLDOWN` | 32 | min tokens between shifts (anti-thrash) |
| `LLAMA_REBALANCE_POLICY` | `adjacent` | `adjacent` = shift to lower-loaded neighbor + prefetch; `cache` = shift to a device that caches the layer (0× WiFi) |
| `LLAMA_REBALANCE_PREFETCH` | 1 | (adjacent policy) grow-blocks of lead to pre-stage the transfer; 0 = synchronous shift |

## Correctness fixes that were prerequisites (enabling, not speedups)

Multi-ubatch (>512 tok) TP prefill crash; transposed V-cache head-split; non-main KV-cache
zero-init; 256-grid quant-block-aligned tensor splits; `llama-bench -sm row` model-load crash
guard; single-device (non-split) graph compute path.

## Known limits

- Cross-process persist covers `-sm layer` only (see above); `-sm row` uses the disk weight cache.
- Concurrent multi-coordinator serving over the peer backend is only partially supported (the
  compute lock serializes backend compute, but the peer topology and all-reduce state aren't
  per-connection isolated). The supported model is one coordinator at a time; persist rebinds on
  restart.

## Correctness note: `-sm row` diff-cache graph-number wrap (fixed)

The `-sm row` diff cache keys stored graphs by a `uint8_t` graph number the client hands out per
distinct topology; it wraps after 256. A wrapped number reused for a new topology used to (a) append
the new graph's segments onto the stale graph's chain on the server, and (b) leave the old
`topo_hash → number` mapping on the client so the old topology could later false-hit and patch the
wrong graph. Fixed: the server replaces (delete-before-store) on the first segment of a fresh batch,
and the client purges stale mappings for a reused number. Validated by forcing an early wrap
(`RPC_GRAPH_WRAP_AT=4`): numbers 0–3 reused 5× each across distinct topologies, output stayed
coherent. No effect below 256 distinct graphs.

---

# Part 2 — Design & analysis: adaptive parallelism

**Thesis.** On a cluster of memory-constrained devices sharing one wireless channel,
single-stream decode is **memory-bandwidth-bound**, and neither classical parallelism axis
helps the way it does in a datacenter: *pipeline* can't parallelize a single stream's
sequential weight read, and *tensor parallelism (TP)* can — but its per-layer all-reduce over
the shared channel costs more than the read it parallelizes. What remains is **pipeline-only,
sized to fit, and adaptively rebalanced as the KV cache grows**, with TP excluded on both
measured and analytical grounds. This yields a hard **single-stream model-size ceiling** set
by one device's memory bandwidth (~7B at ~1 s/token on Pi 4B).

Symbols: `W` = total weight bytes; `w_ℓ` = one layer's weight bytes; `R_bw` = per-device
memory bandwidth (Pi 4B ≈ 3.5 GB/s); `N` = devices; `g` = TP group size;
`AR(g) = AR₀·g^α` = per-layer all-reduce time, `α ≈ 2.1`, `AR₀` the base cost of a `d_model`
f32 exchange over the link (airtime-calibrated); `k` = reduces/layer (≈2). Measured anchors
(7B Q4, N=4): pipeline decode **0.66 t/s**, TP **0.30 t/s**; prefill 1.38 vs 0.40 t/s;
loopback TP decode **6.43 t/s** (21× the WiFi number).

### The premise
Autoregressive decode reads **every weight once per token** → latency ≥ `W / R_eff`. In
pipeline a token traverses stages sequentially (one active at a time), so `R_eff = R_bw` of
*one* device — pipeline doesn't reduce single-token latency, only overlaps *different* tokens
(throughput). TP is the only axis that raises `R_eff` (splits the read `g` ways). So the whole
question is "is TP's all-reduce cheaper than the read it saves?"

### TP has no usable regime here
TP wins a layer iff `(w_ℓ/R_bw)·(1−1/g) > k·AR₀·g^α`. Plugging the hardware (7B: `w_ℓ≈113 MB`,
`w_ℓ/R_bw≈32 ms`; `AR₀≈5 ms`; `k=2`, `α=2.1`) at `g=4`: saved = 32·0.75 = **24 ms**, added =
2·5·4^2.1 ≈ **184 ms** — RHS beats LHS ~8×. TP loses at every `g≥2`, in decode **and** prefill
(prefill's batch amortizes AR's latency term, not the bandwidth term, which dominates over
WiFi — measured TP prefill 0.40 is still 3.5× below pipeline's 1.38).

**The "forced" regime is empty.** TP is only *required* when one layer exceeds one device's
RAM: a 1.6 GB Q4 layer ≈ a ~100B+ model, which reads ~50 GB/token → 15–30 s/token regardless
of split. So the regime that *forces* TP coincides with the regime that's *hopelessly slow
anyway*. (13B's layer is only ~178 MB and fits — TP isn't even forced.)

**Crossover (what flips it).** TP wins at `g=4` iff `AR₀ < 24/(2·18.4) ≈ 0.65 ms`. WiFi
`AR₀≈5 ms`; a **wired gigabit switch** gives ≈0.13 ms → below threshold → **TP becomes
competitive.** So the claim is sharp and hardware-specific: *TP fails because of the shared
channel, not because of TP* (the 21× loopback number is the same statement measured directly).
The one experiment that nails it: peers on an eth switch, re-run N=4 layer-vs-row.

### Adaptive parallelism = KV-growth-driven pipeline rebalancing
The only thing that changes within a request is the KV cache, which grows linearly with
position (`kv/token ≈ 1 MB` for 7B f16). Per-device memory at position `t`:
`M_d(t) = W/N + (kv/token)·t/N + A`, feasible while `M_d(t) ≤ B` (≈1.6 GB). Max context at
fixed `N`: `t_max(N) ≈ (B·N − W − A·N)/(kv/token)` — 7B N=4 ≈ 2600 tok; each **added device
buys ≈1600 more tokens.** So the natural policy is an **elastic pipeline**: start at the
minimal `N` that holds weights + a short context and, as `t` nears `t_max(N)`, recruit a device
or shift a boundary and rebalance. KV distributes as total/N in both PP and TP, so growth
**never resurrects TP**.

**Rebalance cost.** Weights are free to "move" (each device caches all slices locally → a
re-partition is a re-mmap, not a WiFi re-upload). Only KV migrates: a full re-shard on `N→N+1`
moves ~`(t·kv/token)/(N+1)` (~0.4 GB ≈ 80 s at t=2000, N=4 — too expensive to do often), but an
**incremental boundary shift** (move one layer to a neighbor) moves only `t·(kv/token)/n_layer`
= `t·32 KB` (~64 MB ≈ 13 s). Shift boundaries one layer at a time under pressure; recruit a
fresh device only when the whole cluster saturates.

**When it's worth it.** If max context is known, pick `N` up front. Adaptive rebalancing earns
its complexity when context length is unknown/unbounded, when you want to start minimal and
scale on demand, or to reach contexts no static `N` could hold. Paper framing: *elastic
inference whose device footprint grows with the sequence, driven by KV pressure.*

### The single-stream model-size ceiling
Pipeline single-stream decode latency ≈ `W/R_bw` (one device's bandwidth) + hops. For target
latency `L`, the largest usable model is `W ≤ L·R_bw`:

| model (Q4) | W | pipeline decode `W/R_bw` | measured |
|---|---:|---:|---:|
| TinyLlama 1.1B | 0.7 GB | 0.20 s (5.0 t/s) | — |
| **7B** | 3.8 GB | 1.09 s (0.9 t/s) | **0.66 t/s** (+hops/RPC) |
| 13B | 7.5 GB | 2.14 s (0.47 t/s) | — |
| 30B | 17 GB | 4.9 s (0.2 t/s) | — |

At a ~1 s/token interactive bar, `W ≤ 3.5 GB` ⇒ **~7B is the ceiling**, and it's *hard*: TP is
the only axis that could raise `R_eff` and it's excluded. **No parallelism strategy moves this
ceiling on shared WiFi** — it's a property of one device's bandwidth. Escape hatches: faster
per-device bandwidth (raises it linearly); a fast interconnect re-enables TP (`R_eff = N·R_bw`);
or **throughput serving** — pipeline overlaps microbatches across stages, decoupling aggregate
tokens/s from single-stream latency, so big models regain value for *many concurrent* requests
even though each one is slow (this is what the resident serving process in Part 1 enables).

### What the simulator must model
The `graph_partitioning` cost simulator needs: (1) a time-varying memory model `M_d(t)` with
the linear KV term and `t_max(N)`; (2) a rebalance-cost model (boundary-shift vs full re-shard
KV bytes, priced over the airtime-aware link); (3) an elastic policy that picks the rebalance
schedule minimizing stalls subject to `M_d(t) ≤ B`; (4) the TP-exclusion inequality as a guard,
with the wired-switch crossover as a configurable interconnect parameter.

**Airtime-contention term (calibrated).** The all-reduce cost models the actual peer
*all-to-all* collective (N(N−1) sends contending for one 802.11 channel). Calibrated against
the measured peer-to-peer `-sm row` N=2→4 prefill anti-scaling (4.09×): exponent ≈2.1
reproduces it near-exactly, and an iperf3 concurrent-flows sweep independently confirms the
channel is essentially fully airtime-serialized. Wired into `hw_calibration.ipynb` §10 as a
peer-to-peer scenario alongside the centralized one.

### Open questions
- Does incremental boundary-shifting stay ahead of KV growth, or does migration latency itself
  become the bottleneck at long context?
- Can KV migration be lazy/streamed (move cold KV in the background before saturation) to hide
  the ~13 s boundary-shift cost? Is there a device *prefetch* (recruit `N+1` a few hundred
  tokens before `t_max(N)` so migration overlaps decode)?
- Where is the ceiling with *measured* overhead (hops+RPC add ~0.4 s to the 7B 1.09 s floor) —
  is the practical interactive ceiling closer to ~5–6B?
