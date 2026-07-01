# Coordinator: rpi1
## Local execution - no rpc

`./build-rpc/bin/llama-bench -m models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf -p 32,128,256 -n 32,64 -r 5`

| model                  | size       | params | backend | ngl | test  |         t/s |
| ---------------------- | ---------- | ------ | ------- | --: | ----- | ----------: |
| llama 1B Q5_K - Medium | 745.11 MiB | 1.10 B | RPC     |  99 | pp32  | 7.64 ± 0.14 |
| llama 1B Q5_K - Medium | 745.11 MiB | 1.10 B | RPC     |  99 | pp128 | 6.41 ± 0.51 |
| llama 1B Q5_K - Medium | 745.11 MiB | 1.10 B | RPC     |  99 | pp256 | 4.84 ± 0.23 |
| llama 1B Q5_K - Medium | 745.11 MiB | 1.10 B | RPC     |  99 | tg32  | 3.27 ± 0.06 |
| llama 1B Q5_K - Medium | 745.11 MiB | 1.10 B | RPC     |  99 | tg64  | 3.20 ± 0.05 |

# Coordinator: rpi2
## local execution - no rpc (2 runs)
`./bin/llama-bench   -m ~/llama.cpp/models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf  -ngl 11 -p 32,128,256 -n 32,64 -r 5`
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  11 |          pp32 |          7.93 ± 0.00 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  11 |         pp128 |          7.40 ± 0.38 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  11 |         pp256 |          5.61 ± 0.28 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  11 |          tg32 |          3.86 ± 0.05 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  11 |          tg64 |          3.75 ± 0.04 |

`./build-rpc/bin/llama-bench -m ~/llama.cpp/models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf -p 32,128,256 -n 32,64 -r 5`
| model                  |       size | params | backend | ngl |  test |         t/s |
| ---------------------- | ---------: | -----: | ------- | --: | ----: | ----------: |
| llama 1B Q5_K - Medium | 745.11 MiB | 1.10 B | RPC     |  99 |  pp32 | 7.92 ± 0.00 |
| llama 1B Q5_K - Medium | 745.11 MiB | 1.10 B | RPC     |  99 | pp128 | 7.70 ± 0.14 |
| llama 1B Q5_K - Medium | 745.11 MiB | 1.10 B | RPC     |  99 | pp256 | 6.02 ± 0.37 |
| llama 1B Q5_K - Medium | 745.11 MiB | 1.10 B | RPC     |  99 |  tg32 | 3.92 ± 0.03 |
| llama 1B Q5_K - Medium | 745.11 MiB | 1.10 B | RPC     |  99 |  tg64 | 3.88 ± 0.03 |

## 3 offloaded layers to rpi1
`./bin/llama-bench   -m ~/llama.cpp/models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf   --rpc 128.174.61.160:50052   -ngl 3 -p 32,128,256 -n 32,64 -r 5`
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |   3 |          pp32 |          7.05 ± 0.21 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |   3 |         pp128 |          7.20 ± 0.08 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |   3 |         pp256 |          5.95 ± 0.27 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |   3 |          tg32 |          3.29 ± 0.07 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |   3 |          tg64 |          3.22 ± 0.01 |

## all offloaded layers to rpi1
`./bin/llama-bench   -m ~/llama.cpp/models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf   --rpc 128.174.61.160:50052   -ngl 99 -p 32,128,256 -n 32,64 -r 5`
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  99 |          pp32 |          6.52 ± 0.01 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  99 |         pp128 |          6.68 ± 0.52 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  99 |         pp256 |          5.01 ± 0.24 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  99 |          tg32 |          2.33 ± 0.03 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  99 |          tg64 |          2.32 ± 0.03 |

## 11 offloaded layers to rpi1
`./bin/llama-bench   -m ~/llama.cpp/models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf   --rpc 128.174.61.160:50052   -ngl 11 -p 32,128,256 -n 32,64 -r 5`
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  11 |          pp32 |          7.25 ± 0.20 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  11 |         pp128 |          7.01 ± 0.16 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  11 |         pp256 |          6.72 ± 0.04 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  11 |          tg32 |          2.93 ± 0.03 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | RPC        |  11 |          tg64 |          2.93 ± 0.02 |

# Coordinator: Mac
## non-local network (close to AP)
`./build/bin/llama-bench -m models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf --rpc 128.174.61.133:50052,128.174.61.160:50052 -ngl 3 -p 32,128,256 -n 32,64 -r 3`
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | BLAS,RPC   |       4 |          pp32 |         20.18 ± 1.85 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | BLAS,RPC   |       4 |         pp128 |         30.99 ± 2.91 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | BLAS,RPC   |       4 |         pp256 |         35.13 ± 1.04 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | BLAS,RPC   |       4 |          tg32 |          3.12 ± 0.11 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | BLAS,RPC   |       4 |          tg64 |          2.98 ± 0.23 |

## local network (far from AP)
`./build/bin/llama-bench -m models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf --rpc 128.174.61.133:50052,128.174.61.160:50052 -ngl 3 -p 32,128,256 -n 32,64 -r 3`
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | BLAS,RPC   |       4 |          pp32 |         16.84 ± 4.16 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | BLAS,RPC   |       4 |         pp128 |         21.12 ± 2.16 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | BLAS,RPC   |       4 |         pp256 |         22.34 ± 2.50 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | BLAS,RPC   |       4 |          tg32 |          3.29 ± 0.73 |
| llama 1B Q5_K - Medium         | 745.11 MiB |     1.10 B | BLAS,RPC   |       4 |          tg64 |          3.84 ± 0.42 |

---

# Pipeline parallelism sweep — Mac coordinator, AP network
**Branch:** `parallel`  
**Model:** TinyLlama 1.1B Q5_K_M (`-ngl 23 -sm layer`)  
**Prompt sizes:** 32, 128 — **Gen sizes:** 32, 64 — **Reps:** 1  
**Date:** 2026-06-02  
**CSV:** `results/sweep_20260602_161709.csv`

| n_devices | pp32 (t/s) | pp128 (t/s) | tg32 (t/s) | tg64 (t/s) |
|----------:|-----------:|------------:|-----------:|-----------:|
|         1 |       7.56 |        7.47 |       2.59 |       2.52 |
|         2 |       5.80 |        7.09 |       1.92 |       1.79 |
|         4 |       4.92 |        6.20 |       1.33 |       1.25 |
|         8 |       3.63 |        5.36 |       1.05 |       1.08 |
|        16 |       2.22 |        3.95 |      0.157 |      0.155 |

**Observations:**
- Pipeline parallelism hurts throughput at all device counts — network RPC overhead per layer dominates
- Generation (tg) degrades monotonically; 8→16 devices causes a 7× regression (~400 ms/hop × 22 stages)
- Prefill (pp) degrades more gracefully since all tokens in a batch traverse the pipeline together
- Estimated RPi 4B compute: ~5.7 GFLOPS (from 1-device tg: 2.59 t/s × 2.2 GFLOP/token)

**Setup notes:**
- rpi12 excluded: running 32-bit armhf userspace — `size_t` mismatch breaks RPC protocol with 64-bit Mac client
- `GGML_SCHED_MAX_BACKENDS` raised from 16 → 33 in `ggml/src/ggml-backend.cpp` to support N_rpi + 1 CPU backend

---

# Tensor parallelism (`-sm row`) — 4 RPi backends, Mac coordinator

**First working `-sm row` run** after the block-aligned tensor-split fix
(`rpc_get_col_split` + `rpc_get_row_split` in `ggml-rpc.cpp`). Previously crashed
at load (`blk.6.ffn_down`, "tensor write out of bounds") whenever a quantized
tensor's split landed mid-block — only N=2 had ever worked (22 blocks ÷ 2 = aligned).

**Date:** 2026-06-15 — **Backends:** 128.174.61.{160,133,163,134}:50052 — `-ngl 23 -sm row -ts 1,1,1,1`
`llama-cli` single prompt, 33-token prompt + 9 generated.

> ⚠️ Output was still garbage on THIS run (row-split correctness fix not yet in the
> built binary; same `iwarsutter…` as before). Timings are valid regardless (the
> identical ops execute). Correct-output re-run pending a rebuild with both splits fixed.

| phase        | tokens | ms/token | tok/s |
|--------------|-------:|---------:|------:|
| load         |   —    |    —     | 231.9 s total |
| prefill (pp) |  33    |   6252   | 0.16  |
| decode (tg)  |   9    |  54285   | 0.02  |

**Observation — prefill ≫ decode by ~8× (0.16 vs 0.02 t/s), far wider than the
~2× pp/tg gap with pipeline.** `-sm row` does a per-layer all-reduce across all 4
devices on every forward (≈ 2 × 22 = 44 collectives/forward). Prefill **amortises**
those 44 all-reduces over the whole 33-token prompt batch; decode pays the full
all-reduce **latency for every single token** → 54 s/token. Versus pipeline decode
(~3.2 t/s), `-sm row` decode (0.02 t/s) is **~160× slower** — the clearest hardware
demonstration that tensor-parallel *collectives*, not compute, are what kill
single-stream decode over Wi-Fi (exactly what the simulator predicts).

---

## Tensor parallelism (`-sm row`) — N-device sweep (N = 1, 2, 4)

**Date:** 2026-06-16 — **Coordinator:** Mac → rpi{1,2,4}:50052 over the AP —
`llama-bench -ngl 23 -sm row -p 128 -n 8 -r 1`, via `rpi-automation/bench_sweep.sh`.
**First clean run with correct output** ("…the capital city of Greece is Athens")
after the non-main KV-cache zero-init fix — split caches were uninitialized, so
attention on non-main devices read garbage → NaN scores → only the main device's
heads contributed (N=2 limped, N=4 was garbage).

> `-sm row` requires N to divide `num_kv_heads` (TinyLlama = 4), so only N ∈ {1,2,4}
> are valid. N = 3 / 8 give fractional or zero KV-heads per device and crash
> (`ne` underflows → `data+size` overflow in `deserialize_tensor`).

| N (devices) | pp128 (t/s) | tg8 (t/s) | pp ÷ N=1 | tg ÷ N=1 |
|------------:|------------:|----------:|---------:|---------:|
| 1           | 2.71        | 0.22      | 1.00×    | 1.00×    |
| 2           | 1.11        | 0.03      | 0.41×    | 0.14×    |
| 4           | 0.42        | 0.02      | 0.16×    | 0.09×    |

**Anti-scaling — throughput DROPS monotonically as devices are added.** Prefill
2.71 → 1.11 → 0.42 t/s; decode collapses 0.22 → 0.03 t/s from N=1→2. Each forward
does ≈ 2 × 22 = 44 per-layer all-reduces across all N devices over Wi-Fi, and the
collective cost grows with N with **no compute saving to offset it** (the 1.1 B
model already fits one Pi). This is the hardware confirmation that **TP-via-RPC
over Wi-Fi is communication-bound and net-negative** for this workload — the
motivation for partitioning (pipeline + selective TP only where a layer doesn't
fit) that the simulator explores.

> Note on the baseline: N=1 here is a *single device but still Mac-coordinated over
> RPC*, so every layer already pays a Mac↔AP↔rpi round-trip. rpi1 **standalone**
> (no RPC, see `rpi1_local` above) does ~3.3 t/s decode — i.e. the Mac-coordination
> alone costs ~15× on decode before any tensor split. Coordinator placement matters
> as much as N.
