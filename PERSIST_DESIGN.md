# Phase B: cross-process resident weights (opt-in RPC_PERSIST)

Goal: a fresh coordinator process rebinds to weight buffers still resident on the Pi
rpc-servers from a previous process — skipping the WiFi upload, the server-side alloc,
and the per-tensor LOAD_CACHED round-trips. Survives coordinator restarts. 2 GB peers
force reuse-in-place (a server-local copy would double weights → OOM).

## Facts (from lifecycle investigation)
- `rpc_server` is one-per-process, shared by ref across connection threads; buffers survive
  client disconnect. The **client FREE_BUFFER at model teardown** is what frees them.
- `remote_ptr` = server's `ggml_backend_buffer_t` pointer, validated by membership in the
  shared `buffers` set. Changes every process → cannot reuse an old remote_ptr; must rebind
  to the live one.
- Layout is deterministic: {tensor→(buffer-idx, offset, size)} identical across loads of the
  same model+config. ~1 weight buffer per device for -sm layer.
- Each rpc_server IS one device → registry key is just (model_key, size); no device id needed.
- `buffers` has no mutex → add one for the persist mutations.

## Identity: model_key
Computed in llama.cpp load_tensors (cheap): fnv1a over model file path + file size +
n_tensors + split_mode + tensor_split[] + n_gpu_layers. Captures "same model + same layout".
Passed to the RPC backend via proc-address setters (same pattern as create_peer_connection).
Different config → different key → different resident set (correct). Opt-in: only active when
RPC_PERSIST=1 AND llama.cpp calls persist_begin.

## Client hooks (llama.cpp, via get_proc_address)
- `ggml_backend_rpc_persist_begin(uint64_t model_key)` — before the ctx_map buffer-alloc loop.
  Sets g_persist_active=true, g_persist_key. The load window = weight buffers only (KV/act
  allocs happen after persist_end → normal path).
- `ggml_backend_rpc_persist_end()` — after load_all_data. Sends REGISTER for each freshly
  allocated (non-bound) weight buffer; clears g_persist_active.

## Client buffer context additions
- `uint64_t persist_key` (0 = not persistent), `bool skip_upload`.
- alloc_buffer: if g_persist_active → PERSIST_BIND(key,size). resident → wrap live remote_ptr,
  skip_upload=true, persist_key=key. not resident → normal ALLOC_BUFFER, persist_key=key,
  queue for REGISTER at persist_end.
- set_tensor: if skip_upload → return (data already resident).
- free_buffer: if persist_key → PERSIST_DETACH(key,remote_ptr) instead of FREE_BUFFER.

## Server (rpc_server) additions
- `std::unordered_map<uint64_t, std::vector<resident_buf>> persist_registry;`
  resident_buf = { ggml_backend_buffer_t buffer; uint64_t size; bool claimed; uint64_t last_use; }
- `std::mutex persist_mutex;` (guards persist_registry + buffers insert/erase on persist paths)
- persist_bind(key,size): find unclaimed resident buf with matching size → claim, return
  remote_ptr; else 0.
- persist_register(key,remote_ptr): add {buffer, size, claimed=true} (buffer already in `buffers`).
- persist_detach(key,remote_ptr): mark unclaimed (available for rebind); do NOT free.
- free_buffer: if buffer is in persist_registry → retain (return true, don't free). Else as now.

## Wire (new commands before RPC_CMD_COUNT)
- RPC_CMD_PERSIST_BIND: req{model_key,size} rsp{remote_ptr,remote_size}
- RPC_CMD_PERSIST_REGISTER: req{model_key,remote_ptr} rsp none
- RPC_CMD_PERSIST_DETACH: req{model_key,remote_ptr} rsp none

## Increments
- B1: bind+register+detach+retain (single model, single client). Validate: 2 localhost
  processes, 2nd skips upload+alloc, coherent.
- B2: RAM LRU eviction (multi-model, cap RPC_PERSIST_MAX_GB; last_use updated on bind).
- B3: server compute lock for concurrent cross-process clients.

Debug: RPC_DBG_PERSIST=1 logs BIND hit/miss, REGISTER, DETACH, RETAIN, EVICT.
