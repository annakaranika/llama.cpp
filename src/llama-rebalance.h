#pragma once

#include <cstdint>

struct llama_model;
struct llama_kv_cache;
struct llama_context;

// Elastic-rebalancing policy: how the gated rebalancer picks a shift's destination and moves the
// layer's weights. Selected at runtime (LLAMA_REBALANCE_POLICY); extend here to add strategies.
enum llama_rebalance_policy {
    LLAMA_REBALANCE_POLICY_ADJACENT = 0,  // lower-loaded pipeline NEIGHBOR, prefetch the weight transfer
    LLAMA_REBALANCE_POLICY_CACHE    = 1,  // device that already CACHES the layer (0x-WiFi local load), else least-loaded
};

// Rebalance policy hook: invoked once per decode with a MUTABLE model (rebalancing deliberately moves
// weights mid-generation, unlike the rest of inference which treats them as const) + the KV cache +
// this decode's token count. The built-in implementation is llama_rebalance_step; register an
// alternative with llama_set_rebalance_callback. It is a no-op unless LLAMA_REBALANCE_BUDGET_MB is set.
typedef void (*llama_rebalance_fn)(struct llama_model & model, struct llama_kv_cache & kv, int32_t n_tokens);

// The built-in gated policy (adjacent+prefetch or cache-aware, per LLAMA_REBALANCE_POLICY).
void llama_rebalance_step(struct llama_model & model, struct llama_kv_cache & kv, int32_t n_tokens);

// Override the policy invoked each decode (nullptr fn is ignored).
void llama_set_rebalance_callback(struct llama_context * ctx, llama_rebalance_fn fn);
