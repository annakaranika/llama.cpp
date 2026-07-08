#include "llama-rebalance.h"

#include "llama-model.h"
#include "llama-kv-cache.h"
#include "llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

// Gated elastic rebalance: when the most-loaded device's KV footprint exceeds a budget, shift one of
// its layers to another device. Cooldown (LLAMA_REBALANCE_COOLDOWN tokens) avoids thrashing -- every
// shift moves a layer's weights+KV over the link, so we rebalance sparingly. Runs once per decode; a
// no-op unless LLAMA_REBALANCE_BUDGET_MB is set. Placement is re-derived every decode from each
// tensor's ->buffer, so a move is just "relocate the layer's weight+KV tensors + flip dev_layer".
void llama_rebalance_step(llama_model & model, llama_kv_cache & kv, int32_t n_tokens) {
    const auto & hparams = model.hparams;

    static const double budget_mb = getenv("LLAMA_REBALANCE_BUDGET_MB") ? atof(getenv("LLAMA_REBALANCE_BUDGET_MB")) : 0.0;
    static const int    cooldown  = getenv("LLAMA_REBALANCE_COOLDOWN")  ? atoi(getenv("LLAMA_REBALANCE_COOLDOWN"))  : 32;
    // prefetch lead in grow-blocks: pre-stage the shift's weights this many blocks before the
    // predicted budget crossing so the transfer overlaps decode (0 disables -> synchronous shift).
    static const double pf_lead   = getenv("LLAMA_REBALANCE_PREFETCH") ? atof(getenv("LLAMA_REBALANCE_PREFETCH")) : 1.0;
    // policy (LLAMA_REBALANCE_POLICY): ADJACENT (default) shifts to the lower-loaded pipeline
    // neighbor and prefetches the transfer; CACHE shifts to whichever device already holds the
    // layer in its local weight cache (a 0x-WiFi load), falling back to the least-loaded device.
    static const llama_rebalance_policy policy = [] {
        const char * p = getenv("LLAMA_REBALANCE_POLICY");
        if (p && strcmp(p, "cache") == 0) { return LLAMA_REBALANCE_POLICY_CACHE; }
        return LLAMA_REBALANCE_POLICY_ADJACENT;
    }();
    static int rb_tok = 0, rb_last = -1000000;
    static int pf_il = -1;                          // layer currently pre-staged (-1 = none)
    static ggml_backend_dev_t pf_target = nullptr;  // its staged destination
    if (budget_mb > 0.0) {
        const uint32_t cells = kv.size;
        const size_t   elt   = ggml_type_size(kv.type_k) + ggml_type_size(kv.type_v);
        // per-device KV bytes (this decode's capacity) + layer count
        std::map<ggml_backend_dev_t, double> load;
        std::map<ggml_backend_dev_t, int>    cnt;
        for (int il = 0; il < (int) hparams.n_layer; il++) {
            const double b = (double) cells * (hparams.n_embd_k_gqa(il) + hparams.n_embd_v_gqa(il)) * elt;
            load[model.dev_layer(il)] += b / (1024.0*1024.0);
            cnt [model.dev_layer(il)] += 1;
        }
        // most-loaded device + its position in the pipeline order (model.devices is stage order)
        ggml_backend_dev_t dmax = nullptr;
        int imax = -1;
        for (int k = 0; k < (int) model.devices.size(); k++) {
            if (!dmax || load[model.devices[k]] > load[dmax]) { dmax = model.devices[k]; imax = k; }
        }
        if (policy == LLAMA_REBALANCE_POLICY_ADJACENT) {
            // ===== ADJACENT + PREFETCH POLICY (default) =====
            // shift to an ADJACENT pipeline neighbor (keeps stages contiguous -> no extra handoffs),
            // preferring the lower-loaded side; move the EDGE layer toward it (highest layer if going
            // to the next stage, lowest if going to the previous).
            ggml_backend_dev_t prev = imax-1 >= 0 ? model.devices[imax-1] : nullptr;
            ggml_backend_dev_t next = imax+1 < (int) model.devices.size() ? model.devices[imax+1] : nullptr;
            ggml_backend_dev_t target = nullptr; bool up = false;
            if (prev && (!next || load[prev] <= load[next])) { target = prev; up = false; }
            else if (next)                                   { target = next; up = true;  }
            // dmax's EDGE layer toward target (highest if going up, lowest if going down)
            int il = -1;
            if (dmax && target) {
                if (up) { for (int k = (int) hparams.n_layer - 1; k >= 0; k--) if (model.dev_layer(k) == dmax) { il = k; break; } }
                else    { for (int k = 0; k < (int) hparams.n_layer; k++)      if (model.dev_layer(k) == dmax) { il = k; break; } }
            }
            // converge-not-thrash: only shift when the source has >=2 more layers than the target.
            const bool converge = dmax && target && cnt[dmax] >= cnt[target] + 2;

            // PREFETCH: pre-stage the next shift's weights in the background so the ~20 s transfer
            // overlaps decode (WiFi is idle on the compute-bound -sm layer decode); the shift then
            // commits as just a barrier + repoint + tiny KV move. Fires both PREDICTIVELY (projected
            // one grow-block ahead crosses budget, while still under it) and DURING A BURST (already
            // over budget with more shifts to come) -- the cooldown gap before the matching commit is
            // the transfer's lead time, so each shift in a burst gets pipelined, not just the first.
            if (pf_lead > 0.0 && pf_il < 0 && converge && il >= 0) {
                const uint32_t gb = kv.grow_block > 0 ? kv.grow_block : 0;
                const double projected = cells > 0
                    ? load[dmax] * (double) (cells + (uint32_t) (pf_lead * gb)) / (double) cells
                    : load[dmax];
                if (projected > budget_mb && model.prefetch_layer_weights(il, target)) {
                    pf_il = il; pf_target = target;
                    fprintf(stderr, "[rebalance] PREFETCH layer %d %s->%s (background) | cells=%u load=%.1fMB proj=%.1fMB budget=%.1fMB\n",
                            il, ggml_backend_dev_name(dmax), ggml_backend_dev_name(target), cells, load[dmax], projected, budget_mb);
                }
            }

            // COMMIT a shift when the cooldown has elapsed: prefer committing a pre-staged layer (fast),
            // else fall back to a synchronous move of dmax's edge layer.
            if (rb_tok - rb_last >= cooldown) {
                int commit_il = -1; ggml_backend_dev_t commit_dst = nullptr; bool staged = false;
                if (pf_il >= 0 && load[model.dev_layer(pf_il)] > budget_mb) {
                    commit_il = pf_il; commit_dst = pf_target; staged = true;   // its source device crossed budget
                } else if (pf_il < 0 && converge && load[dmax] > budget_mb && il >= 0) {
                    commit_il = il; commit_dst = target;                        // no prefetch available -> sync shift
                }
                if (commit_il >= 0 && commit_dst) {
                    ggml_backend_dev_t src_dev = model.dev_layer(commit_il);
                    size_t wbytes = 0;
                    { char pfx[64]; snprintf(pfx, sizeof(pfx), "blk.%d.", commit_il);
                      for (auto & nt : model.tensors_by_name) if (nt.first.rfind(pfx, 0) == 0) wbytes += ggml_nbytes(nt.second); }
                    const size_t kvbytes = ggml_nbytes(kv.k_l[commit_il]) + ggml_nbytes(kv.v_l[commit_il]);
                    const int64_t ts0 = ggml_time_us();
                    bool committed = staged && model.commit_layer_weights(commit_il);
                    if (!committed) {
                        model.move_layer_weights(commit_il, commit_dst);  // sync fallback
                    }
                    const int64_t ts1 = ggml_time_us();
                    llama_kv_cache_move_layer(kv, model, commit_il, commit_dst);
                    const int64_t ts2 = ggml_time_us();
                    fprintf(stderr, "[rebalance] SHIFT layer %d %s->%s: weights=%.1fMB kv=%.2fMB | total=%.0fms (weights=%.0fms kv=%.0fms) | cells=%u prefetched=%d\n",
                            commit_il, ggml_backend_dev_name(src_dev), ggml_backend_dev_name(commit_dst),
                            wbytes/1e6, kvbytes/1e6, (ts2-ts0)/1e3, (ts1-ts0)/1e3, (ts2-ts1)/1e3, cells, (int) committed);
                    rb_last = rb_tok;
                    pf_il = -1; pf_target = nullptr;
                }
            }
        } else if (policy == LLAMA_REBALANCE_POLICY_CACHE) {
            // ===== CACHE-AWARE POLICY =====
            // Shed dmax's highest-indexed layer to the device that already holds the most of it in
            // its local weight cache (each cached tensor loads locally -> 0x WiFi), tie-broken and
            // cold-fallback by lowest load. Distinct from the adjacent policy: the target is chosen
            // by cache LOCALITY, not pipeline adjacency, and there is no prefetch -- a warm move is
            // already ~free, and a cold move lazy-warms its destination for next time.
            if (rb_tok - rb_last >= cooldown) {
                int il = -1;
                for (int k = (int) hparams.n_layer - 1; k >= 0; k--) if (model.dev_layer(k) == dmax) { il = k; break; }
                ggml_backend_dev_t target = nullptr; int best_hits = -1; double best_load = 1e18; int total_t = 0;
                if (il >= 0) {
                    char pfx[64]; snprintf(pfx, sizeof(pfx), "blk.%d.", il);
                    for (auto & nt : model.tensors_by_name) if (nt.first.rfind(pfx, 0) == 0) total_t++;
                    for (auto * d : model.devices) {
                        if (d == dmax) continue;
                        const int hits = model.layer_cache_hits_on(il, d);
                        if (hits > best_hits || (hits == best_hits && load[d] < best_load)) {
                            target = d; best_hits = hits; best_load = load[d];
                        }
                    }
                }
                if (dmax && target && il >= 0 && load[dmax] > budget_mb && cnt[dmax] >= cnt[target] + 2) {
                    size_t wbytes = 0;
                    { char pfx[64]; snprintf(pfx, sizeof(pfx), "blk.%d.", il);
                      for (auto & nt : model.tensors_by_name) if (nt.first.rfind(pfx, 0) == 0) wbytes += ggml_nbytes(nt.second); }
                    const size_t kvbytes = ggml_nbytes(kv.k_l[il]) + ggml_nbytes(kv.v_l[il]);
                    const int64_t ts0 = ggml_time_us();
                    const int hits = model.move_layer_weights_cached(il, target);
                    const int64_t ts1 = ggml_time_us();
                    llama_kv_cache_move_layer(kv, model, il, target);
                    const int64_t ts2 = ggml_time_us();
                    fprintf(stderr, "[rebalance] SHIFT layer %d %s->%s: weights=%.1fMB kv=%.2fMB | total=%.0fms (weights=%.0fms kv=%.0fms) | cells=%u cache=%d/%d\n",
                            il, ggml_backend_dev_name(dmax), ggml_backend_dev_name(target),
                            wbytes/1e6, kvbytes/1e6, (ts2-ts0)/1e3, (ts1-ts0)/1e3, (ts2-ts1)/1e3, cells, hits, total_t);
                    rb_last = rb_tok;
                }
            }
        }
    }
    rb_tok += n_tokens;
}

void llama_set_rebalance_callback(llama_context * ctx, llama_rebalance_fn fn) {
    if (ctx && fn) {
        ctx->rebalance_cb = fn;
    }
}
