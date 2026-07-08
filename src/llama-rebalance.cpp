#include "llama-rebalance.h"

#include "llama-model.h"
#include "llama-kv-cache.h"
#include "llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

// Log the current per-device layer distribution (how many of the model's layers each device holds).
// Changes only at a shift, so logging it right after each shift captures the full timeline.
static void log_layer_dist(llama_model & model, uint32_t n_layer) {
    std::map<ggml_backend_dev_t, int> d;
    for (int k = 0; k < (int) n_layer; k++) { d[model.dev_layer(k)]++; }
    fprintf(stderr, "[rebalance] dist:");
    for (auto * dev : model.devices) { fprintf(stderr, " %s=%d", ggml_backend_dev_name(dev), d[dev]); }
    fprintf(stderr, "\n");
}

// (capacity-aware) measure one device's LIVE capacity via the RPC live-stats probe: memory available
// right now + how many cores other work is consuming on the host. Falls back to the static
// ggml_backend_dev_memory snapshot (idle-host assumption) for non-RPC devices. Shared building block:
// the elastic placement at load and every rebalance policy read capacity through this one probe.
typedef bool (*rpc_live_stats_t)(ggml_backend_dev_t, uint64_t *, uint64_t *, uint32_t *, float *);

llama_dev_capacity llama_dev_capacity_measure(ggml_backend_dev_t dev) {
    static rpc_live_stats_t live_fn = [] () -> rpc_live_stats_t {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
        return reg ? (rpc_live_stats_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_dev_live_stats") : nullptr;
    }();
    llama_dev_capacity cap = {};
    uint64_t free_b = 0, total_b = 0;
    uint32_t ncpu   = 0;
    float    ext    = 0.0f;
    if (live_fn && live_fn(dev, &free_b, &total_b, &ncpu, &ext)) {
        cap.mem_free_mb  = (double) free_b  / (1024.0*1024.0);
        cap.mem_total_mb = (double) total_b / (1024.0*1024.0);
        cap.n_cpu        = (int) ncpu;
        cap.ext_load     = ext;
        cap.live         = true;
        return cap;
    }
    size_t free_s = 0, total_s = 0;
    ggml_backend_dev_memory(dev, &free_s, &total_s);
    cap.mem_free_mb  = (double) free_s  / (1024.0*1024.0);
    cap.mem_total_mb = (double) total_s / (1024.0*1024.0);
    cap.n_cpu        = 1;
    cap.ext_load     = 0.0;
    cap.live         = false;
    return cap;
}

double llama_dev_compute_frac(const llama_dev_capacity & cap) {
    if (cap.n_cpu <= 0) {
        return 1.0;
    }
    const double f = ((double) cap.n_cpu - cap.ext_load) / (double) cap.n_cpu;
    return std::max(0.05, std::min(1.0, f));
}

// (balanced policy) one planned layer move of a batch: src/dst are indices into model.devices;
// staged means its weights were successfully pre-staged in the background (commit is then ~instant).
struct llama_rebalance_move {
    int  il;
    int  src_i;
    int  dst_i;
    bool staged;
};

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
        if (p && strcmp(p, "cache")    == 0) { return LLAMA_REBALANCE_POLICY_CACHE;    }
        if (p && strcmp(p, "balanced") == 0) { return LLAMA_REBALANCE_POLICY_BALANCED; }
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
                    log_layer_dist(model, hparams.n_layer);
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
                    log_layer_dist(model, hparams.n_layer);
                    rb_last = rb_tok;
                }
            }
        } else if (policy == LLAMA_REBALANCE_POLICY_BALANCED) {
            // ===== CAPACITY-AWARE BALANCED-BATCH POLICY =====
            // Weight each device's layer share by its LIVE capacity (available memory x free
            // compute), re-measured periodically, and correct deviations as ONE batch: when KV
            // pressure is predicted, pre-stage every move needed to put each device at its
            // capacity-weighted share, and commit them together at the budget crossing. A recruit
            // therefore hands the new device its full balanced share in one step (max KV headroom
            // everywhere, far fewer rebalances) instead of draining onto it a layer at a time, and
            // the whole batch transfer overlaps decode so no move stalls inference.
            static const int    measure_every = getenv("LLAMA_REBALANCE_MEASURE")   ? atoi(getenv("LLAMA_REBALANCE_MEASURE"))   : 32;
            static const double margin_mb     = getenv("LLAMA_REBALANCE_MARGIN_MB") ? atof(getenv("LLAMA_REBALANCE_MARGIN_MB")) : 100.0;
            static const int    headroom_tok  = getenv("LLAMA_ELASTIC_KV_HEADROOM") ? atoi(getenv("LLAMA_ELASTIC_KV_HEADROOM")) : 512;
            const int nd = (int) model.devices.size();
            const int nl = (int) hparams.n_layer;

            // PERIODIC MEASUREMENT: live per-device capacity, re-read every measure_every tokens
            // (a few bytes per device on the existing sockets) so decisions track the actual
            // cluster state -- another process eating a device's memory or cores shows up here.
            static std::vector<llama_dev_capacity> caps;
            static int cap_tok = -1000000;
            if ((int) caps.size() != nd || rb_tok - cap_tok >= measure_every) {
                caps.clear();
                for (auto * d : model.devices) { caps.push_back(llama_dev_capacity_measure(d)); }
                cap_tok = rb_tok;
            }

            // CAPACITY-WEIGHTED TARGETS: apportion the layers by live capacity. A device's memory
            // capacity for our layers = free now + what our layers already occupy there (a move
            // returns the source's share, so the total is move-invariant), scaled by the fraction
            // of its compute not consumed by other work; clamped so no target overruns the
            // device's memory at a KV horizon of current cells + a headroom reserve.
            static const double w_layer_mb = [&model, nl] {
                size_t b = 0;
                for (auto & nt : model.tensors_by_name) {
                    if (nt.first.rfind("blk.", 0) == 0) { b += ggml_nbytes(nt.second); }
                }
                return b / (1024.0*1024.0) / std::max(1, nl);
            }();
            const double per_tok_mb  = (double) (hparams.n_embd_k_gqa(0) + hparams.n_embd_v_gqa(0)) * elt / (1024.0*1024.0);
            const double kv_layer_mb = cells * per_tok_mb;
            const double kv_res_mb   = (cells + headroom_tok) * per_tok_mb;
            const uint32_t gb        = kv.grow_block > 0 ? kv.grow_block : 0;
            // the horizon the membership decision (and the staging trigger below) look ahead to:
            // pf_lead grow-blocks, i.e. exactly early enough that a decided batch can transfer
            // hidden behind decode before the pressure actually lands.
            const double proj_cells  = (double) cells + pf_lead * gb;

            // LAZY MEMBERSHIP: apportion the layers over the devices that already hold some ("use
            // the least of the cluster and grow"). An idle device is recruited -- ONE per cycle,
            // best live capacity first -- only when even balanced targets over the current members
            // would break a device's memory or the KV budget at the horizon; the recruit then
            // receives its full capacity-weighted share in the same batch as the rebalance.
            // The whole decision phase is skipped while a staged batch awaits its commit.
            static std::vector<llama_rebalance_move> batch;  // staged, awaiting the budget crossing
            std::vector<double> mem_cap(nd), ideal(nd, 0.0);
            std::vector<int>    target(nd, 0), cap_layers(nd);
            std::vector<char>   member(nd, 0);
            for (int i = 0; i < nd; i++) { member[i] = cnt[model.devices[i]] > 0 ? 1 : 0; }
            double wsum = 0.0;

            // apportion nl layers over the current members by live capacity (memory x free-core
            // fraction, largest remainder, memory-clamped at the horizon). Returns true when the
            // result FITS: no clipped layers left over and no member above the KV budget at the
            // horizon even at its target share.
            auto apportion = [&]() -> bool {
                wsum = 0.0;
                for (int i = 0; i < nd; i++) {
                    ideal[i]  = 0.0;
                    target[i] = 0;
                    const double held = cnt[model.devices[i]] * (w_layer_mb + kv_layer_mb);
                    mem_cap[i]    = std::max(0.0, caps[i].mem_free_mb - margin_mb + held);
                    cap_layers[i] = (int) (mem_cap[i] / (w_layer_mb + kv_res_mb));
                    if (member[i]) { wsum += mem_cap[i] * llama_dev_compute_frac(caps[i]); }
                }
                if (wsum <= 0.0) { return false; }
                int assigned = 0;
                std::vector<int> order;
                for (int i = 0; i < nd; i++) {
                    if (!member[i]) { continue; }
                    ideal[i]  = nl * mem_cap[i] * llama_dev_compute_frac(caps[i]) / wsum;
                    target[i] = (int) ideal[i];
                    assigned += target[i];
                    order.push_back(i);
                }
                // largest-remainder: hand the leftover layers to the largest fractional shares
                std::sort(order.begin(), order.end(), [&](int a, int b) { return ideal[a] - target[a] > ideal[b] - target[b]; });
                for (size_t k = 0; assigned < nl && !order.empty(); k = (k + 1) % order.size()) { target[order[k]]++; assigned++; }
                // memory clamp + redistribute the clipped layers to members with spare headroom
                int excess = 0;
                for (int i = 0; i < nd; i++) {
                    if (member[i] && target[i] > cap_layers[i]) { excess += target[i] - cap_layers[i]; target[i] = cap_layers[i]; }
                }
                while (excess > 0) {
                    int best = -1;
                    for (int i = 0; i < nd; i++) {
                        if (member[i] && target[i] < cap_layers[i] && (best < 0 || ideal[i] - target[i] > ideal[best] - target[best])) { best = i; }
                    }
                    if (best < 0) { return false; }  // memory-infeasible: the members can't hold everything at the horizon
                    target[best]++; excess--;
                }
                int tmax = 0;
                for (int i = 0; i < nd; i++) { tmax = std::max(tmax, target[i]); }
                return tmax * proj_cells * per_tok_mb <= budget_mb;  // budget-feasible at the horizon?
            };

            std::vector<llama_rebalance_move> plan;
            int plan_runs_cur = 0, plan_runs_want = 0;  // pipeline contiguity before/after the plan
            if (batch.empty()) {
                if (!apportion()) {
                    // the active set cannot fit at the horizon -> recruit the best idle device
                    // (one per cycle; if it is still not enough the next cycle recruits another).
                    int best = -1;
                    for (int i = 0; i < nd; i++) {
                        if (!member[i] && (best < 0 || mem_cap[i] * llama_dev_compute_frac(caps[i]) > mem_cap[best] * llama_dev_compute_frac(caps[best]))) { best = i; }
                    }
                    if (best >= 0) {
                        member[best] = 1;
                        fprintf(stderr, "[rebalance] RECRUIT %s (active set cannot fit at horizon cells=%.0f)\n",
                                ggml_backend_dev_name(model.devices[best]), proj_cells);
                        apportion();
                    }
                }

                // PLAN THE BATCH, CONTIGUITY-PRESERVING: every member with a nonzero target gets
                // ONE consecutive block of layers (sizes = targets), so hops/token stays at
                // members-1 forever instead of growing with each recruit (scattered edge-picking
                // gave a recruit 3 separate runs = 6 pipeline boundaries on the 9-Pi cluster).
                // Block order keeps the members' current relative order (sorted by the median of
                // the layers each holds); a member holding nothing (a fresh recruit) is tried at
                // EVERY insertion position and the cheapest wins -- a middle slot typically costs
                // only a couple of extra (hidden) moves but keeps the pipeline contiguous. The
                // plan is then simply every layer whose desired block owner differs from its
                // current one; cascaded boundary shifts fall out of that naturally.
                if (wsum > 0.0) {
                    std::vector<int> ldev(nl, 0);
                    for (int k = 0; k < nl; k++) {
                        for (int i = 0; i < nd; i++) {
                            if (model.devices[i] == model.dev_layer(k)) { ldev[k] = i; break; }
                        }
                    }
                    // members that hold layers, in current pipeline order (median layer index);
                    // members with a target but no layers (recruits) are placed by search below.
                    std::vector<int> held, fresh;
                    for (int i = 0; i < nd; i++) {
                        if (target[i] <= 0) { continue; }
                        if (cnt[model.devices[i]] > 0) { held.push_back(i); } else { fresh.push_back(i); }
                    }
                    std::vector<double> med(nd, 0.0);
                    for (int i : held) {
                        std::vector<int> ls;
                        for (int k = 0; k < nl; k++) { if (ldev[k] == i) { ls.push_back(k); } }
                        med[i] = ls[ls.size()/2];
                    }
                    std::sort(held.begin(), held.end(), [&](int a, int b) { return med[a] < med[b]; });
                    // desired owner per layer for a given block order; returns the move count
                    auto layout = [&](const std::vector<int> & order, std::vector<int> & want) -> int {
                        want.assign(nl, -1);
                        int at = 0, moves = 0;
                        for (int i : order) {
                            for (int t = 0; t < target[i] && at < nl; t++, at++) { want[at] = i; }
                        }
                        for (int k = 0; k < nl; k++) { moves += want[k] >= 0 && want[k] != ldev[k] ? 1 : 0; }
                        return moves;
                    };
                    // insert each fresh member (one per cycle by design) at its cheapest position
                    std::vector<int> order = held, want, cand;
                    for (int f : fresh) {
                        int best_pos = 0, best_moves = -1;
                        for (size_t p = 0; p <= order.size(); p++) {
                            std::vector<int> o = order;
                            o.insert(o.begin() + p, f);
                            const int m = layout(o, cand);
                            if (best_moves < 0 || m < best_moves) { best_moves = m; best_pos = (int) p; }
                        }
                        order.insert(order.begin() + best_pos, f);
                    }
                    layout(order, want);
                    for (int k = 0; k < nl; k++) {
                        if (want[k] >= 0 && want[k] != ldev[k]) { plan.push_back({ k, ldev[k], want[k], false }); }
                    }
                    // contiguity metric: pipeline runs (device changes along the layer axis + 1)
                    plan_runs_cur = 1; plan_runs_want = 1;
                    for (int k = 1; k < nl; k++) {
                        plan_runs_cur  += ldev[k] != ldev[k-1] ? 1 : 0;
                        plan_runs_want += want[k] >= 0 && want[k-1] >= 0 && want[k] != want[k-1] ? 1 : 0;
                    }
                }
            }

            // PREFETCH THE WHOLE BATCH when the projected load at the horizon crosses the budget:
            // every planned layer's weights start moving in the background while decode continues.
            if (pf_lead > 0.0 && batch.empty() && !plan.empty() && cells > 0) {
                const double projected = load[dmax] * proj_cells / (double) cells;
                if (projected > budget_mb) {
                    int staged_n = 0;
                    for (auto & m : plan) {
                        m.staged  = model.prefetch_layer_weights(m.il, model.devices[m.dst_i]);
                        staged_n += m.staged ? 1 : 0;
                    }
                    batch = plan;
                    fprintf(stderr, "[rebalance] PREFETCH-BATCH %zu layers (%d staged) | cells=%u load=%.1fMB proj=%.1fMB budget=%.1fMB | runs %d->%d | targets:",
                            batch.size(), staged_n, cells, load[dmax], projected, budget_mb, plan_runs_cur, plan_runs_want);
                    for (int i = 0; i < nd; i++) { fprintf(stderr, " %d", target[i]); }
                    fprintf(stderr, "\n");
                }
            }

            // COMMIT THE BATCH at the actual budget crossing: staged layers just barrier + repoint
            // (~instant), unstaged ones fall back to a synchronous move; each layer's (tiny) KV
            // moves with it. One cooldown covers the whole batch.
            if (rb_tok - rb_last >= cooldown && load[dmax] > budget_mb) {
                std::vector<llama_rebalance_move> & moves = !batch.empty() ? batch : plan;
                if (!moves.empty()) {
                    const int64_t tb0 = ggml_time_us();
                    int done = 0;
                    for (auto & m : moves) {
                        ggml_backend_dev_t dst_dev = model.devices[m.dst_i];
                        ggml_backend_dev_t src_dev = model.dev_layer(m.il);
                        const int64_t ts0 = ggml_time_us();
                        const bool committed = m.staged && model.commit_layer_weights(m.il);
                        if (!committed && !model.move_layer_weights(m.il, dst_dev)) {
                            continue;
                        }
                        const int64_t ts1 = ggml_time_us();
                        llama_kv_cache_move_layer(kv, model, m.il, dst_dev);
                        const int64_t ts2 = ggml_time_us();
                        fprintf(stderr, "[rebalance] SHIFT layer %d %s->%s: total=%.0fms (weights=%.0fms kv=%.0fms) | prefetched=%d (batch)\n",
                                m.il, ggml_backend_dev_name(src_dev), ggml_backend_dev_name(dst_dev),
                                (ts2-ts0)/1e3, (ts1-ts0)/1e3, (ts2-ts1)/1e3, (int) committed);
                        done++;
                    }
                    fprintf(stderr, "[rebalance] BATCH commit: %d/%zu moves in %.0fms | cells=%u load=%.1fMB budget=%.1fMB\n",
                            done, moves.size(), (ggml_time_us()-tb0)/1e3, cells, load[dmax], budget_mb);
                    log_layer_dist(model, hparams.n_layer);
                    rb_last = rb_tok;
                    batch.clear();
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
