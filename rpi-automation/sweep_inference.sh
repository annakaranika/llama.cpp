#!/usr/bin/env bash
# =============================================================================
# sweep_inference.sh — Benchmark pipeline / tensor parallelism across RPi count
# =============================================================================
#
# For each split mode in --split-modes and each count in --counts, runs
# llama-bench with the configured prompt and generation sizes and appends the
# CSV output to a timestamped results file. Each row in the final CSV is
# prefixed with n_devices so the sweep is straightforward to plot.
#
# Usage:
#   ./sweep_inference.sh [--ips <file>]      [--model <path>]
#                        [--ngl <n>]         [--counts "1 2 4 8"]
#                        [--reps <n>]        [--split-modes "layer row"]
#                        [--split-ratio "1,1"] [--progress]
#
# Defaults (override with the matching flag):
#   --ips          rpi-automation/dprgnet_ips.txt
#   --model        models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf
#   --ngl          23
#   --counts       "1 2 4 8 16"
#   --reps         1
#   --split-modes  "row"
#   --split-ratio  "" (uniform: "1,1" for n=2; "1,1,1" for n=3; …)
#
# --split-modes accepts a space-separated list of "row" and/or "layer":
#   row    -sm row    → tensor parallelism (all-reduce per layer)
#   layer  -sm layer  → pipeline parallelism (one transfer per stage boundary)
#
# --split-ratio is forwarded verbatim to llama-bench's -ts flag, which takes
# *proportions* (not percentages): "1,1" = 50/50, "3,1" = 75/25, "50,50" =
# 50/50 (normalised). A single user-supplied ratio is applied as-is to every
# count in the sweep, so make sure its parts match the largest n you visit.
# Leave it empty to let the script generate a uniform split per device count.
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IPS_FILE="${SCRIPT_DIR}/dprgnet_ips.txt"
MODEL="${REPO_ROOT}/models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
BUILD_DIR="${REPO_ROOT}/build"
RPC_PORT=50052
SSH_USER="pi"

NGL=23
REPS=1
PROMPT_SIZES="32,128"
GEN_SIZES="32,64"
DEVICE_COUNTS=(1 2 4 8 16)
SPLIT_MODES=("row")
SPLIT_RATIO=""     # empty => uniform across n
PROGRESS=0         # 1 => forward --progress to llama-bench

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ips)          IPS_FILE="$2";                            shift 2 ;;
        --model)        MODEL="$2";                               shift 2 ;;
        --ngl)          NGL="$2";                                 shift 2 ;;
        --reps)         REPS="$2";                                shift 2 ;;
        --counts)       IFS=' ' read -ra DEVICE_COUNTS <<< "$2";  shift 2 ;;
        --split-modes)  IFS=' ' read -ra SPLIT_MODES   <<< "$2";  shift 2 ;;
        --split-ratio)  SPLIT_RATIO="$2";                         shift 2 ;;
        --progress)     PROGRESS=1;                               shift   ;;
        *)  echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for sm in "${SPLIT_MODES[@]}"; do
    case "$sm" in
        row|layer) ;;
        *) echo "Invalid --split-modes value: '$sm' (expected 'row' or 'layer')" >&2
           exit 1 ;;
    esac
done

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# All non-loopback IPv4 addresses of this host, best-effort across Linux and
# macOS. Used to detect when one of the IPs in the IPs file is this very
# machine — that one we want to skip rather than SSH to ourselves.
get_local_ips() {
    {
        hostname -I 2>/dev/null | tr ' ' '\n'                               # Linux
        ip -4 -o addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1     # Linux (ip cmd)
        ifconfig 2>/dev/null | awk '/inet /{print $2}'                      # macOS / BSD
    } | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' \
      | grep -v '^127\.' \
      | sort -u
}

# Returns 0 if $1 matches any address in the LOCAL_IPS array.
is_local_ip() {
    local target="$1" ip
    for ip in "${LOCAL_IPS[@]}"; do
        [[ "$ip" == "$target" ]] && return 0
    done
    return 1
}

# Uniform split of proportions for n devices: 1 -> "1", 2 -> "1,1", etc.
# -ts normalises, so equal 1's give a clean exact split with no rounding.
uniform_split_ratio() {
    local n=$1 out="1"
    for (( i=1; i<n; i++ )); do out+=",1"; done
    echo "$out"
}

# row-split at n=1 is meaningless (nothing to shard against); skip it.
should_skip_combo() {
    local sm=$1 n=$2
    [[ "$sm" == "row" && "$n" -eq 1 ]]
}

# Start rpc-server on the first n RPis in parallel and wait for the SSH calls
# to return. The actual daemon launch happens via nohup so the SSH session can
# exit cleanly.
start_servers() {
    local n=$1 i ip
    log "  Starting rpc-server on $n RPi(s)..."
    for (( i=0; i<n; i++ )); do
        ip="${rpi_ips[$i]}"
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
            "${SSH_USER}@${ip}" \
            "pkill rpc-server 2>/dev/null || true; \
             nohup ~/llama.cpp/build-rpc/bin/rpc-server \
               -H ${ip} -p ${RPC_PORT} -m 2000 \
               > /tmp/rpc-server.log 2>&1 &" &
    done
    wait
    sleep 2
}

stop_servers() {
    local n=$1 i
    for (( i=0; i<n; i++ )); do
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
            "${SSH_USER}@${rpi_ips[$i]}" \
            "pkill rpc-server 2>/dev/null || true" &
    done
    wait
}

build_rpc_arg() {
    local n=$1 arg="" i
    for (( i=0; i<n; i++ )); do
        arg+="${rpi_ips[$i]}:${RPC_PORT},"
    done
    echo "${arg%,}"
}

# -----------------------------------------------------------------------------
# RPi discovery
# -----------------------------------------------------------------------------

# Populate LOCAL_IPS so the discovery loop can exclude this machine. The local
# backend already participates in llama-bench directly, so adding it to the
# --rpc list would route locally over TCP for no reason (and SSH-to-self for
# the rpc-server startup is just wasteful).
LOCAL_IPS=()
while IFS= read -r _ip; do LOCAL_IPS+=("$_ip"); done < <(get_local_ips)
(( ${#LOCAL_IPS[@]} > 0 )) && log "Local IPs detected: ${LOCAL_IPS[*]}"

rpi_names=()
rpi_ips=()
while read -r _n _i; do
    [[ -z "$_n" || "${_n:0:1}" == "#" || -z "$_i" ]] && continue
    rpi_names+=("$_n")
    rpi_ips+=("$_i")
done < "$IPS_FILE"
log "Loaded ${#rpi_ips[@]} RPis from $IPS_FILE — checking reachability..."

online_names=()
online_ips=()
for (( i=0; i<${#rpi_ips[@]}; i++ )); do
    if is_local_ip "${rpi_ips[$i]}"; then
        echo "  ⟲ ${rpi_names[$i]} (${rpi_ips[$i]}) — this machine, using local backend"
        continue
    fi
    if ping -c 1 -W 1 "${rpi_ips[$i]}" > /dev/null 2>&1; then
        online_names+=("${rpi_names[$i]}")
        online_ips+=("${rpi_ips[$i]}")
        echo "  ✓ ${rpi_names[$i]} (${rpi_ips[$i]})"
    else
        echo "  ✗ ${rpi_names[$i]} (${rpi_ips[$i]}) — unreachable, skipping"
    fi
done
rpi_names=("${online_names[@]}")
rpi_ips=("${online_ips[@]}")
n_available=${#rpi_ips[@]}
log "$n_available RPis online as RPC backends"
echo ""

# -----------------------------------------------------------------------------
# Sweep planning
# -----------------------------------------------------------------------------

# Drop counts we can't satisfy with the available RPis
valid_counts=()
for c in "${DEVICE_COUNTS[@]}"; do
    [[ $c -le $n_available ]] && valid_counts+=("$c")
done

# Configs to actually run (excluding row@n=1 which is a no-op) and a rough
# wall-clock estimate. Empirically, per-config time grows ~linearly with the
# number of participating devices because each layer's RPC round-trip count
# scales with n. We model it as a small fixed overhead per config (SSH,
# warmup, summary) plus a per-device factor times n; the numbers below come
# from observed runs of (pp32,pp128,tg32,tg64 × REPS=1) and will obviously
# be off by a factor if you change PROMPT_SIZES, GEN_SIZES, or REPS.
EST_BASE_S=60
EST_PER_DEVICE_S=110
n_configs=0
est_total_s=0
for sm in "${SPLIT_MODES[@]}"; do
    for c in "${valid_counts[@]}"; do
        should_skip_combo "$sm" "$c" && continue
        (( ++n_configs ))
        est_total_s=$(( est_total_s + EST_BASE_S + EST_PER_DEVICE_S * c ))
    done
done
est_total_s=$(( est_total_s * REPS ))
est_m=$(( est_total_s / 60 ))

mkdir -p "${REPO_ROOT}/results"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT="${REPO_ROOT}/results/sweep_${TIMESTAMP}.csv"

log "Split modes    : ${SPLIT_MODES[*]}"
log "Configs to run : $n_configs  (rough est. ~${est_m} min — scales with n)"
log "Output         : $OUTPUT"
echo ""

# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------

header_written=0
configs_done=0
total_elapsed_s=0
sweep_start_s=$SECONDS

# Pre-compute the per-config workload weight (= n_devices) so we can update
# the ETA after each config, weighted by remaining work.
remaining_weight=0
for sm_w in "${SPLIT_MODES[@]}"; do
    for c_w in "${valid_counts[@]}"; do
        should_skip_combo "$sm_w" "$c_w" && continue
        remaining_weight=$(( remaining_weight + c_w ))
    done
done
total_weight=$remaining_weight

for sm in "${SPLIT_MODES[@]}"; do
    for n in "${valid_counts[@]}"; do
        should_skip_combo "$sm" "$n" && continue

        # User-supplied ratio wins; otherwise uniform across this n.
        if [[ -n "$SPLIT_RATIO" ]]; then
            ts_arg="$SPLIT_RATIO"
        else
            ts_arg=$(uniform_split_ratio "$n")
        fi

        # llama-bench uses '/' to separate proportions within a single config
        # and ',' to mean "sweep over these values". We want the former, so
        # convert any user-supplied commas (or our own auto-generated ones).
        ts_arg_bench="${ts_arg//,//}"

        log "── sm=$sm  n_devices=$n  ts=$ts_arg_bench  ($((configs_done + 1))/$n_configs) ──"
        config_start_s=$SECONDS
        start_servers "$n"
        RPC_ARG=$(build_rpc_arg "$n")

        # Optional --progress passthrough (forwarded as an array so the
        # llama-bench command stays well-formed when the flag is off).
        progress_flag=()
        (( PROGRESS == 1 )) && progress_flag=(--progress)

        if ! bench_out=$("${BUILD_DIR}/bin/llama-bench" \
                -m "$MODEL"          \
                --rpc "$RPC_ARG"     \
                -ngl "$NGL"          \
                -sm "$sm"            \
                -ts "$ts_arg_bench"  \
                -p "$PROMPT_SIZES"   \
                -n "$GEN_SIZES"      \
                -r "$REPS"           \
                -o csv               \
                "${progress_flag[@]}"); then
            log "  llama-bench failed, skipping"
            stop_servers "$n"
            continue
        fi

        # Prepend an n_devices column to every row, header included.
        tagged=$(echo "$bench_out" | awk -v n="$n" \
            'NR==1 {print "n_devices," $0} NR>1 {print n "," $0}')

        if (( header_written == 0 )); then
            echo "$tagged" >> "$OUTPUT"
            header_written=1
        else
            echo "$tagged" | tail -n +2 >> "$OUTPUT"
        fi

        stop_servers "$n"

        # Per-config timing + running ETA (weighted by remaining n_devices)
        config_elapsed_s=$(( SECONDS - config_start_s ))
        total_elapsed_s=$(( SECONDS - sweep_start_s ))
        (( ++configs_done ))
        remaining_weight=$(( remaining_weight - n ))
        if (( total_weight > 0 )); then
            avg_s_per_unit=$(( total_elapsed_s / (total_weight - remaining_weight) ))
            eta_s=$(( avg_s_per_unit * remaining_weight ))
            log "  ↳ took $(( config_elapsed_s / 60 ))m$(( config_elapsed_s % 60 ))s; \
ETA $(( eta_s / 60 ))m$(( eta_s % 60 ))s remaining"
        fi

        sleep 1
    done
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

if (( header_written == 0 )); then
    log "No results collected — all configs failed or were skipped."
else
    log "Done. Results in $OUTPUT"
    echo ""
    column -t -s',' "$OUTPUT" 2>/dev/null || cat "$OUTPUT"
fi
