#!/bin/bash
# -----------------------------------------------------------------------------
# bench_sweep.sh - sweep `llama-bench -sm row` across device counts (N) for
# hardware calibration. For each N it benchmarks prefill (pp) and decode (tg)
# throughput using the first N RPi servers from the IPs file with a uniform
# tensor split, and appends a markdown table to an output file.
#
# Usage:
#   ./bench_sweep.sh [--ips <file>] [--ns "1,2,4"] [--model <path>]
#                    [-p <n_prompt>] [-n <n_gen>] [-r <reps>] [--ngl <n>]
#                    [--out <file>]
#
# Examples:
#   ./bench_sweep.sh                          # N=1,2,4, pp128/tg8, 1 rep
#   ./bench_sweep.sh --ns "1,2,4" -n 16 -r 2  # tighter decode estimate
#
# Notes:
#   * -sm row needs N to divide the model's num_kv_heads, else some device gets
#     a fractional/zero KV head and the run crashes. TinyLlama has 4 KV heads,
#     so valid N are {1,2,4} (NOT 3, NOT 8).
#   * -sm row decode is all-reduce-over-network bound (~0.02 tok/s), so keep -n
#     small; prefill (-p) is much cheaper.
#   * The rpc-servers must already be running on the Pis (one per IP). Use
#     `make.sh restart` to (re)build and launch them.
# -----------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

IPS_FILE="$SCRIPT_DIR/dprgnet_ips.txt"
MODEL="$REPO_DIR/models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
BENCH="$REPO_DIR/build/bin/llama-bench"
PORT=50052
NS="1,2,4"          # comma-separated device counts to sweep
N_PROMPT=128        # llama-bench -p
N_GEN=8             # llama-bench -n
REPS=1              # llama-bench -r
NGL=23
OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ips)   IPS_FILE="$2"; shift 2 ;;
        --ns)    NS="$2";       shift 2 ;;
        --model) MODEL="$2";    shift 2 ;;
        -p)      N_PROMPT="$2"; shift 2 ;;
        -n)      N_GEN="$2";    shift 2 ;;
        -r)      REPS="$2";     shift 2 ;;
        --ngl)   NGL="$2";      shift 2 ;;
        --out)   OUT="$2";      shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -f "$IPS_FILE" ]] || { echo "ERROR: IPs file not found: $IPS_FILE" >&2; exit 1; }
[[ -x "$BENCH"    ]] || { echo "ERROR: llama-bench not built at $BENCH (build it first)" >&2; exit 1; }
[[ -f "$MODEL"    ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[[ -n "$OUT"      ]] || OUT="$REPO_DIR/bench_smrow_$(date +%Y%m%d_%H%M%S).md"

# Collect server IPs (skip comments / blank lines), same format as make.sh.
ips=()
while read -r name ip; do
    [[ -z "$name" || "${name:0:1}" == "#" || -z "$ip" ]] && continue
    ips+=("$ip")
done < "$IPS_FILE"

: > "$OUT"
{
    echo "# llama-bench -sm row sweep ($(date))"
    echo "# model=$(basename "$MODEL")  ngl=$NGL  -p $N_PROMPT  -n $N_GEN  -r $REPS"
    echo "# servers: ${ips[*]}"
} | tee -a "$OUT"

IFS=',' read -ra n_list <<< "$NS"
for N in "${n_list[@]}"; do
    if (( N > ${#ips[@]} )); then
        echo "## N=$N -- SKIP (only ${#ips[@]} servers in $IPS_FILE)" | tee -a "$OUT"
        continue
    fi
    # First N servers -> comma-separated --rpc list and uniform slash-separated -ts.
    rpc=""; ts=""
    for (( i=0; i<N; i++ )); do
        rpc+="${ips[i]}:${PORT},"
        ts+="1/"
    done
    rpc="${rpc%,}"; ts="${ts%/}"

    echo "## N=$N  (--rpc $rpc  -ts $ts)" | tee -a "$OUT"
    # `|| true`: a bad N (e.g. doesn't divide num_kv_heads) shouldn't abort the sweep.
    "$BENCH" -m "$MODEL" --rpc "$rpc" -ngl "$NGL" -sm row -ts "$ts" \
             -p "$N_PROMPT" -n "$N_GEN" -r "$REPS" -o md 2>&1 | tee -a "$OUT" || true
    echo | tee -a "$OUT"
done

echo "# done -> $OUT" | tee -a "$OUT"
