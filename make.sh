#!/bin/bash
# -----------------------------------------------------------------------------
# make.sh - Build and deploy llama.cpp across RPis
#
# Usage:
#   ./make.sh [--ips <file>] [-n <num>] [--split-ratio <ratio>]
#             [restart] [compile]
#
# Arguments:
#   --ips <file>          IPs file (default: rpi-automation/dprgnet_ips.txt)
#   -n <num>              Number of RPi servers to use (default: all).
#                         Only the first N entries from the file are used.
#   --split-ratio <ratio> Forwarded verbatim to llama-cli's -ts flag (comma-
#                         separated proportions, e.g. "3,1" = 75/25). Default
#                         is a uniform split across the active N servers
#                         ("1,1" for n=2, "1,1,1" for n=3, etc.).
#   restart               SSH into each RPi, pull latest, rebuild, restart
#                         rpc-server.
#   compile               Build llama.cpp locally (Mac).
# -----------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IPS_FILE="${SCRIPT_DIR}/rpi-automation/peer_ips.txt"   # peer: rpi24/rpi25 by default
model="models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
build_dir="build"
N_SERVERS=""
SPLIT_RATIO=""   # empty = uniform across the active server count

restart=""
compile=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ips)          IPS_FILE="$2";    shift 2 ;;
        -n)             N_SERVERS="$2";   shift 2 ;;
        --split-ratio)  SPLIT_RATIO="$2"; shift 2 ;;
        restart)        restart="yes";    shift   ;;
        compile)        compile="yes";    shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Build a default uniform split ratio for n devices (e.g. n=2 -> "1,1").
# llama-cli's -ts takes *proportions*, not percentages, so n equal 1's is the
# cleanest exact representation of a uniform split.
uniform_split_ratio() {
    local n=$1
    local out="1"
    for (( i=2; i<=n; i++ )); do out+=",1"; done
    echo "$out"
}

# Validate -n
if [[ -n "$N_SERVERS" ]]; then
    if ! [[ "$N_SERVERS" =~ ^[0-9]+$ ]] || [[ "$N_SERVERS" -lt 1 ]]; then
        echo "ERROR: -n must be a positive integer, got '$N_SERVERS'" >&2
        exit 1
    fi
fi

if [[ ! -f "$IPS_FILE" ]]; then
    echo "ERROR: IP list not found: $IPS_FILE" >&2; exit 1
fi

rpis=()
while read -r name ip; do
    [[ -z "$name" || "${name:0:1}" == "#" || -z "$ip" ]] && continue
    rpis+=("$ip")
done < "$IPS_FILE"

# Optionally truncate to the first N servers, then settle on a single count.
# After this block, N_SERVERS == ${#rpis[@]} unconditionally.
if [[ -n "$N_SERVERS" ]]; then
    if [[ "$N_SERVERS" -gt "${#rpis[@]}" ]]; then
        echo "ERROR: requested -n $N_SERVERS but only ${#rpis[@]} RPis in $IPS_FILE" >&2
        exit 1
    fi
    rpis=("${rpis[@]:0:$N_SERVERS}")
fi
N_SERVERS=${#rpis[@]}
echo "Using $N_SERVERS RPi server(s): ${rpis[*]}"

if [[ "$restart" == "yes" ]]; then
    git push
    for rpi in "${rpis[@]}"; do
        ssh -t pi@"$rpi" "cd ~/llama.cpp; git pull && ./rpi-automation/make_server.sh $rpi attach" &
    done
fi

if [[ "$compile" == "yes" ]]; then
    (
        cd "$build_dir"
        cmake .. -DGGML_RPC=ON -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=OFF
        cmake --build . --config Release -j$(nproc)
        cd ..
    ) &
fi

wait  # Wait for all background jobs to finish

# Build the --rpc argument
rpc_arg=""
for ip in "${rpis[@]}"; do
    rpc_arg+="${ip}:50052,"
done
# Remove trailing comma
rpc_arg="${rpc_arg%,}"

# Pick the split ratio for this run. User-supplied --split-ratio wins,
# otherwise we fall back to a uniform 1/N split across the active servers.
# When the user supplies a ratio, its part count must match N_SERVERS —
# llama-cli's -ts expects exactly one weight per backend, and a mismatch
# silently misallocates layers (or errors out late in startup).
if [[ -n "$SPLIT_RATIO" ]]; then
    # Count parts: commas + 1 (works for any string without trailing comma)
    n_parts=$(awk -F',' '{print NF}' <<< "$SPLIT_RATIO")
    if [[ "$n_parts" -ne "$N_SERVERS" ]]; then
        echo "ERROR: --split-ratio '$SPLIT_RATIO' has $n_parts parts but $N_SERVERS server(s)." >&2
        echo "       Provide one weight per active server (use -n to match)." >&2
        exit 1
    fi
    ts_arg="$SPLIT_RATIO"
else
    ts_arg=$(uniform_split_ratio "$N_SERVERS")
fi
echo "Split ratio (-ts): $ts_arg"

# echo "GGML_SCHED_DEBUG=2 ./$build_dir/bin/llama-cli -v -m $model --rpc $rpc_arg -ngl 22 -sm row -ts $ts_arg"
# GGML_SCHED_DEBUG=2 ./$build_dir/bin/llama-cli -v -m $model --rpc $rpc_arg -ngl 22 -sm row -ts $ts_arg

# Debug channels (all lines start with "RPCDBG", so `grep RPCDBG` catches them):
#   RPC_DBG_CACHE=1  per-device K/V cache content (nonzero-byte count) after each
#                    attention segment — shows if non-main caches are empty
#   RPC_DBG_STATS=1  per-tensor f32 stats: FULL (all-reduced), PART/ACCUM (per device)
#   RPC_DBG_SHAPE=1  per-device ne/nb/op/view_offs of each tensor
#   RPC_DBG_FILTER   comma-separated OR-list of name patterns to scope STATS/SHAPE
#                    output (substring; trailing '$' anchors to end, so "-1$"
#                    matches block 1 but not block 10/11/.../19)
# Run once with a 2-IP file and once with a 4-IP file (edit $IPS_FILE or pass
# --ips) to diff N=2 vs N=4.
# -no-cnv forces completion mode: -p is the actual prompt to complete, not a
# system prompt that then waits for interactive user input (the default for chat
# models). Set DBG_ENV to a channel above (and pipe to `grep RPCDBG`) to debug.
DBG_ENV=""
prompt="The capital city of Greece is"
echo "$DBG_ENV ./$build_dir/bin/llama-cli -m $model --rpc $rpc_arg -ngl 23 -sm row -ts $ts_arg -no-cnv -p \"$prompt\" -n 24 --temp 0"
env $DBG_ENV ./$build_dir/bin/llama-cli -m $model --rpc $rpc_arg -ngl 23 -sm row -ts $ts_arg -no-cnv -p "$prompt" -n 24 --temp 0

# echo "./$build_dir/bin/llama-bench -m $model --rpc $rpc_arg -ngl 23 -sm row -p 32,128,256 -n 32,64 -r 3"
# ./$build_dir/bin/llama-bench \
#   -m $model \
#   --rpc $rpc_arg \
#   -ngl 23 \
#   -p 32,128,256 \
#   -n 32,64 \
#   -r 3