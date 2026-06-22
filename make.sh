#!/bin/bash
# -----------------------------------------------------------------------------
# make.sh - build + clean-restart peers + run, FROM THE COORDINATOR.
#
# Run it ON rpi1 for the IBSS setup: rpi1 (192.168.4.1) reaches the peers on the
# ad-hoc cell directly and runs llama-cli locally. (Also works on the Mac when the
# Mac is actually on the cluster network.) The build dir is auto-detected:
# build-rpc on the Pis, build on the Mac.
#
# Usage:
#   ./make.sh [--ips <file>] [-n <num>] [--split-ratio <ratio>]
#             [compile] [deploy] [--keep-servers] [-p <prompt>] [-N <ntok>]
#
#   --ips <file>          peer IP list (default rpi-automation/peer_ips.txt)
#   -n <num>              use only the first N peers from the file
#   --split-ratio <r>     -ts proportions (e.g. "3,1"); default uniform 1/N
#   compile               build the coordinator's llama-cli locally
#   deploy                scp this host's ggml-rpc.cpp to each peer + rebuild
#                         rpc-server (IBSS-friendly; peers often lack internet so
#                         git-pull won't work). Use after changing rpc code.
#   --keep-servers        skip the clean-slate peer restart (reuse warm servers)
#   -p <prompt> / -N <n>  prompt / number of tokens to generate
#
# DEFAULT (no flags): clean-slate restart the peer rpc-servers (pkill -9 -x +
# verify exactly 1/peer, via peer_servers.sh) then run llama-cli locally. The
# restart is REQUIRED between client runs -- reusing a server process across runs
# crashes the warmup all-reduce (stale peer/buffer state), and orphaned servers
# cause intermittent GGML_ASSERT(tensor->data >= buffer_start). --keep-servers
# opts out when you know the servers are already a clean, fresh slate.
# -----------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"   # all paths below are repo-root-relative; run from anywhere

IPS_FILE="${SCRIPT_DIR}/rpi-automation/peer_ips.txt"
model="models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
RPC_SRC="ggml/src/ggml-rpc/ggml-rpc.cpp"
# build dir: build-rpc on the Pis (coordinator=rpi1), build on the Mac
if [[ -d "${SCRIPT_DIR}/build-rpc" ]]; then build_dir="build-rpc"; else build_dir="build"; fi

N_SERVERS=""
SPLIT_RATIO=""
prompt="The capital city of Greece is"
ntok=24
compile=""; deploy=""; keep_servers=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ips)          IPS_FILE="$2";    shift 2 ;;
        -n)             N_SERVERS="$2";   shift 2 ;;
        --split-ratio)  SPLIT_RATIO="$2"; shift 2 ;;
        -p)             prompt="$2";      shift 2 ;;
        -N)             ntok="$2";        shift 2 ;;
        compile)        compile="yes";    shift   ;;
        deploy)         deploy="yes";     shift   ;;
        --keep-servers) keep_servers="yes"; shift ;;
        restart)        shift ;;  # accepted for compat: a clean-slate restart is the default
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

uniform_split_ratio() {  # n=2 -> "1,1" (proportions, exact uniform split)
    local n=$1 out="1"
    for (( i=2; i<=n; i++ )); do out+=",1"; done
    echo "$out"
}

if [[ -n "$N_SERVERS" ]]; then
    if ! [[ "$N_SERVERS" =~ ^[0-9]+$ ]] || [[ "$N_SERVERS" -lt 1 ]]; then
        echo "ERROR: -n must be a positive integer, got '$N_SERVERS'" >&2; exit 1
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

if [[ -n "$N_SERVERS" ]]; then
    if [[ "$N_SERVERS" -gt "${#rpis[@]}" ]]; then
        echo "ERROR: requested -n $N_SERVERS but only ${#rpis[@]} peers in $IPS_FILE" >&2; exit 1
    fi
    rpis=("${rpis[@]:0:$N_SERVERS}")
fi
N_SERVERS=${#rpis[@]}
echo "Using $N_SERVERS peer server(s): ${rpis[*]}  (build dir: $build_dir)"

# deploy: push this host's rpc source to each peer + rebuild rpc-server there.
# IBSS-friendly (scp over the cell), unlike the old git-pull (peers have no internet).
if [[ "$deploy" == "yes" ]]; then
    for ip in "${rpis[@]}"; do
        echo "deploy -> $ip"
        scp -o ConnectTimeout=15 "$SCRIPT_DIR/$RPC_SRC" "pi@$ip:~/llama.cpp/$RPC_SRC"
        ssh -n -o ConnectTimeout=15 "pi@$ip" "cd ~/llama.cpp && cmake --build build-rpc --target rpc-server -j\$(nproc)"
    done
fi

# compile: build the coordinator's own llama-cli (no reconfigure -> won't disturb
# the dir's existing cmake cache, e.g. Metal on the Mac).
if [[ "$compile" == "yes" ]]; then
    cmake --build "$build_dir" --target llama-cli -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
fi

# clean-slate restart the peer rpc-servers (the fix for stale-state/orphan crashes)
if [[ -z "$keep_servers" ]]; then
    PEER_IPS="${rpis[*]}" "${SCRIPT_DIR}/rpi-automation/peer_servers.sh" restart \
        || { echo "ERROR: peer servers are not a clean 1/peer slate; aborting run" >&2; exit 1; }
fi

# build the --rpc list
rpc_arg=""
for ip in "${rpis[@]}"; do rpc_arg+="${ip}:50052,"; done
rpc_arg="${rpc_arg%,}"

# pick the split ratio (user --split-ratio wins; must have one weight per peer)
if [[ -n "$SPLIT_RATIO" ]]; then
    n_parts=$(awk -F',' '{print NF}' <<< "$SPLIT_RATIO")
    if [[ "$n_parts" -ne "$N_SERVERS" ]]; then
        echo "ERROR: --split-ratio '$SPLIT_RATIO' has $n_parts parts but $N_SERVERS peer(s)." >&2; exit 1
    fi
    ts_arg="$SPLIT_RATIO"
else
    ts_arg=$(uniform_split_ratio "$N_SERVERS")
fi
echo "Split ratio (-ts): $ts_arg"

# Debug channels (env-gated, see ggml-rpc.cpp): RPC_GRAPH_CACHE=1 (diff cache),
# RPC_DBG_TIMING / RPC_DBG_DIFFCACHE / RPC_DBG_DIFF / RPC_DBG_AR / RPC_DBG_WCACHE.
DBG_ENV=""
echo "$DBG_ENV ./$build_dir/bin/llama-cli -m $model --rpc $rpc_arg -ngl 23 -sm row -ts $ts_arg -no-cnv -p \"$prompt\" -n $ntok --temp 0"
env $DBG_ENV ./"$build_dir"/bin/llama-cli -m "$model" --rpc "$rpc_arg" -ngl 23 -sm row -ts "$ts_arg" -no-cnv -p "$prompt" -n "$ntok" --temp 0
