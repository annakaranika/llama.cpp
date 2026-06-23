#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# ablate.sh — run an A/B ablation over the peer-branch RPC optimization profiles
# defined in rpc_profiles.conf, FROM THE COORDINATOR-side host (Mac), and print a
# comparison table (prefill t/s, decode t/s, output snippet) for the paper.
#
# Each profile's env vars are applied IDENTICALLY to the client AND every rpc-server
# (the gates must match on all nodes), the servers are clean-restarted between
# profiles (see dprgnet-cluster-ops gotchas: pgrep/pkill -x, listen-readiness), and
# perf is parsed from llama.cpp's own llama_perf_context_print lines.
#
# Usage:
#   ./ablate.sh                         # runs: baseline optimized   (the headline A/B)
#   ./ablate.sh optimized opt+fp16 opt+tree opt+tree+fp16
#   ./ablate.sh all                     # every profile in the conf, in file order
#   N=32 PROMPT="Long prompt for prefill" ./ablate.sh baseline optimized
#
# Env overrides: N (decode tokens, default 16), PROMPT, MODEL, RPC_PORT (50052),
#   PEER_IPS / PEER_IPS_FILE (peer list), COORD (ssh alias for the client host,
#   default rpi1-eth), RPC_BIN (server), CLI_BIN (client), CONF (profiles file),
#   REPEAT (runs per profile, default 1; reports the best decode t/s).
# -----------------------------------------------------------------------------
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${CONF:-$SCRIPT_DIR/rpc_profiles.conf}"
IPS_FILE="${PEER_IPS_FILE:-$SCRIPT_DIR/peer_ips.txt}"
RPC_PORT="${RPC_PORT:-50052}"
RPC_MEM_MB="${RPC_MEM_MB:-2000}"
RPC_BIN="${RPC_BIN:-build-rpc/bin/rpc-server}"
CLI_BIN="${CLI_BIN:-build-rpc/bin/llama-cli}"
COORD="${COORD:-rpi1-eth}"
MODEL="${MODEL:-models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf}"
N="${N:-16}"
PROMPT="${PROMPT:-The capital city of Greece is}"
REPEAT="${REPEAT:-1}"
SSH="ssh -n -o ConnectTimeout=12"

peers() {
    if [ -n "${PEER_IPS:-}" ]; then printf '%s\n' $PEER_IPS
    else grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$IPS_FILE" | awk '{print $2}'; fi
}
PEER_LIST="$(peers)"
RPC_CSV="$(echo $PEER_LIST | tr ' ' '\n' | sed "s/\$/:$RPC_PORT/" | paste -sd, -)"

# profile name -> env string ("" if none). Returns 1 if the profile is not in the conf.
profile_env() {
    awk -v want="$1" '
        /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
        { name=$1; $1=""; sub(/^[[:space:]]+/,""); if (name==want) { print; found=1; exit } }
        END { if (!found) exit 1 }' "$CONF"
}
profile_names() { awk '/^[[:space:]]*#/||/^[[:space:]]*$/{next}{print $1}' "$CONF"; }

restart_servers() { # $1 = env string applied to every server
    local env_str="$1"
    for ip in $PEER_LIST; do
        $SSH "pi@$ip" "pkill -9 -x rpc-server 2>/dev/null; while pgrep -x rpc-server>/dev/null; do sleep 0.3; done; \
            cd ~/llama.cpp && (setsid env $env_str $RPC_BIN -H $ip -p $RPC_PORT -m $RPC_MEM_MB </dev/null >/tmp/rpc.log 2>&1 &)" 2>/dev/null
    done
    sleep 2
    # listen-readiness: a client that connects before the listen socket is up gets "closed by peer"
    for ip in $PEER_LIST; do
        $SSH "pi@$ip" "for i in \$(seq 1 20); do (exec 3<>/dev/tcp/$ip/$RPC_PORT) 2>/dev/null && { exec 3>&-; exit 0; }; sleep 0.5; done; exit 1" 2>/dev/null \
            || { echo "  ! $ip not accepting on $RPC_PORT" >&2; return 1; }
    done
}

run_profile() { # $1 = name, $2 = env string ; echoes "prefill_tps|decode_tps|output"
    local name="$1" env_str="$2"
    local best_dec="" best_pre="" out=""
    for ((r=1; r<=REPEAT; r++)); do
        restart_servers "$env_str" || { echo "ERR|ERR|server-restart-failed"; return; }
        # stdout (generation) -> abl_out.txt ; stderr (perf+debug) -> abl_err.txt. Pull the two
        # perf lines and the clean generation back separated by a marker so debug noise can't leak.
        local cap; cap="$($SSH "$COORD" "cd ~/llama.cpp && env $env_str $CLI_BIN -m $MODEL --rpc $RPC_CSV \
            -ngl 23 -sm row -ts 1,1,1,1 -no-cnv -p '$PROMPT' -n $N --temp 0 2>/tmp/abl_err.txt >/tmp/abl_out.txt; \
            grep -E 'prompt eval time|:[[:space:]]*eval time' /tmp/abl_err.txt; echo '##OUT##'; tr '\n' ' ' </tmp/abl_out.txt" 2>/dev/null)"
        local perf="${cap%%##OUT##*}" gen="${cap#*##OUT##}"
        local pre dec
        pre="$(printf '%s\n' "$perf" | awk -F'[(,]' '/prompt eval time/{for(i=1;i<=NF;i++) if($i~/tokens per second/){gsub(/[^0-9.]/,"",$i); print $i}}' | head -1)"
        dec="$(printf '%s\n' "$perf" | awk -F'[(,]' '/eval time/ && !/prompt/{for(i=1;i<=NF;i++) if($i~/tokens per second/){gsub(/[^0-9.]/,"",$i); print $i}}' | head -1)"
        out="$(printf '%s' "$gen" | sed 's/^ *//; s/  */ /g' | cut -c1-66)"
        # keep the best (highest) decode t/s across repeats
        if [ -n "$dec" ] && { [ -z "$best_dec" ] || awk "BEGIN{exit !($dec>$best_dec)}"; }; then best_dec="$dec"; best_pre="$pre"; fi
    done
    echo "${best_pre:-?}|${best_dec:-?}|${out:-<no output / crash>}"
}

# ---- which profiles to run ----
if [ "$#" -eq 0 ]; then set -- baseline optimized; fi
if [ "$1" = "all" ]; then set -- $(profile_names); fi

echo "ablation: profiles=[$*]  N=$N  peers=[$PEER_LIST]  coord=$COORD  repeat=$REPEAT"
echo "prompt: \"$PROMPT\""
printf '\n| %-14s | %-18s | %-10s | %-10s | %s\n' "profile" "env" "prefill t/s" "decode t/s" "output"
printf '|%s|%s|%s|%s|%s\n' "----------------" "--------------------" "------------" "------------" "----------------------"
for name in "$@"; do
    env_str="$(profile_env "$name")" || { echo "| $name : NOT IN $CONF"; continue; }
    res="$(run_profile "$name" "$env_str")"
    pre="${res%%|*}"; rest="${res#*|}"; dec="${rest%%|*}"; out="${rest#*|}"
    printf '| %-14s | %-18s | %-10s | %-10s | %s\n' "$name" "${env_str:-<all opt>}" "$pre" "$dec" "$out"
done
# leave a clean slate (fully-optimized servers) afterwards
restart_servers "" >/dev/null 2>&1 || true
