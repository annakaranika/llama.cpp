#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# peer_servers.sh — manage the peer-branch rpc-servers on the dprgnet IBSS
# cluster FROM THE COORDINATOR (Mac). Peers come from peer_ips.txt and are
# reached over SSH (the ~/.ssh/config ProxyJump through rpi1's eth port handles
# the 192.168.4.* IBSS addresses).
#
# Bakes in the hard-won gotchas (see memory dprgnet-cluster-ops):
#   * EXACT-NAME process matching only: `pgrep -x` / `pkill -x`, NEVER `-f`.
#     "rpc-server" appears in the ssh command line itself, so `pgrep -f` /
#     `pkill -f` self-match the remote shell -> phantom counts, and `pkill -9 -f`
#     kills its own ssh shell (exit 255 + incomplete cleanup).
#   * `setsid` rpc-servers are ORPHANS that outlive the launching ssh. If a launch
#     ssh is killed (stuck/aborted) the remote daemon keeps running; stale orphans
#     pile up and a new client hits a mix of fresh+stale servers with mismatched
#     buffer pointers -> intermittent GGML_ASSERT(tensor->data >= buffer_start).
#     The fix is to ALWAYS kill by name and verify the count before launching.
#   * `ssh -n` (don't eat the loop's stdin) + ConnectTimeout (macOS has no timeout).
#
# Usage:
#   ./peer_servers.sh status     # rpc-server count per peer (exact-name)
#   ./peer_servers.sh stop       # kill all, verify 0/peer
#   ./peer_servers.sh start      # launch one/peer, verify exactly 1/peer
#   ./peer_servers.sh restart    # clean slate (stop then start) -- REQUIRED between client runs
#
# Peer list: $PEER_IPS (space-separated) if set, else the 2nd column of peer_ips.txt.
# Env overrides: RPC_PORT (50052), RPC_MEM_MB (2000),
#                RPC_BIN (build-rpc/bin/rpc-server), PEER_IPS_FILE, PEER_IPS,
#                RPC_SERVER_ENV (extra env prepended to the server, e.g. "RPC_DBG_WCACHE=1")
# -----------------------------------------------------------------------------
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IPS_FILE="${PEER_IPS_FILE:-$SCRIPT_DIR/peer_ips.txt}"
RPC_PORT="${RPC_PORT:-50052}"
RPC_MEM_MB="${RPC_MEM_MB:-2000}"
RPC_BIN="${RPC_BIN:-build-rpc/bin/rpc-server}"
RPC_SERVER_ENV="${RPC_SERVER_ENV:-}"
SSH="ssh -n -o ConnectTimeout=10"

# explicit list (PEER_IPS) wins, so a caller like make.sh can pass its active -n subset
peers() {
    if [ -n "${PEER_IPS:-}" ]; then
        printf '%s\n' $PEER_IPS
    else
        grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$IPS_FILE" | awk '{print $2}'
    fi
}

# exact-name count via a tagged line (command substitution over the proxy can drop output)
count_on() { $SSH "pi@$1" 'echo "CNT $(pgrep -x rpc-server | wc -l)"' 2>/dev/null | awk '/^CNT/{print $2; exit}'; }

stop_all() {
    for ip in $(peers); do
        # kill, then re-check up to ~3s in case one is slow to die
        $SSH "pi@$ip" 'pkill -9 -x rpc-server 2>/dev/null; for i in 1 2 3 4 5 6; do pgrep -x rpc-server >/dev/null || break; sleep 0.5; pkill -9 -x rpc-server 2>/dev/null; done' 2>/dev/null
        echo "  $ip: stopped (count=$(count_on "$ip"))"
    done
}

# is the rpc-server on $1 actually ACCEPTING connections on RPC_PORT? (process-up via pgrep
# is not enough -- a client that connects before the listen socket is ready gets "Connection
# closed by peer" at warmup. The peer checks its own port locally so it works whether the
# coordinator is rpi1 (on-cell) or the Mac (behind the proxy).
listening_on() {
    # the server binds to the peer's own IBSS IP ($1), not loopback -- probe that.
    $SSH "pi@$1" "for i in \$(seq 1 20); do
        (exec 3<>/dev/tcp/$1/$RPC_PORT) 2>/dev/null && { echo LISTEN; exec 3>&-; exit 0; }
        sleep 0.5
      done; echo NOLISTEN" 2>/dev/null | awk '/LISTEN|NOLISTEN/{print; exit}'
}

start_all() {
    for ip in $(peers); do
        $SSH "pi@$ip" "cd ~/llama.cpp && ($RPC_SERVER_ENV setsid $RPC_BIN -H $ip -p $RPC_PORT -m $RPC_MEM_MB </dev/null >/tmp/rpc.log 2>&1 &)" 2>/dev/null
    done
    sleep 2
    local ok=1
    for ip in $(peers); do
        local n; n="$(count_on "$ip")"
        local l; l="$(listening_on "$ip")"  # poll until the port accepts (or ~10s timeout)
        if [ "$n" = "1" ] && [ "$l" = "LISTEN" ]; then
            echo "  $ip: OK (1 rpc-server, listening)"
        else
            echo "  $ip: BAD (count=${n:-?} listen=${l:-?}) -- not a clean ready slate"; ok=0
        fi
    done
    if [ "$ok" = "1" ]; then
        echo "clean slate: exactly one rpc-server per peer, all accepting connections"
    else
        echo "WARNING: NOT a clean ready slate -- a client run may hit stale state / GGML_ASSERT" >&2
        return 1
    fi
}

status_all() { for ip in $(peers); do echo "  $ip: $(count_on "$ip") rpc-server"; done; }

case "${1:-}" in
    status)  status_all ;;
    stop)    stop_all ;;
    start)   start_all ;;
    restart) stop_all; start_all ;;
    *) echo "usage: $0 {status|stop|start|restart}"; exit 2 ;;
esac
