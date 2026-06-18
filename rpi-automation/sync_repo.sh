#!/bin/zsh
# Sync all RPis to the same git commit that is checked out on this machine.
# Fetches from origin on each RPi and hard-resets to the local HEAD hash.
# If the hash changed, rebuilds build-rpc in parallel across all updated RPis.
#
# Usage:
#   ./rpi-automation/sync_repo.sh [--ips <file>] [--repo <path>] [--branch <name>]
#                                  [--no-rebuild] [--force-rebuild]
#
# Defaults:
#   --ips        rpi-automation/dprgnet_ips.txt
#   --repo       ~/llama.cpp               (path on each RPi)
#   --branch     current branch on this machine
#   --no-rebuild skip recompilation even when hash changed

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IPS_FILE="${SCRIPT_DIR}/dprgnet_ips.txt"
REMOTE_REPO="~/llama.cpp"
SSH_USER="pi"
REBUILD=1
FORCE_REBUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ips)           IPS_FILE="$2";    shift 2 ;;
        --repo)          REMOTE_REPO="$2"; shift 2 ;;
        --branch)        BRANCH="$2";      shift 2 ;;
        --no-rebuild)    REBUILD=0;        shift 1 ;;
        --force-rebuild) FORCE_REBUILD=1;  shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$IPS_FILE" ]]; then
    echo "ERROR: IP list not found: $IPS_FILE" >&2; exit 1
fi

TARGET_HASH=$(git rev-parse HEAD)
TARGET_BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

echo "Target commit : $TARGET_HASH"
echo "Target branch : $TARGET_BRANCH"
echo "Remote repo   : $REMOTE_REPO"
echo ""

log() { echo "[$(date '+%H:%M:%S')] $*"; }
ssh_cmd() { ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${SSH_USER}@$1" "$2"; }

RPI_NAMES=()
RPIS=()
while read -r _n _i; do
    [[ -z "$_n" || "${_n[1]}" == "#" || -z "$_i" ]] && continue
    RPI_NAMES+=("$_n")
    RPIS+=("$_i")
done < "$IPS_FILE"

ok=0; skipped=0; failed=0

n_nodes=${#RPIS[@]}
# ~15s per RPi: git fetch + checkout + reset over WiFi
est_s=$(( n_nodes * 15 ))
est_m=$(( est_s / 60 )); est_rem=$(( est_s % 60 ))
echo "Nodes to sync  : $n_nodes"
echo "Estimated time : ${est_m}m ${est_rem}s (sync only)"
echo ""

# Track which RPis need a rebuild (hash actually changed)
rebuild_names=()
rebuild_ips=()

for (( i=1; i<=${#RPIS[@]}; i++ )); do
    name="${RPI_NAMES[$i]}"
    ip="${RPIS[$i]}"
    log "Syncing $name ($ip)..."

    result=$(ssh_cmd "$ip" "
        set -e
        cd $REMOTE_REPO
        prev=\$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
        git fetch origin 2>&1 | tail -1
        git checkout $TARGET_BRANCH 2>/dev/null || git checkout -b $TARGET_BRANCH origin/$TARGET_BRANCH
        git reset --hard $TARGET_HASH
        now=\$(git rev-parse --short HEAD)
        if [[ \"\$prev\" != \"\$now\" ]]; then
            echo \"UPDATED: \$prev -> \$now on \$(git rev-parse --abbrev-ref HEAD)\"
        else
            echo \"OK: \$now (already up to date) on \$(git rev-parse --abbrev-ref HEAD)\"
        fi
    " 2>&1) && {
        log "  $result"
        (( ++ok ))
        if [[ $REBUILD -eq 1 ]] && { [[ "$result" == *"UPDATED:"* ]] || [[ $FORCE_REBUILD -eq 1 ]]; }; then
            rebuild_names+=("$name")
            rebuild_ips+=("$ip")
        fi
    } || {
        if echo "$result" | grep -q "Connection\|timeout\|refused"; then
            log "  SKIP: unreachable"
            (( ++skipped ))
        else
            log "  FAIL: $result"
            (( ++failed ))
        fi
    }
done

echo ""
echo "Sync done — $ok synced, $skipped unreachable, $failed failed."

# ── Rebuild ──────────────────────────────────────────────────────────────────

if [[ $REBUILD -eq 0 ]] || [[ ${#rebuild_ips[@]} -eq 0 ]]; then
    [[ ${#rebuild_ips[@]} -eq 0 ]] && echo "All RPis already on target hash — no rebuild needed."
    exit 0
fi

n_rebuild=${#rebuild_ips[@]}
echo ""
echo "${n_rebuild} RPi(s) updated — rebuilding build-rpc in parallel (~10 min):"
for name in "${rebuild_names[@]}"; do echo "  $name"; done
echo ""

BUILD_CMD="cd $REMOTE_REPO/build-rpc && \
    cmake .. -DGGML_RPC=ON -DGGML_VULKAN=OFF -DGGML_METAL=OFF \
             -DGGML_CUDA=OFF -DGGML_NATIVE=OFF 2>&1 | tail -1 && \
    cmake --build . --config Release -j4 --clean-first 2>&1 | tail -3"

build_pids=()
for (( i=1; i<=${#rebuild_ips[@]}; i++ )); do
    name="${rebuild_names[$i]}"
    ip="${rebuild_ips[$i]}"
    (
        out=$(ssh_cmd "$ip" "$BUILD_CMD" 2>&1)
        rc=$?
        if [[ $rc -eq 0 ]]; then
            log "  BUILD OK  $name"
        else
            log "  BUILD FAIL $name: $out"
        fi
    ) &
    build_pids+=($!)
done

wait
echo ""
echo "Rebuild complete."
