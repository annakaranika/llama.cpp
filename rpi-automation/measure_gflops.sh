#!/usr/bin/env bash
# =============================================================================
# measure_gflops.sh — Collect per-device MUL_MAT throughput across the RPi fleet
# =============================================================================
#
# Runs `test-backend-ops perf` on every reachable device in the IP list (over
# SSH, concurrently) and collects the raw output plus device metadata into a
# timestamped results directory. Feed that directory to parse_gflops.py to build
# a tidy CSV + per-device profile for the heterogeneous-device dataset.
#
# Each device runs its OWN local CPU benchmark — no RPC / network traffic during
# the measurement — so running them concurrently is safe and independent.
#
# Usage:
#   ./measure_gflops.sh [--ips <file>] [--op <OP>] [--threads <n>]
#                       [--jobs <n>] [--seq] [--repo <path>] [--build <dir>]
#
# Defaults:
#   --ips      rpi-automation/dprgnet_ips.txt
#   --op       MUL_MAT       (full perf sweep for that op; matches rpi1/rpi2)
#   --threads  4             (OMP_NUM_THREADS on each device)
#   --jobs     8             (max concurrent devices; --seq forces 1)
#   --repo     ~/llama.cpp   (remote repo path)
#   --build    build-rpc     (remote build dir under repo)
#
# Output:  results/gflops/<timestamp>/
#   raw/<name>.txt   raw test-backend-ops output (+ metadata header)
#   meta/<name>.env  device metadata (key=val)
#   devices.csv      assembled metadata table
#   RUN_INFO.txt     op / threads / ips / local git hash / date
# =============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IPS_FILE="${SCRIPT_DIR}/dprgnet_ips.txt"
OP="MUL_MAT"
THREADS=4
JOBS=8
SSH_USER="pi"
REMOTE_REPO="~/llama.cpp"
BUILD_DIR="build-rpc"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ips)     IPS_FILE="$2";    shift 2 ;;
    --op)      OP="$2";          shift 2 ;;
    --threads) THREADS="$2";     shift 2 ;;
    --jobs)    JOBS="$2";        shift 2 ;;
    --seq)     JOBS=1;           shift 1 ;;
    --repo)    REMOTE_REPO="$2"; shift 2 ;;
    --build)   BUILD_DIR="$2";   shift 2 ;;
    -h|--help) sed -n '4,33p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -f "$IPS_FILE" ]] || { echo "ERROR: IP list not found: $IPS_FILE" >&2; exit 1; }

BIN="${REMOTE_REPO}/${BUILD_DIR}/bin/test-backend-ops"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${REPO_ROOT}/results/gflops/${TIMESTAMP}"
mkdir -p "$OUTDIR/raw" "$OUTDIR/meta"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes)

# Parse IP file ("name ip" lines, # comments) into parallel arrays
NAMES=(); IPS=()
while read -r name ip _; do
  [[ -z "${name:-}" || "${name:0:1}" == "#" || -z "${ip:-}" ]] && continue
  NAMES+=("$name"); IPS+=("$ip")
done < "$IPS_FILE"

N=${#NAMES[@]}
[[ $N -gt 0 ]] || { echo "ERROR: no devices parsed from $IPS_FILE" >&2; exit 1; }

LOCAL_HASH="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
{
  echo "timestamp  : $TIMESTAMP"
  echo "op         : $OP"
  echo "threads    : $THREADS"
  echo "ips_file   : $IPS_FILE"
  echo "remote_bin : $BIN"
  echo "local_hash : $LOCAL_HASH"
  echo "devices    : $N"
} > "$OUTDIR/RUN_INFO.txt"

log "Measuring $OP on $N device(s) from $(basename "$IPS_FILE"), up to $JOBS at a time."
log "Output dir : $OUTDIR"
log "Each device runs a full perf sweep (~8-12 min). Batches: ~$(( (N + JOBS - 1) / JOBS ))."

# Remote metadata collector (single-quoted heredoc: evaluated entirely on device)
read -r -d '' META_SNIPPET <<'EOS' || true
echo "model=$(tr -d '\000' < /proc/device-tree/model 2>/dev/null || echo unknown)"
echo "cores=$(nproc 2>/dev/null || echo nan)"
echo "mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo nan)"
echo "git_hash=$(git -C ~/llama.cpp rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "temp_c=$(vcgencmd measure_temp 2>/dev/null | grep -oE '[0-9.]+' || echo nan)"
echo "clock_arm_hz=$(vcgencmd measure_clock arm 2>/dev/null | grep -oE '[0-9]+$' || echo nan)"
echo "throttled=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2 || echo nan)"
EOS

run_device() {
  local name="$1" ip="$2"
  local raw="$OUTDIR/raw/${name}.txt"
  local env="$OUTDIR/meta/${name}.env"

  # 1) metadata + binary existence
  local meta
  meta=$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" \
           "$META_SNIPPET; echo \"bin_exists=\$([[ -x $BIN ]] && echo 1 || echo 0)\"" 2>/dev/null) || {
    log "  $name ($ip): UNREACHABLE"
    { echo "name=$name"; echo "ip=$ip"; echo "status=unreachable"; } > "$env"; return
  }

  if ! grep -q '^bin_exists=1' <<< "$meta"; then
    log "  $name ($ip): test-backend-ops NOT built ($BIN) — skipping (run sync_repo.sh?)"
    { echo "name=$name"; echo "ip=$ip"; echo "status=no_binary"; printf '%s\n' "$meta"; } > "$env"; return
  fi

  local temp_before
  temp_before=$(grep '^temp_c=' <<< "$meta" | cut -d= -f2)

  # 2) the benchmark (the slow part)
  log "  $name ($ip): running $OP sweep..."
  local bench rc
  bench=$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" \
            "OMP_NUM_THREADS=${THREADS} ${BIN} perf -o ${OP} -b CPU 2>&1"); rc=$?
  if [[ $rc -ne 0 ]]; then
    log "  $name ($ip): benchmark FAILED (rc=$rc)"
    { echo "name=$name"; echo "ip=$ip"; echo "status=bench_failed"; printf '%s\n' "$meta"; } > "$env"
    printf '%s\n' "$bench" > "$raw"; return
  fi

  # 3) temp / throttle state AFTER the sustained load (catches throttling)
  local after
  after=$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" \
            "echo temp_after_c=\$(vcgencmd measure_temp 2>/dev/null | grep -oE '[0-9.]+' || echo nan); \
             echo throttled_after=\$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2 || echo nan)" 2>/dev/null) || after=""

  # 4) write raw (metadata header + benchmark) and env
  {
    echo "# device=$name ip=$ip op=$OP threads=$THREADS ts=$TIMESTAMP"
    echo "# temp_before_c=$temp_before"
    printf '%s\n' "$after" | sed 's/^/# /'
    printf '%s\n' "$bench"
  } > "$raw"

  {
    echo "name=$name"; echo "ip=$ip"; echo "status=ok"; echo "temp_before_c=$temp_before"
    printf '%s\n' "$meta" | grep -v '^bin_exists='
    printf '%s\n' "$after"
  } > "$env"

  local rows; rows=$(grep -c "${OP}(" "$raw" 2>/dev/null || echo 0)
  log "  $name ($ip): DONE — $rows rows"
}

# Launch in batches of $JOBS (bash 3.2-compatible: no `wait -n`)
i=0
while [[ $i -lt $N ]]; do
  pids=()
  for (( j=0; j<JOBS && i<N; j++, i++ )); do
    run_device "${NAMES[$i]}" "${IPS[$i]}" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
done

# Assemble devices.csv from per-device env files
CSV="$OUTDIR/devices.csv"
echo "name,ip,status,model,cores,mem_kb,git_hash,temp_before_c,temp_after_c,clock_arm_hz,throttled,throttled_after" > "$CSV"
for env in "$OUTDIR"/meta/*.env; do
  [[ -f "$env" ]] || continue
  unset name ip status model cores mem_kb git_hash temp_c temp_before_c temp_after_c clock_arm_hz throttled throttled_after
  while IFS='=' read -r k v; do [[ -n "$k" ]] && printf -v "$k" '%s' "$v"; done < "$env"
  echo "${name:-},${ip:-},${status:-},\"${model:-}\",${cores:-},${mem_kb:-},${git_hash:-},${temp_before_c:-${temp_c:-}},${temp_after_c:-},${clock_arm_hz:-},${throttled:-},${throttled_after:-}" >> "$CSV"
done

echo ""
log "All done. Raw: $OUTDIR/raw/   Metadata: $CSV"
log "Next: python3 ${SCRIPT_DIR}/parse_gflops.py $OUTDIR"
