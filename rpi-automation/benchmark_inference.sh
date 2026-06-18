#!/bin/bash
# -----------------------------------------------------------------------------
# benchmark_inference.sh
#
# Benchmarks distributed PP inference latency across N RPis.
# Measures: time-to-first-token (TTFT) and tokens/sec (generation throughput).
# Sweeps over number of devices and logs results to CSV.
#
# Usage:
#   ./scripts/benchmark_inference.sh [ap|mesh] [num_runs]
#
# Arguments:
#   ap|mesh    Network mode label (default: ap). Affects output filename only;
#              you must have already configured the network before running.
#   num_runs   Number of inference runs per configuration (default: 5)
#
# Output:
#   results/inference_<mode>_<timestamp>.csv
#
# Requirements:
#   - RPi rpc-servers already running (use make.sh restart)
#   - llama-cli built locally in build-rpc/
#   - Model file present at $MODEL path
# -----------------------------------------------------------------------------
set -e

# ---- Configuration ----
RPIS=(
    "172.16.107.154"
    "172.16.112.192"
    # Add more as available
)
RPC_PORT=50052
MODEL="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"   # Q4 for better quality signal
BUILD_DIR="build-rpc"
PROMPT="Describe the key challenges of running large language models on IoT devices."
N_PREDICT=50       # tokens to generate
N_WARMUP=2         # warmup runs before measuring (discarded)
MODE="${1:-ap}"
NUM_RUNS="${2:-5}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="results"
OUTPUT="${OUTPUT_DIR}/inference_${MODE}_${TIMESTAMP}.csv"

mkdir -p "$OUTPUT_DIR"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- Ensure rpc-servers are running on all RPis ----
start_servers() {
    local n_devices="$1"
    log "Starting rpc-servers on $n_devices RPis..."
    for ((i=0; i<n_devices; i++)); do
        rpi="${RPIS[$i]}"
        ssh -o StrictHostKeyChecking=no pi@"$rpi" \
            "pkill -9 rpc-server 2>/dev/null || true; \
             nohup ./test/dprg_llama.cpp/llama.cpp/${BUILD_DIR}/bin/rpc-server \
             -H $rpi -p $RPC_PORT -m 2048 > /tmp/rpc-server.log 2>&1 &" &
    done
    wait
    sleep 3  # give servers time to start
}

stop_servers() {
    log "Stopping rpc-servers..."
    for rpi in "${RPIS[@]}"; do
        ssh -o StrictHostKeyChecking=no pi@"$rpi" "pkill -9 rpc-server 2>/dev/null || true" &
    done
    wait
}

# ---- Build RPC argument for N devices ----
build_rpc_arg() {
    local n="$1"
    local rpc_arg=""
    for ((i=0; i<n; i++)); do
        rpc_arg+="${RPIS[$i]}:${RPC_PORT},"
    done
    echo "${rpc_arg%,}"
}

# ---- Run one inference and extract timing ----
run_inference() {
    local rpc_arg="$1"
    local n_layers="$2"   # total GPU layers to offload via RPC

    # Run llama-cli and capture timing output
    OUTPUT_RAW=$(./${BUILD_DIR}/bin/llama-cli \
        -m "$MODEL" \
        --rpc "$rpc_arg" \
        -ngl "$n_layers" \
        -sm row \
        -p "$PROMPT" \
        -n "$N_PREDICT" \
        --no-display-prompt \
        -t 4 \
        2>&1)

    # Extract TTFT (time to first token) in ms
    TTFT=$(echo "$OUTPUT_RAW" | grep "time to first token" | grep -oP '[\d.]+(?= ms)' | head -1 || echo "")
    # Extract generation speed in tokens/sec
    TPS=$(echo "$OUTPUT_RAW" | grep "eval time" | grep -oP '[\d.]+(?= tokens/s)' | head -1 || echo "")

    echo "${TTFT},${TPS}"
}

# ---- Write CSV header ----
echo "network_mode,n_devices,run,ttft_ms,tokens_per_sec" > "$OUTPUT"
log "Results -> $OUTPUT"

# ---- Sweep over device counts ----
MAX_DEVICES=${#RPIS[@]}
for ((n=1; n<=MAX_DEVICES; n++)); do
    RPC_ARG=$(build_rpc_arg "$n")
    N_LAYERS=23   # TinyLlama has 22 transformer layers + 1 output

    start_servers "$n"

    # Warmup runs
    log "Warming up ($n devices, $N_WARMUP runs)..."
    for ((w=0; w<N_WARMUP; w++)); do
        run_inference "$RPC_ARG" "$N_LAYERS" > /dev/null 2>&1 || true
    done

    # Measured runs
    log "Benchmarking: $n device(s), $NUM_RUNS runs..."
    for ((r=1; r<=NUM_RUNS; r++)); do
        RESULT=$(run_inference "$RPC_ARG" "$N_LAYERS")
        TTFT=$(echo "$RESULT" | cut -d',' -f1)
        TPS=$(echo "$RESULT" | cut -d',' -f2)
        echo "${MODE},${n},${r},${TTFT},${TPS}" >> "$OUTPUT"
        log "  run $r: TTFT=${TTFT}ms, ${TPS} tok/s"
    done

    stop_servers
    sleep 2
done

log "Benchmark complete. Results in $OUTPUT"
echo ""
echo "=== Summary ==="
column -t -s',' "$OUTPUT"
