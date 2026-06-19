#!/bin/bash
# sd_experiment.sh -- Single (S) vs Distributed (D) tensor-parallel sweep.
#
# Goal: find where splitting the model across Pis BEATS running it on one Pi.
#   S = llama-bench local on THIS one Pi (no RPC)            -> single-machine baseline
#   D = llama-bench --rpc <N cell peers> -sm row            -> N=4 tensor-parallel
# Sweeps prefill (-p, "pp" = prompt processing) and decode (-n, "tg" = token-gen).
#
# Run ON the coordinator Pi (rpi1): it has the model files AND --rpc reach to the
# cell peers. From a laptop:
#   ssh pi@<rpi1-eth> 'bash ~/llama.cpp/rpi-automation/sd_experiment.sh'
#
# Expectations:
#   - tg (decode):  D loses (all-reduce latency >> 1/N compute savings).
#   - pp (prefill): D overtakes S past some prompt length (compute-bound).
#   - llama-7b: S = OOM on 1.8 GB/Pi, D runs (~1 GB/Pi) -> the memory-bound win.
#
# Override anything via env: RPC=, PP=, TG=, REPS=, TINY=, SEVENB=
set -u
REPO="$HOME/llama.cpp"
BIN="$REPO/build-rpc/bin/llama-bench"
RPC="${RPC:-192.168.4.24:50052,192.168.4.25:50052,192.168.4.20:50052,192.168.4.22:50052}"
PP="${PP:-64,256,512}"
TG="${TG:-8}"
REPS="${REPS:-2}"
TINY="${TINY:-$REPO/models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf}"
SEVENB="${SEVENB:-$REPO/models/llama-7b.Q4_K_M.gguf}"

if [ ! -x "$BIN" ]; then
    echo ">> building llama-bench (one-time)..."
    cmake --build "$REPO/build-rpc" -j"$(nproc)" --target llama-bench >/tmp/bench_build.log 2>&1 \
        && echo "   ok" || { echo "   BUILD FAILED:"; tail -4 /tmp/bench_build.log; exit 1; }
fi

run() {  # $1=label  $2=model-path  $3=extra args
    echo
    echo "########## $1 : $(basename "$2") ##########"
    if [ ! -f "$2" ]; then echo "  (model not found: $2 -- copy it here first)"; return; fi
    # llama-bench prints a markdown table; the data rows start with '|'. A failure
    # (e.g. OOM loading a too-big model on one Pi) prints no table -> note it.
    if ! $BIN -m "$2" -p "$PP" -n "$TG" -r "$REPS" $3 2>/tmp/bench_err.log | grep -E "^\|"; then
        echo "  (no result -- likely OOM / load failure; tail of stderr:)"
        tail -3 /tmp/bench_err.log | sed 's/^/    /'
    fi
}

echo "==== S vs D sweep on $(uname -n) ===="
echo "pp=$PP  tg=$TG  reps=$REPS  |  D = N=4 over the cell ($RPC)"
run "S  single-Pi"  "$TINY"   ""
run "D  N=4 cell"   "$TINY"   "--rpc $RPC -sm row -ts 1,1,1,1"
run "S  single-Pi"  "$SEVENB" ""
run "D  N=4 cell"   "$SEVENB" "--rpc $RPC -sm row -ts 1,1,1,1"
echo
echo ">> done"
