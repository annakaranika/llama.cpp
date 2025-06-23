#!/bin/bash
# -----------------------------------------------------------------------------
# make.sh - Script to manage building and deploying llama.cpp on multiple RPIs
#
# Features:
#   - Supports two main operations via arguments: 'restart' and 'compile'
#   - 'restart': Pushes latest code and restarts the rpc-server on each RPI
#   - 'compile': Configures and builds the project with specific CMake options
#   - Automatically constructs the --rpc argument for llama-cli using RPI IPs
#   - Runs llama-cli with the specified model and RPC endpoints
#
# Usage:
#   ./make.sh [restart] [compile]
#
# Arguments:
#   restart   - SSH into each RPI, kill existing rpc-server, and restart it
#   compile   - Run CMake configuration and build the project
#
# Environment:
#   - Expects SSH access to each RPI listed in the 'rpis' array
#   - Assumes model file path is correct and llama.cpp is set up on each RPI
#
# Example:
#   ./make.sh restart compile
# -----------------------------------------------------------------------------
set -e

rpis=("172.16.107.154" "172.16.112.192")
model="models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
build_dir="build-rpc"

# Parse arguments
restart=""
compile=""
for arg in "$@"; do
    case "$arg" in
        restart)
            restart="yes"
            ;;
        compile)
            compile="yes"
            ;;
    esac
done

if [[ "$restart" == "yes" ]]; then
    # SSH into each server and restart the rpc-server
    git push
    for rpi in "${rpis[@]}"; do
        ssh -t pi@"$rpi" "pkill -9 rpc-server; cd ~/test/dprg_llama.cpp/llama.cpp && ./make_server.sh $rpi attach" &
    done
fi

if [[ "$compile" == "yes" ]]; then
    (
        cd $build_dir
        cmake .. -DGGML_RPC=ON -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF
        cmake --build . --config Release
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

# GGML_SCHED_DEBUG=2 ./$build_dir/bin/llama-cli -v -m $model --rpc $rpc_arg -ngl 23 -sm row
./$build_dir/bin/llama-cli -m $model --rpc $rpc_arg -ngl 23 -sm row
