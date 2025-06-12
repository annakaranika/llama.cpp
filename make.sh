#!/bin/bash
set -e

rpis=("172.16.107.154" "172.16.112.192")

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
    # Example: SSH into each server and restart the rpc-server
    for rpi in "${rpis[@]}"; do
        ssh anna@"$rpi" "pkill -9 rpc-server; cd ~/test/dprg_llama.cpp/llama.cpp && ./make_server.sh $rpi" &
    done
    wait
fi

if [[ "$compile" == "yes" ]]; then
    cd build
    cmake .. -DGGML_RPC=ON -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF
    cmake --build . --config Release
    cd ..
fi

# GGML_SCHED_DEBUG=2 ./build/bin/llama-cli -v -m models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf --rpc 172.16.107.154:50052,172.16.112.192:50052 -ngl 23
./build/bin/llama-cli -v -m models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf --rpc 172.16.107.154:50052,172.16.112.192:50052 -ngl 23 -sm row
