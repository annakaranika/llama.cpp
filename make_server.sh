#!/bin/bash
set -e

cd ~/test/dprg_llama.cpp/llama.cpp/
git pull

cd build-rpc
cmake .. -DGGML_RPC=ON -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF
cmake --build . --config Release
killall -9 rpc-server

rpi1=172.16.107.154
rpi7=172.16.112.192

# Parse arguments
arg="$1"
attach="$2"
if [[ "$arg" == "rpi1" ]]; then
    server_ip="$rpi1"
elif [[ "$arg" == "rpi7" ]]; then
    server_ip="$rpi7"
else
    server_ip="$arg"
fi

# Create tmux session 'llm' if it doesn't exist
tmux has-session -t llm 2>/dev/null || tmux new-session -d -s llm

# Create window 1 if it doesn't exist
if ! tmux list-windows -t llm | grep -q "^1:"; then
    tmux new-window -t llm:1 -n server
fi

# Send the server command to window 1
tmux send-keys -t llm:1 "./build-rpc/bin/rpc-server -H \"$server_ip\" -m 1000" C-m

# Optionally attach to tmux session
if [[ "$attach" == "attach" ]]; then
    tmux select-window -t llm:1
    tmux attach-session -t llm
fi

