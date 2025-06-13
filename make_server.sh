#!/bin/bash
# -----------------------------------------------------------------------------
# make_server.sh
#
# This script automates the process of updating, building, and running the
# rpc-server for llama.cpp in a tmux session. It supports launching the server
# on specified IP addresses (e.g., rpi1, rpi7, or a custom IP) and optionally
# attaches to the tmux session.
#
# Steps performed:
#   1. Navigates to the llama.cpp directory and pulls the latest changes.
#   2. Builds the project with RPC enabled and other accelerators disabled.
#   3. Kills any running rpc-server processes.
#   4. Determines the server IP based on the first argument (rpi1, rpi7, or custom).
#   5. Ensures a tmux session named 'llm' exists, and creates a window for the server.
#   6. Starts the rpc-server in the tmux window with the specified IP.
#   7. Optionally attaches to the tmux session if the second argument is 'attach'.
#
# Usage:
#   ./make_server.sh <rpi1|rpi7|custom_ip> [attach]
#
# Arguments:
#   <rpi1|rpi7|custom_ip>   Target server IP or alias ('rpi1' or 'rpi7').
#   [attach]                (Optional) Attach to the tmux session after starting.
#
# Example:
#   ./make_server.sh rpi1 attach
# -----------------------------------------------------------------------------

set -e

cd ~/test/dprg_llama.cpp/llama.cpp/
git pull

cd build-rpc
cmake .. -DGGML_RPC=ON -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF
cmake --build . --config Release
killall -9 rpc-server || true

rpi1="172.16.107.154"
rpi7="172.16.112.192"

# Parse arguments
host="$1"
if [ "$host" = "rpi1" ]; then
    server_ip="$rpi1"
elif [ "$host" = "rpi7" ]; then
    server_ip="$rpi7"
else
    server_ip="$host"
fi

tmux="$2"
if [ "$tmux" = "attach" ]; then
    # Create tmux session 'llm' if it doesn't exist
    tmux has-session -t llm 2>/dev/null || tmux new-session -d -s llm

    # Create window 1 if it doesn't exist
    if ! tmux list-windows -t llm | grep -q "^1:"; then
        tmux new-window -t llm:1 -n server
    fi

    tmux select-window -t llm:1
    # tmux attach-session -t llm

    # Send the server command to window 1
    tmux send-keys -t llm:1 "./build-rpc/bin/rpc-server -H $server_ip -m 1000" C-m
else
    ./bin/rpc-server -H $server_ip -m 1000
fi
