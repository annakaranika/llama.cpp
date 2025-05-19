#!/bin/bash
set -e
cd build-rpc
cmake .. -DGGML_RPC=ON -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF
cmake --build . --config Release
cd ..
./build-rpc/bin/llama-cli -m models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf --rpc 172.16.107.154:50052,172.16.112.192:50052 -ngl 23
