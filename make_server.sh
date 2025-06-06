#!/bin/bash
set -e

cd ~/test/dprg_llama.cpp/llama.cpp/build-rpc
cmake .. -DGGML_RPC=ON -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF
cmake --build . --config Release
# killall -9 llama-server
