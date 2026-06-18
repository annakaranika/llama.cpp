sudo apt update && sudo apt install git cmake -y
rm -rf llama.cpp
git clone https://github.com/annakaranika/llama.cpp.git
cd llama.cpp
git checkout parallel
rm -rf build-rpc
mkdir build-rpc
cd build-rpc
cmake .. -DGGML_RPC=ON -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=OFF
# The following may fail on Raspberry Pi and make it hang
cmake --build . --config Release -j$(nproc)
