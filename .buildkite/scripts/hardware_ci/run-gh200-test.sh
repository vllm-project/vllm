#!/bin/bash

# This script build the GH200 docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Skip the new torch installation during build since we are using the specified version for arm64 in the Dockerfile
python3 use_existing_torch.py

# Try building the docker image
DOCKER_BUILDKIT=1 docker build . \
  --file docker/Dockerfile \
  --target vllm-openai \
  --platform "linux/arm64" \
  -t gh200-test \
  --build-arg max_jobs=66 \
  --build-arg nvcc_threads=2 \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg torch_cuda_arch_list="9.0+PTX"

# Setup cleanup
remove_docker_container() { docker rm -f gh200-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Debug: inspect the built image before running inference
docker run --rm --entrypoint="" gh200-test bash -c '
    echo "===== Python & pip info ====="
    python3 --version
    pip --version

    echo ""
    echo "===== Torch installation details ====="
    pip show torch
    echo ""
    python3 -c "import torch; print(f\"torch version: {torch.__version__}\"); print(f\"torch file: {torch.__file__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\")" 2>&1 || echo "FAILED to import torch"

    echo ""
    echo "===== libtorch_cuda.so search ====="
    find / -name "libtorch_cuda*.so*" 2>/dev/null || echo "libtorch_cuda.so NOT FOUND anywhere"

    echo ""
    echo "===== Torch lib directory contents ====="
    TORCH_DIR=$(python3 -c "import pathlib, torch; print(pathlib.Path(torch.__file__).parent / \"lib\")" 2>/dev/null)
    if [ -n "$TORCH_DIR" ] && [ -d "$TORCH_DIR" ]; then
        ls -la "$TORCH_DIR"/libtorch* 2>/dev/null || echo "No libtorch* files in $TORCH_DIR"
        ls -la "$TORCH_DIR"/*.so* 2>/dev/null | head -30
    else
        echo "Could not determine torch lib directory"
    fi

    echo ""
    echo "===== vllm._C check ====="
    python3 -c "import vllm._C; print(\"vllm._C loaded OK\")" 2>&1 || echo "FAILED to import vllm._C"

    echo ""
    echo "===== LD_LIBRARY_PATH ====="
    echo "$LD_LIBRARY_PATH"

    echo ""
    echo "===== NVIDIA / CUDA runtime info ====="
    nvidia-smi 2>&1 || echo "nvidia-smi not available"
    nvcc --version 2>&1 || echo "nvcc not available"

    echo ""
    echo "===== Installed packages (torch-related) ====="
    pip list 2>/dev/null | grep -iE "torch|cuda|vllm|flash"

    echo ""
    echo "===== ldd on vllm._C ====="
    VLLM_C_SO=$(python3 -c "import vllm._C as c; print(c.__file__)" 2>/dev/null)
    if [ -n "$VLLM_C_SO" ]; then
        ldd "$VLLM_C_SO" 2>&1
    else
        # Try to find it manually
        find / -name "_C*.so" -path "*/vllm/*" 2>/dev/null | while read f; do
            echo "Found: $f"
            ldd "$f" 2>&1
        done
    fi
'

# Run the image and test offline inference
docker run -e HF_TOKEN -e VLLM_WORKER_MULTIPROC_METHOD=spawn -v /root/.cache/huggingface:/root/.cache/huggingface --name gh200-test --gpus=all --entrypoint="" gh200-test bash -c '
    python3 examples/basic/offline_inference/generate.py --model meta-llama/Llama-3.2-1B
'
