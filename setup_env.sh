# Install Nvidia driver -- setup_gpu_1.sh
  % ./setup_gpu_1.sh
  # content: 
    % sudo apt update
    % sudo apt install -y nvidia-utils-535
    % sudo add-apt-repository restricted
    % sudo add-apt-repository universe
    % sudo apt update
    % sudo apt install -y nvidia-driver-535
    % sudo nvidia-smi -mig 0
    % sudo reboot

# Setup disk drive -- setup_machine.sh
  % ./setup_machine.sh <device> <mount_dir> <user:group>
  % sudo cp /etc/fstab /etc/fstab.backup
  % sudo nano /etc/fstab   # Format: UUID=YOUR_UUID_HERE   /holly   ext4   defaults,nofail   0   2

# Add public key in github

# Set up CUDA toolkit, Python, pip, torch, and etc
  % ./setup_gpu_2.sh
  # content:
    % curl -LsSf https://astral.sh/uv/install.sh | sh

    % bash
    % source $HOME/.local/bin/env

    % uv venv --python 3.12 --seed --managed-python

    % wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    % sudo dpkg -i cuda-keyring_1.1-1_all.deb
    % sudo apt-get update
    % sudo apt-get install -y ccache
    % sudo apt-get install -y cuda-toolkit-12-8

    % sudo ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
    % echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
    % echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    % source ~/.bashrc

    % cd /holly
    % git clone git@github.com:hcasalet/villum.git
    % cd villum
    % source .venv/bin/activate

    % uv pip install -U pip
    % uv pip install "torch==2.10.0+cu128" --index-url https://download.pytorch.org/whl/cu128
    % uv pip install packaging setuptools wheel ninja cmake numpy nixl setuptools_scm

# Build
  % rm -rf build/ .eggs/ *.egg-info
  % find . -name "__pycache__" -type d -prune -exec rm -rf {} +
  % export CUDA_HOME=/usr/local/cuda
  % export PATH="${CUDA_HOME}/bin:${PATH}"
  % export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
  % export TORCH_CUDA_ARCH_LIST="8.0"
  % export MAX_JOBS=1
  % export NVCC_THREADS=1
  % python use_existing_torch.py
  % pip install -r requirements/build.txt
  # native-code build path
  % CCACHE_NOHASHDIR=true pip install -e . --no-build-isolation -v
  # Python-only path
  % VLLM_USE_PRECOMPILED=1 uv pip install --editable .

  # for debugging: uv pip install -e . --no-build-isolation -v 2>&1 | tee /tmp/vllm_build.log

# Checking (Build)
  % lspci |grep -i nvidia       # shows if there is GPU in hardware
  % ls /dev/nvidia*             # if returns nothing, driver is not installed
  % lsmod | grep -i nvidia || echo "nvidia kernel module not loaded".  # confirming if driver module not loaded
  % nvidia-smi -q | grep -A2 "MIG Mode"  # Expects MIG Current : Disable, Pending : Disabled
  % sudo mount -a
  % mount | grep holly  # verify that /holly survives reboot
  # rc is expected to be 0
  % sudo /holly/villum/.venv/bin/python - #<<'PY'  
    import ctypes
    lib = ctypes.CDLL("libcuda.so.1")
    cuInit = lib.cuInit
    cuInit.argtypes = [ctypes.c_uint]
    cuInit.restype = ctypes.c_int
    print("cuInit rc =", cuInit(0))
    PY 
  # Checking if any errors in build:
  % grep -nE "error:|fatal|ptxas|nvcc fatal" /tmp/vllm_build.log | head -n 120

# Run
## Multi-GPU
  % export MODEL="facebook/opt-125m"
  % export HF_TOKEN=""

  # prefill
  % VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname -I | awk '{print $1}') \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    UCX_NET_DEVICES=all \
    CUDA_VISIBLE_DEVICES=0 \
    vllm serve "$MODEL" \
    --port 8100 \
    --enforce-eager \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'

  # decode
  % VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname -I | awk '{print $1}') \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    UCX_NET_DEVICES=all \
    CUDA_VISIBLE_DEVICES=0 \
    vllm serve "$MODEL" \
    --port 8200 \
    --enforce-eager \
    --gpu-memory-utilization 0.80 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'

  # proxy
  % python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --port 8192 \
    --prefiller-hosts <PREFILL_IP> \
    --prefiller-ports 8100 \
    --decoder-hosts <DECODE_IP> \
    --decoder-ports 8200

## Single-GPU
  # prefill
  % CUDA_VISIBLE_DEVICES=0 \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    vllm serve "$MODEL" \
    --port 8100 \
    --enforce-eager \
    --gpu-memory-utilization 0.35 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'

  # decode
  % CUDA_VISIBLE_DEVICES=0 \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
    vllm serve "$MODEL" \
    --port 8200 \
    --enforce-eager \
    --gpu-memory-utilization 0.35 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'

  # proxy
  % python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --port 8192 \
    --prefiller-hosts localhost \
    --prefiller-ports 8100 \
    --decoder-hosts localhost \
    --decoder-ports 8200

# Test
  # disaggregate
  % curl http://localhost:8192/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "facebook/opt-125m",
    "prompt": "Continue this story in detail: A scientist opened a door beneath the ocean and found",
    "max_tokens": 100,
    "temperature": 0.7
  }' | python -m json.tool

  # one server
  % curl http://127.0.0.1:8100/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Continue this story in detail: A scientist opened a door beneath the ocean and found",
    "max_tokens": 100,
    "temperature": 0.7
  }' | python -m json.tool

# Some leftover
  % export UCX_TLS=all

# Versions to keep
 -- Python 3.12.13
 -- pip 26.0.1 from /holly/villum/.venv/lib/python3.12/site-packages/pip (python 3.12)
      
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.288.01             Driver Version: 535.288.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A30                     Off | 00000000:25:00.0 Off |                    0 |
| N/A   26C    P0              28W / 165W |      0MiB / 24576MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0

python - <<'PY'
import torch, torchvision
print(torch.__version__)
print(torchvision.__version__)
print(torch.version.cuda)
PY  