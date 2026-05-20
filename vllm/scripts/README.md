Information about prefill/decode disaggregate serving: https://docs.vllm.ai/en/stable/features/disagg_prefill/#development

# setup

### nvidia
Install nvidia tools (setup_gpu_1.sh)
```
sudo apt update
sudo apt install -y nvidia-utils-535
sudo add-apt-repository restricted
sudo add-apt-repository universe
sudo apt update
sudo apt install -y nvidia-driver-535
sudo nvidia-smi -mig 0
sudo reboot
```

### vllm
https://docs.vllm.ai/en/latest/getting_started/quickstart/#prerequisites

Install UV & restart shell
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install nvidia toolchain (cuda-keyring for downloading nvidia software)
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8 ccache

echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Install dependencies via python
```
source .venv/bin/activate
uv pip install -U pip
uv pip install "torch==2.10.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install packaging setuptools wheel ninja cmake numpy nixl
uv pip install -U "triton==3.5"
```

Install VLLM
```
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

```
export MODEL="facebook/opt-125m"
export HF_TOKEN=""
```

Start prefill service
```
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
CUDA_VISIBLE_DEVICES=0 \
vllm serve "$MODEL" \
--port 8100 \
--enforce-eager \
--gpu-memory-utilization 0.6 \
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
```

Start decode service
```
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve "$MODEL" \
--port 8200 \
--enforce-eager \
--gpu-memory-utilization 0.35 \
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'
```

Start the proxy
```
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --port 8192 \
    --prefiller-hosts localhost \
    --prefiller-ports 8100 \
    --decoder-hosts localhost \
    --decoder-ports 8200
```