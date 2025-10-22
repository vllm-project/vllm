# Set Environment

1. Docker Image
`
rocm/ali-private:ubuntu22.04_rocm6.4.3.127_aiter_6b586ae_vllm_5b842c2_20250911
`
2. Install dependencies

```bash
pip install flash-linear-attention
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
python3 setup.py install
```

Make sure your torch viersion is new than 2.8

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

## Launch Server

* Launch serve with TP=8 and EP

```bash
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
VLLM_ROCM_USE_AITER_MHA=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
    --port 8000 --tensor-parallel-size 8 --max-model-len 262114 --enable-expert-parallel
```

* Launch serve with Multi-token prediction

```bash
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
VLLM_ROCM_USE_AITER_MHA=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
    --port 8000 --tensor-parallel-size 4 --max-model-len 262114 \
    --force-eager \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```
