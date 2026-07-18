# GLM-5.2-NVFP4 — Low-Latency (LL) serving recipe · 8×B300 · TP8 · MTP=5

Single-node 8×B300 (sm_100/103), pure tensor parallel, NVFP4 weights, fp8 KV cache,
MTP=5 speculative decode, model runner v2.

Validated on branch `glm5.2-LL` @ `8d407aee1` (vllm `0.1.dev422+g8d407aee1`).

## Versions

| Component | Version |
|---|---|
| vLLM | `0.1.dev422+g8d407aee1` (branch `glm5.2-LL`) |
| flashinfer-python | `0.6.15` |
| Model | `nvidia/GLM-5.2-NVFP4` (`modelopt_fp4`) |

## Build (once)

```bash
# in the vllm source tree, branch glm5.2-LL, with the venv activated
# flashinfer 0.6.15 (its cubin package only ships <=0.6.13, so remove it and let
# 0.6.15 JIT/fetch its own kernels at runtime)
pip install 'flashinfer-python==0.6.15'
rm -rf "$(python -c 'import site;print(site.getsitepackages()[0])')"/flashinfer_cubin*
# the flashinfer upgrade displaces the NCCL pin — restore it
pip install --no-deps 'nvidia-nccl-cu13==2.30.7'
# build tools
pip install cmake setuptools_rust setuptools_scm

# rebuild extensions in place (no dep churn; scope to Blackwell; Rust frontend skipped)
CUDA_HOME=/usr/local/cuda-13.0 \
TORCH_CUDA_ARCH_LIST="10.0 10.3" MAX_JOBS=96 \
VLLM_USE_PRECOMPILED=0 VLLM_USE_PRECOMPILED_RUST=0 \
pip install -e . --no-deps --no-build-isolation
```

flashinfer JIT needs `nvcc`/`ninja` on `PATH` and `CUDA_HOME` set.

## Server

```bash
export VLLM_USE_V2_MODEL_RUNNER=1          # model runner v2
export VLLM_DEEP_GEMM_WARMUP=skip

vllm serve nvidia/GLM-5.2-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
  --max-model-len 32768 --max-num-batched-tokens 16384 --max-num-seqs 256 \
  --no-enable-prefix-caching --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --kernel-config '{"ir_op_priority":{"rms_norm":["vllm_c","native"],"fused_add_rms_norm":["vllm_c","native"]}}' \
  --host 0.0.0.0 --port 8000
```
