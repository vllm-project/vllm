# GLM-5.2-NVFP4 — Low-Latency (LL) serving recipe · 8×B300 · TP8 · MTP=5

Single-node 8×B300 (sm_100/103), pure tensor parallel, NVFP4 weights, fp8 KV cache,
MTP=5 speculative decode, model runner v2.

Validated on branch `glm5.2-LL` @ `8d407aee1` (vllm `0.1.dev422+g8d407aee1`).

## Versions

| Component | Version |
|---|---|
| vLLM | `0.1.dev422+g8d407aee1` (branch `glm5.2-LL`) |
| flashinfer-python | `0.6.15` |
| torch | `2.11.0+cu130` |
| nvidia-nccl-cu13 | `2.30.7` |
| CUDA | `13.0` |
| Python | `3.12` |
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

Ensure the serve environment has `nvcc`/`ninja` on `PATH` and `CUDA_HOME` set —
otherwise flashinfer's runtime JIT fails with `FileNotFoundError: ninja`.

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

First cold start is dominated by flashinfer autotune, which runs on every launch
(not cached to disk).

## Client sweep (concurrency 1→16)

```bash
for c in 1 2 4 8 16; do
  python -m sglang.bench_serving \
    --backend vllm --host localhost --port 8000 \
    --model nvidia/GLM-5.2-NVFP4 \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --random-range-ratio 1.0 \
    --num-prompts $((8*c)) --max-concurrency $c --warmup-requests $c \
    --output-file sweep_c${c}.jsonl
done
```

## Results (8192 in / 1024 out, total tok/s)

| concurrency | mnbt=8192 | mnbt=16384 |
|--:|--:|--:|
| 1 | 4891 | 4897 |
| 2 | 7369 | 7573 |
| 4 | 8731 | 11054 |
| 8 | 6623 | 15198 |
| 16 | 3682 | **18872** |

For 8192-token inputs, `--max-num-batched-tokens 16384` is required: at 8192 a single
prefill fills the entire batch budget, so prefills serialize and throughput tapers
past concurrency 4; at 16384 throughput scales monotonically (18.9k tok/s at c=16;
median TTFT/TPOT/E2E there = 1.07 s / 6.15 ms / 7.5 s).

## Notes

- Spec-decode acceptance ≈6.0 under `--dataset-name random` is a synthetic artifact
  (repetitive output is trivially predictable); a real prompt gives ≈2.0, so the
  numbers above are a best-case-for-MTP ceiling.
- The GLM-5.2 tokenizer's native `TokenizersBackend` may not load under vanilla
  `transformers`; the benchmark client needs an HF-loadable (fast) tokenizer for
  correct token accounting.
