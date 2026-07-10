# AMMO optimizations — nvidia/Gemma-4-26B-A4B-NVFP4 on RTX PRO 6000 Blackwell (sm_120)

Two shipped, accuracy-gated vLLM optimizations discovered by an automated
kernel-optimization campaign. Both are **opt-in via env flags that default OFF**,
so this branch is a strict superset of the base tree — with both flags unset it is
byte-for-byte behaviourally identical to upstream `ee0da84ab`.

| Op | Flag | What | Correctness (vs pure-bf16) |
|----|------|------|-----------------------------|
| `fp8-qkv`     | `VLLM_QKV_FP8_REQUANT=1`  | FP8-e4m3 requant of dense `self_attn.qkv_proj`, in-tree SM120 FP8 CUTLASS GEMM | cosine 0.99930 / relL2 0.0372 / GSM8K 0.4238 |
| `w8a16-oproj` | `VLLM_OPROJ_FP8_W8A16=1`  | Weight-only FP8 W8A16 (FP8-Marlin) on dense `self_attn.o_proj`                 | cosine 0.99964 / relL2 0.0269 / GSM8K 0.3995 |

**Accuracy floor** (hard gate for these lossy ops): GSM8K ≥ 0.3873 (pure-bf16 baseline − 1.0pp). Both pass.

## Headline result (end-to-end, full-model latency sweep)

| Config | bs1 latency | bs8 latency | bs1 speedup | bs8 speedup |
|--------|-------------|-------------|-------------|-------------|
| baseline (both flags OFF) | 6.4746 s | 8.9865 s | 1.000× | 1.000× |
| `fp8-qkv` only            | 6.0396 s | 8.7040 s | 1.072× | 1.032× |
| **both (stacked)**        | **5.8308 s** | **8.4063 s** | **1.110×** | **1.069×** |

Base commit: `ee0da84ab9e04ac7610e28580af62c365e898389` (vLLM `v0.24.0` tag).
Workload: input-len 1024, output-len 1024, TP=1, dtype bf16 (weights NVFP4).

> These are **true end-to-end** numbers from `vllm bench latency`, NOT isolated-kernel
> microbenchmarks. `w8a16-oproj`'s standalone GEMM is actually *slower* cold-cache; the
> E2E win comes from halved HBM weight traffic during real decode.

---

## Reproduce it yourself on a g7e instance

Target: **`g7e.2xlarge`** (1× RTX PRO 6000 Blackwell, sm_120, ~96 GB) in a region
that has it (e.g. `us-west-2`). A larger g7e (more vCPU) builds vLLM faster but the
GPU/latency numbers are per-GPU so they will match.

### 0. Prereqs
- Ubuntu 22.04/24.04 DLAMI (or any image with NVIDIA driver ≥ the CUDA-13 line + Docker).
- ~40 GB free disk for the model, ~60 GB for the vLLM source build.
- A HuggingFace token if you haven't cached the model (the model repo is public but HF
  rate-limits anonymous pulls). **Never commit your token** — export it in the shell only.

### 1. Clone this branch
```bash
git clone --branch ammo/gemma4-nvfp4-fp8-qkv-oproj \
  https://github.com/access2rohit/vllm.git
cd vllm
git log --oneline -3   # expect: w8a16-oproj, fp8-qkv, then base ee0da84ab
```

### 2. Build vLLM from source (matches the campaign build)
Use the official vLLM build container so the toolchain matches. From the repo root:
```bash
export HF_TOKEN=<your_hf_token>   # shell-only, do NOT persist to a file in the repo

# Build in the vllm CUDA build image (adjust tag to the CUDA-13 build image you use):
docker run --gpus all -it --rm \
  -v "$PWD":/workspace/vllm -w /workspace/vllm \
  -e HF_TOKEN="$HF_TOKEN" \
  nvcr.io/nvidia/pytorch:25.06-py3 bash -lc '
    pip install -e . --no-build-isolation
    python -c "import vllm; print(\"vLLM\", vllm.__version__)"
  '
```
(If you already have a working vLLM dev environment for sm_120, just
`pip install -e . --no-build-isolation` in it — no container needed.)

### 3. Warm the model cache once (optional but avoids first-run download skew)
```bash
huggingface-cli download nvidia/Gemma-4-26B-A4B-NVFP4 --quiet
```

### 4. Run the three latency configs
Run each config; compare the `Avg latency` line. `--num-iters 10` matches the campaign.

```bash
COMMON="--model nvidia/Gemma-4-26B-A4B-NVFP4 --tensor-parallel-size 1 \
  --max-model-len 4096 --input-len 1024 --output-len 1024 --num-iters 10 --trust-remote-code"

# (a) BASELINE — both flags OFF
vllm bench latency $COMMON --batch-size 1
vllm bench latency $COMMON --batch-size 8

# (b) fp8-qkv only
VLLM_QKV_FP8_REQUANT=1 vllm bench latency $COMMON --batch-size 1
VLLM_QKV_FP8_REQUANT=1 vllm bench latency $COMMON --batch-size 8

# (c) BOTH stacked
VLLM_QKV_FP8_REQUANT=1 VLLM_OPROJ_FP8_W8A16=1 vllm bench latency $COMMON --batch-size 1
VLLM_QKV_FP8_REQUANT=1 VLLM_OPROJ_FP8_W8A16=1 vllm bench latency $COMMON --batch-size 8
```

Expect ≈ **1.11× at bs1** and ≈ **1.07× at bs8** for config (c) vs (a). Run-to-run
latency noise on a shared box is ~0.5%; the wins are well above that (Welch p<0.01).

### 5. Verify accuracy (GSM8K) — confirms the ops are lossy-but-safe
Using `lm-eval` (or your GSM8K harness of choice), greedy, 1319 questions,
max-tokens 1024:
```bash
pip install lm-eval

# baseline
lm_eval --model vllm \
  --model_args pretrained=nvidia/Gemma-4-26B-A4B-NVFP4,tensor_parallel_size=1,trust_remote_code=True,max_model_len=4096 \
  --tasks gsm8k --batch_size auto

# stacked (accuracy must stay >= 0.3873)
VLLM_QKV_FP8_REQUANT=1 VLLM_OPROJ_FP8_W8A16=1 lm_eval --model vllm \
  --model_args pretrained=nvidia/Gemma-4-26B-A4B-NVFP4,tensor_parallel_size=1,trust_remote_code=True,max_model_len=4096 \
  --tasks gsm8k --batch_size auto
```
Note: GSM8K greedy churns ~1pp run-to-run; the gate is an **absolute floor of 0.3873**,
not a per-run delta. Both ops clear it (0.4238 / 0.3995 measured).

### 6. Confirm the kernels actually fired (anti-no-op)
The flags log a one-time marker at model load. Grep the server/engine stderr:
```
"VLLM_QKV_FP8_REQUANT active"      # fp8-qkv
"VLLM_OPROJ_FP8_W8A16 active"      # w8a16-oproj
```
Or profile with nsys and look for `marlin::Marlin` launches (30 = one per decoder layer's o_proj):
```bash
nsys profile -o /tmp/opt vllm bench latency \
  <COMMON, both flags=1, --batch-size 1>
nsys stats --report cuda_gpu_kern_sum /tmp/opt.nsys-rep | grep -i marlin
```

---

## Caveats (honest)
- This was scoped to the **dense attention projections** only. The MoE expert GEMMs and
  dense MLP (the bulk of the model's compute/weight mass) were **not** optimized — there
  is very likely more speedup available there.
- `w8a16-oproj`'s isolated-kernel micro-benchmark is a NULL (Marlin is slower standalone
  at tiny M); the E2E win is real and comes from reduced HBM traffic in decode.
- GSM8K at ~0.39 is a low absolute score for this model/harness config; "above floor"
  measures that the op didn't *degrade* accuracy, not that the model is strong at GSM8K.

See `fp8-qkv_validation.md`, `w8a16-oproj_validation.md`, and `AMMO_REPORT.md` in this
directory for the full gate-by-gate evidence.
