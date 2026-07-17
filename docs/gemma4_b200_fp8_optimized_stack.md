# Gemma4 B200 FP8 Optimized vLLM Path

This branch adds a Gemma4-specific optimized FP8 path for B200/SM100 in vLLM.

The easiest deployment artifact is a prebuilt vLLM runtime image containing this
branch's rebuilt native extension and Python routing files.

```text
vllm-gemma4-b200-fp8-opt:<commit_sha>
```

A tested image was built from `vllm/vllm-openai:latest` with only the optimized
native vLLM extension and routing files patched into the installed runtime
package. It keeps the runtime image's existing FlashAttention/FlashInfer stack.

## What This Enables

- Gemma4 gated-MLP fast path for FP8 weights.
- CUTLASS SM100 tactic dispatches tuned for Gemma4 shapes.
- Fused gated-MLP GELU duplicate epilogue.
- Fused row-amax path for the activated gated output before the down projection.
- Python routing from `Gemma4MLP` into the new custom ops when the model and quantization layout match.

The changes are native CUDA/CUTLASS changes, not Python-only changes. A source
build or this prebuilt image is required.

## Fast Path: Use A Prebuilt Image

Use this when you want the optimized runtime without rebuilding vLLM.

```bash
docker run --gpus all --ipc=host --network=host \
  -e HF_HOME=/models \
  -v /path/to/model/cache:/models \
  vllm-gemma4-b200-fp8-opt:<commit_sha> \
  bash
```

Inside the container:

```bash
export HF_HOME=/models
export HF_HUB_CACHE=${HF_HOME}/hub
export OMP_NUM_THREADS=8
unset PYTHONPATH

python3 - <<'PY'
import torch
import vllm._C_stable_libtorch  # noqa: F401

for name in [
    "cutlass_scaled_mm_gemma4_gated",
    "cutlass_scaled_mm_gemma4_gated_amax",
]:
    print(name, hasattr(torch.ops._C, name))
    assert hasattr(torch.ops._C, name), name

print("registration_ok")
PY

vllm serve RedHatAI/gemma-4-31B-it-FP8-Dynamic \
  --served-model-name gemma-4 \
  --host 0.0.0.0 \
  --port 8080 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.9292 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --attention-backend FLASHINFER \
  --attention-config '{"use_trtllm_attention": true}' \
  --hf-overrides '{"text_config":{"use_bidirectional_attention":null}}' \
  --speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":3,"draft_tensor_parallel_size":1}' \
  --generation-config vllm
```

## What Runtime-Only Means

The standard `vllm/vllm-openai:latest` image is mainly a runtime image. It is
built for running an already-compiled vLLM installation with `vllm serve`.

This branch changes native CUDA/CUTLASS code under `csrc/`. Those changes only
become active after the native vLLM extensions are rebuilt. If you run the normal
vLLM image without rebuilding or without using the custom image above, you will
still be running the old precompiled kernels.

So the clean user-facing path is:

1. Build this branch once in a build-capable environment.
2. Package the built vLLM into an image.
3. Users run that image exactly like normal vLLM.

## Step-by-Step: Build From This Branch

Use this path if you want to build the optimized vLLM yourself instead of using a prebuilt image.

1. Clone vLLM and check out the optimized branch:

```bash
git clone <vllm_repo_url> vllm
cd vllm
git checkout gemma4-b200-fp8-optimized-stack
```

2. Start a build-capable CUDA/PyTorch container.

Use an image that contains a compiler, CUDA toolkit, Python headers, CMake/Ninja support, and a PyTorch/CUDA stack compatible with the target runtime. For example, on Slurm/Pyxis:

```bash
srun \
  --partition=<partition> \
  --account=<account> \
  --gres=gpu:1 \
  --cpus-per-task=56 \
  --mem=0 \
  --container-image=nvcr.io#nvidia/pytorch:26.06-py3 \
  --container-mounts=$PWD:/workspace/vllm \
  --container-workdir=/workspace/vllm \
  --pty bash
```

3. Build/install vLLM from the branch:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e . --no-build-isolation
```

If your environment requires the stable libtorch extension to be built explicitly,
use:

```bash
python3 tools/generate_cmake_presets.py --force-overwrite
cmake --preset release
cmake --build --preset release --target _C_stable_libtorch.abi3.so
cmake --install cmake-build-release --prefix "$PWD" --component _C_stable_libtorch
```

4. Verify that the optimized custom ops are registered:

```bash
python3 - <<'PY'
import torch
import vllm._C_stable_libtorch  # noqa: F401

for name in [
    "cutlass_scaled_mm_gemma4_gated",
    "cutlass_scaled_mm_gemma4_gated_amax",
]:
    print(name, hasattr(torch.ops._C, name))
    assert hasattr(torch.ops._C, name), name

print("registration_ok")
PY
```

Expected output:

```text
cutlass_scaled_mm_gemma4_gated True
cutlass_scaled_mm_gemma4_gated_amax True
registration_ok
```

5. Run vLLM with the optimized Gemma4 FP8 path:

```bash
vllm serve RedHatAI/gemma-4-31B-it-FP8-Dynamic \
  --served-model-name gemma-4 \
  --host 0.0.0.0 \
  --port 8080 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.9292 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --attention-backend FLASHINFER \
  --attention-config '{"use_trtllm_attention": true}' \
  --hf-overrides '{"text_config":{"use_bidirectional_attention":null}}' \
  --speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":3,"draft_tensor_parallel_size":1}' \
  --generation-config vllm
```

## Step-by-Step: Package As A User-Friendly Image

This is the best path for other users.

1. Build vLLM from this branch in a build-capable image.
2. Verify the optimized ops are registered.
3. Bake the resulting installed vLLM package and native `.so` files into a runtime image.
4. Tag the image with the source commit SHA:

```bash
vllm-gemma4-b200-fp8-opt:<commit_sha>
```

5. Users then run the image like normal vLLM:

```bash
vllm serve RedHatAI/gemma-4-31B-it-FP8-Dynamic ...
```

## Required Runtime Flags

- Hardware: B200 / SM100.
- Model weights: `RedHatAI/gemma-4-31B-it-FP8-Dynamic`.
- KV cache: FP8, via `--kv-cache-dtype fp8`.
- Attention backend: FlashInfer, via `--attention-backend FLASHINFER`.
- Gemma4 HF override:

```bash
--hf-overrides '{"text_config":{"use_bidirectional_attention":null}}'
```

For speculative decoding, use the Gemma assistant draft model:

```bash
--speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":3,"draft_tensor_parallel_size":1}'
```

## Verification Performed

- Native op registration passed inside the exported image:
  - `cutlass_scaled_mm_gemma4_gated True`
  - `cutlass_scaled_mm_gemma4_gated_amax True`
- Direct-image smoke validation passed with `100/100` requests completed.
- Server used:
  - RedHatAI FP8 weights: `RedHatAI/gemma-4-31B-it-FP8-Dynamic`
  - FlashInfer attention backend
  - TRT-LLM attention config
  - FP8 KV cache
  - Gemma assistant MTP speculative decoding with 3 draft tokens
- Nsight profiling was collected for the optimized path during development.

Packaging note: the branch is the source of truth; the easiest user-facing
artifact is a prebuilt image tagged with the source commit SHA.
