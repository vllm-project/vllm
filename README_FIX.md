# Fix: init_fp8_kv_scales() crash during sleep mode wake-up

## Problem

When using vLLM sleep mode with FP8 KV cache (`--kv-cache-dtype fp8`) and tag-split wake-up:

```bash
POST /sleep?level=1           → 200 OK
POST /wake_up?tags=weights    → 200 OK
POST /wake_up?tags=kv_cache   → CRASH
```

The crash occurs because `init_fp8_kv_scales()` iterates over `self.kv_caches`
which is a nested `list[list[Tensor]]` structure. During partial wake-up,
some layers may contain unallocated tensors (`numel() == 0`), and calling
`.zero_()` on these raises an `AttributeError` or CUDA error.

## Fix

The patch adds a guard in `init_fp8_kv_scales()` that skips unallocated tensors
by checking `cache_tensor.numel() > 0` before calling `.zero_()`.

## Installation

### Option 1: Dockerfile (recommended)

```dockerfile
COPY fix_fp8_kv_scales.patch /tmp/
RUN patch -p1 -d /opt/venv/lib/python3.12/site-packages < /tmp/fix_fp8_kv_scales.patch
```

### Option 2: Runtime patch

```bash
docker exec <container> patch -p1 -d /opt/venv/lib/python3.12/site-packages \
  < /tmp/fix_fp8_kv_scales.patch
```

### Option 3: sed (if patch is not available)

```bash
sed -i \
  '/kv_caches = getattr(self, "kv_caches", \[\])/a\        # Guard against unallocated tensors during partial wake-up (tags=kv_cache)' \
  '/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py

sed -i \
  's/if cache_tensor is not None:/if cache_tensor is not None and cache_tensor.numel() > 0:/' \
  '/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
```

## Affected Files

- `vllm/v1/worker/gpu_model_runner.py`

## Verification

After applying the patch, sleep mode wake-up should complete without errors:

```bash
POST /sleep?level=1           → 200 OK
POST /wake_up                 → 200 OK (no crash)
```
