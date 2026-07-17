---
name: check-precompiled-wheels
description: Check if pre-compiled wheels are available for the current commit before using VLLM_USE_PRECOMPILED=1.
---

# Checking Pre-Compiled Wheels Availability

When installing vLLM with `VLLM_USE_PRECOMPILED=1`, wheels must already be built for your commit. This skill verifies availability before installation to avoid failures or unexpected compilation.

## Quick Check

```bash
# Get your current commit
git rev-parse HEAD

# Check if wheels are built at https://wheels.vllm.ai/{commit}/{cuda-variant}/vllm/
# Common variants: cu118, cu121, cu124

# Example: manually check cu121 wheels for commit abc123def456
curl -sI https://wheels.vllm.ai/abc123def456/cu121/vllm/ | head -1
# If you get "200 OK", wheels are available. If "404 Not Found", they're not ready yet.
```

## Using in a Script

```bash
#!/bin/bash
COMMIT=$(git rev-parse HEAD)
VARIANT="cu121"

# Check if wheels exist
if curl -sf "https://wheels.vllm.ai/${COMMIT}/${VARIANT}/vllm/" > /dev/null 2>&1; then
    echo "✅ Wheels available for $VARIANT"
    VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
else
    echo "❌ Wheels not ready yet. Compiling..."
    uv pip install -e . --torch-backend=auto
fi
```

## CUDA Variants

- `cu118` — CUDA 11.8
- `cu121` — CUDA 12.1  
- `cu124` — CUDA 12.4

Check your CUDA version: `nvidia-smi` or `nvcc --version`

## Troubleshooting

**Wheels take 5-15 minutes to build** after a commit is pushed. If they're not immediately available, wait a few minutes and check again.

**Check the wheels directory directly:**
```bash
COMMIT=$(git rev-parse HEAD)
echo "https://wheels.vllm.ai/${COMMIT}/cu121/vllm/"
# Open in browser or curl it
```

## When to use VLLM_USE_PRECOMPILED=1

✅ **DO use it** when wheels are confirmed available (curl returns 200)
❌ **DON'T use it** when wheels are not ready (404) — falls back to slow compilation
