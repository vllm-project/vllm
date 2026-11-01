---
name: check-precompiled-wheels
description: Check if pre-compiled wheels are available for the current commit before using VLLM_USE_PRECOMPILED=1.
---

# Checking Pre-Compiled Wheels Availability

When installing vLLM with `VLLM_USE_PRECOMPILED=1`, wheels must already be built for your commit. This skill verifies availability before installation to avoid failures or unexpected compilation.

## Quick Check

```bash
# Detect your system's CUDA version
CUDA_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
# Convert to wheel variant (e.g., 12.1 → cu121, 11.8 → cu118)
VARIANT="cu${CUDA_VERSION//./}"

# Get your current commit
COMMIT=$(git rev-parse HEAD)

# Check if wheels are built (only worth checking if commit is recent)
COMMIT_AGE=$(($(date +%s) - $(git log -1 --format=%ct)))
if [ $COMMIT_AGE -lt 7200 ]; then  # Within 2 hours
    curl -sI "https://wheels.vllm.ai/${COMMIT}/${VARIANT}/vllm/" | head -1
    # If you get "200 OK", wheels are available. If "404 Not Found", they're not ready yet.
fi
```

## Using in a Script

```bash
#!/bin/bash
COMMIT=$(git rev-parse HEAD)
COMMIT_AGE=$(($(date +%s) - $(git log -1 --format=%ct)))

# Detect CUDA version from nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    VARIANT="cu${CUDA_VERSION//./}"
else
    echo "⚠️  nvidia-smi not found, skipping wheel check"
    uv pip install -e . --torch-backend=auto
    exit 0
fi

# Only check wheels for recent commits (avoid unnecessary requests)
if [ $COMMIT_AGE -lt 7200 ]; then  # 2 hours
    if curl -sf "https://wheels.vllm.ai/${COMMIT}/${VARIANT}/vllm/" > /dev/null 2>&1; then
        echo "✅ Wheels available for $VARIANT"
        VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
    else
        echo "❌ Wheels not ready yet. Compiling..."
        uv pip install -e . --torch-backend=auto
    fi
else
    echo "⏭️  Commit is older than 2 hours, wheels likely built. Installing..."
    VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
fi
```

## Troubleshooting

**nvidia-smi not found:** The script skips wheel checks on machines without NVIDIA GPUs. Install normally without `VLLM_USE_PRECOMPILED=1`.

**Wheels take 5-15 minutes to build** after a commit is pushed. For commits older than 2 hours, wheels are assumed to be available and installation proceeds with `VLLM_USE_PRECOMPILED=1`. For newer commits, the script checks before installing.

**Manually check a specific commit:**
```bash
COMMIT=$(git rev-parse HEAD)
VARIANT="cu121"  # Replace with your CUDA variant
curl -sI "https://wheels.vllm.ai/${COMMIT}/${VARIANT}/vllm/" | head -1
# "200 OK" = available, "404 Not Found" = not ready yet
```
