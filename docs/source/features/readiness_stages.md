# Model-Ready Autoscaling: `/readiness/stages` Endpoint

## Overview

The `/readiness/stages` endpoint exposes vLLM's internal engine readiness
state to autoscalers, enabling **model-ready autoscaling** — a policy that
correctly accounts for LLM-specific startup latency and KV-cache memory
constraints.

## Motivation

Existing autoscalers treat GPU allocation as equivalent to service readiness.
For LLM serving, this is wrong: a replica passes through weight loading,
KV-cache allocation, and graph capture before it can serve SLO-compliant
tokens. This causes elasticity loss: requests arrive but cannot be served.

Two structural problems require model-aware information:

**Pillar 1 — Readiness delay:** The autoscaler must know `residual_delay_s`
(how long until model-ready) to decide when to start pre-warming replicas.

**Pillar 2 — KV-cache memory:** Even a model-ready replica cannot serve a
burst if `hbm_available_gib` is insufficient for the request's KV-cache.
The autoscaler must check `M_active + M_s + M_KV ≤ M_budget` before
committing to a warm replica count.

## Usage

```bash
curl http://localhost:8000/readiness/stages
```

**Response when model-ready:**
```json
{
  "stage": "model_ready",
  "model_ready": true,
  "residual_delay_s": 0.0,
  "hbm_available_gib": 151.49,
  "kv_blocks_free": 2836640,
  "kv_cache_tokens": 2836640,
  "timestamp": 1781390000.0
}
```

**Response during startup:**
```json
{
  "stage": "loading_weights",
  "model_ready": false,
  "residual_delay_s": 56.0,
  "hbm_available_gib": null,
  "kv_blocks_free": null,
  "kv_cache_tokens": null,
  "timestamp": 1781389960.0
}
```

## Sub-Stage Ordering

Stages progress in this order:

| Stage | Description | Default residual¹ |
|-------|-------------|-------------------|
| `initializing` | Engine not yet loading | ∞ |
| `engine_init` | V1 engine constructor entered | ~87s |
| `loading_weights` | Loading checkpoint shards to GPU | ~56s |
| `weights_loaded` | All weights in GPU, KV-cache pending | ~6s |
| `kv_cache_ready` | KV-cache pool profiled and allocated | ~0.5s |
| `graph_capture_done` | CUDA/HIP graph capture complete | ~0s |
| `model_ready` | Serving SLO-compliant tokens | 0s |

¹ **Default residual delays are illustrative defaults only**, calibrated from
measurements on AMD MI300X with a 7B-parameter model (T_s ≈ 91s total).
They will differ for other hardware, model sizes, and configurations:

| Setup | Approx. T_s |
|-------|-------------|
| 7B, TP=1, AMD MI300X (warm page cache) | ~91s |
| 7B, TP=1, AMD MI300X (cold NFS) | ~171s |
| 72B, TP=4, AMD MI300X (warm page cache) | ~475s |
| Your deployment | Measure with `startup_path.json` |

**Most autoscalers should ignore `residual_delay_s` entirely** and simply
poll until `model_ready: true`. The residual is only useful for
*proactive* autoscalers that want to start warming replicas before a
predicted burst arrives (e.g., T_s seconds in advance).

### Calibrating residuals for your deployment

If you need accurate residuals, measure your own startup traces:

```python
import json, time, requests

# Record timestamps at each stage transition
start = time.time()
while True:
    r = requests.get("http://localhost:8000/readiness/stages").json()
    print(f"{time.time()-start:.1f}s  stage={r['stage']}")
    if r["model_ready"]:
        break
    time.sleep(5)
```

The `vllm serve` output also logs per-phase timing in structured form
when `--log-level=info` is set. The `residual_delay_s` values in
`_DEFAULT_RESIDUAL_S` (in `health.py`) can be overridden by operators
who patch the file with their own measurements.

## Two autoscaler patterns

### Pattern 1 — Polling (simple, recommended)

Poll `/readiness/stages` every few seconds and wait for `model_ready: true`.
No residual calibration needed.

```python
import requests, time

def wait_until_model_ready(url: str, timeout_s: float = 600) -> bool:
    """Block until model-ready or timeout. Returns True if ready."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/readiness/stages", timeout=5).json()
            print(f"  stage={r['stage']}  model_ready={r['model_ready']}")
            if r["model_ready"]:
                return True
        except Exception:
            pass
        time.sleep(10)
    return False
```

### Pattern 2 — Proactive pre-warming (advanced)

Use `residual_delay_s` to fire pre-warming T_s seconds before a predicted
burst. Requires calibrated residuals for your deployment.

```python
import requests

def should_prewarm(url: str, slo_s: float, burst_rps: float) -> bool:
    """Return True if model-ready pre-warming is needed."""
    r = requests.get(f"{url}/readiness/stages", timeout=5).json()

    # Check residual delay vs SLO (calibrate residual for your setup)
    if r["residual_delay_s"] >= slo_s:
        return True  # Must pre-warm: model won't be ready in time

    # Check HBM headroom for KV-cache (Pillar 2)
    if r["hbm_available_gib"] is not None:
        # Rough estimate: 0.05 GiB per concurrent request
        kv_needed = burst_rps * slo_s * 0.05
        if r["hbm_available_gib"] < kv_needed:
            return False  # Not enough HBM even if model-ready

    return not r["model_ready"]
```

See the full controller implementation (with EMA-based burst prediction and
HBM-aware replica packing) at:
https://github.com/zhihuidu-amd/model-ready-autoscaling-llm

## Experimental Results

Validated on AMD MI300X (192 GiB HBM) with Qwen2.5-7B:

| Policy | Normalized λ_T | Description |
|--------|---------------|-------------|
| Compute-ready (existing) | 0.626 | Starts serving before model-ready |
| Model-ready (this endpoint) | 0.011 | Waits for `model_ready: true` |

The endpoint eliminates elasticity loss from premature serving without any
changes to the autoscaler's core logic — just poll until `model_ready: true`.

## References

- HiPC 2026: "Model-Ready Autoscaling for LLM Serving"
- GitHub: https://github.com/zhihuidu-amd/model-ready-autoscaling-llm
- Related issue: vllm-project/vllm#6073
