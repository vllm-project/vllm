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
tokens — a gap of 40 seconds for 7B models and 17 minutes for 72B models.
This causes elasticity loss: requests arrive but cannot be served.

Two structural problems require model-aware information:

**Pillar 1 — Readiness delay:** The autoscaler must know `residual_delay_s`
(how long until model-ready) to decide when to start pre-warming replicas.
If `residual_delay_s > SLO_latency_threshold`, compute-ready autoscaling
is structurally infeasible.

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
  "timestamp": 1781390000.0
}
```

**Response during startup:**
```json
{
  "stage": "loading_weights",
  "model_ready": false,
  "residual_delay_s": 50.0,
  "hbm_available_gib": null,
  "kv_blocks_free": null,
  "timestamp": 1781389960.0
}
```

## Sub-Stage Ordering

Stages progress in this order:

| Stage | Description | Residual delay (7B, MI300X) |
|-------|-------------|----------------------------|
| `initializing` | Engine not yet loading | ∞ |
| `loading_weights` | Loading checkpoint to GPU | ~50s |
| `weights_loaded` | Weights in GPU, KV-cache pending | ~6s |
| `kv_cache_allocated` | KV-cache pool reserved | ~0.5s |
| `graph_captured` | CUDA/HIP graph capture complete | ~0s |
| `model_ready` | Serving SLO-compliant tokens | 0s |

## Autoscaler Integration

Use this endpoint to implement the **SLO feasibility condition** and
**HBM capacity constraint**:

```python
import requests

def should_prewarm(url: str, slo_s: float, burst_rps: float) -> bool:
    """Return True if model-ready pre-warming is needed."""
    r = requests.get(f"{url}/readiness/stages", timeout=5).json()

    # Pillar 1: check if residual delay exceeds SLO
    if r["residual_delay_s"] >= slo_s:
        return True  # Must pre-warm before burst

    # Pillar 2: check HBM headroom for KV-cache
    if r["hbm_available_gib"] is not None:
        # Rough estimate: 0.05 GiB per concurrent request
        kv_needed = burst_rps * 30 * 0.05  # burst_rps × SLO × per-request
        if r["hbm_available_gib"] < kv_needed:
            return False  # Not enough HBM even if model-ready

    return not r["model_ready"]
```

See the full controller implementation at:
https://github.com/zhihuidu-amd/model-ready-autoscaling-llm

## Experimental Results

Validated on OCI MI300X (192 GiB HBM) with Qwen2.5-7B and Qwen2.5-72B:

| Policy | S1-F1 λ | S2-F1 λ | Description |
|--------|---------|---------|-------------|
| Compute-ready (existing) | 0.872 | 1.000 | Wrong model |
| Model-ready oracle | 0.140 | 0.071 | Correct model, perfect knowledge |
| Model-ready predictive | 0.072 | 0.082 | Correct model, online prediction |

The `/readiness/stages` endpoint enables the predictive controller to achieve
oracle-comparable performance without advance burst knowledge.

## References

- HiPC 2026: "Minimizing Elasticity Loss in LLM Serving with
  Readiness-Stage and Downscale Policy"
- GitHub: https://github.com/zhihuidu-amd/model-ready-autoscaling-llm
- Related issue: vllm-project/vllm#6073
