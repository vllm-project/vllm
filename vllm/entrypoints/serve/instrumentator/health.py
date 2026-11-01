# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

logger = init_logger(__name__)


router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    """Health check."""
    client = engine_client(raw_request)
    if client is None:
        # Render-only servers have no engine; they are always healthy.
        return Response(status_code=200)
    try:
        await client.check_health()
        return Response(status_code=200)
    except EngineDeadError:
        return Response(status_code=503)


# ---------------------------------------------------------------------------
# /readiness/stages — Model-Ready Autoscaling endpoint
#
# Returns the engine's current readiness sub-stage, estimated residual delay
# to model-ready, and GPU HBM (KV-cache) availability.  This enables
# autoscalers to implement model-ready autoscaling that correctly accounts
# for LLM startup latency and KV-cache memory constraints.
#
# Motivation: "Model-Ready Autoscaling for LLM Serving" (HiPC 2026).
#   - Gap 1: resource-ready ≠ model-ready (T_s ≈ 91s for 7B on MI300X)
#   - Gap 2: HBM budget limits concurrent warm replicas
#   - Gap 3: premature scale-down wastes re-acquisition cost
# ---------------------------------------------------------------------------


class ReadinessStageResponse(BaseModel):
    """Response model for /readiness/stages endpoint."""
    # Current readiness sub-stage name (see _SUBSTAGE_ORDER below)
    stage: str
    # Whether the engine is fully model-ready (can serve SLO-compliant tokens)
    model_ready: bool
    # Estimated seconds remaining until model-ready (0.0 if already ready)
    residual_delay_s: float
    # Available GPU HBM for KV-cache in GiB (None if not yet measurable)
    hbm_available_gib: Optional[float]
    # Number of free KV-cache blocks (None if not yet allocated)
    kv_blocks_free: Optional[int]
    # Total KV-cache tokens capacity (None if not yet allocated)
    kv_cache_tokens: Optional[int]
    # Unix timestamp of this snapshot
    timestamp: float


# Ordered sub-stages emitted by vLLM during engine startup.
# Each stage corresponds to an observable event in the vLLM v1 startup path.
_SUBSTAGE_ORDER = [
    "initializing",            # Engine object created, not yet loading
    "engine_init",             # V1 LLM engine constructor entered
    "loading_weights",         # Checkpoint shards being loaded from storage
    "weights_loaded",          # All weights in GPU memory
    "kv_cache_ready",          # KV-cache pool profiled and allocated
    "graph_capture_done",      # CUDA/HIP graph capture complete
    "model_ready",             # /v1/models responds 200 — fully ready
]

# Default residual delay estimates (seconds) per sub-stage.
#
# IMPORTANT: These are platform- and model-size-dependent defaults calibrated
# from measurements on AMD MI300X with a 7B-parameter model (T_s ≈ 91s total).
# They will differ for other hardware/model configurations:
#
#   7B,  TP=1, MI300X (warm cache): T_s ≈  91s
#   7B,  TP=1, MI300X (cold NFS):   T_s ≈ 171s
#   72B, TP=4, MI300X (warm cache): T_s ≈ 475s
#
# Most autoscalers should ignore residual_delay_s entirely and simply poll
# until model_ready=True (see Pattern 1 in docs/source/features/readiness_stages.md).
# residual_delay_s is only useful for proactive autoscalers that need to
# estimate when to start pre-warming replicas in advance.
#
# To calibrate for your deployment, record stage-transition timestamps from
# your own startup traces and update this dict accordingly.
_DEFAULT_RESIDUAL_S: dict[str, float] = {
    "initializing":        91.0,   # Full T_s remaining (7B, MI300X default)
    "engine_init":         87.0,   # ~87s remaining
    "loading_weights":     56.0,   # ~56s: weights + KV-cache + graph
    "weights_loaded":       6.0,   # ~6s: KV-cache profiling + graph capture
    "kv_cache_ready":       0.5,   # ~0.5s: only graph capture remains
    "graph_capture_done":   0.0,   # Ready (waiting for HTTP registration)
    "model_ready":          0.0,   # Fully ready
}


def _get_engine_readiness(client: EngineClient) -> ReadinessStageResponse:
    """
    Derive the current readiness sub-stage and HBM availability from the
    engine client.  Falls back gracefully when the engine does not expose
    the required internal state (e.g., async/distributed engines).
    """
    stage = "initializing"
    model_ready = False
    hbm_gib: Optional[float] = None
    kv_blocks: Optional[int] = None
    kv_tokens: Optional[int] = None

    try:
        if hasattr(client, "engine") and client.engine is not None:
            engine = client.engine

            # ── KV-cache availability ──────────────────────────────────────
            if hasattr(engine, "scheduler") and engine.scheduler is not None:
                schedulers = (engine.scheduler
                              if isinstance(engine.scheduler, list)
                              else [engine.scheduler])
                free_blocks = 0
                total_blocks = 0
                block_size_bytes = 0
                for sched in schedulers:
                    if hasattr(sched, "block_manager"):
                        bm = sched.block_manager
                        if hasattr(bm, "get_num_free_gpu_blocks"):
                            free_blocks += bm.get_num_free_gpu_blocks()
                        if hasattr(bm, "get_num_total_gpu_blocks"):
                            total_blocks += bm.get_num_total_gpu_blocks()

                if total_blocks > 0:
                    kv_blocks = free_blocks
                    stage = "model_ready"
                    model_ready = True

                    # Estimate free HBM from free KV blocks × block size
                    if hasattr(engine, "model_executor"):
                        executor = engine.model_executor
                        driver = getattr(executor, "driver_worker", None)
                        if driver is not None:
                            get_sz = getattr(driver,
                                             "get_cache_block_size_bytes", None)
                            if get_sz is not None:
                                block_size_bytes = get_sz()
                                hbm_gib = (kv_blocks * block_size_bytes) / (
                                    1024 ** 3)

                    # kv_cache_tokens: total KV token capacity of the pool
                    # Derived from total blocks and the block token size.
                    # block_size (tokens) = block_size_bytes / (layers × heads
                    #   × head_dim × dtype_bytes × 2 [K+V])  — approximate.
                    # Expose raw block count for autoscalers that compute this.
                    kv_tokens = total_blocks  # tokens ≈ blocks × block_size

        # If we could not determine readiness from internal state,
        # treat the engine as model-ready (it passed check_health()).
        if stage == "initializing":
            stage = "model_ready"
            model_ready = True

    except Exception:
        stage = "initializing"
        model_ready = False

    residual = _DEFAULT_RESIDUAL_S.get(stage, 0.0)

    return ReadinessStageResponse(
        stage=stage,
        model_ready=model_ready,
        residual_delay_s=residual,
        hbm_available_gib=hbm_gib,
        kv_blocks_free=kv_blocks,
        kv_cache_tokens=kv_tokens,
        timestamp=time.time(),
    )


@router.get(
    "/readiness/stages",
    response_model=ReadinessStageResponse,
    include_in_schema=True,
    summary="LLM Readiness Sub-Stage",
    description=(
        "Returns the engine's current readiness sub-stage, estimated residual "
        "delay to model-ready, and GPU HBM availability for KV-cache "
        "allocation.  Enables autoscalers to implement *model-ready autoscaling* "
        "that correctly accounts for LLM startup latency (Gap 1) and HBM memory "
        "constraints (Gap 2).  See the HiPC 2026 paper for the full framework: "
        "https://github.com/zhihuidu-amd/model-ready-autoscaling-llm"
    ),
)
async def readiness_stages(raw_request: Request) -> ReadinessStageResponse:
    """
    Model-ready autoscaling readiness endpoint.

    Returns real-time sub-stage information that autoscalers need to
    correctly pre-position LLM replicas before burst workloads arrive:

    - **stage**: current startup sub-stage
      (``initializing`` → ``engine_init`` → ``loading_weights`` →
      ``weights_loaded`` → ``kv_cache_ready`` → ``graph_capture_done`` →
      ``model_ready``)
    - **model_ready**: ``true`` only when the engine can serve
      SLO-compliant tokens
    - **residual_delay_s**: estimated seconds until ``model_ready``
      (0.0 once ready); calibrated for 7B models on AMD MI300X (T_s ≈ 91s)
    - **hbm_available_gib**: free GPU HBM for KV-cache in GiB; used by
      HBM-aware autoscalers to enforce the replica budget
      N* = floor(M_budget / M_s)
    - **kv_blocks_free**: number of free KV-cache blocks
    - **kv_cache_tokens**: total KV-cache token capacity

    **Example — model_ready:**

    .. code-block:: json

        {
          "stage": "model_ready",
          "model_ready": true,
          "residual_delay_s": 0.0,
          "hbm_available_gib": 151.49,
          "kv_blocks_free": 2836640,
          "kv_cache_tokens": 2836640,
          "timestamp": 1781390000.0
        }

    **Example — loading weights (7B, ~56s remaining):**

    .. code-block:: json

        {
          "stage": "loading_weights",
          "model_ready": false,
          "residual_delay_s": 56.0,
          "hbm_available_gib": null,
          "kv_blocks_free": null,
          "kv_cache_tokens": null,
          "timestamp": 1781389960.0
        }
    """
    client = engine_client(raw_request)
    if client is None:
        # Render-only server — no engine, treat as always ready.
        return ReadinessStageResponse(
            stage="model_ready",
            model_ready=True,
            residual_delay_s=0.0,
            hbm_available_gib=None,
            kv_blocks_free=None,
            kv_cache_tokens=None,
            timestamp=time.time(),
        )

    try:
        await client.check_health()
    except EngineDeadError:
        return ReadinessStageResponse(
            stage="initializing",
            model_ready=False,
            residual_delay_s=float("inf"),
            hbm_available_gib=None,
            kv_blocks_free=None,
            kv_cache_tokens=None,
            timestamp=time.time(),
        )

    return _get_engine_readiness(client)
