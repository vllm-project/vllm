# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early FlashInfer table loading for TP4 specialized decode."""

from pathlib import Path

from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    preload_shared_autotune_cache,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.cudagraph_utils import is_piecewise_cudagraph_available


def _can_resolve_early_autotune_key(vllm_config: VllmConfig, model: nn.Module) -> bool:
    model_config = vllm_config.model_config
    if model_config is not None and model_config.original_max_model_len == -1:
        # KV sizing can change the context limit after model load.
        return False
    compilation = vllm_config.compilation_config
    mode = compilation.cudagraph_mode
    # Under the specialized backend's requirements, unavailable piecewise
    # capture is the remaining graph-mode downgrade during KV initialization.
    return (
        mode is None
        or not mode.requires_piecewise_compilation()
        or is_piecewise_cudagraph_available(compilation, model)
    )


def preload_rank_flashinfer_autotune_cache(
    vllm_config: VllmConfig, model: nn.Module
) -> Path | None:
    """Load the legacy table before specialized decode's first tuning call."""
    from flashinfer.autotuner import AutoTuner

    from vllm.distributed.parallel_state import get_world_group

    # The block size is in the key. The executor normally resolves it after
    # load_model, but specialized warmup runs during model-state construction.
    current_platform.update_block_size_for_backend(vllm_config)
    return preload_shared_autotune_cache(
        vllm_config if _can_resolve_early_autotune_key(vllm_config, model) else None,
        AutoTuner.get(),
        get_world_group(),
    )
