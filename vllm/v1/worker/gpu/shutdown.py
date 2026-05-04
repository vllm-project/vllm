# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def free_before_shutdown(vllm_config: VllmConfig) -> None:
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
    from vllm.v1.worker.workspace import reset_workspace_manager

    cache_config = vllm_config.cache_config
    cache_config.num_gpu_blocks = None

    compilation_config = vllm_config.compilation_config
    for layer in compilation_config.static_forward_context.values():
        if hasattr(layer, "kv_cache"):
            kv_cache = layer.kv_cache
            layer.kv_cache = (
                torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
            )
        if hasattr(layer, "impl"):
            if hasattr(layer.impl, "_k_scale_cache"):
                layer.impl._k_scale_cache = None
            if hasattr(layer.impl, "_v_scale_cache"):
                layer.impl._v_scale_cache = None
    compilation_config.static_forward_context.clear()

    _ROPE_DICT.clear()
    reset_workspace_manager()
