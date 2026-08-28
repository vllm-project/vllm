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
    # Sever each layer's alias of the KV allocation before dropping the
    # registry that holds the layer handles. The layers are submodules of the
    # model, so clearing the registry alone leaves the whole KV buffer alive
    # for as long as anything still references the model -- which, in-process,
    # outlives the engine. Mirrors GPUModelRunner V1's
    # _cleanup_profiling_kv_cache and this runner's own
    # _teardown_profiling_state.
    for layer in compilation_config.static_forward_context.values():
        if hasattr(layer, "kv_cache"):
            kv_cache = layer.kv_cache
            # Mamba layers bind a tuple of per-state views, attention a tensor.
            layer.kv_cache = (
                torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
            )
            del kv_cache
        # Quantized KV cache scale views (int8/fp8 per-token-head).
        impl = getattr(layer, "impl", None)
        if impl is not None:
            if hasattr(impl, "_k_scale_cache"):
                impl._k_scale_cache = None
            if hasattr(impl, "_v_scale_cache"):
                impl._v_scale_cache = None
    compilation_config.static_forward_context.clear()

    _ROPE_DICT.clear()
    reset_workspace_manager()
