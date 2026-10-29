# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm the selected Mamba2 SSD prefill scan."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@torch.inference_mode()
def mamba_ssd_warmup(runner: "GPUModelRunner") -> None:
    """Warm the SSD scan of every Mamba2 layer."""
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    forward_context = runner.vllm_config.compilation_config.static_forward_context
    for group in runner.kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.first_spec
        if not (
            isinstance(spec, MambaSpec)
            and spec.mamba_type == MambaAttentionBackendEnum.MAMBA2
        ):
            continue
        for layer_name in group.layer_names:
            layer = forward_context[layer_name]
            if isinstance(layer, MambaMixer2):
                layer._warmup_ssd_kernels()
