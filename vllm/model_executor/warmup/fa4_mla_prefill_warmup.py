# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 MLA prefill for every distinct layer config.

The prefill backend's registry warmup only sees the target model's
full-attention config. Sliding-window layers, NoPE layers (RoPE width folded
into qk_nope, so a different V stride) and drafters with their own head count
compile different FA4 variants, and the split-KV combine kernel is never
reached by compile-only forward warmup.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    MLADims,
)
from vllm.model_executor.warmup.runner_utils import target_and_draft_models
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.prefill.flash_attn import (
        FA4MLALayerConfig,
        FA4MLAPrefillKernel,
    )
    from vllm.v1.worker.gpu_worker import Worker


def _fa4_mla_layer_configs(worker: "Worker") -> list["FA4MLALayerConfig"]:
    # flash_attn imports FA4 at module scope, which is absent on some platforms.
    from vllm.v1.attention.backends.mla.prefill.flash_attn import FA4MLALayerConfig

    configs: dict[tuple[int, int, int, int, int], FA4MLALayerConfig] = {}
    for model in target_and_draft_models(worker):
        for layer in model.modules():
            if (
                not isinstance(layer, MLAAttention)
                or getattr(layer.prefill_backend, "vllm_flash_attn_version", None) != 4
            ):
                continue
            window = layer.sliding_window or 0
            key = (
                layer.num_heads,
                layer.qk_nope_head_dim,
                layer.qk_rope_head_dim,
                layer.v_head_dim,
                window,
            )
            if key in configs:
                continue
            configs[key] = FA4MLALayerConfig(
                num_heads=layer.num_heads,
                mla_dims=MLADims(
                    q_lora_rank=layer.q_lora_rank,
                    kv_lora_rank=layer.kv_lora_rank,
                    qk_nope_head_dim=layer.qk_nope_head_dim,
                    qk_rope_head_dim=layer.qk_rope_head_dim,
                    v_head_dim=layer.v_head_dim,
                ),
                window_size=(window - 1, 0) if window > 0 else None,
            )
    return list(configs.values())


def fa4_mla_prefill_warmup(worker: "Worker") -> None:
    if worker.model_runner.is_pooling_model:
        return
    configs = _fa4_mla_layer_configs(worker)
    if not configs:
        return

    from vllm.v1.attention.backends.mla.prefill.flash_attn import (
        _FA4_MLA_PREFILL_KERNEL,
    )

    keys = _FA4_MLA_PREFILL_KERNEL.get_warmup_keys(worker.vllm_config, configs)
    _FA4_MLA_PREFILL_KERNEL.compile_many(keys)
    _warm_split_kv_combine(keys)


def _warm_split_kv_combine(
    compile_keys: Sequence["FA4MLAPrefillKernel.CompileKey"],
) -> None:
    """Compile the split-KV combine kernel, which compile-only forward
    warmup skips. Mirror FA4's call at the end of _flash_attn_fwd, not the
    public flash_attn_combine: the cache key also covers varlen, the dynamic
    split count from the scheduler metadata and no seqused / virtual batch
    index. FA4's num_splits_heuristic caps auto splits at
    num_SMs // num_heads, which is <= 32 for >= 5 heads per rank on B200, so
    ceil(log2(num_splits)) is always the kernel's floor of 5."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd_combine

    combine_configs = {
        (key.q_dtype, key.v_shape[-1], key.q_shape[1])
        for key in compile_keys
        if key.num_splits != 1
    }
    device = torch.device(current_platform.device_type)
    # FakeTensorMode populates the native cache without launching the kernel.
    with FakeTensorMode():
        for dtype, v_head_dim, num_heads in combine_configs:
            for return_lse in (True, False):
                _flash_attn_fwd_combine(
                    torch.empty(
                        (2, 1, num_heads, v_head_dim),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.empty(
                        (2, num_heads, 1), dtype=torch.float32, device=device
                    ).transpose(-1, -2),
                    torch.empty((1, num_heads, v_head_dim), dtype=dtype, device=device),
                    torch.empty(
                        (num_heads, 1), dtype=torch.float32, device=device
                    ).transpose(-1, -2)
                    if return_lse
                    else None,
                    cu_seqlens=torch.empty((2,), dtype=torch.int32, device=device),
                    num_splits_dynamic_ptr=torch.empty(
                        (1,), dtype=torch.int32, device=device
                    ),
                )
