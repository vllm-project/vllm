# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 MLA prefill for every distinct layer config.

The prefill backend's registry warmup only sees the target model's
full-attention config. Sliding-window layers, NoPE layers (RoPE width folded
into qk_nope, so a different V stride) and drafters with their own head count
compile different FA4 variants. The sweep collects those configs and runs
them through the same kernel warmup path as the registry.
"""

from typing import TYPE_CHECKING

from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    MLADims,
)
from vllm.model_executor.warmup.runner_utils import target_and_draft_models

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.prefill.flash_attn import FA4MLALayerConfig
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
