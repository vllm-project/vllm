# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile hybrid GDN packed-decode and MRoPE Triton keys."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def _is_non_empty_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and value.numel() > 0


def _packed_gdn_configs(model: torch.nn.Module, model_dtype: torch.dtype):
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        PackedGdnDecodeWarmupConfig,
    )

    configs: list[PackedGdnDecodeWarmupConfig] = []
    for layer in model.modules():
        if not all(
            hasattr(layer, attr)
            for attr in (
                "num_k_heads",
                "num_v_heads",
                "head_k_dim",
                "head_v_dim",
                "tp_size",
                "A_log",
                "dt_bias",
                "kv_cache",
            )
        ):
            continue
        kv_cache = layer.kv_cache
        if not isinstance(kv_cache, (tuple, list)) or len(kv_cache) < 2:
            continue
        state = kv_cache[1]
        if not _is_non_empty_tensor(state):
            continue

        H = int(layer.num_k_heads) // int(layer.tp_size)
        HV = int(layer.num_v_heads) // int(layer.tp_size)
        K = int(layer.head_k_dim)
        V = int(layer.head_v_dim)
        configs.append(
            PackedGdnDecodeWarmupConfig(
                mixed_qkv_dtype=model_dtype,
                a_dtype=model_dtype,
                b_dtype=model_dtype,
                A_log_dtype=layer.A_log.dtype,
                dt_bias_dtype=layer.dt_bias.dtype,
                output_dtype=model_dtype,
                state_dtype=state.dtype,
                scale=K**-0.5,
                stride_mixed_qkv_tok=2 * H * K + HV * V,
                stride_a_tok=HV,
                stride_b_tok=HV,
                stride_init_state_token=state.stride(0),
                stride_final_state_token=state.stride(0),
                stride_indices_seq=1,
                H=H,
                HV=HV,
                K=K,
                V=V,
                use_qk_l2norm_in_kernel=True,
            )
        )
    return configs


def _mrope_configs(model: torch.nn.Module, model_dtype: torch.dtype):
    from vllm.model_executor.layers.rotary_embedding.mrope import (
        MropeWarmupConfig,
        MRotaryEmbedding,
    )

    configs: list[MropeWarmupConfig] = []
    for module in model.modules():
        rotary_emb = getattr(module, "rotary_emb", None)
        if not isinstance(rotary_emb, MRotaryEmbedding):
            continue
        section = rotary_emb.mrope_section
        if section is None or len(section) != 3:
            continue
        if not all(
            hasattr(module, attr) for attr in ("num_heads", "num_kv_heads", "head_dim")
        ):
            continue
        cache_dtype = rotary_emb.cos_sin_cache.dtype
        configs.append(
            MropeWarmupConfig(
                q_dtype=model_dtype,
                k_dtype=model_dtype,
                cos_dtype=cache_dtype,
                sin_dtype=cache_dtype,
                n_qh=int(module.num_heads),
                n_kh=int(module.num_kv_heads),
                head_size=int(module.head_dim),
                rotary_dim=int(rotary_emb.rotary_dim),
                mrope_section=(int(section[0]), int(section[1]), int(section[2])),
                is_interleaved=bool(rotary_emb.mrope_interleaved),
            )
        )
    return configs


def hybrid_gdn_mamba_mrope_warmup(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    model = worker.get_model()
    model_dtype = worker.vllm_config.model_config.dtype

    from vllm.model_executor.layers.rotary_embedding.mrope import _MROPE_KERNEL
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        _PACKED_GDN_DECODE_KERNEL,
    )

    _PACKED_GDN_DECODE_KERNEL.warmup(_packed_gdn_configs(model, model_dtype))
    _MROPE_KERNEL.warmup(_mrope_configs(model, model_dtype))
