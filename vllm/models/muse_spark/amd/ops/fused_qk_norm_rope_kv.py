# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.muse_spark.common.ops.fused_qk_norm_rope_kv import (
    fused_qk_norm_rope_kv_insert,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)


def rocm_fused_qk_norm_rope_kv_insert(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    layer_name: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> None:
    if torch.compiler.is_compiling():
        torch.ops.vllm.muse_spark_fused_qk_norm_rope_kv_insert(
            qkv,
            positions,
            cos_sin_cache,
            _encode_layer_name(layer_name),
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            eps,
        )
    else:
        fused_qk_norm_rope_kv_insert(
            qkv,
            positions,
            cos_sin_cache,
            layer_name,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            eps,
        )


def _muse_spark_fused_qk_norm_rope_kv_insert(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    layer_name: LayerNameType,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> None:
    fused_qk_norm_rope_kv_insert(
        qkv,
        positions,
        cos_sin_cache,
        _resolve_layer_name(layer_name),
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps,
    )


def _muse_spark_fused_qk_norm_rope_kv_insert_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    layer_name: LayerNameType,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> None:
    return


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="muse_spark_fused_qk_norm_rope_kv_insert",
        op_func=_muse_spark_fused_qk_norm_rope_kv_insert,
        fake_impl=_muse_spark_fused_qk_norm_rope_kv_insert_fake,
        mutates_args=["qkv"],
    )
