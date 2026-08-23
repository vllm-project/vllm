# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opaque piecewise-split custom op for the Kimi-K3 KDA mixer.

The KDA recurrence reads forward-context metadata, launches causal-conv / KDA
kernels, and mutates the per-request conv/SSM cache in place. Tracing that
stateful path through torch.compile (or capturing it as many eager Triton
launches) is what made conc=1 decode launch-bound.

Wrapping the mixer as one custom op, and listing it in
``CompilationConfig._attention_ops``, is the same pattern as Qwen GDN
(``qwen_gdn_attention_core``) and Bailing V3 KDA: inductor treats the mixer
as opaque, piecewise CUDA graphs split here, and the surrounding in_proj /
o_proj GEMMs stay inside captured graph segments.

Input/output projections stay in the caller so they remain compiled.
"""

from __future__ import annotations

import torch

from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)


def kimi_k3_kda_attention(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the KDA mixer (conv + recurrence + gated RMSNorm) in-place."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer._forward(
        mixed_qkv=mixed_qkv,
        g1=g1,
        g2=g2,
        beta=beta,
        core_attn_out=core_attn_out,
    )


def kimi_k3_kda_attention_fake(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="kimi_k3_kda_attention",
    op_func=kimi_k3_kda_attention,
    mutates_args=["core_attn_out"],
    fake_impl=kimi_k3_kda_attention_fake,
)
