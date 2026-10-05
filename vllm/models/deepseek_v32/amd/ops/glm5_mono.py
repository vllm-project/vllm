# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-private custom op: the mono path of one GLM-5.2 decoder layer
(``MonoLive.mono_forward`` of the active ``mono.dispatch.Glm5MonoDecode``; the FlyDSL
ops are not tensors). The step decision and vLLM's fallback layer stay outside the op.
The first mono layer's ``fused_allreduce_rms_norm`` may write hidden_states / residual
in place; both outputs are fresh (never an alias of an input)."""

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def _glm5_mono_decode_layer(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.models.deepseek_v32.amd.mono.dispatch import active

    return active().forward_layer_impl(layer_idx, positions, hidden_states, residual)


def _glm5_mono_decode_layer_fake(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del positions, layer_idx
    return torch.empty_like(hidden_states), torch.empty_like(residual)


direct_register_custom_op(
    op_name="glm5_mono_decode_layer",
    op_func=_glm5_mono_decode_layer,
    mutates_args=["hidden_states", "residual"],
    fake_impl=_glm5_mono_decode_layer_fake,
)


def glm5_mono_decode_layer(positions, hidden_states, residual, layer_idx: int):
    return torch.ops.vllm.glm5_mono_decode_layer(
        positions, hidden_states, residual, layer_idx
    )
