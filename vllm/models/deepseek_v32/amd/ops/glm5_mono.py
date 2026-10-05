# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-private custom op for the GLM-5.2 decode MonoKernel layer. The FlyDSL op
objects are not tensors, so the op looks the active ``mono.dispatch.Glm5MonoDecode`` up
by registry.

Contract:
  * the op is only the mono path of one layer (``MonoLive.mono_forward``); the per-step
    go / no-go decision and the fallback to vLLM's layer happen outside it, in
    ``Glm5MonoDecode.forward_layer``;
  * ``mutates_args=["hidden_states", "residual"]``: on the first mono layer
    ``fused_allreduce_rms_norm`` may write either in place; later layers only read;
  * both outputs are fresh (never an alias of an input): ``x_out`` is allocated per
    launch and the zeros output is the layer's own persistent zero buffer.
The fake implementation returns fresh tensors of the same shapes / dtypes."""

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
