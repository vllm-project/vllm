# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request activation steering custom op and hook-point definitions.

Registered as ``torch.ops.vllm.apply_steering`` so that torch.compile
treats the operation as an opaque splitting point.  The real Python
implementation executes at runtime between compiled graph segments,
reading the live buffer values rather than baked-in constants.
"""

from enum import Enum

import torch

from vllm.utils.torch_utils import direct_register_custom_op


class SteeringHookPoint(str, Enum):
    """Positions in a decoder layer where steering can be applied.

    All hook points operate on the residual stream tensor.
    """

    PRE_ATTN = "pre_attn"
    """After input_layernorm, before self_attn."""

    POST_ATTN = "post_attn"
    """After post_attention_layernorm, before pre_feedforward_layernorm."""

    POST_MLP_PRE_LN = "post_mlp_pre_ln"
    """After mlp, before post_feedforward_layernorm."""

    POST_MLP_POST_LN = "post_mlp_post_ln"
    """After post_feedforward_layernorm."""


# Buffer attribute names on decoder layer modules, keyed by hook point.
HOOK_POINT_TABLE_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "steering_table_pre_attn",
    SteeringHookPoint.POST_ATTN: "steering_table_post_attn",
    SteeringHookPoint.POST_MLP_PRE_LN: "steering_table_post_mlp_pre_ln",
    SteeringHookPoint.POST_MLP_POST_LN: "steering_table_post_mlp_post_ln",
}

HOOK_POINT_VECTOR_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "steering_vector_pre_attn",
    SteeringHookPoint.POST_ATTN: "steering_vector_post_attn",
    SteeringHookPoint.POST_MLP_PRE_LN: "steering_vector_post_mlp_pre_ln",
    SteeringHookPoint.POST_MLP_POST_LN: "steering_vector_post_mlp_post_ln",
}

# Valid hook point string values for validation.
VALID_HOOK_POINT_NAMES: frozenset[str] = frozenset(hp.value for hp in SteeringHookPoint)

DEFAULT_HOOK_POINT = SteeringHookPoint.POST_MLP_PRE_LN


def apply_steering(
    hidden_states: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
) -> torch.Tensor:
    """Apply per-request activation steering via indexed gather.

    ``steering_table`` is a per-layer buffer of shape
    ``(max_configs + 2, hidden_size)`` where row 0 is always zeros
    (prefill/no-steering sentinel), row 1 holds the global-only vector,
    and rows 2+ hold combined global + per-request vectors.

    ``steering_index`` is a shared buffer of shape ``(max_tokens,)``
    mapping each token position to its steering table row.  Updated
    in-place by the model runner before each forward pass.
    """
    return hidden_states + steering_table[steering_index[: hidden_states.shape[0]]].to(
        hidden_states.dtype
    )


def apply_steering_fake(
    hidden_states: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
) -> torch.Tensor:
    """FX-tracing fake — correct shape, no computation."""
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="apply_steering",
    op_func=apply_steering,
    fake_impl=apply_steering_fake,
    mutates_args=[],
)
