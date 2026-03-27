# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-only activation steering custom op.

Registered as ``torch.ops.vllm.apply_steering`` so that torch.compile
treats the operation as an opaque splitting point.  The real Python
implementation executes at runtime between compiled graph segments,
reading the live buffer values rather than baked-in constants.
"""

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def apply_steering(
    hidden_states: torch.Tensor,
    steering_vector: torch.Tensor,
    decode_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply decode-only activation steering.

    ``decode_mask`` is a persistent buffer (shape ``(max_tokens, 1)``)
    with 1.0 for decode positions and 0.0 elsewhere.  It is updated
    in-place by the model runner before each forward pass.
    """
    return (
        hidden_states
        + decode_mask[: hidden_states.shape[0]]
        * steering_vector.to(hidden_states.dtype)
    )


def apply_steering_fake(
    hidden_states: torch.Tensor,
    steering_vector: torch.Tensor,
    decode_mask: torch.Tensor,
) -> torch.Tensor:
    """FX-tracing fake — correct shape, no computation."""
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="apply_steering",
    op_func=apply_steering,
    fake_impl=apply_steering_fake,
    mutates_args=[],
)
