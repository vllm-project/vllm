# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation steering custom op.

Registered as ``torch.ops.vllm.apply_steering`` so that torch.compile
treats the entire operation as an opaque node.  The Python implementation
executes at runtime and reads ``num_decode_tokens`` from the live
:class:`~vllm.forward_context.ForwardContext`, avoiding the value being
baked into the compiled graph as a constant.
"""

import torch

from vllm.forward_context import get_num_decode_tokens
from vllm.utils.torch_utils import direct_register_custom_op


def apply_steering(
    hidden_states: torch.Tensor,
    steering_vector: torch.Tensor,
) -> torch.Tensor:
    """Apply decode-only activation steering.

    Only the first ``num_decode_tokens`` rows of *hidden_states* are
    modified.  When the forward context is unavailable (e.g. during
    profiling / warmup), the default is **0** — no tokens are steered,
    which is the safe choice for decode-only steering.
    """
    num_decode_tokens = get_num_decode_tokens(default=0)
    decode_mask = (
        torch.arange(hidden_states.shape[0], device=hidden_states.device)
        < num_decode_tokens
    ).unsqueeze(1)
    return hidden_states + decode_mask.to(hidden_states.dtype) * steering_vector.to(hidden_states.dtype)


def apply_steering_fake(
    hidden_states: torch.Tensor,
    steering_vector: torch.Tensor,
) -> torch.Tensor:
    """FX-tracing fake — returns the correct shape without reading context."""
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="apply_steering",
    op_func=apply_steering,
    fake_impl=apply_steering_fake,
)
