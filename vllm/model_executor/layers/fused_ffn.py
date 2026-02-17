# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused FFN custom op for piecewise CUDA graph with dynamic-shape FFN.

Registers `vllm::fused_silu_ffn` as an opaque custom op so that the FFN
(MLP) forward becomes a graph-splitting point in piecewise CUDA graph mode.
This lets the FFN run eagerly with the actual (unpadded) batch size,
reducing activation memory from unnecessary batch padding.
"""

import torch

from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op


# ---------------------------------------------------------------------------
# Custom op implementations
# ---------------------------------------------------------------------------
def fused_silu_ffn(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    """Real implementation — called eagerly at runtime.

    Looks up the MLP module by *layer_name* from the forward context
    (registered in static_forward_context during model init) and delegates
    to its forward method. Temporarily sets use_direct_call=True to avoid
    recursion (this op is only invoked when use_direct_call is False).
    """
    forward_context = get_forward_context()
    mlp = forward_context.no_compile_layers[layer_name]
    mlp.use_direct_call = True
    try:
        return mlp(x)
    finally:
        mlp.use_direct_call = False


def fused_silu_ffn_fake(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    """Fake implementation — used by torch.compile tracing.

    Returns an empty tensor with the same shape/dtype/device as the input,
    since the FFN output has the same hidden_size as the input.
    """
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="fused_silu_ffn",
    op_func=fused_silu_ffn,
    fake_impl=fused_silu_ffn_fake,
)
