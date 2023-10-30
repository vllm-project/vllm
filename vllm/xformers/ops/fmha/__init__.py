# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Tuple, Type, Union

import torch

#from . import cutlass, decoder, flash, small_k, triton
from .attn_bias import AttentionBias
from .common import (
    AttentionFwOpBase,
    Inputs,
)
from .dispatch import _dispatch_fw, _ensure_op_supports_or_raise


def memory_efficient_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Optional[Type[AttentionFwOpBase]] = None,
) -> torch.Tensor:
    """
    Calculates the forward pass of :attr:`xformers.ops.memory_efficient_attention`.
    """
    return _memory_efficient_attention_forward(
        Inputs(
            query=query, key=key, value=value, p=p, attn_bias=attn_bias, scale=scale
        ),
        op=op,
    )

def _memory_efficient_attention_forward(
    inp: Inputs, op: Optional[Type[AttentionFwOpBase]]
) -> torch.Tensor:
    inp.validate_inputs()
    output_shape = inp.normalize_bmhk()
    if op is None:
        op = _dispatch_fw(inp, False)
    else:
        _ensure_op_supports_or_raise(ValueError, "memory_efficient_attention", op, inp)

    out, *_ = op.apply(inp, needs_gradient=False)
    return out.reshape(output_shape)

