# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark draft's first norm (its first layer's pre_qkv_proj_norm of the
BF16 input rows) as one CuTe kernel, bit for bit the stock norm
(``MuseSpark1xRMSNorm.forward``: ``fused_rmsnorm_fwd`` into BF16).

It is the row norm of ``common/row_norm.py`` at 8192 columns, one CTA of 256
threads per row; the weight loads before the PDL wait and the dependents
launch at the grid's end.
"""

import torch
from torch import nn

from ..common.host import check_operand, pdl_supported, rows_view
from ..common.row_norm import H, RowNorm, RowNormConstants, row_norm_constants


def input_norm_constants(norm: nn.Module, *, enable_pdl: bool) -> RowNormConstants:
    """The constants of the stock norm ``norm`` (a MuseSpark1xRMSNorm or its
    input-scaled variant over 8192 columns), or ValueError where the kernel
    does not reproduce it or does not take its weight."""
    return row_norm_constants(norm, enable_pdl=enable_pdl)


class InputNorm:
    """The draft's first norm, compiled at construction (before CUDA graph
    capture): ``run(x)`` returns ``norm(x, out_dtype=bfloat16)`` for
    contiguous BF16 rows ``x`` [m, 8192]. ``constants``: its
    ``input_norm_constants`` where the caller has them. Graph-capturable."""

    def __init__(
        self, norm: nn.Module, constants: RowNormConstants | None = None
    ) -> None:
        if constants is None:
            constants = input_norm_constants(norm, enable_pdl=pdl_supported())
        self.norm = RowNorm(constants, norm.weight if norm.has_weight else None)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """The norm of ``x``, contiguous BF16 [m, 8192], into a new BF16
        tensor."""
        m = x.shape[0]
        check_operand(x, (m, H), torch.bfloat16, 16, "x")
        out = torch.empty_like(x)
        if m:
            self.norm.launch(rows_view(x, H), rows_view(out, H), m)
        return out
