# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The BF16 edge layers' ``pre_expert_norm`` of the pooler's gathered
``[b, 4096]`` BF16 rows as one CuTe kernel, bit for bit the stock norm
(``fused_rmsnorm_fwd`` with the layer's weight after the norm).

It is the row norm of ``common/row_norm.py`` at 4096 columns, one CTA of 256
threads per row; the weight loads before the PDL wait and the dependents
launch at the grid's end.
"""

import cutlass.cute as cute
import torch

from ..common import host
from ..common.row_norm import RowNorm, RowNormConstants

POOLED = 4096


def rows_view(tensor: torch.Tensor) -> cute.Tensor:
    """The dynamic CuTe view of contiguous BF16 rows of 4096 the norm takes."""
    return host.rows_view(tensor, POOLED)


class ExpertNorm:
    """One BF16 edge layer's ``pre_expert_norm`` (its weight applied after the
    norm), compiled and bound at construction: ``launch(rows, out, m)``
    writes ``out = norm(rows)`` for the first ``m`` rows, the stock norm's
    bits. Graph-capturable; allocates nothing."""

    def __init__(
        self, weight: torch.Tensor, eps: float, gain_center: float, *, enable_pdl: bool
    ) -> None:
        host.check_operand(weight, (POOLED,), torch.bfloat16, 16, "the norm weight")
        constants = RowNormConstants(
            width=POOLED,
            eps=float(eps),
            gain_center=float(gain_center),
            input_scale=False,
            weight_fp32=False,
            enable_pdl=bool(enable_pdl),
        )
        self.norm = RowNorm(constants, weight)

    def launch(self, rows: cute.Tensor, out: cute.Tensor, m: int) -> None:
        """The norm of the first ``m`` rows of ``rows`` into ``out`` (views
        from ``rows_view``, bound by the caller, at least ``m`` rows each)."""
        self.norm.launch(rows, out, m)
