# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The first decoder layer's input as one CuTe kernel: the embedded tokens
into the FP32 stream (and the aux state) and the layer's pre_qkv_proj_norm of
the stream into the attention input.

Uses the row norm of ``common/row_norm.py``, bit for bit with the stock norm.
With PDL the kernel releases its dependents right after its wait; the QKV front
waits for this grid before it reads the attention input.
"""

import torch
from torch import nn

from ..common.host import check_operand, pdl_supported, rows_view
from ..common.row_norm import H, RowNorm, row_norm_constants


class LayerFront:
    """The first layer's front: what the stock forward does with the embedded
    tokens ``x`` before the QKV front, bitwise.

    ``run(x, stream, out, aux)`` writes ``stream = x.float()``, ``aux =
    stream`` (an aux layer) and ``out = norm(stream)`` in BF16, where ``norm``
    is the layer's pre_qkv_proj_norm on its fused-kernel path. ``x`` is BF16
    (the Omni class's eager embedding) or FP32 (the text class's in-graph
    embedding). Both input dtypes' kernels compile at construction (before
    CUDA graph capture); then it is graph-capturable and allocates nothing.
    """

    INPUT_DTYPES = (torch.bfloat16, torch.float32)

    def __init__(self, norm: nn.Module, *, aux: bool) -> None:
        self.aux = aux
        self.enable_pdl = pdl_supported()
        weight = norm.weight if norm.has_weight else None
        self.norms = {
            dtype: RowNorm(
                row_norm_constants(
                    norm,
                    enable_pdl=self.enable_pdl,
                    x_fp32=dtype == torch.float32,
                    stream=True,
                    aux=aux,
                    release_early=True,
                ),
                weight,
            )
            for dtype in self.INPUT_DTYPES
        }

    @classmethod
    def build(cls, ctx, layer: nn.Module, rows: int) -> "LayerFront":
        front = cls(
            layer.self_attn.pre_qkv_proj_norm, aux=layer.layer_idx in ctx.aux_layers
        )
        front.warm_up(ctx.device, rows)
        return front

    def warm_up(self, device: torch.device, rows: int) -> None:
        """Run both input dtypes' kernels once at ``rows`` rows (before CUDA
        graph capture)."""
        stream = torch.empty(rows, H, dtype=torch.float32, device=device)
        out = torch.empty(rows, H, dtype=torch.bfloat16, device=device)
        aux = torch.empty_like(stream) if self.aux else None
        for dtype in self.INPUT_DTYPES:
            x = torch.zeros(rows, H, dtype=dtype, device=device)
            self.run(x, stream, out, aux)
        torch.accelerator.synchronize(device)

    def run(
        self,
        x: torch.Tensor,
        stream: torch.Tensor,
        out: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> None:
        rows = x.shape[0]
        if x.dtype not in self.INPUT_DTYPES:
            raise ValueError(f"x must be one of {self.INPUT_DTYPES}")
        # Each thread moves 8 consecutive elements per access: 32 bytes for
        # FP32 rows, 16 for BF16.
        for tensor, dtype, name in (
            (x, x.dtype, "x"),
            (stream, torch.float32, "stream"),
            (out, torch.bfloat16, "out"),
        ):
            check_operand(tensor, (rows, H), dtype, 8 * dtype.itemsize, name)
        if (aux is not None) != self.aux:
            raise ValueError(f"aux is {'required' if self.aux else 'not built'}")
        if aux is not None:
            check_operand(aux, (rows, H), torch.float32, 32, "aux")
        if rows == 0:
            return
        self.norms[x.dtype].launch(
            rows_view(x, H),
            rows_view(out, H),
            rows,
            rows_view(stream, H),
            rows_view(aux, H) if aux is not None else None,
        )
