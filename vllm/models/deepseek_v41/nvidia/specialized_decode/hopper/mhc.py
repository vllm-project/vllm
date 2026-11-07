# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mHC pre of the Hopper executor's boundaries: ``mhc_pre_delayed_overlap``
with the CuTe DSL mixes kernel (``mhc_cute``) in place of the DeepGEMM prenorm
GEMM, on the executor's own side stream.

The layer input (the collapse of the streams with the carried pre-mix,
normalized) is ready on the caller's stream; the coefficients (post mix,
residual mix, next pre-mix) on the side stream, after the caller's work.
Both are the stock mHC epilogue's (``MHC_PRE_NORM_KERNEL``), in its "input"
and "stats" modes."""

from typing import TYPE_CHECKING

import torch
from cutlass import Int32, Int64

from . import mhc_cute

if TYPE_CHECKING:
    from ...model import DeepseekV4DecoderLayer


def _address(t: torch.Tensor) -> Int64:
    return Int64(t.data_ptr())


class SeamPre:
    """The mHC pre of one seam (attention or FFN entry) of one layer."""

    def __init__(self, layer: "DeepseekV4DecoderLayer", ffn: bool) -> None:
        seam = "ffn" if ffn else "attn"
        self.fn = getattr(layer, f"hc_{seam}_fn")
        self.scale = getattr(layer, f"hc_{seam}_scale")
        self.base = getattr(layer, f"hc_{seam}_base")
        self.norm = layer.ffn_norm if ffn else layer.attn_norm
        self.fields = dict(
            hidden_size=mhc_cute.HIDDEN,
            rms_eps=layer.rms_norm_eps,
            hc_pre_eps=layer.hc_eps,
            hc_sinkhorn_eps=layer.hc_eps,
            hc_post_mult_value=layer.hc_post_alpha,
            sinkhorn_repeat=layer.hc_sinkhorn_iters,
            norm_eps=self.norm.variance_epsilon,
            hc_mult=mhc_cute.HC,
            save_pre_mix=True,
            rms_numel=mhc_cute.HC * mhc_cute.HIDDEN,
        )

    @staticmethod
    def applies(layer: "DeepseekV4DecoderLayer") -> bool:
        """Whether the layer has the kernel's shapes: 4 hc streams of 5120 and
        FP32 ``fn`` of both seams."""
        shape = (mhc_cute.MIXES, mhc_cute.HC * mhc_cute.HIDDEN)
        return getattr(layer, "hc_mult", None) == mhc_cute.HC and all(
            isinstance(fn := getattr(layer, f"hc_{seam}_fn", None), torch.Tensor)
            and fn.dtype == torch.float32
            and tuple(fn.shape) == shape
            and fn.is_contiguous()
            for seam in ("attn", "ffn")
        )

    def __call__(
        self,
        residual: torch.Tensor,
        pre_mix: torch.Tensor | None,
        stream: torch.cuda.Stream,
        layer_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Post mix, residual mix, layer input and next pre-mix of the streams
        ``residual`` [M, 4, 5120]; with ``layer_input``, that input as given
        (the fused all-reduce wrote it). ``pre_mix`` None selects stream zero
        (the first layer)."""
        from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL
        from vllm.models.common.ops.moe_tail_fusion.primitives import (
            current_cu_stream,
        )

        m, hc, hidden = residual.shape
        assert residual.is_contiguous() and residual.dtype == torch.bfloat16
        post = residual.new_empty(m, hc, dtype=torch.float32)
        comb = residual.new_empty(m, hc * hc, dtype=torch.float32)
        next_pre = torch.empty_like(post)
        compute_input = layer_input is None
        if layer_input is None:
            layer_input = residual.new_empty(m, hidden)
        mix = residual.new_empty(
            mhc_cute.MIX_SPLITS, m, mhc_cute.MIXES, dtype=torch.float32
        )
        sqr = residual.new_empty(mhc_cute.MIX_SPLITS, m, dtype=torch.float32)

        def epilogue(split_mode: str) -> None:
            MHC_PRE_NORM_KERNEL(
                mix,
                sqr,
                self.scale,
                self.base,
                residual,
                post,
                comb,
                layer_input,
                self.norm.weight,
                pre_mix if pre_mix is not None else post,
                next_pre,
                layer_input,  # Unused aux output in split modes.
                use_pre_mix_in=pre_mix is not None,
                split_mode=split_mode,
                **self.fields,
            )

        if compute_input:
            epilogue("input")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mhc_cute.compiled(mhc_cute.tiles_of(m))(
                _address(self.fn),
                _address(residual),
                _address(mix),
                _address(sqr),
                Int32(m),
                current_cu_stream(),
            )
            epilogue("stats")
        for tensor in (residual, mix, sqr, layer_input):
            tensor.record_stream(stream)
        return post.unsqueeze(-1), comb.view(m, hc, hc), layer_input, next_pre

    def warm_up(self, rows: list[int], first: bool = False) -> None:
        """Compile the kernels of calls of ``rows`` rows (``first``: the first
        layer's, without a carried pre-mix)."""
        from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL

        for tiles in {mhc_cute.tiles_of(m) for m in rows}:
            mhc_cute.compiled(tiles)
        for use_pre_mix_in in {not first, True}:
            for split_mode in ("input", "stats"):
                MHC_PRE_NORM_KERNEL.compile(
                    MHC_PRE_NORM_KERNEL.dispatch(
                        n_splits=mhc_cute.MIX_SPLITS,
                        use_pre_mix_in=use_pre_mix_in,
                        split_mode=split_mode,
                        **self.fields,
                    )
                )
