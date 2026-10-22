# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint dense linears with a small-M CuTe GEMV fast path.

NVFP4 (W4A16): the GEMV (``nvfp4_gemv``) reads the very tensors the FlashInfer
CuTe-DSL kernel was prepared with (ModelOpt-packed weight, 128x4-swizzled
scales), so dispatching per token count costs no extra weight memory; larger
M stays on FlashInfer. BF16: the same GEMV kernel on BF16 weights where it
beats cuBLAS, cuBLAS otherwise.
"""

from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCuteDslNvFp4W4A16LinearKernel,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from . import nvfp4_gemv

logger = init_logger(__name__)


def _swizzled_scale_buffer(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The flat 128x4-swizzled scale buffer behind FlashInfer's strided view of
    it (no copy)."""
    nbytes = (n + 127) // 128 * 128 * ((k // 16 + 3) // 4 * 4)
    storage = weight_scale.untyped_storage()
    assert (
        weight_scale.dtype == torch.float8_e4m3fn
        and weight_scale.storage_offset() == 0
        and storage.nbytes() == nbytes
    ), "FlashInfer NVFP4 scales are not the flat 128x4-swizzled buffer"
    return torch.empty(0, dtype=torch.uint8, device=weight_scale.device).set_(
        storage, 0, (nbytes,), (1,)
    )


class FlintNvFp4LinearKernel(FlashInferCuteDslNvFp4W4A16LinearKernel):
    """FlashInfer CuTe-DSL W4A16 GEMM with a small-M CuTe GEMV fast path."""

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        n, k = layer.output_size_per_partition, layer.input_size_per_partition
        assert (
            getattr(layer, "weights_padding_cols", 0) == 0
            and layer.weight.dtype == torch.uint8
            and tuple(layer.weight.shape) == (n, k // 2)
            and layer.weight.is_contiguous()
        ), "FlashInfer NVFP4 weight is not in the GEMV layout"
        layer.flint_weight_scale = _swizzled_scale_buffer(layer.weight_scale, n, k)
        # Host copy of the global scale for the large-M cuBLAS alpha.
        layer.flint_global_scale = float(layer.weight_global_scale.item())
        layer.flint_bf16 = False
        layer.flint_pdl = True
        layer.flint_nk = (n, k)
        _setup_gemv(layer, n, k, True, bf16=False)

    def apply_weights(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert bias is None
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.shape[0] >= _DEQUANT_MIN_M:
            out = _dequant_gemm(layer, x_2d)
            return out.view(*x.shape[:-1], out.shape[-1])
        if _use_gemv(layer, x_2d, silu_mul=False):
            out = nvfp4_gemv.nvfp4_gemv(
                x_2d,
                layer.weight,
                layer.flint_weight_scale,
                layer.weight_global_scale,
            )
            return out.view(*x.shape[:-1], out.shape[-1])
        return super().apply_weights(layer, x, bias)


class FlintBf16LinearMethod(UnquantizedLinearMethod):
    """BF16 linear: CuTe GEMV for small M, cuBLAS otherwise."""

    def __init__(self, pdl: bool = True) -> None:
        super().__init__()
        self.pdl = pdl

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        w = layer.weight
        n, k = w.shape
        assert w.dtype == torch.bfloat16 and w.is_contiguous()
        layer.flint_bf16 = True
        layer.flint_pdl = self.pdl
        layer.flint_nk = (n, k)
        _setup_gemv(layer, n, k, self.pdl, bf16=True)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert bias is None
        x_2d = x.reshape(-1, x.shape[-1])
        if _use_gemv(layer, x_2d, silu_mul=False):
            out = nvfp4_gemv.bf16_gemv(x_2d, layer.weight, pdl=self.pdl)
            return out.view(*x.shape[:-1], out.shape[-1])
        return super().apply(layer, x, bias)


# Token counts from which exact BF16 dequantization + cuBLAS beats FlashInfer
# CuTe-DSL in the full model (GB200). At 8192 tokens it wins in isolation
# (o_proj 107 -> 86 us) but loses in-model (dequant ~9 us under contention).
_DEQUANT_MIN_M = 16384


def _dequant_gemm(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``bf16(fp32(x @ bf16(e2m1 * e4m3).T) * global)`` via cuBLAS: the global
    scale is the GEMM alpha, applied to the FP32 accumulator."""
    w = nvfp4_gemv.dequantize_rows(layer.weight, layer.flint_weight_scale)
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)
    matmul = torch.backends.cuda.matmul
    prev = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = False  # FP32 split-K sums
    try:
        torch.addmm(out, x, w.t(), beta=0, alpha=layer.flint_global_scale, out=out)
    finally:
        matmul.allow_bf16_reduced_precision_reduction = prev
    return out


def _setup_gemv(layer: nn.Module, n: int, k: int, pdl: bool, bf16: bool) -> None:
    """Enable (and compile) the GEMV where its tuned table lists the shape."""
    layer.flint_gemv = nvfp4_gemv.tuned(n, k, bf16=bf16)
    layer.flint_gemv_silu_mul = layer.output_partition_sizes == [
        n // 2
    ] * 2 and nvfp4_gemv.tuned(n, k, silu_mul=True, bf16=bf16)
    for silu_mul in (False, True):
        if layer.flint_gemv_silu_mul if silu_mul else layer.flint_gemv:
            nvfp4_gemv.warmup(n, k, silu_mul=silu_mul, pdl=pdl, bf16=bf16)
            logger.info_once(
                "Flint dense %s%s (N=%d, K=%d) -> Flint CuTe GEMV for small M",
                "BF16" if bf16 else "W4A16",
                " + SiLU-and-mul" if silu_mul else "",
                n,
                k,
            )


def _gemv_input_ok(x: torch.Tensor) -> bool:
    # 16-byte vector loads of activation rows.
    return x.stride(-1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0


def _use_gemv(layer: nn.Module, x: torch.Tensor, silu_mul: bool) -> bool:
    """Whether the tuned dispatch table picks the GEMV for this M."""
    flag = layer.flint_gemv_silu_mul if silu_mul else layer.flint_gemv
    if not flag or x.shape[0] == 0:
        return False
    n, k = layer.flint_nk
    if nvfp4_gemv.pick_nt(n, k, x.shape[0], silu_mul, layer.flint_bf16) is None:
        return False
    assert x.dtype == torch.bfloat16 and _gemv_input_ok(x)
    return True


def gate_up_silu_mul(layer: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    """``silu_and_mul(layer(x))`` in one GEMV for small M, else None."""
    if not (
        getattr(layer, "flint_gemv_silu_mul", False)
        and x.dim() == 2
        and _use_gemv(layer, x, silu_mul=True)
    ):
        return None
    if layer.flint_bf16:
        return nvfp4_gemv.bf16_gemv(x, layer.weight, silu_mul=True, pdl=layer.flint_pdl)
    return nvfp4_gemv.nvfp4_gemv(
        x,
        layer.weight,
        layer.flint_weight_scale,
        layer.weight_global_scale,
        silu_mul=True,
        pdl=layer.flint_pdl,
    )


_DENSE_SUFFIXES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


# BF16 layers launched without PDL: the 4x larger weight prefetch of a BF16
# o_proj slows the attention tail (+0.18 ms per d1 forward with PDL).
_BF16_NO_PDL_SUFFIXES = ("o_proj",)


def install_flint_dense_kernels(model: nn.Module) -> None:
    """Route the dense linears of ``model`` through the Flint GEMV (launched
    with PDL except the BF16 ``_BF16_NO_PDL_SUFFIXES``): NVFP4 linears get
    ``FlintNvFp4LinearKernel``, unquantized BF16 projections
    (``_DENSE_SUFFIXES``) ``FlintBf16LinearMethod``. Must run before weights
    are processed after loading."""
    count = 0
    for module in model.modules():
        if not isinstance(module, LinearBase):
            continue
        method: Any = getattr(module, "quant_method", None)
        kernel = getattr(method, "kernel", None)
        if type(kernel) is FlashInferCuteDslNvFp4W4A16LinearKernel:
            method.kernel = FlintNvFp4LinearKernel(kernel.config)
            count += 1
        elif type(method) is UnquantizedLinearMethod and module.prefix.endswith(
            _DENSE_SUFFIXES
        ):
            pdl = not module.prefix.endswith(_BF16_NO_PDL_SUFFIXES)
            module.quant_method = FlintBf16LinearMethod(pdl)
            count += 1
    logger.info("Flint dense GEMV installed on %d linear layers", count)


_MERGED_SHARDS = {
    ".q_proj.": ".qkv_proj.",
    ".k_proj.": ".qkv_proj.",
    ".v_proj.": ".qkv_proj.",
    ".gate_proj.": ".gate_up_proj.",
    ".up_proj.": ".gate_up_proj.",
}


def check_merged_global_scales(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pass checkpoint tensors through, raising if the NVFP4 global scales of
    shards fused into one linear (q/k/v, gate/up) differ: the fused layer keeps
    a single global scale."""
    seen: dict[str, torch.Tensor] = {}
    for name, tensor in weights:
        if name.endswith(".weight_scale_2"):
            for shard, merged in _MERGED_SHARDS.items():
                if shard in name:
                    key = name.replace(shard, merged)
                    ref = seen.setdefault(key, tensor)
                    if not torch.equal(ref.reshape(()), tensor.reshape(())):
                        raise ValueError(
                            f"{name}: global scale {tensor.item()} differs from "
                            f"the other shards of {key} ({ref.item()})"
                        )
                    break
        yield name, tensor
