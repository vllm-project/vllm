# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 skinny GEMM selection for tiny-M decode on SM100.

Dispatch is purely by local ``(N, K)`` shape and token count ``M`` — the
module name plays no role. The 20 (shape, M) cells in
:data:`DSV41_PROJECTIONS_SM100` were measured on B200 at TP8 shapes over a
rotating-weight CUDA graph protocol; only cells that beat the production
baseline enter the table, so unlisted shapes and token counts keep the
original kernels. Winners:

- MXFP8 projections (fused_wqa_wkv, wo_a chain, wo_b): the Triton SIMT GEMM,
  which folds the per-32-element E8M0 activation quantization into the GEMM
  (``round_mx``), matching the baseline MXFP8 quantization exactly.
- BF16 projections (indexer weights_proj/wk, lm_head): CuTe skinny GEMM, with
  one SIMT cell (weights_proj M4).
- compressor fused_wkv_wgate scores (BF16 -> FP32): ll_bf16 dotprod with
  per-M block sizes instead of ``torch.mm(out_dtype=torch.float32)``.

The BF16 modules are installed by :func:`enable_dsv41_low_latency_gemm` as
quant-method replacements (Kimi-K3 precedent); the MXFP8 and fp32-score
paths dispatch at their model-layer call sites through the ``try_*`` helpers,
which return None to fall back to the original code. Gating: SM100 only,
(N, K, M) table hit, dtype/stride runtime checks, per-call
``VLLM_BATCH_INVARIANT`` short-circuit, and no install at all when LoRA is
enabled (the model-layer branches bypass module forward, where adapters
hook). CuTe configs are precompiled via ``request_warmup_configs``; SIMT/ll
kernels are exercised once at enable time so Triton/CuTe compilation is done
before CUDA graph capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import LLBf16Gemm
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .ops.mxfp8_skinny_gemm import Mxfp8SimtGemmConfig, mxfp8_simt_gemm


@dataclass(frozen=True, slots=True)
class Dsv41ProjectionSpec:
    cute_configs: tuple[tuple[int, SkinnyGemmConfig], ...] = ()
    simt_configs: tuple[tuple[int, Mxfp8SimtGemmConfig], ...] = ()
    ll_dotprod_bs: tuple[tuple[int, int], ...] = ()
    mxfp8: bool = False
    name: str = ""  # optional debug label; never used for dispatch


# Keyed by local (N, K). Measured on B200 over M={1, 2, 4, 8, 16}; only the
# winning M cells are kept. BF16 activations entering an MXFP8 SIMT cell are
# re-quantized in-kernel; if the SIMT winner changes, the table must be
# re-measured.
DSV41_PROJECTIONS_SM100: dict[tuple[int, int], Dsv41ProjectionSpec] = {
    (1792, 5120): Dsv41ProjectionSpec(
        simt_configs=((1, Mxfp8SimtGemmConfig(1, 4)),),
        mxfp8=True,
        name="attn.fused_wqa_wkv",
    ),
    (1024, 4096): Dsv41ProjectionSpec(
        simt_configs=(
            (1, Mxfp8SimtGemmConfig(1, 1)),
            (2, Mxfp8SimtGemmConfig(1, 4)),
            (4, Mxfp8SimtGemmConfig(4, 2)),
        ),
        mxfp8=True,
        name="attn.wo_a",
    ),
    (5120, 1024): Dsv41ProjectionSpec(
        simt_configs=(
            (1, Mxfp8SimtGemmConfig(1, 8)),
            (2, Mxfp8SimtGemmConfig(2, 8)),
        ),
        mxfp8=True,
        name="attn.wo_b",
    ),
    (32, 5120): Dsv41ProjectionSpec(
        cute_configs=(
            (1, SkinnyGemmConfig(1, 160, 4, 1, 8, static_k=5120)),
            (2, SkinnyGemmConfig(2, 160, 4, 1, 8, static_k=5120)),
        ),
        simt_configs=((4, Mxfp8SimtGemmConfig(2, 4)),),
        name="attn.indexer.weights_proj",
    ),
    (128, 512): Dsv41ProjectionSpec(
        cute_configs=(
            (1, SkinnyGemmConfig(1, 64, 2, 2, 8)),
            (2, SkinnyGemmConfig(2, 64, 2, 2, 8)),
            (4, SkinnyGemmConfig(4, 64, 2, 2, 8)),
        ),
        name="attn.indexer.wk",
    ),
    (1024, 5120): Dsv41ProjectionSpec(
        ll_dotprod_bs=((1, 256), (2, 128), (4, 128)),
        name="attn.compressor.fused_wkv_wgate",
    ),
    (512, 5120): Dsv41ProjectionSpec(
        ll_dotprod_bs=((1, 256), (2, 256), (4, 128)),
        name="attn.compressor.fused_wkv_wgate",
    ),
    (16160, 5120): Dsv41ProjectionSpec(
        cute_configs=(
            (1, SkinnyGemmConfig(1, 64, 4, 1, 8, static_k=5120)),
            (2, SkinnyGemmConfig(2, 64, 4, 1, 8, static_k=5120)),
        ),
        name="lm_head",
    ),
}

_ACTIVE = False


def _is_sm100() -> bool:
    return current_platform.is_device_capability((10, 0))


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok_bf16(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and x.stride(1) == 1
        and (x.shape[0] == 1 or _is_packed_row_major(x))
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


def _runtime_ok_simt(
    x: torch.Tensor,
    w: torch.Tensor,
    sw: torch.Tensor | None,
    sx: torch.Tensor | None,
) -> bool:
    if not (
        x.dim() == 2
        and x.stride(1) == 1
        and x.is_cuda
        and w.dim() == 2
        and w.is_contiguous()
        and w.is_cuda
        and x.shape[1] == w.shape[1]
    ):
        return False
    if sw is None:
        return x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and sx is None
    return (
        w.dtype == torch.float8_e4m3fn
        and x.dtype == (torch.float8_e4m3fn if sx is not None else torch.bfloat16)
        and sw.dtype == torch.uint8
        and sw.dim() == 1
    )


def _dsv41_bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    spec = DSV41_PROJECTIONS_SM100.get((weight.shape[0], weight.shape[1]))
    if spec is not None and _runtime_ok_bf16(x, weight):
        m = x.shape[0]
        config = dict(spec.cute_configs).get(m)
        if config is not None and shape_dynamic_skinny_gemm.is_available():
            return shape_dynamic_skinny_gemm(x, weight, config)
        simt = dict(spec.simt_configs).get(m)
        if simt is not None:
            return mxfp8_simt_gemm(x, weight, None, None, simt)
    return torch.nn.functional.linear(x, weight)


def _dsv41_bf16_gemm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="dsv41_bf16_gemm",
    op_func=_dsv41_bf16_gemm,
    fake_impl=_dsv41_bf16_gemm_fake,
)


def _dsv41_simt_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    sw: torch.Tensor | None,
    sx: torch.Tensor | None,
    bm: int,
    bn: int,
    warps: int,
    out_fp32: bool,
) -> torch.Tensor:
    return mxfp8_simt_gemm(x, w, sw, sx, Mxfp8SimtGemmConfig(bm, bn, warps), out_fp32)


def _dsv41_simt_gemm_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    sw: torch.Tensor | None,
    sx: torch.Tensor | None,
    bm: int,
    bn: int,
    warps: int,
    out_fp32: bool,
) -> torch.Tensor:
    return x.new_empty(
        (x.shape[0], w.shape[0]),
        dtype=torch.float32 if out_fp32 else torch.bfloat16,
    )


direct_register_custom_op(
    op_name="dsv41_simt_gemm",
    op_func=_dsv41_simt_gemm,
    fake_impl=_dsv41_simt_gemm_fake,
)


class _TunedLlBf16Gemm(LLBf16Gemm):
    """LLBf16Gemm pinned to one measured (M, K, bs) compile key."""

    def __init__(
        self, compile_key: LLBf16Gemm.CompileKey, *, prefetch_pdl_weights: bool
    ) -> None:
        super().__init__(prefetch_pdl_weights=prefetch_pdl_weights)
        self._compile_key = compile_key

    def dispatch(self, *, M: int, K: int, N: int) -> LLBf16Gemm.CompileKey:
        return self._compile_key


_ll_dotprod_instances: dict[tuple[int, int, int], LLBf16Gemm] = {}


def _ll_dotprod(x: torch.Tensor, w: torch.Tensor, bs: int) -> torch.Tensor:
    key = (x.shape[0], x.shape[1], bs)
    instance = _ll_dotprod_instances.get(key)
    if instance is None:
        instance = _TunedLlBf16Gemm(
            LLBf16Gemm.CompileKey(backend="dotprod", M=x.shape[0], K=x.shape[1], bs=bs),
            prefetch_pdl_weights=x.shape[0] == 1,
        )
        _ll_dotprod_instances[key] = instance
    return instance(x, w)


def _dsv41_ll_fp32_gemm(x: torch.Tensor, w: torch.Tensor, bs: int) -> torch.Tensor:
    return _ll_dotprod(x, w, bs)


def _dsv41_ll_fp32_gemm_fake(x: torch.Tensor, w: torch.Tensor, bs: int) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float32)


direct_register_custom_op(
    op_name="dsv41_ll_fp32_gemm",
    op_func=_dsv41_ll_fp32_gemm,
    fake_impl=_dsv41_ll_fp32_gemm_fake,
)


def try_dsv41_skinny_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    sw: torch.Tensor | None = None,
    sx: torch.Tensor | None = None,
    out_fp32: bool = False,
) -> torch.Tensor | None:
    """Measured skinny GEMM for ``x @ w.T``, or None to fall back.

    Dispatch is by ``(w.shape[0], w.shape[1])`` and ``x.shape[0]``; ``sw`` /
    ``sx`` select the MXFP8 SIMT cells and ``out_fp32`` the ll_bf16 dotprod
    cells.
    """
    if envs.VLLM_BATCH_INVARIANT or not _ACTIVE or not _is_sm100() or w.dim() != 2:
        return None
    spec = DSV41_PROJECTIONS_SM100.get((w.shape[0], w.shape[1]))
    if spec is None or x.dim() != 2:
        return None
    if out_fp32:
        bs = dict(spec.ll_dotprod_bs).get(x.shape[0])
        if bs is None or not _runtime_ok_bf16(x, w) or not ll_bf16.is_available():
            return None
        return torch.ops.vllm.dsv41_ll_fp32_gemm(x, w, bs)
    config = dict(spec.simt_configs).get(x.shape[0])
    if config is None or not _runtime_ok_simt(x, w, sw, sx):
        return None
    return torch.ops.vllm.dsv41_simt_gemm(
        x, w, sw, sx, config.bm, config.bn, config.warps, out_fp32
    )


def _mxfp8_layer_operands(layer: nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
    weight = getattr(layer, "weight", None)
    scale = getattr(layer, "weight_scale", None)
    if (
        weight is None
        or scale is None
        or weight.dtype != torch.float8_e4m3fn
        or weight.dim() != 2
        or weight.stride() != (1, weight.shape[0])
    ):
        return None
    return weight.t(), scale


def try_fused_wqa_wkv_gemm(
    layer: nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor | None:
    operands = _mxfp8_layer_operands(layer)
    if operands is None:
        return None
    return try_dsv41_skinny_gemm(hidden_states, operands[0], sw=operands[1])


def try_wo_b_gemm(
    wo_b: nn.Module, z: torch.Tensor, *, einsum_recipe: tuple[int, int, int]
) -> torch.Tensor | None:
    """Skinny wo_b GEMM plus the RowParallel forward tail, or None.

    The replacement sits before the reduce, so SP (``reduce_results=False``)
    is unaffected.
    """
    if einsum_recipe != (1, 1, 32) or getattr(wo_b, "bias", None) is not None:
        return None
    operands = _mxfp8_layer_operands(wo_b)
    if operands is None:
        return None
    output = try_dsv41_skinny_gemm(z, operands[0], sw=operands[1])
    if output is None:
        return None
    if wo_b.reduce_results and wo_b.tp_size > 1:
        return tensor_model_parallel_all_reduce(output)
    return output


def try_compressor_kv_score_gemm(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    return try_dsv41_skinny_gemm(hidden_states, weight, out_fp32=True)


def try_wo_a_chain_gemm(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor | None:
    """Inverse-RoPE + wo_a GEMM chain for M in {1, 2, 4}, or None.

    Runs the production inverse-RoPE kernel unquantized and lets the SIMT
    GEMM apply the per-32-element E8M0 quantization in-kernel — the same
    rounding math as the quantized ``(1, 1, 32)`` einsum baseline, verified
    over input scales 1e-5..100.
    """
    if (
        einsum_recipe != (1, 1, 32)
        or not tma_aligned_scales
        or n_groups != 1
        or heads_per_group * (nope_dim + rope_dim) != 4096
        or o_lora_rank != 1024
        or o.dim() != 3
        or o.dtype != torch.bfloat16
    ):
        return None
    weight = getattr(wo_a, "weight", None)
    swizzled = getattr(wo_a, "_dsv41_wo_a_scale", None)
    if (
        weight is None
        or swizzled is None
        or weight.shape != (1, 1024, 4096)
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
    ):
        return None
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    rotated, _ = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=einsum_recipe[2],
        tma_aligned_scales=tma_aligned_scales,
        quantize=False,
    )
    x = rotated.view(o.shape[0], -1)
    return try_dsv41_skinny_gemm(x, weight[0], sw=swizzled)


class _Dsv41LowLatencyApply:
    """Mixin: run the shape-dispatched custom op, else the base method."""

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return torch.ops.vllm.dsv41_bf16_gemm(x, layer.weight)
        return super().apply(layer, x, bias)  # type: ignore[misc]


class Dsv41LowLatencyLinearMethod(_Dsv41LowLatencyApply, UnquantizedLinearMethod):
    pass


class Dsv41LowLatencyEmbeddingMethod(_Dsv41LowLatencyApply, UnquantizedEmbeddingMethod):
    pass


def _warmup_measured_kernels() -> None:
    """Compile the SIMT/ll cells once so capture never hits a first call."""
    if not torch.accelerator.is_available():
        return
    device = torch.device(f"cuda:{torch.accelerator.current_device_index()}")
    for (n, k), spec in DSV41_PROJECTIONS_SM100.items():
        for m, config in spec.simt_configs:
            x = torch.zeros((m, k), dtype=torch.bfloat16, device=device)
            if spec.mxfp8:
                w = torch.zeros((n, k), dtype=torch.float8_e4m3fn, device=device)
                rows, cols = (n + 127) // 128 * 128, (k + 127) // 128 * 4
                sw = torch.zeros(rows * cols, dtype=torch.uint8, device=device)
            else:
                w, sw = torch.zeros((n, k), dtype=torch.bfloat16, device=device), None
            mxfp8_simt_gemm(x, w, sw, None, config)
    if ll_bf16.is_available():
        for (n, k), spec in DSV41_PROJECTIONS_SM100.items():
            for m, bs in dict.fromkeys(spec.ll_dotprod_bs):
                x = torch.zeros((m, k), dtype=torch.bfloat16, device=device)
                w = torch.zeros((n, k), dtype=torch.bfloat16, device=device)
                _ll_dotprod(x, w, bs)


def enable_dsv41_low_latency_gemm(module: nn.Module, vllm_config: VllmConfig) -> None:
    """Install shape-dispatched low-latency GEMMs for unquantized BF16 modules.

    Modules are matched by type, an exactly-unquantized method, and a local
    ``(N, K)`` present in :data:`DSV41_PROJECTIONS_SM100`. No-ops off SM100,
    for non-BF16 models, or when LoRA is enabled.
    """
    global _ACTIVE
    if vllm_config.lora_config is not None:
        return
    dtype = vllm_config.model_config.dtype
    if dtype != torch.bfloat16 or not current_platform.is_device_capability((10, 0)):
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        is_linear = (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        )
        is_head = (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        )
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
            continue
        spec = DSV41_PROJECTIONS_SM100.get((weight.shape[0], weight.shape[1]))
        if spec is None or not (spec.cute_configs or spec.simt_configs):
            # ll-only rows (compressor scores) dispatch at the model layer.
            continue
        if is_linear:
            child.quant_method = Dsv41LowLatencyLinearMethod()
        else:
            child.quant_method = Dsv41LowLatencyEmbeddingMethod()
        warmup_configs.update(config for _, config in spec.cute_configs)

    if shape_dynamic_skinny_gemm.is_available() and warmup_configs:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)
    _warmup_measured_kernels()
    _ACTIVE = True


def prepare_dsv41_wo_a_scales(model: nn.Module) -> None:
    """Rearrange wo_a weights per-row packed scales for the SIMT chain once.

    The packed per-row UE8M0 layout (``(1, 1024, 32)``) belongs to the
    DeepGEMM BMM path; the SIMT kernel needs the F8_128x4 swizzle, so it is
    unpacked and swizzled here, off the hot path, into a non-persistent
    buffer read by :func:`try_wo_a_chain_gemm`.
    """
    from .ops.mxfp8_skinny_gemm import swizzle_wo_a_packed_scale

    if not _ACTIVE:
        return
    for child in model.modules():
        wo_a = getattr(child, "wo_a", None)
        if wo_a is None:
            continue
        weight = getattr(wo_a, "weight", None)
        packed = getattr(wo_a, "weight_scale", None)
        if (
            weight is None
            or packed is None
            or weight.shape != (1, 1024, 4096)
            or weight.dtype != torch.float8_e4m3fn
            or packed.shape != (1, 1024, 32)
            or packed.dtype.is_floating_point
        ):
            continue
        wo_a.register_buffer(
            "_dsv41_wo_a_scale",
            swizzle_wo_a_packed_scale(packed[0]),
            persistent=False,
        )
