# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-batch routed-expert GEMMs for DeepSeek-V4.1 decode (1-8 rows).

At concurrency 1 with DSpark, every target MoE layer applies its routed experts
to six rows (384 experts, top-6; at TP4 every rank holds all experts at
intermediate 576, padded to 640). ``TrtLlmMxfp4ExpertsModular`` runs FlashInfer's
TRT-LLM-gen routing kernel and two batched MXFP8 x MXFP4 GEMMs; at this size they
stream the weights of about 22 experts per layer, so their cost is HBM time plus
fixed per-kernel latency. This module computes the same values from the same
tensors with CuTe DSL kernels sized for at most eight rows::

    router logits ─► front ───────────────────────┐ (top-k, MXFP8 input, routing)
    top-k ids, MXFP8 input ─► route ──────────────┤
                                                  ▼
                         GEMM1 ─► h (MXFP8) ─► GEMM2 ─► unweighted permuted rows
                                                          │
       UnfinalizedMoEOutput (fused all-reduce + mHC) ◄────┘

* ``routed_expert_route_cute.py`` builds the expert-major, 8-row-tile
  permutation and the per-tile record both GEMMs read, as FlashInfer's routing
  kernel does with precomputed ids and an 8-token tile.
* ``routed_expert_gemm_cute.py`` holds GEMM1 and GEMM2: persistent 128 x 8
  ``tcgen05`` block-scaled tiles over the (weight tile, token tile) work items
  of the record, with evict-first hints on the expert weights. GEMM1 skips the
  zero padding tile of the 576-wide slice. Their numerics are those of the
  TRT-LLM-gen ``t128x8`` tactics.
* ``routed_expert_front_cute.py`` replaces the DSv4 top-k (Triton), the MXFP8
  quantization of the routed input and the route kernel with one launch whose
  outputs are bitwise those of the three; it is built for the target's 384
  experts and top-6.

The kernels read the tensors vLLM prepares for the TRT-LLM-gen MXFP4 backend:
``w13`` [E, 1280, 2560] u8 (interleaved up/gate rows shuffled for 128-row
epilogue tiles) with [E, 1280, 160] scales in the 128x4 layout, ``w2``
[E, 5120, 320] u8 with [E, 5120, 20] scales, and the per-expert SwiGLU clamp.
Each kernel is compiled once per device when the layers are bound (shapes,
expert counts and the routing record are launch arguments) and launched on the
current stream with programmatic dependent launch, so CUDA graphs capture
them like any other kernel.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_fake_stream, make_ptr

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

from ...ops.cute_dsl.primitives import current_cu_stream
from .routed_expert_front_cute import RoutedExpertFront
from .routed_expert_gemm_cute import (
    RoutedExpertGemm1,
    RoutedExpertGemm2,
    tile_scale_bytes,
)
from .routed_expert_route_cute import RoutedExpertRoute

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEPrepareAndFinalizeModular,
    )

logger = init_logger(__name__)

HIDDEN = 5120
SLICE = 576
PADDED = 640
# The fused front is built for exactly the target layers' routing.
EXPERTS = 384
TOP_K = 6
# The route kernel and the GEMMs take any multiple of 32 local experts up to
# MAX_EXPERTS, and any top-k whose 8 rows fit the route kernel.
MAX_EXPERTS = 384
MAX_TOP_K = 8
TILE = 8
MAX_ROWS = 8


def max_ctas(rows: int, top_k: int, experts: int) -> int:
    """The most 8-row token tiles any routing of ``rows * top_k`` slots needs."""
    slots = rows * top_k
    filled = min(experts, slots)
    return filled + (slots - filled) // TILE


def cta_words(ctas: int) -> int:
    """``ctas`` rounded up to a multiple of four (the record's table width)."""
    return (ctas + 3) // 4 * 4


def route_words(ctas: int) -> int:
    """int32 length of the routing record for ``ctas`` token tiles."""
    return 4 + 2 * cta_words(ctas) + TILE * ctas


@dataclass
class FrontConfig:
    """The router's DSv4 top-k parameters the fused front applies."""

    bias: torch.Tensor  # FP32 [384] score correction bias
    bias_vl: torch.Tensor | None  # FP32 [384] image-token bias, None without
    image_sentinel_lo: int  # first of the five image sentinel token ids
    scale: float  # routed scaling factor of the normalized weights


@dataclass
class Routing:
    """What GEMM1 and GEMM2 read, from ``route`` or ``front``."""

    x_quant: torch.Tensor  # MXFP8 activations [rows, 5120]
    x_scale: torch.Tensor  # their UE8M0 scales, linear layout
    topk_ids: torch.Tensor  # int32 [rows, top_k]
    topk_weights: torch.Tensor  # FP32 [rows, top_k]
    permuted: torch.Tensor  # int32 [rows, top_k], slot -> permuted row
    record: torch.Tensor  # int32 routing record (routed_expert_route_cute.py)
    ctas: int  # token tiles the GEMMs' records allow for


class RoutedExpertGemm:
    """The route kernel (or fused front) and GEMM pair bound to one MoE
    layer's TRT-LLM-gen MXFP4 experts (see the module docstring)."""

    def __init__(
        self,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        clamp: torch.Tensor,
        top_k: int,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        quant_config: FusedMoEQuantConfig,
    ):
        check_weights(w13, w13_scale, w2, w2_scale, clamp, top_k)
        self.experts = int(w13.size(0))
        self.top_k = top_k
        index = w13.device.index
        self.device_index = (
            torch.accelerator.current_device_index() if index is None else index
        )
        # Compile the kernels now, not inside a CUDA graph capture.
        _route_kernel(self.device_index)
        _gemm1_kernel(self.device_index)
        _gemm2_kernel(self.device_index)
        # The modular kernel's input quantization, reused so the activations
        # match its path bit for bit.
        self.prepare_finalize = prepare_finalize
        self.quant_config = quant_config
        self.front_config: FrontConfig | None = None
        self.w13, self.w13_scale, self.w2, self.w2_scale = w13, w13_scale, w2, w2_scale
        self.clamp = clamp

    def set_front(self, config: FrontConfig) -> None:
        """Use the fused front with the router's ``config``."""
        _front_kernel(self.device_index, config.bias_vl is not None)
        self.front_config = config

    def accepts(self, x: torch.Tensor, runner: torch.nn.Module) -> bool:
        """Whether a call with router input ``x`` can take these kernels: 1-8
        BF16 rows, and the runner's expert weights still the tensors bound here
        (a weight reload replaces them)."""
        return (
            1 <= x.size(0) <= MAX_ROWS
            and x.dtype == torch.bfloat16
            and tuple(x.shape) == (x.size(0), HIDDEN)
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and runner.routed_experts.w13_weight.data_ptr() == self.w13.data_ptr()
        )

    def route(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Routing:
        """Quantize ``x`` as the modular kernel does and launch the route kernel
        on the router's top-k (``topk_weights`` FP32)."""
        rows, top_k = topk_ids.shape
        x_quant, x_scale, *_ = self.prepare_finalize.prepare(
            x,
            topk_weights,
            topk_ids,
            self.experts,
            None,
            False,
            self.quant_config,
            defer_input_quant=False,
        )
        assert x_scale is not None and x_scale.is_contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        ctas = max_ctas(rows, top_k, self.experts)
        permuted = torch.empty((rows, top_k), dtype=torch.int32, device=x.device)
        record = torch.empty(route_words(ctas), dtype=torch.int32, device=x.device)
        _route_kernel(self.device_index)(
            *(_pointer(cutlass.Int32, t, 4) for t in (ids, permuted, record)),
            Int32(rows),
            Int32(top_k),
            Int32(self.experts),
            current_cu_stream(),
        )
        return Routing(x_quant, x_scale, ids, topk_weights, permuted, record, ctas)

    def front(
        self, x: torch.Tensor, logits: torch.Tensor, input_ids: torch.Tensor | None
    ) -> Routing:
        """Top-k, MXFP8 quantization and routing record in one launch.

        ``x``: BF16 router input [rows, 5120]; ``logits``: FP32 router logits
        [rows, 384]; ``input_ids``: [rows] int32 or int64, read only with image
        routing.
        """
        config = self.front_config
        assert config is not None
        rows = x.size(0)
        device = x.device
        ctas = max_ctas(rows, TOP_K, self.experts)
        topk_ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
        topk_weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=device)
        x_quant = torch.empty((rows, HIDDEN), dtype=torch.float8_e4m3fn, device=device)
        x_scale = torch.empty((rows, HIDDEN // 32), dtype=torch.uint8, device=device)
        permuted = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
        record = torch.empty(route_words(ctas), dtype=torch.int32, device=device)
        image = config.bias_vl is not None
        ids = topk_ids
        if image:
            assert input_ids is not None
            ids = input_ids
        _front_kernel(self.device_index, image)(
            _pointer(cutlass.Float32, logits, 16),
            _pointer(cutlass.Float32, config.bias, 16),
            _pointer(cutlass.Float32, config.bias_vl if image else config.bias, 16),
            _pointer(cutlass.Int32, ids, 4),
            Int32(int(ids.dtype == torch.int64)),
            Int32(config.image_sentinel_lo),
            Float32(config.scale),
            _pointer(cutlass.Uint32, x, 16),
            _pointer(cutlass.Int32, topk_ids, 16),
            _pointer(cutlass.Float32, topk_weights, 16),
            _pointer(cutlass.Uint8, x_quant, 16),
            _pointer(cutlass.Uint8, x_scale, 16),
            _pointer(cutlass.Int32, permuted, 16),
            _pointer(cutlass.Int32, record, 16),
            Int32(rows),
            current_cu_stream(),
        )
        return Routing(x_quant, x_scale, topk_ids, topk_weights, permuted, record, ctas)

    def gemms(self, routing: Routing) -> UnfinalizedMoEOutput:
        """Launch GEMM1 and GEMM2 right after the kernel that built ``routing``."""
        return self.gemm2(routing, *self.gemm1(routing))

    def gemm1(self, routing: Routing) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch GEMM1; returns the intermediate H and its block scales (one
        128-row scale atom per token tile, rows 0-7 used)."""
        device = routing.x_quant.device
        padded_rows = TILE * routing.ctas
        h_values = torch.empty((padded_rows, PADDED), dtype=torch.uint8, device=device)
        h_scales = torch.empty(
            routing.ctas * tile_scale_bytes(PADDED), dtype=torch.uint8, device=device
        )
        _gemm1_kernel(self.device_index)(
            _pointer(cutlass.Float4E2M1FN, self.w13, 32),
            _pointer(cutlass.Float8E8M0FNU, self.w13_scale, 16),
            _pointer(cutlass.Float8E4M3FN, routing.x_quant, 16),
            _pointer(cutlass.Uint32, routing.x_scale, 16),
            _pointer(cutlass.Uint16, h_values, 16),
            _pointer(cutlass.Uint8, h_scales, 16),
            _pointer(cutlass.Int32, routing.record, 16),
            _pointer(cutlass.Float32, self.clamp, 16),
            Int32(routing.x_quant.size(0)),
            Int32(self.experts),
            Int32(cta_words(routing.ctas)),
            Int32(padded_rows),
            Int32(_sm_count(self.device_index)),
            current_cu_stream(),
        )
        return h_values, h_scales

    def gemm2(
        self, routing: Routing, h_values: torch.Tensor, h_scales: torch.Tensor
    ) -> UnfinalizedMoEOutput:
        """Launch GEMM2 on GEMM1's outputs."""
        rows, top_k = routing.topk_ids.shape
        gemm2_rows = torch.empty(
            (rows * top_k * TILE, HIDDEN),
            dtype=torch.bfloat16,
            device=routing.x_quant.device,
        )
        _gemm2_kernel(self.device_index)(
            _pointer(cutlass.Float4E2M1FN, self.w2, 32),
            _pointer(cutlass.Float8E8M0FNU, self.w2_scale, 16),
            _pointer(cutlass.Float8E4M3FN, h_values, 16),
            _pointer(cutlass.Float8E8M0FNU, h_scales, 16),
            _pointer(cutlass.BFloat16, gemm2_rows, 16),
            _pointer(cutlass.Int32, routing.record, 16),
            Int32(TILE * routing.ctas),
            Int32(self.experts),
            Int32(cta_words(routing.ctas)),
            Int32(_sm_count(self.device_index)),
            current_cu_stream(),
        )
        return UnfinalizedMoEOutput(
            gemm2_permuted=gemm2_rows,
            expert_weights=routing.topk_weights.contiguous().view(rows, top_k),
            expanded_idx_to_permuted_idx=routing.permuted,
        )


def _pointer(dtype: type[cutlass.Numeric], tensor: torch.Tensor, align: int):
    """A CuTe DSL global-memory pointer to ``tensor``'s data."""
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _fake(dtype: type[cutlass.Numeric], align: int):
    """A placeholder pointer for compilation."""
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


@cache
def _route_kernel(device_index: int):
    """``RoutedExpertRoute`` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertRoute(),
            _fake(cutlass.Int32, 4),
            _fake(cutlass.Int32, 4),
            _fake(cutlass.Int32, 4),
            Int32(1),
            Int32(TOP_K),
            Int32(MAX_EXPERTS),
            make_fake_stream(),
        )


@cache
def _front_kernel(device_index: int, image: bool):
    """``RoutedExpertFront`` compiled for a device, with or without image
    routing."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertFront(image),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Int32, 4),
            Int32(0),
            Int32(0),
            Float32(1.0),
            _fake(cutlass.Uint32, 16),
            _fake(cutlass.Int32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Uint8, 16),
            _fake(cutlass.Uint8, 16),
            _fake(cutlass.Int32, 16),
            _fake(cutlass.Int32, 16),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _gemm1_kernel(device_index: int):
    """``RoutedExpertGemm1`` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertGemm1(HIDDEN, SLICE, PADDED),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E8M0FNU, 16),
            _fake(cutlass.Float8E4M3FN, 16),
            _fake(cutlass.Uint32, 16),
            _fake(cutlass.Uint16, 16),
            _fake(cutlass.Uint8, 16),
            _fake(cutlass.Int32, 16),
            _fake(cutlass.Float32, 16),
            Int32(1),
            Int32(MAX_EXPERTS),
            Int32(4),
            Int32(TILE),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _gemm2_kernel(device_index: int):
    """``RoutedExpertGemm2`` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertGemm2(HIDDEN, PADDED),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E8M0FNU, 16),
            _fake(cutlass.Float8E4M3FN, 16),
            _fake(cutlass.Float8E8M0FNU, 16),
            _fake(cutlass.BFloat16, 16),
            _fake(cutlass.Int32, 16),
            Int32(TILE),
            Int32(MAX_EXPERTS),
            Int32(4),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _sm_count(device_index: int) -> int:
    """Streaming multiprocessors of a device: the persistent GEMMs' CTAs."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def check_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    clamp: torch.Tensor | None,
    top_k: int,
) -> None:
    """Raise ``ValueError`` unless the weights have the layout the kernels read
    and the routing fits the route kernel."""
    experts = w13.size(0)
    for tensor, shape in (
        (w13, (experts, 2 * PADDED, HIDDEN // 2)),
        (w13_scale, (experts, 2 * PADDED, HIDDEN // 32)),
        (w2, (experts, HIDDEN, PADDED // 2)),
        (w2_scale, (experts, HIDDEN, PADDED // 32)),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.element_size() != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"expected a contiguous {shape} 1-byte tensor, got "
                f"{tuple(tensor.shape)} {tensor.dtype}"
            )
    if w13.data_ptr() % 32 or w2.data_ptr() % 32:
        raise ValueError("packed expert weights must be 32-byte aligned")
    if w13_scale.data_ptr() % 16 or w2_scale.data_ptr() % 16:
        raise ValueError("weight scales must be 16-byte aligned")
    if experts % 32 or not 0 < experts <= MAX_EXPERTS:
        raise ValueError(f"{experts} local experts")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top-k {top_k}")
    if (
        clamp is None
        or clamp.dtype != torch.float32
        or tuple(clamp.shape) != (experts,)
    ):
        raise ValueError("no FP32 [E] SwiGLU clamp limit")
    # GEMM1 never reads w13's padding tile (intermediate 576..639), so the
    # padding must be zero, as must the matching K columns of w2.
    if (
        torch.count_nonzero(w13.view(torch.uint8)[:, 2 * SLICE :]).item()
        or torch.count_nonzero(w2.view(torch.uint8)[:, :, SLICE // 2 :]).item()
    ):
        raise ValueError("the intermediate padding of w13/w2 is not zero")


def _unsupported_reason(runner: torch.nn.Module) -> str | None:
    """Why the MoE runner's routed experts cannot take the GEMM pair."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
        TrtLlmMxfp4ExpertsModular,
    )

    kernel = runner.routed_experts.quant_method.moe_kernel
    impl = kernel.fused_experts if kernel is not None else None
    if not isinstance(impl, TrtLlmMxfp4ExpertsModular):
        return f"experts kernel {type(impl).__name__}"
    config = impl.moe_config
    quant = impl.quant_config
    if config.activation != MoEActivation.SILU:
        return f"activation {config.activation}"
    if impl.gemm1_alpha is not None or impl.gemm1_beta is not None:
        return "gemm1 alpha/beta"
    if impl.w1_bias is not None or impl.w2_bias is not None:
        return "expert biases"
    if impl.local_num_experts != config.num_experts:
        return "expert parallelism"
    if runner.routed_scaling_factor != 1.0:
        return "routed scaling"
    if quant.quant_dtype != "mxfp8" or quant.is_scale_swizzled:
        return "activations are not MXFP8 with linear scales"
    return None


def _front_config(runner: torch.nn.Module) -> FrontConfig | str:
    """The fused front's configuration for the runner's router, or why its
    routing is not the plain DSv4 top-k the front computes."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        FusedTopKBiasRouter,
    )

    router = runner.router
    if not isinstance(router, FusedTopKBiasRouter):
        return f"router {type(router).__name__}"
    if router.scoring_func != "sqrtsoftplus" or not router.renormalize:
        return "routing is not renormalized sqrt-softplus"
    if router.top_k != TOP_K or router.global_num_experts != EXPERTS:
        return f"top-{router.top_k} of {router.global_num_experts} experts"
    if router._hash_indices_table is not None or router.num_fused_shared_experts:
        return "hash routing or fused shared experts"
    if runner.gate is None or runner._fse_fuse_gate:
        return "no plain router gate"
    bias, bias_vl = router.e_score_correction_bias, router.bias_vl
    for tensor in (bias, bias_vl):
        if tensor is not None and (
            tensor.dtype != torch.float32
            or tuple(tensor.shape) != (EXPERTS,)
            or not tensor.is_contiguous()
        ):
            return f"bias is not FP32 [{EXPERTS}]"
    if bias is None:
        return "no score correction bias"
    if router.image_sentinel_lo <= 0:
        bias_vl = None
    return FrontConfig(
        bias=bias.data,
        bias_vl=None if bias_vl is None else bias_vl.data,
        image_sentinel_lo=0 if bias_vl is None else router.image_sentinel_lo,
        scale=float(router.routed_scaling_factor),
    )


def front_applies(gemm: RoutedExpertGemm, runner: torch.nn.Module) -> bool:
    """Whether the fused front may replace the router's top-k this call: the
    router's biases are the tensors the front reads, and no routed-expert
    capture hook (which only ``select_experts`` feeds) is set."""
    config = gemm.front_config
    router = runner.router
    if (
        config is None
        or router.capture_fn is not None
        or router._routing_replay_out is not None
    ):
        return False
    bias, bias_vl = router.e_score_correction_bias, router.bias_vl
    return (
        bias is not None
        and bias.data_ptr() == config.bias.data_ptr()
        and (
            config.bias_vl is None
            or (bias_vl is not None and bias_vl.data_ptr() == config.bias_vl.data_ptr())
        )
    )


def build_routed_expert_gemms(
    runners: list[torch.nn.Module],
) -> list[RoutedExpertGemm | None]:
    """A ``RoutedExpertGemm`` for each MoE runner that supports it (None for the
    others), with the fused front where the router allows it. Needs the MoE
    kernels that the loader's per-layer quantization finalize builds."""
    reasons: Counter[str] = Counter()
    gemms: list[RoutedExpertGemm | None] = []
    fronts = 0
    for runner in runners:
        gemm = None
        reason = _unsupported_reason(runner)
        if reason is not None:
            reasons[reason] += 1
        else:
            routed = runner.routed_experts
            kernel = routed.quant_method.moe_kernel
            impl = kernel.fused_experts
            try:
                gemm = RoutedExpertGemm(
                    routed.w13_weight,
                    impl.w1_scale,
                    routed.w2_weight,
                    impl.w2_scale,
                    impl.gemm1_clamp_limit,
                    impl.topk,
                    kernel.prepare_finalize,
                    impl.quant_config,
                )
            except ValueError as error:
                reasons[str(error)] += 1
        if gemm is not None:
            front = _front_config(runner)
            if isinstance(front, str):
                reasons[f"fused front: {front}"] += 1
            else:
                gemm.set_front(front)
                fronts += 1
        gemms.append(gemm)
    # A layer that keeps FlashInfer under decode_backend=latency_specialized is a
    # deployment problem (e.g. unexpected weights), so it warns.
    log = logger.warning if reasons else logger.info
    log(
        "DeepSeek-V4.1 low-batch routed-expert GEMMs on %d of %d MoE layers "
        "(%d with the fused front)%s.",
        sum(gemm is not None for gemm in gemms),
        len(runners),
        fronts,
        "; other layers keep FlashInfer: "
        + "; ".join(f"{reason} ({n})" for reason, n in reasons.items())
        if reasons
        else "",
    )
    return gemms
