# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-batch FFN half of a MiniMax-M3 MoE layer (1-8 decode rows).

From the post-attention norm's output to the ``MoEOutput`` the fused
finalize + all-reduce + norm kernel (``ops/finalize_allreduce_norm.py``)
takes, the stock chain runs about eleven small kernels. Here::

    x ─► gate (stock) ─► front ─► FC1 ─► FC2 input ─► FC2 ─► gemm2_permuted
    │                    (top-4, NVFP4 x, record)
    └─(aux stream)─► MXFP8 x ─► FC1 (split K) ─► SwiGLU, MXFP8 h ─► FC2
                                                         └─► shared_output

* ``front_cute.py``: the routing (top-4 of ``sigmoid(logit) + bias``, the
  weights from the recovered scores), the routed input's per-token NVFP4 and
  the routing record of ``routed_expert_route_cute.py`` in one launch, and the
  per-token NVFP4 of FC1's BF16 output;
* ``gemm_cute.py``: FC1 (with SwiGLU-OAI) and FC2, persistent ``tcgen05``
  NVFP4 GEMMs over the record's (weight tile, token tile) items;
* ``shared_cute.py``: the MXFP8 shared expert on the same GEMM.

The kernels read the weights vLLM prepares for the TRT-LLM-gen NVFP4 experts
and the FlashInfer MXFP8 linears in place, are compiled once per device when a
layer is bound, and launch with programmatic dependent launch, so CUDA graphs
capture them like any other kernel. The routing weights are BF16, as
the monolithic TRT-LLM-gen experts return them and the fused finalize reads
them.
"""

from __future__ import annotations

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_fake_stream, make_ptr

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_gemm import (  # noqa: E501
    TILE,
    cta_words,
    max_ctas,
    route_words,
    width,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_gemm import (  # noqa: E501
    WIDTHS as GEMM_WIDTHS,
)
from vllm.utils.torch_utils import aux_stream

from .front_cute import (
    EXPERTS,
    H_TILE_BYTES,
    HIDDEN,
    INTERMEDIATE,
    MAX_ROWS,
    TOP_K,
    LowBatchFront,
)
from .gemm_cute import RoutedFc1, RoutedFc2
from .shared_cute import (
    SPLIT_K,
    SPLITS,
    SharedAct,
    SharedFc1,
    SharedFc2,
    SharedInputQuant,
)
from .shared_cute import tile_scale_bytes as shared_tile_scale_bytes

logger = init_logger(__name__)

# The SwiGLU-OAI constants FC1's epilogue is compiled with.
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
SWIGLU_BETA = 1.0
# K tile and pipeline stages of the routed GEMMs (shared memory per CTA:
# 21 KB per stage), their persistent CTAs (None: one per SM), and the shared
# FC1's stages (36 KB per stage). Measured on B300 and B200 (four real layers,
# 1, 4 and 8 rows) against FC1 4/6 stages, FC2 3/6/8, shared FC1 3-5 and 96-128
# routed CTAs: FC2 at 4 stages gains 6% at one row; no other setting moves a
# row count by 3% or more, and fewer routed CTAs slow eight rows.
ROUTED_FC1_CTA_K = 256
ROUTED_FC2_CTA_K = 256
ROUTED_FC1_STAGES = 8
ROUTED_FC2_STAGES = 4
ROUTED_CTAS: int | None = None
SHARED_FC1_STAGES = 6
# Persistent CTAs of the shared expert's GEMMs (144 and 48 work items), which
# leave SMs to the routed GEMMs beside them (the critical path). On B300 at
# 1, 4 and 8 rows, within 3% of the best of 144/48, 72/48, 48/24 and 24/16.
SHARED_FC1_CTAS = 36
SHARED_FC2_CTAS = 24
# The shared expert's 8-row tiles, and its records' table widths.
SHARED_ROW_TILES = MAX_ROWS // TILE
SHARED_FC1_WORDS = cta_words(SPLITS * SHARED_ROW_TILES)
SHARED_FC2_WORDS = cta_words(SHARED_ROW_TILES)
# The front's builds: the routed-expert GEMMs' row counts up to MAX_ROWS.
WIDTHS = tuple(w for w in GEMM_WIDTHS if w <= MAX_ROWS)
assert WIDTHS[-1] == MAX_ROWS


class LowBatchFfn:
    """The low-batch FFN bound to one ``MiniMaxM3MoE`` layer's weights."""

    def __init__(self, moe: torch.nn.Module) -> None:
        routed = moe.experts.routed_experts
        quant = routed.quant_method.moe_kernel.fused_experts.quant_config
        self.gate = moe.gate
        self.bias = moe.e_score_correction_bias
        self.scale = float(moe.routed_scaling_factor)
        self.routed = routed
        self.w13, self.w13_scale = routed.w13_weight, quant.w1_scale
        self.w2, self.w2_scale = routed.w2_weight, quant.w2_scale
        self.alpha1, self.alpha2 = quant.g1_alphas, quant.g2_alphas
        gate_up, down = moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj
        self.shared_linears = (gate_up, down)
        self.gate_up, self.gate_up_scale = gate_up.weight, gate_up.weight_scale
        self.down, self.down_scale = down.weight, down.weight_scale
        device = self.w13.device
        self.device_index = device.index
        # The shared expert's routing records for each row count: FC1's tile
        # s r + t is rows 8t .. of K split s (r row tiles); FC2's tile t rows
        # 8t ...
        fc1, fc2 = [], []
        for rows in range(1, MAX_ROWS + 1):
            row_tiles = -(-rows // TILE)
            limits = [min(TILE * (t + 1), rows) for t in range(row_tiles)]
            fc1.append(
                _record(
                    [s for s in range(SPLITS) for _ in range(row_tiles)],
                    [
                        TILE * s * row_tiles + limit
                        for s in range(SPLITS)
                        for limit in limits
                    ],
                    SHARED_FC1_WORDS,
                )
            )
            fc2.append(_record([0] * row_tiles, limits, SHARED_FC2_WORDS))
        self.shared_fc1_records = torch.tensor(fc1, dtype=torch.int32, device=device)
        self.shared_fc2_records = torch.tensor(fc2, dtype=torch.int32, device=device)
        # Compile the kernels now, not inside a CUDA graph capture.
        for built in WIDTHS:
            _front(self.device_index, built)
        for compile_kernel in _KERNELS:
            compile_kernel(self.device_index)
        self.aux = aux_stream()
        self.input_ready = torch.cuda.Event()
        self.shared_ready = torch.cuda.Event()

    def accepts(self, x: torch.Tensor) -> bool:
        """1-32 contiguous BF16 rows of the hidden size, and the routed and
        shared expert weights still the tensors bound here (a weight reload
        replaces them)."""
        return (
            x.dim() == 2
            and 1 <= x.size(0) <= MAX_ROWS
            and x.size(1) == HIDDEN
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and self.routed.w13_weight.data_ptr() == self.w13.data_ptr()
            and self.routed.w2_weight.data_ptr() == self.w2.data_ptr()
            and self.shared_linears[0].weight.data_ptr() == self.gate_up.data_ptr()
            and self.shared_linears[1].weight.data_ptr() == self.down.data_ptr()
        )

    def run(self, x: torch.Tensor) -> MoEOutput:
        """What ``MiniMaxM3MoE.forward_unfinalized(x)`` returns, launched on the
        current stream (the shared expert on the aux stream, joined before
        returning)."""
        rows = x.size(0)
        device = x.device
        main = torch.cuda.current_stream()
        self.input_ready.record(main)
        with torch.cuda.stream(self.aux):
            self.input_ready.wait(self.aux)
            shared_output = self.shared(x)
            self.shared_ready.record(self.aux)

        logits, _ = self.gate(x)
        tiles = max_ctas(rows, TOP_K, EXPERTS)
        h_rows = TILE * tiles
        words = Int32(cta_words(tiles))
        weights = torch.empty((rows, TOP_K), dtype=torch.bfloat16, device=device)
        permuted = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
        # The record, then one FC1 item count per tile (zeroed by the front).
        record = torch.empty(
            route_words(tiles) + cta_words(tiles), dtype=torch.int32, device=device
        )
        x_quant = torch.empty((rows, HIDDEN // 2), dtype=torch.uint8, device=device)
        x_scale = torch.empty((rows, HIDDEN // 16), dtype=torch.uint8, device=device)
        x_pt = torch.empty(rows, dtype=torch.float32, device=device)
        h = torch.empty((h_rows, INTERMEDIATE), dtype=torch.bfloat16, device=device)
        h_quant = torch.empty(
            (h_rows, INTERMEDIATE // 2), dtype=torch.uint8, device=device
        )
        h_scale = torch.empty(tiles * H_TILE_BYTES, dtype=torch.uint8, device=device)
        h_pt = torch.empty(h_rows, dtype=torch.float32, device=device)
        output = torch.empty((h_rows, HIDDEN), dtype=torch.bfloat16, device=device)
        stream = current_cu_stream()
        ctas = Int32(ROUTED_CTAS or _sm_count(self.device_index))
        _front(self.device_index, width(rows))(
            _ptr(cutlass.Float32, logits),
            _ptr(cutlass.Float32, self.bias),
            Float32(self.scale),
            _ptr(cutlass.Uint32, x),
            _ptr(cutlass.Uint16, weights),
            _ptr(cutlass.Uint32, x_quant),
            _ptr(cutlass.Uint8, x_scale),
            _ptr(cutlass.Float32, x_pt),
            _ptr(cutlass.Int32, permuted),
            _ptr(cutlass.Int32, record),
            Int32(rows),
            stream,
        )
        _fc1(self.device_index)(
            _ptr(cutlass.Float4E2M1FN, self.w13, 32),
            _ptr(cutlass.Float8E4M3FN, self.w13_scale),
            _ptr(cutlass.Float4E2M1FN, x_quant, 32),
            _ptr(cutlass.Uint8, x_quant),
            _ptr(cutlass.Uint32, x_scale),
            _ptr(cutlass.Float32, x_pt),
            _ptr(cutlass.Float32, self.alpha1),
            _ptr(cutlass.Uint32, h),
            _ptr(cutlass.Uint32, h_quant),
            _ptr(cutlass.Uint8, h_scale),
            _ptr(cutlass.Float32, h_pt),
            _ptr(cutlass.Int32, record),
            _ptr(cutlass.Int32, record[route_words(tiles) :]),
            Int32(rows),
            Int32(EXPERTS),
            words,
            Int32(h_rows),
            ctas,
            stream,
        )
        _fc2(self.device_index)(
            _ptr(cutlass.Float4E2M1FN, self.w2, 32),
            _ptr(cutlass.Float8E4M3FN, self.w2_scale),
            _ptr(cutlass.Float4E2M1FN, h_quant, 32),
            _ptr(cutlass.Float8E4M3FN, h_scale),
            _ptr(cutlass.Float32, h_pt),
            _ptr(cutlass.Float32, self.alpha2),
            _ptr(cutlass.BFloat16, output),
            _ptr(cutlass.Int32, record),
            Int32(h_rows),
            Int32(EXPERTS),
            words,
            ctas,
            stream,
        )
        self.shared_ready.wait(main)
        return MoEOutput(
            routed=UnfinalizedMoEOutput(
                gemm2_permuted=output,
                expert_weights=weights,
                expanded_idx_to_permuted_idx=permuted,
            ),
            shared_output=shared_output,
        )

    def shared(self, x: torch.Tensor) -> torch.Tensor:
        """The shared expert's BF16 output (``shared_cute.py``), on the current
        stream."""
        rows = x.size(0)
        row_tiles = -(-rows // TILE)
        tiles = SPLITS * row_tiles
        device = x.device
        x8 = torch.empty(row_tiles * TILE * HIDDEN, dtype=torch.uint8, device=device)
        x8_scale = torch.empty(
            tiles * shared_tile_scale_bytes(SPLIT_K), dtype=torch.uint8, device=device
        )
        partial = torch.empty(
            tiles * TILE * 2 * INTERMEDIATE, dtype=torch.float32, device=device
        )
        h8 = torch.empty(
            row_tiles * TILE * INTERMEDIATE, dtype=torch.uint8, device=device
        )
        h8_scale = torch.empty(
            row_tiles * shared_tile_scale_bytes(INTERMEDIATE),
            dtype=torch.uint8,
            device=device,
        )
        output = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        stream = current_cu_stream()
        index = self.device_index
        sms = _sm_count(index)
        _shared_input(index)(
            _ptr(cutlass.Uint32, x),
            _ptr(cutlass.Uint32, x8),
            _ptr(cutlass.Uint8, x8_scale),
            Int32(rows),
            Int32(row_tiles),
            stream,
        )
        _shared_fc1(index)(
            _ptr(cutlass.Float8E4M3FN, self.gate_up),
            _ptr(cutlass.Float8E8M0FNU, self.gate_up_scale),
            _ptr(cutlass.Float8E4M3FN, x8),
            _ptr(cutlass.Float8E8M0FNU, x8_scale),
            _ptr(cutlass.Float32, partial),
            _ptr(cutlass.Int32, self.shared_fc1_records[rows - 1]),
            Int32(row_tiles),
            Int32(SHARED_FC1_WORDS),
            Int32(min(sms, SHARED_FC1_CTAS)),
            stream,
        )
        _shared_act(index)(
            _ptr(cutlass.Float32, partial),
            _ptr(cutlass.Uint32, h8),
            _ptr(cutlass.Uint8, h8_scale),
            Int32(rows),
            Int32(row_tiles),
            stream,
        )
        _shared_fc2(index)(
            _ptr(cutlass.Float8E4M3FN, self.down),
            _ptr(cutlass.Float8E8M0FNU, self.down_scale),
            _ptr(cutlass.Float8E4M3FN, h8),
            _ptr(cutlass.Float8E8M0FNU, h8_scale),
            _ptr(cutlass.BFloat16, output),
            _ptr(cutlass.Int32, self.shared_fc2_records[rows - 1]),
            Int32(rows),
            Int32(row_tiles),
            Int32(SHARED_FC2_WORDS),
            Int32(min(sms, SHARED_FC2_CTAS)),
            stream,
        )
        return output


def _record(experts: list[int], limits: list[int], words: int) -> list[int]:
    """A static routing record (``routed_expert_route_cute.py``'s first
    words) of ``len(experts)`` tiles in a table of ``words``."""
    pad = [0] * (words - len(experts))
    return [TILE * len(experts), len(experts), 0, 0] + experts + pad + limits + pad


def _ptr(dtype: type[cutlass.Numeric], tensor: torch.Tensor, align: int = 16):
    """A CuTe DSL global-memory pointer to ``tensor``'s data."""
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _fake(dtype: type[cutlass.Numeric], align: int = 16):
    """A placeholder pointer for compilation."""
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


@cache
def _front(device_index: int, built: int):
    """``LowBatchFront`` for calls of up to ``built`` rows."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            LowBatchFront(built),
            _fake(cutlass.Float32),
            _fake(cutlass.Float32),
            Float32(1.0),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint16),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint8),
            _fake(cutlass.Float32),
            _fake(cutlass.Int32),
            _fake(cutlass.Int32),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _fc1(device_index: int):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedFc1(HIDDEN, INTERMEDIATE, ROUTED_FC1_CTA_K, ROUTED_FC1_STAGES),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Uint8),
            _fake(cutlass.Uint32),
            _fake(cutlass.Float32),
            _fake(cutlass.Float32),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint8),
            _fake(cutlass.Float32),
            _fake(cutlass.Int32),
            _fake(cutlass.Int32),
            Int32(1),
            Int32(EXPERTS),
            Int32(4),
            Int32(TILE * TOP_K),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _fc2(device_index: int):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedFc2(
                HIDDEN,
                INTERMEDIATE,
                2 * INTERMEDIATE // 128,  # FC1's m tiles: a tile's ready count
                ROUTED_FC2_CTA_K,
                ROUTED_FC2_STAGES,
            ),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float32),
            _fake(cutlass.Float32),
            _fake(cutlass.BFloat16),
            _fake(cutlass.Int32),
            Int32(TILE * TOP_K),
            Int32(EXPERTS),
            Int32(4),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _shared_input(device_index: int):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            SharedInputQuant(),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint8),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _shared_fc1(device_index: int):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            SharedFc1(num_ab_stage=SHARED_FC1_STAGES),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float8E8M0FNU),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float8E8M0FNU),
            _fake(cutlass.Float32),
            _fake(cutlass.Int32),
            Int32(1),
            Int32(SHARED_FC1_WORDS),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _shared_act(device_index: int):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            SharedAct(),
            _fake(cutlass.Float32),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint8),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _shared_fc2(device_index: int):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            SharedFc2(),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float8E8M0FNU),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float8E8M0FNU),
            _fake(cutlass.BFloat16),
            _fake(cutlass.Int32),
            Int32(1),
            Int32(1),
            Int32(SHARED_FC2_WORDS),
            Int32(1),
            make_fake_stream(),
        )


_KERNELS = (
    _fc1,
    _fc2,
    _shared_input,
    _shared_fc1,
    _shared_act,
    _shared_fc2,
)


@cache
def _sm_count(device_index: int) -> int:
    """Streaming multiprocessors of a device: the persistent GEMMs' CTAs."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def unsupported_reason(moe: torch.nn.Module) -> str | None:
    """Why ``moe`` cannot take the low-batch FFN, or None."""
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsMonolithic,
    )

    if getattr(moe, "use_mega_moe", False) or moe.shared_experts is None:
        return "MegaMoE or no shared expert"
    if moe.e_score_correction_bias is None or moe.scoring_func != "sigmoid":
        return "routing is not sigmoid with a score correction bias"
    if moe.top_k != TOP_K:
        return f"top-{moe.top_k}"
    routed = moe.experts.routed_experts
    kernel = getattr(routed.quant_method, "moe_kernel", None)
    impl = getattr(kernel, "fused_experts", None)
    if not isinstance(impl, TrtLlmNvFp4ExpertsMonolithic):
        return f"experts kernel {type(impl).__name__}"
    if not impl.per_token_activation:
        return "activations are not per-token NVFP4"
    config = impl.moe_config
    if (
        impl.local_num_experts != EXPERTS
        or config.num_experts != EXPERTS
        or config.hidden_dim != HIDDEN
        or impl.intermediate_size_per_partition != INTERMEDIATE
    ):
        return "layer shape"
    if (
        getattr(config, "swiglu_limit", None) != SWIGLU_LIMIT
        or getattr(config, "swiglu_alpha", None) != SWIGLU_ALPHA
        or getattr(config, "swiglu_beta", None) != SWIGLU_BETA
    ):
        return "SwiGLU-OAI constants"
    quant = impl.quant_config
    for name, tensor, shape in (
        ("w13", routed.w13_weight, (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2)),
        ("w13 scales", quant.w1_scale, (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16)),
        ("w2", routed.w2_weight, (EXPERTS, HIDDEN, INTERMEDIATE // 2)),
        ("w2 scales", quant.w2_scale, (EXPERTS, HIDDEN, INTERMEDIATE // 16)),
    ):
        if (
            tensor is None
            or tuple(tensor.shape) != shape
            or tensor.element_size() != 1
            or not tensor.is_contiguous()
            or tensor.data_ptr() % 32
        ):
            return f"{name} layout"
    for alphas in (quant.g1_alphas, quant.g2_alphas):
        if (
            alphas is None
            or alphas.dtype != torch.float32
            or tuple(alphas.shape) != (EXPERTS,)
            or not alphas.is_contiguous()
        ):
            return "no FP32 per-expert global scales"
    bias = moe.e_score_correction_bias
    if bias.dtype != torch.float32 or tuple(bias.shape) != (EXPERTS,):
        return "bias is not FP32 [128]"
    shared = moe.shared_experts
    act = shared.act_fn
    if (act.swiglu_limit, act.alpha, act.beta) != (
        SWIGLU_LIMIT,
        SWIGLU_ALPHA,
        SWIGLU_BETA,
    ):
        return "shared expert SwiGLU-OAI constants"
    for linear, n, k in (
        (shared.gate_up_proj, 2 * INTERMEDIATE, HIDDEN),
        (shared.down_proj, HIDDEN, INTERMEDIATE),
    ):
        # FlashInfer's MXFP8 linears keep K-major E4M3 weights and E8M0
        # scales in the 128x4 layout.
        kernel_name = type(getattr(linear.quant_method, "kernel", None)).__name__
        weight, scale = linear.weight, getattr(linear, "weight_scale", None)
        if (
            not (kernel_name.startswith("FlashInfer") and "Mxfp8" in kernel_name)
            or weight.dtype != torch.float8_e4m3fn
            or tuple(weight.t().shape) != (n, k)
            or not weight.t().is_contiguous()
            or scale is None
            or scale.dtype != torch.uint8
            or scale.numel() != n * k // 32
        ):
            return f"shared expert {kernel_name}"
    return None


def build_low_batch_ffn(moe: torch.nn.Module) -> LowBatchFfn | None:
    """The low-batch FFN of a ``MiniMaxM3MoE`` with processed weights, or None
    when the layer cannot take it. Compiles the kernels (outside any CUDA
    graph capture). For callers that can run without it, like a benchmark; the
    model raises with ``unsupported_reason`` instead."""
    reason = unsupported_reason(moe)
    if reason is not None:
        logger.warning_once("MiniMax-M3 low-batch FFN not used: %s.", reason)
        return None
    return LowBatchFfn(moe)
