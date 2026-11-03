# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-batch FFN half of a MiniMax-M3 MoE layer (1-8 decode rows).

From the post-attention norm's output to the ``MoEOutput`` the fused
finalize + all-reduce + norm kernel (``ops/finalize_allreduce_norm.py``)
takes, the stock chain runs about eleven small kernels. Here::

    x ─► gate (stock) ─► front ─► FC1 ─► FC2 input ─► FC2 ─► gemm2_permuted
    │                    (top-4, NVFP4 x, record)
    └─(aux stream)─► shared expert (stock) ─────────────────► shared_output

* ``front_cute.py``: the routing (top-4 of ``sigmoid(logit) + bias``, the
  weights from the recovered scores), the routed input's per-token NVFP4 and
  the routing record of ``routed_expert_route_cute.py`` in one launch, and the
  per-token NVFP4 of FC1's BF16 output;
* ``gemm_cute.py``: FC1 (with SwiGLU-OAI) and FC2, persistent ``tcgen05``
  NVFP4 GEMMs over the record's (weight tile, token tile) items.

The kernels read the expert weights vLLM prepares for the TRT-LLM-gen NVFP4
experts in place, are compiled once per device when a layer is bound, and
launch on the current stream with programmatic dependent launch, so CUDA
graphs capture them like any other kernel. The routing weights are BF16, as
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
    cta_words,
    max_ctas,
    route_words,
)
from vllm.utils.torch_utils import aux_stream

from .front_cute import (
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    MAX_ROWS,
    TILE,
    TOP_K,
    Fc2InputQuant,
    LowBatchFront,
)
from .gemm_cute import RoutedFc1, RoutedFc2

logger = init_logger(__name__)

# The SwiGLU-OAI constants FC1's epilogue is compiled with.
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
SWIGLU_BETA = 1.0


class LowBatchFfn:
    """The low-batch FFN bound to one ``MiniMaxM3MoE`` layer's weights."""

    def __init__(self, moe: torch.nn.Module) -> None:
        routed = moe.experts.routed_experts
        quant = routed.quant_method.moe_kernel.fused_experts.quant_config
        self.gate = moe.gate
        self.shared_experts = moe.shared_experts
        self.bias = moe.e_score_correction_bias
        self.scale = float(moe.routed_scaling_factor)
        self.routed = routed
        self.w13, self.w13_scale = routed.w13_weight, quant.w1_scale
        self.w2, self.w2_scale = routed.w2_weight, quant.w2_scale
        self.alpha1, self.alpha2 = quant.g1_alphas, quant.g2_alphas
        index = self.w13.device.index
        self.device_index = (
            torch.accelerator.current_device_index() if index is None else index
        )
        # Compile the kernels now, not inside a CUDA graph capture.
        for compile_kernel in (_front, _fc2_input, _fc1, _fc2):
            compile_kernel(self.device_index)
        self.aux = aux_stream()
        self.input_ready = torch.cuda.Event()
        self.shared_ready = torch.cuda.Event()

    def accepts(self, x: torch.Tensor) -> bool:
        """1-8 contiguous BF16 rows of the hidden size, and the expert weights
        still the tensors bound here (a weight reload replaces them)."""
        return (
            x.dim() == 2
            and 1 <= x.size(0) <= MAX_ROWS
            and x.size(1) == HIDDEN
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and self.routed.w13_weight.data_ptr() == self.w13.data_ptr()
            and self.routed.w2_weight.data_ptr() == self.w2.data_ptr()
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
            shared_output = self.shared_experts(x)
            self.shared_ready.record(self.aux)

        logits, _ = self.gate(x)
        tiles = max_ctas(rows, TOP_K, EXPERTS)
        h_rows = TILE * tiles
        words = Int32(cta_words(tiles))
        weights = torch.empty((rows, TOP_K), dtype=torch.bfloat16, device=device)
        permuted = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
        record = torch.empty(route_words(tiles), dtype=torch.int32, device=device)
        x_quant = torch.empty((rows, HIDDEN // 2), dtype=torch.uint8, device=device)
        x_scale = torch.empty((rows, HIDDEN // 16), dtype=torch.uint8, device=device)
        x_pt = torch.empty(rows, dtype=torch.float32, device=device)
        h = torch.empty((h_rows, INTERMEDIATE), dtype=torch.bfloat16, device=device)
        h_quant = torch.empty(
            (h_rows, INTERMEDIATE // 2), dtype=torch.uint8, device=device
        )
        h_scale = torch.empty(
            tiles * 128 * INTERMEDIATE // 16, dtype=torch.uint8, device=device
        )
        h_pt = torch.empty(h_rows, dtype=torch.float32, device=device)
        output = torch.empty((h_rows, HIDDEN), dtype=torch.bfloat16, device=device)
        stream = current_cu_stream()
        ctas = Int32(_sm_count(self.device_index))
        _front(self.device_index)(
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
            _ptr(cutlass.Uint32, x_scale),
            _ptr(cutlass.Float32, x_pt),
            _ptr(cutlass.Float32, self.alpha1),
            _ptr(cutlass.Uint32, h),
            _ptr(cutlass.Int32, record),
            Int32(rows),
            Int32(EXPERTS),
            words,
            Int32(h_rows),
            ctas,
            stream,
        )
        _fc2_input(self.device_index)(
            _ptr(cutlass.Uint32, h),
            _ptr(cutlass.Int32, record),
            _ptr(cutlass.Uint32, h_quant),
            _ptr(cutlass.Uint8, h_scale),
            _ptr(cutlass.Float32, h_pt),
            Int32(tiles),
            words,
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


def _ptr(dtype: type[cutlass.Numeric], tensor: torch.Tensor, align: int = 16):
    """A CuTe DSL global-memory pointer to ``tensor``'s data."""
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _fake(dtype: type[cutlass.Numeric], align: int = 16):
    """A placeholder pointer for compilation."""
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


@cache
def _front(device_index: int):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            LowBatchFront(),
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
def _fc2_input(device_index: int):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            Fc2InputQuant(),
            _fake(cutlass.Uint32),
            _fake(cutlass.Int32),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint8),
            _fake(cutlass.Float32),
            Int32(TOP_K),
            Int32(4),
            make_fake_stream(),
        )


@cache
def _fc1(device_index: int):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedFc1(HIDDEN, INTERMEDIATE),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Uint32),
            _fake(cutlass.Float32),
            _fake(cutlass.Float32),
            _fake(cutlass.Uint32),
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
            RoutedFc2(HIDDEN, INTERMEDIATE),
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
def _sm_count(device_index: int) -> int:
    """Streaming multiprocessors of a device: the persistent GEMMs' CTAs."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _unsupported_reason(moe: torch.nn.Module) -> str | None:
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
            return f"{name} {tuple(tensor.shape)} {tensor.dtype}"
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
    return None


def build_low_batch_ffn(moe: torch.nn.Module) -> LowBatchFfn | None:
    """The low-batch FFN of a ``MiniMaxM3MoE`` with processed weights, or None
    when the layer cannot take it. Compiles the kernels (outside any CUDA
    graph capture)."""
    reason = _unsupported_reason(moe)
    if reason is not None:
        logger.warning_once("MiniMax-M3 low-batch FFN not used: %s.", reason)
        return None
    return LowBatchFfn(moe)
