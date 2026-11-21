# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import os
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterExperts,
    AiterInputChoice,
    AiterInputFormat,
    aiter_moe_input_format,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)

logger = init_logger(__name__)

# Receive-row bounds are rounded up to this many rows (as SGLang's MORI path does).
_RECV_ROW_ALIGN = 32


@functools.cache
def _aiter_supports_mxfp8_dispatch() -> bool:
    """Whether this AITER both emits and consumes MXFP8 activations.

    The two halves ship separately (ROCm/aiter#4954 is the receive side), and
    neither has a capability flag. An older per_1x32 quant without `scale_type`
    returns FP32 scales that the MoE would read as e8m0 bytes, and an older MoE
    re-quantizes the FP8 input as if it were BF16: both silently corrupt.
    """
    try:
        import inspect

        import aiter.fused_moe as aiter_fused_moe
        from aiter import QuantType, get_hip_quant

        can_emit = (
            "scale_type"
            in inspect.signature(get_hip_quant(QuantType.per_1x32)).parameters
        )
        can_consume = (
            "hidden_states.dtype == dtypes.fp8 and a1_scale is not None"
            in inspect.getsource(aiter_fused_moe)
        )
        return can_emit and can_consume
    except Exception:
        return False


def _mori_epv2_runs_async_ll() -> bool:
    """MORI_ENABLE_SDMA=1 runs MORI EPv2 on its AsyncLL kernels (vllm#65128)."""
    return os.environ.get("MORI_ENABLE_SDMA", "0").lower() in ("1", "true")


def mori_epv2_dispatch_format(
    moe_config: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig | None,
    *,
    async_ll: bool | None = None,
    internode: bool = False,
) -> AiterInputChoice:
    """The activation format MORI EPv2 dispatches for this MoE layer.

    MORI EPv2 sends what AITER's fused MoE consumes, quantized before the
    all2all, so less crosses the fabric and the experts skip quantizing the
    received rows; BF16 where AITER has no fixed answer or MORI cannot carry it.
    The MORI op is sized and the AITER experts configured from this one
    function. AsyncLL and the internode kernels carry no packed FP4, so MXFP4
    goes as BF16 there.
    """
    if async_ll is None:
        async_ll = _mori_epv2_runs_async_ll()
    if not moe_config.use_mori_epv2_kernels:
        choice = AiterInputChoice(AiterInputFormat.BF16, "not MORI EPv2")
    elif not moe_config.rocm_aiter_fmoe_enabled or moe_config.moe_backend not in (
        "auto",
        "aiter",
    ):
        choice = AiterInputChoice(
            AiterInputFormat.BF16, "the experts are not AITER fused MoE"
        )
    else:
        choice = aiter_moe_input_format(moe_config, quant_config)

    if choice.format == AiterInputFormat.MXFP4 and (async_ll or internode):
        # MXFP8 would need AITER's A8W4 kernels, which exist only for gate/up
        # interleaved weights; AITER answers MXFP4 only for separated ones.
        transport = "AsyncLL" if async_ll else "the internode kernels"
        choice = AiterInputChoice(
            AiterInputFormat.BF16, f"{choice.reason}, which {transport} cannot carry"
        )
    if choice.format == AiterInputFormat.MXFP8 and not _aiter_supports_mxfp8_dispatch():
        choice = AiterInputChoice(
            AiterInputFormat.BF16,
            f"{choice.reason}, but this AITER cannot emit and consume mxfp8 "
            "(needs per_1x32 quant with scale_type and ROCm/aiter#4954)",
            actionable=True,
        )
    log = logger.warning_once if choice.actionable else logger.info_once
    log("MORI EPv2 dispatch dtype=%s (%s)", choice.format.value, choice.reason)
    return choice


# Element dtype and elements per scale of each quantized dispatch format.
_SCALES = {
    AiterInputFormat.FP8: (torch.float32, 128),
    AiterInputFormat.MXFP8: (torch.float8_e8m0fnu, 32),
    AiterInputFormat.MXFP4: (torch.float8_e8m0fnu, 32),
}


def mori_epv2_dispatch_args(fmt: AiterInputFormat, hidden_dim: int) -> dict[str, Any]:
    """MORI EPv2 op config for dispatching `fmt` with a BF16 combine."""
    if fmt == AiterInputFormat.BF16:
        return {}
    from vllm.platforms import current_platform

    scale_dtype, group = _SCALES[fmt]
    return dict(
        dispatch_data_type=(
            torch.float4_e2m1fn_x2
            if fmt == AiterInputFormat.MXFP4
            else current_platform.fp8_dtype()
        ),
        combine_data_type=torch.bfloat16,
        scale_dim=hidden_dim // group,
        scale_type_size=scale_dtype.itemsize,
    )


def _quantize_for_dispatch(
    x: torch.Tensor, fmt: AiterInputFormat
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 rows the way AITER's fused MoE would on receipt."""
    from aiter import QuantType, get_hip_quant

    from vllm.platforms import current_platform

    fp8 = current_platform.fp8_dtype()
    scale_dtype, group = _SCALES[fmt]
    if x.shape[0] == 0:
        cols = x.shape[1] // 2 if fmt == AiterInputFormat.MXFP4 else x.shape[1]
        dtype = torch.float4_e2m1fn_x2 if fmt == AiterInputFormat.MXFP4 else fp8
        return (
            torch.empty(0, cols, dtype=dtype, device=x.device),
            torch.empty(0, x.shape[1] // group, dtype=scale_dtype, device=x.device),
        )
    if fmt == AiterInputFormat.FP8:
        return get_hip_quant(QuantType.per_1x128)(x, quant_dtype=fp8)
    # The scales stay unshuffled: AITER's fused MoE sorts them itself when it
    # receives pre-quantized MX input.
    if fmt == AiterInputFormat.MXFP8:
        return get_hip_quant(QuantType.per_1x32)(
            x, quant_dtype=fp8, shuffle=False, scale_type=torch.float8_e8m0fnu
        )
    return get_hip_quant(QuantType.per_1x32)(x, shuffle=False)


class MoriEPv2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    def __init__(
        self,
        mori_op: Any,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_local_experts: int,
        rank_expert_offset: int,
        dispatch_format: AiterInputFormat = AiterInputFormat.BF16,
    ):
        super().__init__()
        # Set from mori_epv2_dispatch_format, which also sized the MORI op.
        self.dispatch_format = dispatch_format
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_local_experts = num_local_experts
        self.rank_expert_offset = rank_expert_offset
        self._padding_expert_ids: torch.Tensor | None = None
        self._routing = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts):
        if fused_experts.moe_config.in_dtype != torch.bfloat16:
            raise ValueError(
                "MORI EPv2 only dispatches BF16 activations; got "
                f"{fused_experts.moe_config.in_dtype}. Set VLLM_MORI_EP_V2=0 to "
                "use MORI EPv1."
            )
        quant_dtype = fused_experts.quant_config.quant_dtype
        if self.dispatch_format != AiterInputFormat.BF16:
            # Only AITER fused MoE takes the quantized formats as they are, and
            # it must have made the same choice.
            if (
                not isinstance(fused_experts, AiterExperts)
                or fused_experts.expects_unquantized_inputs
            ):
                raise ValueError(
                    f"MORI EPv2 dispatches {self.dispatch_format.value} activations "
                    "quantized for AITER fused MoE, but "
                    f"{type(fused_experts).__name__} does not take them. Use AITER "
                    "fused MoE experts "
                    "(VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1), or set "
                    "VLLM_MORI_EP_V2=0 for MORI EPv1, which also requires AITER "
                    "fused MoE."
                )
        elif quant_dtype is not None and not fused_experts.expects_unquantized_inputs:
            raise ValueError(
                "MORI EPv2 dispatches unquantized BF16 activations, but "
                f"{type(fused_experts).__name__} expects {quant_dtype} "
                "activations quantized before dispatch. Use AITER fused MoE experts "
                "(VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1), or set "
                "VLLM_MORI_EP_V2=0 for MORI EPv1, which also requires AITER fused MoE."
            )

        # OCP MX (e.g. MXFP4) reports a per-tensor group shape but scales every
        # 1x32 block, so only FP8 / INT8 activations count as per tensor.
        quant_config = fused_experts.quant_config
        if quant_config.is_per_tensor and (
            quant_config.use_fp8_w8a8 or quant_config.use_int8_w8a8
        ):
            raise ValueError(
                "MORI EPv2 does not support per-tensor activation quantization: "
                "the experts would quantize the received rows, which AITER's "
                "per-tensor quant does not support. Set VLLM_MORI_EP_V2=0 to use "
                "MORI EPv1, which quantizes before dispatch."
            )

    def allocate_fused_expert_output(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> torch.Tensor | None:
        # Experts that reduce top-k themselves write straight into the arena
        # rows combine sends, and combine skips its copy-in.
        if not isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP):
            return None
        buf = self.mori_op.combine_in_view()
        if (
            len(shape) != 2
            or buf.dtype != dtype
            or buf.device != device
            or shape[1] != buf.shape[1]
            or shape[0] > buf.shape[0]
        ):
            return None
        return buf[: shape[0]]

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self):
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return False

    def _route_padding_locally(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # DP padding tokens carry expert id -1 (VLLM_MOE_SKIP_PADDING) and zero
        # weight. MORI EPv2 has no invalid-expert slot and its FlyDSL dispatch
        # faults on -1, so send them to this rank's own experts instead.
        if self._padding_expert_ids is None:
            slots = torch.arange(topk_ids.shape[1], device=topk_ids.device)
            self._padding_expert_ids = (
                self.rank_expert_offset + slots % self.num_local_experts
            ).to(topk_ids.dtype)
        return torch.where(topk_ids < 0, self._padding_expert_ids, topk_ids)

    def _recv_rows_bound(self, a1: torch.Tensor, capacity: int) -> int:
        """Host-side upper bound on the rows this rank can receive this step.

        MORI receives each source token at most once per rank, so the receive
        count is at most the tokens all dispatchers send: the sum of the DP
        batches, times the dispatchers per DP rank (each TP rank may send the
        same batch). Slicing the capacity-sized buffers (world_size *
        max_num_tokens rows) to this bound keeps the row masking, the experts
        and combine's copy-in off the unused tail. The bound is rounded up to a
        multiple of _RECV_ROW_ALIGN to limit the distinct expert shapes.
        """
        dp_meta = (
            get_forward_context().dp_metadata
            if is_forward_context_available()
            else None
        )
        if dp_meta is None:
            return capacity
        tokens = dp_meta.num_tokens_across_dp_cpu
        n_dp = tokens.numel()
        dp_rank = get_dp_group().rank_in_group
        if dp_rank >= n_dp or a1.shape[0] > int(tokens[dp_rank]):
            # Metadata that does not cover this rank's own batch is stale; a
            # bound derived from it could cut off received rows.
            logger.debug_once(
                "MORI EPv2: DP token counts do not cover this rank's batch; "
                "using the full receive buffer."
            )
            return capacity
        if n_dp > 0 and self.num_dispatchers_ % n_dp == 0:
            bound = self.num_dispatchers_ // n_dp * int(tokens.sum())
        else:
            bound = self.num_dispatchers_ * int(tokens.max())
        bound = max(bound, 1)
        bound = -(-bound // _RECV_ROW_ALIGN) * _RECV_ROW_ALIGN
        return min(capacity, bound)

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        del num_experts, expert_map, defer_input_quant
        assert not apply_router_weight_on_input, (
            "MORI EPv2 does not support apply_router_weight_on_input=True"
        )
        if a1.dtype != torch.bfloat16:
            raise ValueError(f"MORI EPv2 requires BF16 dispatch input, got {a1.dtype}")

        fmt = self.dispatch_format
        hidden_dim = a1.shape[-1]
        scale = None
        if fmt != AiterInputFormat.BF16:
            a1, scale = _quantize_for_dispatch(a1, fmt)
        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
            self._routing,
        ) = self.mori_op.dispatch(
            a1,
            topk_weights.to(torch.float32),
            scale,
            self._route_padding_locally(topk_ids.to(torch.int32)),
            return_routing=True,
        )
        if fmt != AiterInputFormat.BF16:
            # MORI returns the scales as int32 words, in rows it may pad (to
            # 128 B on its HIP backend); AITER reads the leading columns.
            scale_dtype, group = _SCALES[fmt]
            dispatch_scale = dispatch_scale.view(scale_dtype)[:, : hidden_dim // group]
        rows = self._recv_rows_bound(a1, dispatch_a1.shape[0])
        if rows < dispatch_a1.shape[0]:
            dispatch_a1 = dispatch_a1[:rows]
            dispatch_weights = dispatch_weights[:rows]
            dispatch_ids = dispatch_ids[:rows]
            if dispatch_scale is not None:
                dispatch_scale = dispatch_scale[:rows]
        if dispatch_scale is not None and not dispatch_scale.is_contiguous():
            dispatch_scale = dispatch_scale.contiguous()
        # The experts get no received-row count and read every bounded row, and
        # MORI reuses its buffers, so make the unused rows inert: zero
        # activations and weights, routed to expert -1.
        unused = (
            torch.arange(dispatch_a1.shape[0], device=dispatch_a1.device)
            >= dispatch_recv_token_num
        ).unsqueeze(1)
        # Quantized rows are cleared bytewise (FP4 has no masked_fill kernel);
        # zero bytes are zero values, and finite scales.
        for buf in (dispatch_a1, dispatch_scale):
            if buf is not None:
                buf = buf.view(torch.uint8) if buf.itemsize == 1 else buf
                buf.masked_fill_(unused, 0)
        dispatch_weights.masked_fill_(unused, 0)
        dispatch_ids.masked_fill_(unused, -1)
        return (
            dispatch_a1,
            dispatch_scale,
            None,
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        del topk_weights, topk_ids, apply_router_weight_on_input
        del weight_and_reduce_impl
        if self._routing is None:
            raise RuntimeError("MORI EPv2 combine called without a dispatch")
        # combine takes its token mapping from the routing, not from indices.
        result = self.mori_op.combine(fused_expert_output, routing=self._routing)[0]
        output.copy_(result[: output.shape[0]])
        if not torch.cuda.is_current_stream_capturing():
            self._routing = None
