# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mori
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.mori_dtypes import (
    FP8_BLOCK_SIZE,
    MXFP4_BLOCK_SIZE,
    CombineDtype,
    DispatchDtype,
)
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


class MoriEPHTPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    MoRI EP high-throughput Prepare/Finalize.

    Uses the monolithic ``op.dispatch`` / ``op.combine`` path (IntraNode /
    InterNodeV1 / InterNodeV1LL kernels). Supports bf16/fp8/fp4 dispatch and
    bf16/fp8 combine. Each DBO micro-batch uses its own EpDispatchCombineOp
    because the op's receive buffers are internal/non-reentrant.
    """

    def __init__(
        self,
        handle_factory,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        dispatch_dtype: DispatchDtype = DispatchDtype.bf16,
        combine_dtype: CombineDtype = CombineDtype.bf16,
        global_to_physical: torch.Tensor | None = None,
        physical_to_global: torch.Tensor | None = None,
        local_expert_global_ids: torch.Tensor | None = None,
    ):
        super().__init__()
        # Lazily create one op per ubatch id; mori recv buffers are internal so
        # two in-flight DBO micro-batches must not share an op.
        self._handle_factory = handle_factory
        self._ops: dict[int, mori.ops.EpDispatchCombineOp] = {}
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.dispatch_dtype = dispatch_dtype
        self.combine_dtype = combine_dtype
        self.use_fp8_dispatch = dispatch_dtype == DispatchDtype.fp8
        self.use_fp4_dispatch = dispatch_dtype == DispatchDtype.fp4
        # EPLB routing tables (identity until EPLB is enabled).
        self.global_to_physical = (
            global_to_physical.to(torch.int32)
            if global_to_physical is not None
            else None
        )
        self.physical_to_global = physical_to_global
        self.local_expert_global_ids = local_expert_global_ids

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

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

    # ----- helpers -------------------------------------------------------

    def _op(self) -> "mori.ops.EpDispatchCombineOp":
        uid = dbo_current_ubatch_id()
        op = self._ops.get(uid)
        if op is None:
            op = self._handle_factory(uid)
            self._ops[uid] = op
        return op

    def _map_global_to_physical_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.global_to_physical is None:
            return topk_ids
        return self.global_to_physical[topk_ids]

    def _quantize(
        self,
        a1: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Quantize activations for dispatch according to dispatch_dtype.

        When defer_input_quant is True the expert kernel quantizes internally,
        so we dispatch the raw (bf16/fp16) activations.
        """
        if defer_input_quant or self.dispatch_dtype == DispatchDtype.bf16:
            return a1, None

        num_token, hidden_size = a1.shape

        if self.use_fp4_dispatch:
            from aiter import QuantType, get_hip_quant

            if num_token > 0:
                fp4_quant_func = get_hip_quant(QuantType.per_1x32)
                return fp4_quant_func(a1, shuffle=False)
            # aiter can mishandle the token==0 case at e2e; build empties.
            empty_a1 = torch.empty(
                (0, hidden_size // 2),
                dtype=torch.float4_e2m1fn_x2,
                device=a1.device,
            )
            empty_scale = torch.empty(
                (0, hidden_size // MXFP4_BLOCK_SIZE),
                dtype=torch.float8_e8m0fnu,
                device=a1.device,
            )
            return empty_a1, empty_scale

        # FP8 dispatch
        from aiter import QuantType, get_hip_quant

        fp8_dtype = current_platform.fp8_dtype()
        if num_token == 0:
            empty_a1 = torch.empty((0, hidden_size), dtype=fp8_dtype, device=a1.device)
            empty_scale = torch.empty(
                (0, hidden_size // FP8_BLOCK_SIZE),
                dtype=torch.float32,
                device=a1.device,
            )
            return empty_a1, empty_scale
        if quant_config.is_block_quantized:
            quant_func = get_hip_quant(QuantType.per_1x128)
            return quant_func(a1, quant_dtype=fp8_dtype)
        if quant_config.is_per_act_token:
            quant_func = get_hip_quant(QuantType.per_Token)
            return quant_func(a1, quant_dtype=fp8_dtype)
        return a1, None

    def _build_prepare_result(self, dispatch_out: tuple) -> mk.PrepareResultType:
        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = dispatch_out
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
        )
        return (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        )

    # ----- prepare / finalize -------------------------------------------

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
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True now."
        )
        a1, scale = self._quantize(a1, quant_config, defer_input_quant)
        dispatch_ids = self._map_global_to_physical_ids(topk_ids)

        op = self._op()
        dispatch_out = op.dispatch(a1, topk_weights, scale, dispatch_ids)
        return self._build_prepare_result(dispatch_out)

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        num_token = output.shape[0]
        combine_ids = self._map_global_to_physical_ids(topk_ids)

        op = self._op()
        # Combine on-wire quantization (fp8 blockwise/direct-cast) is configured
        # in EpDispatchCombineConfig.quant_type; the returned buffer is bf16.
        result = op.combine(fused_expert_output, None, combine_ids)[0]
        output.copy_(result[:num_token])
