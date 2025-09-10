# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP)
from vllm.triton_utils import tl, triton
from vllm.utils import has_triton_kernels

logger = init_logger(__name__)

if has_triton_kernels():
    try:
        import triton_kernels.swiglu
        from triton_kernels.matmul_ogs import (FnSpecs, FusedActivation,
                                               matmul_ogs)
        from triton_kernels.routing import (RoutingData, routing,
                                            routing_from_bitmatrix)
        from triton_kernels.tensor import Bitmatrix
    except ModuleNotFoundError:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible.")

if TYPE_CHECKING:
    from triton_kernels.matmul_ogs import PrecisionConfig


@triton.jit
def populate_bitmatrix_kernel(
        topk_ids,
        topk_row_stride,
        topk_col_stride,
        bm,  # bitmatrix
        bm_row_stride,
        bm_col_stride,
        num_rows,  # topk_ids rows 
        num_cols: tl.constexpr,  # topk_ids cols
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr):

    pid = tl.program_id(0)
    m_start = pid * BLOCK_M

    offsets = tl.arange(0, BLOCK_M)
    mask = offsets < (num_rows - m_start)

    topk_ptrs = topk_ids + m_start * topk_row_stride + offsets * topk_row_stride
    bm_ptrs = bm + m_start * bm_row_stride + offsets * bm_row_stride

    one = tl.cast(1, dtype=tl.uint32)

    for i in tl.range(num_cols):
        topk = tl.load(topk_ptrs, mask=mask, other=-1)
        bm_mask = topk != -1
        bm_offset = topk // 32
        rem = topk % 32
        bits = tl.where(topk == -1, 0, one << rem)
        bm_load_store_ptrs = bm_ptrs + bm_offset * bm_col_stride

        existing_bits = tl.load(bm_load_store_ptrs, mask=bm_mask, other=0)
        bits |= existing_bits

        tl.store(bm_load_store_ptrs, bits, mask=bm_mask)
        topk_ptrs += topk_col_stride


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    w1_precision: Optional["PrecisionConfig"] = None,
    w2_precision: Optional["PrecisionConfig"] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:

    routing_data, gather_idx, scatter_idx = routing(gating_output,
                                                    topk,
                                                    sm_first=not renormalize)

    return triton_kernel_fused_experts(
        None,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        w1_precision=w1_precision,
        w2_precision=w2_precision,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape)


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx
    scatter_indx,  # ScatterIndx
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    w1_precision: Optional["PrecisionConfig"] = None,
    w2_precision: Optional["PrecisionConfig"] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:

    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert w1_bias is None or w1_bias.dtype == torch.float32
    assert w2_bias is None or w2_bias.dtype == torch.float32

    # Shape check, only check non-mxfp4
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    act = FusedActivation(
        FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
        (swiglu_alpha, swiglu_limit), 2)
    gammas = routing_data.gate_scal if routing_data else None

    intermediate_cache1 = matmul_ogs(
        hidden_states,
        w1,
        w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=w1_precision,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act)

    intermediate_cache3 = matmul_ogs(
        intermediate_cache1,
        w2,
        w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=w2_precision,
        gammas=None if apply_router_weight_on_input else gammas,
        y=output_tensor,
    )
    return intermediate_cache3


class BaseOAITritonExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(self, moe: FusedMoEConfig, w1_precision: "PrecisionConfig",
                 w2_precision: "PrecisionConfig",
                 w1_bias: Optional[torch.Tensor],
                 w2_bias: Optional[torch.Tensor]):

        super().__init__(moe.quant_config)
        self.w1_precision = w1_precision
        self.w2_precision = w2_precision
        self.w1_bias = w1_bias
        self.w2_bias = w2_bias

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceNoOP()

    def _make_bitmatrix(self, topk_ids: torch.Tensor, num_local_experts: int):

        # Following code from the topk_forward function at,
        # https://github.com/triton-lang/triton/blob/7871be232696d2112f7030e467ec35f47db543b9/python/triton_kernels/triton_kernels/topk.py#L9
        cdiv = lambda a, b: (a + b - 1) // b
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_S = 128

        n_rows, _ = topk_ids.size()
        n_cols = num_local_experts
        n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
        n_cols_words = n_cols_pad // 32
        bitmatrix = torch.empty((n_cols_words, cdiv(n_rows, 32) * 32),
                                dtype=torch.uint32,
                                device=topk_ids.device)
        bitmatrix = torch.transpose(bitmatrix, 0, 1)[:n_rows]

        grid = [cdiv(n_rows, BLOCK_M)]
        populate_bitmatrix_kernel[grid](topk_ids, topk_ids.stride(0),
                                        topk_ids.stride(1), bitmatrix,
                                        bitmatrix.stride(0),
                                        bitmatrix.stride(1), topk_ids.size(0),
                                        topk_ids.size(1), BLOCK_M, BLOCK_N)

        s_blocks = cdiv(n_cols, BLOCK_S)
        s_cols = s_blocks * BLOCK_S
        scratchpad = torch.zeros((s_cols, ),
                                 dtype=torch.int32,
                                 device=topk_ids.device)

        bitmatrix_shape = [n_rows, n_cols_words * 32]
        bitmatrix_shape_max = [n_rows, None]
        return Bitmatrix(bitmatrix,
                         shape=bitmatrix_shape,
                         shape_max=bitmatrix_shape_max,
                         scratchpad=scratchpad)

    def _make_routing_data(
        self, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
        num_local_experts: int
    ) -> tuple[RoutingData, torch.Tensor, torch.Tensor]:
        """
        Return RoutingData, GatherIndx and ScatterIndx required for
        matmul_ogs.
        """
        topk_ids = topk_ids.to(torch.int16)
        topk_weights = topk_weights.to(torch.bfloat16)

        bitmatrix: Bitmatrix = self._make_bitmatrix(topk_ids,
                                                    num_local_experts)

        num_topk = topk_ids.size(1)
        routing_data, gather_indx, scatter_indx = routing_from_bitmatrix(
            bitmatrix,
            topk_weights,
            topk_ids,
            n_expts_tot=num_local_experts,
            n_expts_act=num_topk)
        return (routing_data, gather_indx, scatter_indx)


class OAITritonExperts(BaseOAITritonExperts):

    def __init__(
        self,
        moe: FusedMoEConfig,
        w1_precision: "PrecisionConfig",
        w2_precision: "PrecisionConfig",
        w1_bias: Optional[torch.Tensor],
        w2_bias: Optional[torch.Tensor],
    ):
        super().__init__(moe, w1_precision, w2_precision, w1_bias, w2_bias)

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.Standard,
                mk.FusedMoEActivationFormat.Standard)

    def supports_chunking(self) -> bool:
        return True

    def workspace_shapes(
        self, a: torch.Tensor, aq: torch.Tensor, M: int, N: int, K: int,
        topk: int, global_num_experts: int, local_num_experts: int,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        # workspace are allocated inside the kernel
        workspace1 = (M, K)
        workspace2 = (0, 0)
        output = (M, K)
        return (workspace1, workspace2, output, a.dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):

        assert expert_map is not None
        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        num_local_experts = w1.size(0)
        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, num_local_experts)

        # Make bitmatrix and RoutingData
        print("In OAITritonExperts ...")
        experts_output = triton_kernel_fused_experts(
            None,
            hidden_states,
            w1,
            w2,
            routing_data,
            gather_indx,
            scatter_indx,
            activation=activation,
            apply_router_weight_on_input=False,
            use_fp8_w8a8=False,
            per_channel_quant=False,
            global_num_experts=num_local_experts,
            expert_map=None,  # applied already
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=self.w1_bias,
            w2_bias=self.w2_bias,
            w1_precision=self.w1_precision,
            w2_precision=self.w2_precision,
            a1_scale=a1q_scale,
            a2_scale=a2_scale)

        print(
            f"experts output : {experts_output.shape} | output : {output.shape}"
        )
        output.copy_(experts_output, non_blocking=True)
        torch.cuda.synchronize()
        assert not torch.isnan(output).any()


class BatchedOAITritonExperts(BaseOAITritonExperts):

    def __init__(
        self,
        moe: FusedMoEConfig,
        max_num_tokens: int,
        num_dispatchers: int,
        w1_precision: "PrecisionConfig",
        w2_precision: "PrecisionConfig",
        w1_bias: Optional[torch.Tensor],
        w2_bias: Optional[torch.Tensor],
    ):
        super().__init__(moe, w1_precision, w2_precision, w1_bias, w2_bias)
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.BatchedExperts,
                mk.FusedMoEActivationFormat.BatchedExperts)

    def supports_chunking(self) -> bool:
        return False

    def workspace_shapes(
        self, a: torch.Tensor, aq: torch.Tensor, M: int, N: int, K: int,
        topk: int, global_num_experts: int, local_num_experts: int,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        # workspace are allocated inside the kernel
        assert a.dim() == 2
        num_dp = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = self.max_num_tokens
        workspace2 = (0, 0, 0)
        output = (num_experts, max_num_tokens * num_dp, N)
        return (output, workspace2, output, a.dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):
        print("In BatchedOAITritonExperts ...")
        return triton_kernel_fused_experts(
            output,
            hidden_states,
            w1,
            w2,
            None,
            None,
            None,
            activation=activation,
            apply_router_weight_on_input=False,
            use_fp8_w8a8=False,
            per_channel_quant=False,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=self.w1_bias,
            w2_bias=self.w2_bias,
            w1_precision=self.w1_precision,
            w2_precision=self.w2_precision,
            a1_scale=a1q_scale,
            a2_scale=a2_scale)
