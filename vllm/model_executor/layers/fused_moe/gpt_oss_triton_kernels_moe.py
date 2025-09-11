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
        from triton_kernels.routing import (RoutingData, prune_routing,
                                            routing, routing_from_bitmatrix)
        from triton_kernels.tensor import Bitmatrix
    except ModuleNotFoundError:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible.")

if TYPE_CHECKING:
    from triton_kernels.matmul_ogs import PrecisionConfig
"""
code reference:
https://github.com/triton-lang/triton/blob/dd1bbc52b34d202dfe5ffea1e04fb16166c5c04e/python/triton_kernels/bench/distributed.py#L264
"""


@triton.jit
def pack_bitmatrix(
    bitmatrix,
    expt_indx,
    n_rows,
    n_cols,
    n_expts_act,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    sentinel: tl.constexpr,
):
    """
    Packs expt_indx into a bitmatrix.
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(expt_indx + offsets, mask=mask, other=sentinel)
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)
    iters = tl.cdiv(sentinel, BLOCK_SIZE_K)
    for i in range(iters):
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        x = tl.where(div[:, :, None] == offs[None, None, :],
                     (one << rem)[:, :, None], 0)
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:,
                                               None] * n_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)


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

    def _make_routing_data(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_local_experts: int,
        num_global_experts: int,
    ) -> tuple[RoutingData, torch.Tensor, torch.Tensor]:

        topk_ids = torch.where(topk_ids == -1, num_local_experts, topk_ids)

        topk_ids = topk_ids.to(torch.int16)
        topk_weights = topk_weights.to(torch.bfloat16)
        ep_size = num_global_experts // num_local_experts

        # Recover bitmatrix for local experts
        BLOCK_SIZE_M = 512
        BLOCK_SIZE_K = 32
        # The sentinel value is chunk_size + 1 instead of chunk_size to ensure
        # the bitmatrix buffer doesn't overflow. For example, cdiv(32, 32) is
        # 1, while the 32th bit is on the second column.
        sentinel = num_local_experts + 1
        n_cols = triton.cdiv(sentinel, BLOCK_SIZE_K)
        n_rows, num_topk = topk_ids.size()

        bitmatrix = torch.zeros((n_rows, n_cols),
                                dtype=torch.uint32,
                                device=topk_ids.device)
        grid = (triton.cdiv(n_rows, BLOCK_SIZE_M), )

        pack_bitmatrix[grid](
            bitmatrix,
            topk_ids,
            n_rows,
            n_cols,
            num_topk,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            sentinel=sentinel,
        )

        bitmatrix_shape = [
            n_rows, triton.cdiv(num_local_experts, BLOCK_SIZE_K) * 32
        ]
        bitmatrix_shape_max = [n_rows, None]
        bitmatrix = Bitmatrix(bitmatrix,
                              shape=bitmatrix_shape,
                              shape_max=bitmatrix_shape_max,
                              scratchpad=None)
        expt_scal, expt_indx, bitmatrix = prune_routing(
            topk_weights, topk_ids, bitmatrix, num_global_experts, ep_size)
        routing_data, gather_indx, scatter_indx = routing_from_bitmatrix(
            bitmatrix, expt_scal, expt_indx, num_local_experts, num_topk)

        return routing_data, gather_indx, scatter_indx


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

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts
        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, local_num_experts, global_num_experts)

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
            global_num_experts=local_num_experts,
            expert_map=None,  # applied already
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=self.w1_bias,
            w2_bias=self.w2_bias,
            w1_precision=self.w1_precision,
            w2_precision=self.w2_precision,
            a1_scale=a1q_scale,
            a2_scale=a2_scale)

        output.copy_(experts_output, non_blocking=True)
