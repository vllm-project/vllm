# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_triton_kernels
from .bitonic_sort import bitonic_sort_warp_size_descending

logger = init_logger(__name__)

if has_triton_kernels():
    try:
        import triton_kernels.swiglu
        from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, matmul_ogs
        from triton_kernels.routing import RoutingData, routing, routing_from_bitmatrix
        from triton_kernels.tensor import Bitmatrix
    except (AttributeError, ImportError) as e:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible. Error: %s",
            e,
        )


@triton.jit
def pack_bitmatrix(
    bitmatrix,
    topk_ids,
    n_rows,  # n_rows in bitmatrix / topk_ids
    bm_cols: tl.constexpr,  # n int32_t bitpacks in bitmatrix
    n_expts_act,  # num_topk
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Packs topk_ids into a bitmatrix.
    code reference:
    https://github.com/triton-lang/triton/blob/dd1bbc52b34d202dfe5ffea1e04fb16166c5c04e/python/triton_kernels/bench/distributed.py#L264
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    # Iterate through all the relevant bitmatrix columns.
    for i in range(bm_cols):
        # When BLOCK_SIZE_K=32, offs is just the column index.
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        # All topks that need to go into this column has the correct bit set.
        # Other bits are 0. x is a 2D tensor.
        x = tl.where(
            div[:, :, None] == offs[None, None, :], (one << rem)[:, :, None], 0
        )
        # Reduce x to get a single int32_t bitpack.
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)

@triton.autotune(
    configs=[
        triton.Config({"ROWS_PER_PID": r}, num_warps=num_warps, num_stages=num_stages)
        for r in [1, 2, 4, 8, 16, 32]
        for num_warps in [1, 2, 4, 8, 16]
        for num_stages in [1, 2, 3]
    ],
    key=["N", "topk"],
    cache_results=True,
)
@triton.jit
def _topk_softmax_kernel(
    logits_ptr,
    weights_ptr,
    indices_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    topk: tl.constexpr,
    stride_lm,
    stride_ln,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    topk_padded: tl.constexpr,
    RENORM: tl.constexpr,
    ROWS_PER_PID: tl.constexpr,
    num_stages: tl.constexpr,
    USE_BITONIC: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, topk_padded)
    mask_n = offs_n < N
    store_mask = offs_k < topk
    warp_size: tl.constexpr = 32

    # impl topk<=2 and RENORM specialization by tl.constexpr,
    # same as constexpr if in C++17
    if topk == 1:
        for row_idx in tl.range(pid, M, num_programs, num_stages, warp_specialize=True):
            if BLOCK_N != N:
                logits = tl.load(
                    logits_ptr + row_idx * stride_lm + offs_n * stride_ln,
                    mask=mask_n,
                    other=float("-inf"),
                )
            else:
                logits = tl.load(
                    logits_ptr + row_idx * stride_lm + offs_n * stride_ln,
                )

            if not RENORM:
                row_sub_max = logits - tl.max(logits, axis=0)
                numerator = tl.exp(row_sub_max)
                denominator = tl.sum(numerator, axis=0)
                logits = numerator / denominator

            cur_max = 1 if RENORM else tl.max(logits, axis=0)
            cur_idx = tl.argmax(logits, axis=0)

            tl.store(weights_ptr + row_idx * stride_wm + 0 * stride_wk, cur_max)
            tl.store(indices_ptr + row_idx * stride_im + 0 * stride_ik, cur_idx)

    elif topk == 2:
        for row_idx in tl.range(pid, M, num_programs, num_stages, warp_specialize=True):
            if BLOCK_N != N:
                logits = tl.load(
                    logits_ptr + row_idx * stride_lm + offs_n * stride_ln,
                    mask=mask_n,
                    other=float("-inf"),
                )
            else:
                logits = tl.load(
                    logits_ptr + row_idx * stride_lm + offs_n * stride_ln,
                )

            if not RENORM:
                row_sub_max = logits - tl.max(logits, axis=0)
                numerator = tl.exp(row_sub_max)
                denominator = tl.sum(numerator, axis=0)
                logits = numerator / denominator

            val0 = tl.max(logits, axis=0)
            idx0 = tl.argmax(logits, axis=0)
            logits = tl.where(offs_n == idx0, float("-inf"), logits)
            val1 = tl.max(logits, axis=0)
            idx1 = tl.argmax(logits, axis=0)

            if RENORM:
                max_val = tl.maximum(val0, val1)
                exp0 = tl.exp(val0 - max_val)
                exp1 = tl.exp(val1 - max_val)
                val0 = exp0 / (exp0 + exp1)
                val1 = exp1 / (exp0 + exp1)

            tl.store(weights_ptr + row_idx * stride_wm, val0)
            tl.store(indices_ptr + row_idx * stride_im, idx0)
            tl.store(weights_ptr + row_idx * stride_wm + 1 * stride_wk, val1)
            tl.store(indices_ptr + row_idx * stride_im + 1 * stride_ik, idx1)

    else:
        rows = tl.arange(0, ROWS_PER_PID)
        for row_idx in tl.range(
            pid * ROWS_PER_PID,
            M,
            num_programs * ROWS_PER_PID,
            num_stages,
            warp_specialize=True,
        ):
            topk_vals = tl.full(
                [ROWS_PER_PID, topk_padded], float("-inf"), dtype=tl.float32
            )
            topk_idxs = tl.zeros([ROWS_PER_PID, topk_padded], dtype=tl.int32)
            row_indices = row_idx + rows  # [ROWS_PER_POD,]
            row_mask = row_indices < M

            # broadcast to [ROWS_PER_PID, BLOCKN]
            ptr_off = (
                logits_ptr
                + row_indices[:, None] * stride_lm
                + offs_n[None, :] * stride_ln
            )
            logits = tl.load(ptr_off)

            if not RENORM:
                logits = tl.softmax(logits, dim=1, keep_dims=True)

            if N == BLOCK_N and warp_size == N:
                # TODO(ijpq): we should enable this sort when N <= warp_size,
                # but need align tensor's layout with warp.
                # leverage PTX to sort warp_size experts to bypass sharedmemory
                idx = tl.arange(0, warp_size)[None, :]
                idxs = tl.broadcast_to(idx, (ROWS_PER_PID, warp_size))
                sorted_val, sorted_idx = bitonic_sort_warp_size_descending(
                    val=logits, idx=idxs
                )  # [ROWS_PER_PID, 32]
                tl.static_assert(sorted_val.shape == (ROWS_PER_PID, warp_size))
            else:
                # XXX: may use topk from triton_kernels
                for k in tl.static_range(topk):
                    cur_max = tl.max(
                        logits, axis=1, keep_dims=True
                    )  # [ROWS_PER_PID, 1]
                    cur_idx = tl.argmax(logits, axis=1, keep_dims=True)

                    k_mask = offs_k == k
                    topk_vals = tl.where(
                        k_mask, cur_max, topk_vals
                    )  # [ROWS_PER PID, 1], [ROWS_PER PID, topkpadded]
                    topk_idxs = tl.where(k_mask, cur_idx, topk_idxs)

                    mask_selected = (
                        cur_idx == offs_n[None, :]
                    )  # [ROWSPERPID,1] [1,BLOCKN]
                    logits = tl.where(mask_selected, float("-inf"), logits)

            if RENORM:
                if USE_BITONIC:
                    topk_col_mask = (
                        tl.arange(0, warp_size)[None, :] < topk
                    )  # [1, warp_size]
                    masked_val = tl.where(topk_col_mask, sorted_val, float("-inf"))
                    masked_val = tl.softmax(masked_val, dim=1)
                    sorted_val = tl.where(
                        topk_col_mask, masked_val, sorted_val
                    )
                else:
                    topk_vals = topk_vals - tl.max(
                        topk_vals, axis=1, keep_dims=True
                    )  # [ROWSPERPID, topkpadded] - [ROWSPERPID,1]
                    numerator = tl.exp(topk_vals)
                    denominator = tl.sum(
                        numerator, axis=1, keep_dims=True
                    )  # [ROWSPERPID,1]
                    topk_vals = numerator / denominator  # [ROWSPERPID,topkpadded]

            # WB
            if USE_BITONIC:
                offs_warp_size = tl.arange(0, warp_size)
                store_col_mask = offs_warp_size < topk
                tl.store(
                    weights_ptr
                    + row_indices[:, None] * stride_wm
                    + offs_warp_size[None, :] * stride_wk,
                    sorted_val,
                    mask=row_mask[:, None] & store_col_mask[None, :],
                )
                tl.store(
                    indices_ptr
                    + row_indices[:, None] * stride_im
                    + offs_warp_size[None, :] * stride_ik,
                    sorted_idx,
                    mask=row_mask[:, None] & store_col_mask[None, :],
                )
            else:
                tl.store(
                    weights_ptr
                    + row_indices[:, None] * stride_wm  # [ROWSPERPID,1]
                    + offs_k[None, :] * stride_wk,  # [1, topkpadded]
                    topk_vals,
                )
                tl.store(
                    indices_ptr
                    + row_indices[:, None] * stride_im
                    + offs_k[None, :] * stride_ik,
                    topk_idxs,
                )
               

def fused_topk_softmax(
    router_logits: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = router_logits.shape  # num_tokens, num_experts
    weights = torch.empty((M, topk), device=router_logits.device, dtype=router_logits.dtype)
    indices = torch.empty((M, topk), device=router_logits.device, dtype=torch.int32)

    BLOCK_N = triton.next_power_of_2(N)  # num_padded_experts
    topk_padded = triton.next_power_of_2(topk)
    BLOCK_M = triton.next_power_of_2(M)
    warp_size = 32

    grid = lambda META: (triton.cdiv(M, META["ROWS_PER_PID"]),)

    _topk_softmax_kernel[grid](
        logits_ptr=router_logits,
        weights_ptr=weights,
        indices_ptr=indices,
        M=M,
        N=N,
        topk=topk,
        stride_lm=router_logits.stride(0),
        stride_ln=router_logits.stride(1),
        stride_wm=weights.stride(0),
        stride_wk=weights.stride(1),
        stride_im=indices.stride(0),
        stride_ik=indices.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        topk_padded=topk_padded,
        RENORM=renormalize,
        USE_BITONIC=False
    )

    return weights, indices

def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    quant_config: FusedMoEQuantConfig | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    topk_weights, topk_indices = fused_topk_softmax(gating_output, topk, renormalize)
    routing_data, gather_idx, scatter_idx = make_routing_data(topk_indices, topk_weights, num_local_experts=gating_output.shape[-1])

    output = torch.empty_like(hidden_states)

    return triton_kernel_fused_experts(
        output,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        topk=topk,
        activation=activation,
        quant_config=quant_config,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx
    scatter_indx,  # ScatterIndx
    topk: int,
    activation: str = "silu",
    quant_config: FusedMoEQuantConfig | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    intermediate_cache: torch.Tensor | None = None,
    a1q_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
    assert quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32

    # Shape check, only check non-mxfp4
    assert hidden_states.ndim == 2
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    batch_dim = 1
    M, K = hidden_states.shape[-2:]
    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    if intermediate_cache is None:
        intermediate_cache = torch.empty(
            (batch_dim, M * topk, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    # Add batch_dim to output buffer because matmul_ogs expects 3D output
    intermediate_cache = _resize_cache(
        intermediate_cache, (batch_dim, M * topk, N // 2)
    )
    output_tensor = _resize_cache(output_tensor, (batch_dim, M, K))

    act = FusedActivation(
        FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
        (swiglu_alpha, swiglu_limit),
        2,
    )
    gammas = routing_data.gate_scal if routing_data else None

    matmul_ogs(
        hidden_states,
        w1,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=quant_config.w1_precision,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act,
        y=intermediate_cache,
    )

    matmul_ogs(
        intermediate_cache.view(M * topk, N // 2),
        w2,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=quant_config.w2_precision,
        gammas=None if apply_router_weight_on_input else gammas,
        y=output_tensor,
    )
    output_tensor = output_tensor.view(M, K)
    return output_tensor


def make_routing_data(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
) -> tuple["RoutingData", torch.Tensor, torch.Tensor]:
    topk_ids = topk_ids.to(torch.int16)
    topk_weights = topk_weights.to(torch.bfloat16)

    n_rows, num_topk = topk_ids.size()

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32

    bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)  # n_bitpacks
    bitmatrix = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
    )

    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    bitmatrix_shape = [n_rows, bm_cols * 32]
    bitmatrix_shape_max = [n_rows, None]
    bitmatrix = Bitmatrix(
        bitmatrix, shape=bitmatrix_shape, shape_max=bitmatrix_shape_max, scratchpad=None
    )

    # matmul_ogs expects invalid topk_weights to be -1s
    topk_weights = torch.where(topk_ids == -1, -1.0, topk_weights)
    routing_data, gather_indx, scatter_indx = routing_from_bitmatrix(
        bitmatrix, topk_weights, topk_ids, num_local_experts, num_topk
    )

    return routing_data, gather_indx, scatter_indx


class BaseOAITritonExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(self, quant_config: FusedMoEQuantConfig):
        super().__init__(quant_config)

    def supports_expert_map(self) -> bool:
        return True

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """
        Extract the MoE problem size from the given tensor arguments:
        - a: The hidden states, input to the MoE layer.
        - w1: The first set of expert weights.
        - w2: The second set of expert weights.
        - topk_ids: The topk ids.
        Note: extracting the problem shape from the weight and activation
        tensors is not obvious.  It needs to be done this way specifically
        due to subtle issues with particular kernels, e.g. the int4 kernels
        divide the trailing dimension by two, so it's not "correct" to
        extract N or K from the trailing dimension of w1 or w2.  Similarly,
        some kernels transpose the weights, so this needs to be kept in mind.
        Note: This implementation covers most cases. However, if experts
        require a specialized implementation, like MarlinExperts, they are free
        to override this function.
        """
        assert w1.dim() == 3 and w2.dim() == 3
        E, _, N = w1.size()
        K = a1.size(-1)

        assert a1.dim() == 2
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        M = a1.size(0)

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Weight application and reduction happens in the fused_experts kernel.
        return TopKWeightAndReduceNoOP()

    def _make_routing_data(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_local_experts: int,
    ) -> tuple["RoutingData", torch.Tensor, torch.Tensor]:
        return make_routing_data(topk_ids, topk_weights, num_local_experts)


class OAITritonExperts(BaseOAITritonExperts):
    def __init__(self, quant_config: FusedMoEQuantConfig):
        # TODO (varun) : Enable activation quantization
        assert quant_config.use_mxfp4_w4a16, "Supports only mxfp4_w4a16"
        super().__init__(quant_config)

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # workspace are allocated inside the kernel
        workspace1 = (0, 0)
        workspace2 = (M * topk, N // 2)
        output = (M, K)
        return (workspace1, workspace2, output)

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
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )

        topk = topk_ids.size(1)
        triton_kernel_fused_experts(
            output,
            hidden_states,
            w1,
            w2,
            routing_data,
            gather_indx,
            scatter_indx,
            topk=topk,
            activation=activation,
            quant_config=self.quant_config,
            apply_router_weight_on_input=False,
            global_num_experts=local_num_experts,
            expert_map=None,  # applied already
            intermediate_cache=workspace2,
            a1q_scale=a1q_scale,
        )


class UnfusedOAITritonExperts(BaseOAITritonExperts):
    """
    A Triton based MoE expert class that operates on expert standard
    format and explicitly keeps the activation and reduction (moe_sum) steps
    unfused from the matmul_ogs kernel. This exposes injection points
    for activation and moe_sum.

    One use case for it is to inject LoRA modules on the activation and moe_sum.
    """

    def __init__(self, quant_config: FusedMoEQuantConfig):
        # TODO (varun) : Enable activation quantization
        assert quant_config.use_mxfp4_w4a16, "Supports only mxfp4_w4a16"
        super().__init__(quant_config)

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # workspace are allocated inside the kernel
        workspace1 = (M * topk, N // 2)
        workspace2 = (M * topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor):
        ops.moe_sum(input, output)

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
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if self.quant_config is None:
            self.quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )

        topk = topk_ids.size(1)

        # type check, uint8 means mxfp4
        assert hidden_states.dtype == torch.bfloat16
        assert (
            self.quant_config.w1_bias is None
            or self.quant_config.w1_bias.dtype == torch.float32
        )
        assert (
            self.quant_config.w2_bias is None
            or self.quant_config.w2_bias.dtype == torch.float32
        )

        # Shape check, only check non-mxfp4
        assert hidden_states.ndim == 2
        assert hidden_states.shape[-1] == w1.shape[-2]
        assert w2.shape[-1] == w1.shape[1]

        batch_dim = 1
        M, K = hidden_states.shape
        E, _, N = w1.shape

        if global_num_experts == -1:
            global_num_experts = E

        # Note that the output tensor might be in workspace13
        intermediate_cache1 = _resize_cache(workspace2, (batch_dim, M * topk, N))
        intermediate_cache3 = _resize_cache(workspace2, (batch_dim, M * topk, K))
        intermediate_cache2 = _resize_cache(workspace13, (M * topk, N // 2))

        gammas = routing_data.gate_scal if routing_data else None

        matmul_ogs(
            hidden_states,
            w1,
            self.quant_config.w1_bias,
            routing_data,
            gather_indx=gather_indx,
            precision_config=self.quant_config.w1_precision,
            gammas=gammas if apply_router_weight_on_input else None,
            fused_activation=None,
            y=intermediate_cache1,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        # matmul_ogs grouped reduction fuse sum across multiple experts:
        # y[dst_ind // n_expts_act, :] += x[src_ind, :]
        # Need to set n_expts_act to 1 to unfuse moe_sum
        routing_data.n_expts_act = 1

        matmul_ogs(
            intermediate_cache2,
            w2,
            self.quant_config.w2_bias,
            routing_data,
            scatter_indx=scatter_indx,
            precision_config=self.quant_config.w2_precision,
            gammas=None if apply_router_weight_on_input else gammas,
            y=intermediate_cache3,
        )

        self.moe_sum(intermediate_cache3.view(-1, topk, K), output)
