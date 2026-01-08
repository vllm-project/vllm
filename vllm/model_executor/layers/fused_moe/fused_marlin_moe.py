# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE utilities for GPTQ."""

from collections.abc import Callable

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    batched_moe_align_block_size,
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    apply_moe_activation,
    disable_inplace,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_moe_intermediate_size,
    marlin_quant_input,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types


def _fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    num_topk: int,
    quant_type: ScalarType,
    apply_router_weight_on_input: bool,
    expert_map: torch.Tensor | None,
    block_size_m: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    activation: str = "silu",
    activation_func: Callable[
        [str, torch.Tensor, torch.Tensor], None
    ] = apply_moe_activation,
    input_global_scale1: torch.Tensor | None = None,
    input_global_scale2: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    g_idx1: torch.Tensor | None = None,
    g_idx2: torch.Tensor | None = None,
    sort_indices1: torch.Tensor | None = None,
    sort_indices2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    intermediate_cache13: torch.Tensor | None = None,
    intermediate_cache2: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
    is_k_full: bool = True,
) -> torch.Tensor:
    assert hidden_states.ndim == 2
    M, K = hidden_states.size()
    N = marlin_moe_intermediate_size(w1, w2)
    w13_num_shards = 1 if "no_mul" in activation else 2
    if workspace is None:
        workspace = marlin_make_workspace_new(hidden_states.device, 4)

    if intermediate_cache13 is None:
        intermediate_cache13 = torch.empty(
            (M * num_topk * max(w13_num_shards * N, K),),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    if intermediate_cache2 is None:
        intermediate_cache2 = torch.empty(
            (M * num_topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    intermediate_cache1 = _resize_cache(
        intermediate_cache13, (M * num_topk, w13_num_shards * N)
    )

    intermediate_cache3 = _resize_cache(intermediate_cache13, (M * num_topk, K))

    intermediate_cache2 = _resize_cache(intermediate_cache2, (M * num_topk, N))

    a_scales1 = None
    gate_up_input = hidden_states
    if input_dtype == torch.int8:
        gate_up_input, a_scales1 = marlin_quant_input(hidden_states, input_dtype)
        if input_global_scale1 is not None:
            a_scales1 = a_scales1 * input_global_scale1
    elif input_dtype == torch.float8_e4m3fn:
        gate_up_input, a_scales1 = marlin_quant_input(hidden_states, input_dtype)

    intermediate_cache1 = ops.moe_wna16_marlin_gemm(
        gate_up_input,
        intermediate_cache1,
        w1,
        bias1,
        w1_scale,
        a_scales1,
        global_scale1,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=num_topk,
        mul_topk_weights=apply_router_weight_on_input,
        b_q_type=quant_type,
        size_m=M,
        size_n=w13_num_shards * N,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    activation_func(
        activation,
        intermediate_cache2,
        intermediate_cache1.view(-1, w13_num_shards * N),
    )

    if output is None:
        output = intermediate_cache3

    if expert_map is not None:
        output.zero_()

    a_scales2 = None
    if input_dtype == torch.int8:
        intermediate_cache2, a_scales2 = marlin_quant_input(
            intermediate_cache2, input_dtype
        )
        if input_global_scale2 is not None:
            a_scales2 = a_scales2 * input_global_scale2
    elif input_dtype == torch.float8_e4m3fn:
        intermediate_cache2, a_scales2 = marlin_quant_input(
            intermediate_cache2, input_dtype
        )

    output = ops.moe_wna16_marlin_gemm(
        intermediate_cache2,
        output,
        w2,
        bias2,
        w2_scale,
        a_scales2,
        global_scale2,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=not apply_router_weight_on_input,
        b_q_type=quant_type,
        size_m=M * num_topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    return output


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: str = "silu",
    activation_func: Callable[
        [str, torch.Tensor, torch.Tensor], None
    ] = apply_moe_activation,
    moe_sum: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    expert_map: torch.Tensor | None = None,
    input_global_scale1: torch.Tensor | None = None,
    input_global_scale2: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    g_idx1: torch.Tensor | None = None,
    g_idx2: torch.Tensor | None = None,
    sort_indices1: torch.Tensor | None = None,
    sort_indices2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    intermediate_cache13: torch.Tensor | None = None,
    intermediate_cache2: torch.Tensor | None = None,
    is_k_full: bool = True,
    output: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
    inplace: bool = False,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - g_idx1 (torch.Tensor|None): The first set of act_order indices.
    - g_idx2 (torch.Tensor|None): The second set of act_order indices.
    - sort_indices1 (torch.Tensor|None): The first act_order input
        permutation.
    - sort_indices2 (torch.Tensor|None): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (torch.Tensor|None): Optional zero points to be used for w1.
    - w2_zeros (torch.Tensor|None): Optional zero points to be used for w2.
    - num_bits (bool): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """

    if inplace:
        assert output is None, "Conflicting request"

    quant_type = ScalarType.from_id(quant_type_id)
    assert quant_type in [
        scalar_types.uint4,
        scalar_types.uint8b128,
        scalar_types.uint4b8,
        scalar_types.float8_e4m3fn,
        scalar_types.float4_e2m1f,
    ]

    bit4_scalar_types = [
        scalar_types.uint4,
        scalar_types.uint4b8,
        scalar_types.float4_e2m1f,
    ]
    num_bits = 4 if quant_type in bit4_scalar_types else 8

    M, K = hidden_states.size()
    E = w1.size(0)
    topk = topk_ids.size(1)

    # Check constraints.
    assert w1.size(1) * 16 == K, "Hidden size mismatch w1"
    assert w2.size(2) // (num_bits // 2) == K, "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert num_bits in [4, 8]
    assert topk_weights.dtype == torch.float32

    # M block size selection logic
    # TODO: tune this further for specific models
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

    if input_dtype is not None and input_dtype.itemsize == 1:
        block_size_m = max(block_size_m, 16)

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size_m,
        global_num_experts,
        expert_map,
        ignore_invalid_experts=True,
    )

    assert activation is not None
    moe_output = _fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        bias1=bias1,
        bias2=bias2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        num_topk=topk,
        quant_type=quant_type,
        apply_router_weight_on_input=apply_router_weight_on_input,
        expert_map=expert_map,
        block_size_m=block_size_m,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        activation=activation,
        activation_func=activation_func,
        input_global_scale1=input_global_scale1,
        input_global_scale2=input_global_scale2,
        global_scale1=global_scale1,
        global_scale2=global_scale2,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=w1_zeros,
        w2_zeros=w2_zeros,
        workspace=workspace,
        intermediate_cache13=intermediate_cache13,
        intermediate_cache2=intermediate_cache2,
        output=None,
        input_dtype=input_dtype,
        is_k_full=is_k_full,
    ).view(-1, topk, K)

    if output is None:
        if inplace and not disable_inplace():
            output = hidden_states
        else:
            output = torch.empty_like(hidden_states)

    if moe_sum is None:
        return torch.sum(moe_output.view(-1, topk, K), dim=1, out=output)
    else:
        return moe_sum(moe_output, output)


def batched_fused_marlin_moe(
    hidden_states: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: str | None = "silu",
    expert_map: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    g_idx1: torch.Tensor | None = None,
    g_idx2: torch.Tensor | None = None,
    sort_indices1: torch.Tensor | None = None,
    sort_indices2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    intermediate_cache13: torch.Tensor | None = None,
    intermediate_cache2: torch.Tensor | None = None,
    is_k_full: bool = True,
    output: torch.Tensor | None = None,
    inplace: bool = False,
) -> torch.Tensor:
    """
    This function massages the inputs so the batched hidden_states can be
    presented as a 2D contiguous tensor that could be used with
    _fused_marlin_moe.

    Note that both batched_fused_marlin_moe and fused_marlin_moe ultimately
    use `ops.moe_wna16_marlin_gemm` for the gemm operation and
    `ops.moe_mna16_marlin_gemm` supports only 2D contiguous hidden_states.
    Note that the moe_align_block_size function indicates,
        - What rows of the A matrix (hidden_states) to access during the
        matmul, via sorted_ids output.
        - What expert_id to use for each block matmul, via expert_ids ouptut.

    In the batched version, the tokens are already grouped/batched by experts
    they subscribe to. Due to this, we can represent the batched hidden_states
    tensor of shape [B, MAX_TOKENS_PER_BATCH, K] as a 2D tensor of shape,
    [B * MAX_TOKENS_PER_BATCH, K]. We may treat this a 2D contiguous tensor
    with topk=1 as each token (row in the tensor) subscribes to exactly one
    expert_id (which is the batch_id). With the expert_num_tokens tensor, that
    indicates how many tokens are actually valid in each batch, the
    batched_moe_align_block_size function constructs the sorted_ids and
    expert_ids tensors, so only relevant/valid rows of A (hidden_states)
    are accessed and are processed with the correct expert_ids.
    """

    assert hidden_states.ndim == 3, (
        f"hidden states must be batched. e.g. [B, MAX_TOKENS, K]."
        f"But got {hidden_states.size()}"
    )
    if inplace:
        assert output is None, "Conflicting request."

    quant_type = ScalarType.from_id(quant_type_id)
    assert quant_type in [
        scalar_types.uint4,
        scalar_types.uint8b128,
        scalar_types.uint4b8,
        scalar_types.float8_e4m3fn,
        scalar_types.float4_e2m1f,
    ]

    bit4_scalar_types = [
        scalar_types.uint4,
        scalar_types.uint4b8,
        scalar_types.float4_e2m1f,
    ]
    num_bits = 4 if quant_type in bit4_scalar_types else 8

    B, BATCH_TOKENS_MAX, K = hidden_states.size()
    M = hidden_states.view(-1, K).size(0)
    E = w1.size(0)

    # Check constraints.
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert expert_num_tokens.size(0) == E
    assert B == E, (
        "Batch must be as big as number of experts as the tokens"
        "are sorted into the batch/expert they belong to"
    )
    assert w1.size(1) * 16 == K, "Hidden size mismatch w1"
    assert w2.size(2) // (num_bits // 2) == K, "Hidden size mismatch w2"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert num_bits in [4, 8]

    # Technically, the tokens are already separated by their expert ids.
    # Hidden-States can just be squeezed to have just 2 dimensions,
    # [B * MAX_TOKENS, K] and top_k can be interpreted as just 1.
    topk = 1

    # TODO(varun) : Choose a decent block size like in fused_marlin_moe
    block_size_m = 64

    sorted_token_ids, expert_ids, num_tokens_post_padded = batched_moe_align_block_size(
        max_tokens_per_batch=BATCH_TOKENS_MAX,
        block_size=block_size_m,
        expert_num_tokens=expert_num_tokens,
    )

    if output is None and inplace:
        output = hidden_states

    # TODO (varun): This can be avoided by plumbing the marlin kernel to
    # ignore topk_weights when topk_weights_ptr is a nullptr.
    topk_weights = torch.ones(
        (M, topk), device=hidden_states.device, dtype=torch.float32
    )

    assert activation is not None
    output = _fused_marlin_moe(
        hidden_states=hidden_states.view(-1, K),
        w1=w1,
        w2=w2,
        bias1=bias1,
        bias2=bias2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        num_topk=topk,
        quant_type=quant_type,
        apply_router_weight_on_input=apply_router_weight_on_input,
        activation=activation,
        expert_map=expert_map,
        block_size_m=block_size_m,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        global_scale1=global_scale1,
        global_scale2=global_scale2,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=w1_zeros,
        w2_zeros=w2_zeros,
        workspace=workspace,
        intermediate_cache13=intermediate_cache13,
        intermediate_cache2=intermediate_cache2,
        output=output.view(-1, K) if output is not None else output,
        is_k_full=is_k_full,
    )

    output = output.view(B, BATCH_TOKENS_MAX, K)

    return output


class MarlinExpertsBase(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
        w13_g_idx: torch.Tensor | None = None,
        w2_g_idx: torch.Tensor | None = None,
        w13_g_idx_sort_indices: torch.Tensor | None = None,
        w2_g_idx_sort_indices: torch.Tensor | None = None,
        is_k_full: bool = True,
    ):
        # TODO (varun) : Enable activation quantization
        assert (
            quant_config.use_mxfp4_w4a16
            or quant_config.use_nvfp4_w4a16
            or quant_config.use_int4_w4a16
            or quant_config.use_fp8_w8a16
        ), "Supports only {mxfp,nvfp,int}4_w4a16 or fp8_w8a16"
        self.w13_g_idx = w13_g_idx
        self.w2_g_idx = w2_g_idx
        self.w13_g_idx_sort_indices = w13_g_idx_sort_indices
        self.w2_g_idx_sort_indices = w2_g_idx_sort_indices
        self.is_k_full = is_k_full
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.has_device_capability((7, 5))

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # TODO(rob): add int4, mxfp4, int8 as integrations
        # are migrated to use the oracle one-by-one.
        SUPPORTED_W = [
            kFp8Static128BlockSym,
            kFp8StaticChannelSym,
            kFp8StaticTensorSym,
            kNvfp4Static,
        ]
        return weight_key in SUPPORTED_W

    @staticmethod
    def _supports_activation(activation: str) -> bool:
        return activation in [
            "silu",
            "gelu",
            "swigluoai",
            "silu_no_mul",
            "gelu_no_mul",
            "relu2_no_mul",
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @property
    def quant_type_id(self) -> int:
        # uint4b8 will be set for int4 weight and float4_e2m1f will be used for mxfp4
        if self.quant_config.use_int4_w4a16:
            return scalar_types.uint4b8.id
        elif self.quant_config.use_mxfp4_w4a16 or self.quant_config.use_nvfp4_w4a16:
            return scalar_types.float4_e2m1f.id
        elif (
            self.quant_config.use_fp8_w8a16
            and current_platform.fp8_dtype() == torch.float8_e4m3fn
        ):
            return scalar_types.float8_e4m3fn.id
        else:
            raise NotImplementedError("Unsupported quantization type.")

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        assert w1.dim() == 3 and w2.dim() == 3

        E = w1.size(0)
        K = a1.size(-1)
        N = marlin_moe_intermediate_size(w1, w2)

        if a1.dim() == 2:
            # Make sure we are using the correct a1 (pre-permute).
            assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
            M = a1.size(0)
        else:
            assert a1.dim() == 3
            assert a1.size(0) == E, f"{a1.size(0)} == {E}"
            M = a1.size(1)  # This is max_num_tokens

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk


class MarlinExperts(MarlinExpertsBase):
    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

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
        activation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Modular Kernel provisions output buffer from workspace1. However in
        # the fused_marlin_moe() function, the final torch.sum(), is defined
        # essentially as,
        # `torch.sum(workspace1, dim=1, out=output)`
        # Having overlapping input and output tensors for torch.sum seems
        # error prone and depends on how the torch.sum is implemented.
        # For this reason we swap let the output buffer provision from
        # workspace2.

        # Workspace/IntermediateCache allocation matching fused_marlin_moe()
        # workspace1 = (M * topk * max(2 * N, K),)
        # workspace2 = (M * topk, N)

        # Workspace/IntermediateCache allocation accounting for output buffer
        # provisioning
        workspace1 = (M * topk, max(N, K))
        workspace2 = (M * topk * max(2 * N, K),)
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
        assert self.w1_scale is not None
        assert self.w2_scale is not None
        return fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            bias1=self.w1_bias,
            bias2=self.w2_bias,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_scale1=self.g1_alphas,
            global_scale2=self.g2_alphas,
            quant_type_id=self.quant_type_id,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            activation=activation,
            activation_func=self.activation,
            moe_sum=self.moe_sum,
            expert_map=expert_map,
            output=output,
            # Workspaces are swapped in workspace_shapes() to account for proper
            # output buffer allocation. Please refer to workspace_shapes().
            intermediate_cache13=workspace2,
            intermediate_cache2=workspace13,
            g_idx1=self.w13_g_idx,
            g_idx2=self.w2_g_idx,
            sort_indices1=self.w13_g_idx_sort_indices,
            sort_indices2=self.w2_g_idx_sort_indices,
            is_k_full=self.is_k_full,
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        ops.moe_sum(input, output)


class BatchedMarlinExperts(MarlinExpertsBase):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
        w13_g_idx: torch.Tensor | None = None,
        w2_g_idx: torch.Tensor | None = None,
        w13_g_idx_sort_indices: torch.Tensor | None = None,
        w2_g_idx_sort_indices: torch.Tensor | None = None,
        is_k_full: bool = True,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
            w13_g_idx=w13_g_idx,
            w2_g_idx=w2_g_idx,
            w13_g_idx_sort_indices=w13_g_idx_sort_indices,
            w2_g_idx_sort_indices=w2_g_idx_sort_indices,
            is_k_full=is_k_full,
        )

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def supports_chunking(self) -> bool:
        return False

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert self.num_dispatchers is not None
        assert self.max_num_tokens is not None
        num_dispatchers = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = self.max_num_tokens
        workspace13 = (num_experts * max_num_tokens * num_dispatchers, max(K, N * 2))
        workspace2 = (num_experts * max_num_tokens * num_dispatchers, N)
        output = (num_experts, max_num_tokens * num_dispatchers, K)
        return (workspace13, workspace2, output)

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
        assert expert_tokens_meta is not None, "Num valid tokens per batch is required"
        return batched_fused_marlin_moe(
            hidden_states=hidden_states,
            expert_num_tokens=expert_tokens_meta.expert_num_tokens,
            w1=w1,
            w2=w2,
            bias1=self.w1_bias,
            bias2=self.w2_bias,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            quant_type_id=self.quant_type_id,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            activation=activation,
            expert_map=expert_map,
            output=output,
            intermediate_cache13=workspace13,
            intermediate_cache2=workspace2,
            g_idx1=self.w13_g_idx,
            g_idx2=self.w2_g_idx,
            sort_indices1=self.w13_g_idx_sort_indices,
            sort_indices2=self.w2_g_idx_sort_indices,
            is_k_full=self.is_k_full,
        )
