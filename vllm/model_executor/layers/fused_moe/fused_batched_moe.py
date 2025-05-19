# SPDX-License-Identifier: Apache-2.0
"""Fused batched MoE kernel."""
from typing import Optional

import torch
import triton
import triton.language as tl

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_config_dtype_str, try_get_optimal_moe_config)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache


@triton.jit
def moe_mmk(
        a_ptrs,
        b_ptrs,
        K,
        expert_id,
        a_scale_ptr,
        b_scale_ptr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_ak,
        stride_bk,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Offsets and masks
        offs_m,
        offs_n,
        mask_m,
        # Block size for block-wise quantization
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        # Meta-parameters
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        compute_type: tl.constexpr,
        use_w8a8: tl.constexpr,
        use_w8a16: tl.constexpr):

    offs_k = tl.arange(0, BLOCK_K)

    if use_w8a16:
        b_scale_ptrs = b_scale_ptr + expert_id * stride_bse + offs_n[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            offs_bsn = offs_n // group_n
            b_scale_ptrs = (b_scale_ptr + expert_id * stride_bse +
                            offs_bsn * stride_bsn)
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + expert_id)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_K),
                    other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # We accumulate along the K dimension.
        if use_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=mask_m,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:,
                                                      None] * b_scale[None, :]
            else:
                if use_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if use_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    return accumulator


@triton.jit
def expert_triton_kernel(
        a_ptr,  #[max_tokens, K]
        b_ptr,  #[K, N]
        c_ptr,  #[max_tokens, N]
        expert_id,
        compute_type: tl.constexpr,
        # Dimensions
        M,
        N,
        K,
        # Quantization data
        a_scale_ptr,
        b_scale_ptr,
        b_zp_ptr,
        # strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Blockwise quantization data
        group_n,
        group_k,
        # Quantization schemes
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr,
        # Kernel config
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr):

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) % N
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = moe_mmk(
        a_ptrs,
        b_ptrs,
        K,
        expert_id,
        a_scale_ptr,
        b_scale_ptr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_ak,
        stride_bk,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Offsets and masks
        offs_m,
        offs_n,
        mask_m,
        # Block size for block-wise quantization
        group_n,
        group_k,
        # Meta-parameters
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a16)

    # store in C
    offs_cn = tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = mask_m[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def batched_triton_kernel(
        a_ptr,  # [E, max_num_tokens, K]
        b_ptr,  # [E, K, N]
        c_ptr,  # [E, max_num_tokens, N]
        expert_num_tokens,  # [E]
        compute_type: tl.constexpr,
        # Dimensions
        max_num_tokens,
        K,
        N,
        # Quantization data
        a_scale_ptr,
        b_scale_ptr,
        b_zp_ptr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_ae,
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_ce,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Blockwise quantization data
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        # Quantization schemes
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr,
        # Kernel config
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr):
    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # Early exit
        return

    pid_mn = tl.program_id(axis=1)
    #num_pid_m = tl.cdiv(max_num_tokens, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    cta_m_start = pid_m * BLOCK_M
    cta_n_start = pid_n * BLOCK_N
    if cta_m_start >= e_num_tokens:
        # Early exit
        return

    cta_m_size = min(BLOCK_M, e_num_tokens - cta_m_start)
    cta_n_size = min(BLOCK_N, N - cta_n_start)

    a_ptr = a_ptr + expert_id * stride_ae + cta_m_start * stride_am
    b_ptr = b_ptr + expert_id * stride_be + cta_n_start * stride_bn
    c_ptr = (c_ptr + expert_id * stride_ce + cta_m_start * stride_cm +
             cta_n_start * stride_cn)

    expert_triton_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        expert_id,
        compute_type,
        cta_m_size,  # M
        cta_n_size,  # N
        K,  # K
        a_scale_ptr,
        b_scale_ptr,
        b_zp_ptr,
        # Strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Blockwise quantization data
        group_n,
        group_k,
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        # Kernel config
        BLOCK_M,
        BLOCK_N,
        BLOCK_K)


def invoke_moe_batched_triton_kernel(
        A: torch.Tensor,  # [E, max_tokens, K]
        B: torch.Tensor,  # [E, K, N]
        C: torch.Tensor,  # [E, max_tokens, N]
        expert_num_tokens: torch.Tensor,  # [E]
        compute_type: tl.dtype,
        # Quantization data
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        B_zp: torch.Tensor,
        # Quantization schemes
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        config: dict[str, int],
        block_shape: Optional[list[int]] = None):

    assert not use_int4_w4a16
    max_num_tokens = A.size(1)
    K = A.size(2)
    N = C.size(2)

    BLOCK_M = config['BLOCK_SIZE_M']
    BLOCK_N = config['BLOCK_SIZE_N']
    BLOCK_K = config['BLOCK_SIZE_K']
    assert (torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()
            or max_num_tokens % BLOCK_M == 0)

    grid = (expert_num_tokens.size(0), triton.cdiv(max_num_tokens, BLOCK_M) *
            triton.cdiv(B.size(1), BLOCK_N))

    batched_triton_kernel[grid](
        A,
        B,
        C,
        expert_num_tokens,
        compute_type,
        # Dimensions
        max_num_tokens,
        K,
        N,
        # Quantization data
        A_scale,
        B_scale,
        B_zp,
        # Strides
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        # Blockwise quantization data
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        # Kernel config
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K)


class BatchedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    A reference prepare/finalize class that reorganizes the tokens into
    expert batched format, i.e. E x max_num_tokens x K.  This is the format
    that the PPLX dispatch/combine kernels use.
    """

    def __init__(self, max_num_tokens: Optional[int], world_size: int,
                 dp_size: int, rank: int):
        super().__init__()
        self.world_size = world_size
        self.dp_size = dp_size
        self.rank = rank
        self.max_num_tokens = max_num_tokens

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert a1.dim() == 2
        assert topk_ids.dim() == 2
        assert topk_ids.size(0) == a1.size(0)

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, \
                "apply_router_weight_on_input is only implemented for topk=1"
            a1.mul_(topk_weights.to(a1.dtype))

        num_tokens, hidden_dim = a1.size()
        topk = topk_ids.size(1)

        if self.max_num_tokens is None:
            tokens_per_expert = torch.bincount(topk_ids.view(-1),
                                               minlength=num_experts)
            self.max_num_tokens = int(tokens_per_expert.max().item())
        else:
            tokens_per_expert = torch.zeros(num_experts,
                                            dtype=torch.int,
                                            device=a1.device)

        assert num_experts % self.world_size == 0

        num_local_experts = num_experts // self.world_size

        b_a1 = torch.zeros(
            (num_local_experts, self.max_num_tokens, hidden_dim),
            dtype=a1.dtype,
            device=a1.device)

        first_expert = num_local_experts * self.rank
        last_expert = first_expert + num_local_experts

        for expert_id in range(first_expert, last_expert):
            topks = torch.any(topk_ids == expert_id, dim=1).flatten()
            rows = torch.count_nonzero(topks.flatten())
            b_a1[expert_id -
                 first_expert, :rows, :] = a1[:topks.numel()][topks]
            tokens_per_expert[expert_id - first_expert] = rows

        return b_a1, a1_scale, tokens_per_expert

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        num_tokens = topk_ids.size(0)
        num_local_experts = fused_expert_output.size(0)
        K = fused_expert_output.size(-1)
        assert output.size(0) == num_tokens and output.size(1) == K

        output.fill_(0)

        first_expert = num_local_experts * self.rank
        last_expert = first_expert + num_local_experts

        for expert_id in range(first_expert, last_expert):
            matching_tokens = topk_ids == expert_id
            topks = torch.any(matching_tokens, dim=1).flatten()
            rows = torch.count_nonzero(topks)
            rhs = fused_expert_output[expert_id - first_expert, :rows, :]
            if not apply_router_weight_on_input:
                rhs.mul_(topk_weights[matching_tokens].view(rhs.size(0), 1))
            output[topks] = output[topks] + rhs


class BatchedExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    A reference MoE expert class that operates on expert batched format,
    i.e. E x max_num_tokens x K.  This is the format that the pplx
    dispatch/combine kernels use.
    """

    def __init__(
        self,
        world_size: int,
        dp_size: int,
        max_num_tokens: Optional[int] = None,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        block_shape: Optional[list[int]] = None,
        block_m: Optional[int] = None,
    ):
        super().__init__()
        assert block_shape is None
        assert block_m is None
        assert not use_fp8_w8a8, "NYI"
        assert not use_int8_w8a8, "NYI"
        assert not use_int8_w8a16, "NYI"
        assert not use_int4_w4a16, "NYI"
        self.max_num_tokens = max_num_tokens
        self.world_size = world_size
        self.dp_size = dp_size

    def workspace_shapes(
        self,
        a: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
    ) -> tuple[int, int, torch.dtype]:
        assert a.dim() == 2
        num_dp = self.world_size // self.dp_size
        max_num_tokens = a.size(
            0) if self.max_num_tokens is None else self.max_num_tokens
        #print(f"WORKSPACE {max_num_tokens} {num_dp}")
        workspace13 = num_experts * max_num_tokens * num_dp * K
        workspace2 = max_num_tokens * num_dp * N
        return (workspace13, workspace2, a.dtype)

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
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
        expert_num_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert hidden_states.dim() == 3
        assert expert_num_tokens is not None
        hidden_dim = hidden_states.size(-1)

        if self.max_num_tokens is None:
            max_num_tokens = hidden_states.size(1)
        else:
            max_num_tokens = self.max_num_tokens

        num_dp = self.world_size // self.dp_size
        num_experts = global_num_experts
        out = _resize_cache(workspace13,
                            (num_experts, max_num_tokens * num_dp, hidden_dim))
        num_local_experts = w1.size(0)
        assert num_local_experts == w1.size(0), (
            f"{num_local_experts} == {w1.size(0)}")

        N = w1.size(1) // 2

        # Not cudagraph friendly
        assert (torch.compiler.is_compiling()
                or torch.cuda.is_current_stream_capturing()
                or torch.all(expert_num_tokens <= max_num_tokens * num_dp)), (
                    f"{expert_num_tokens} <= {max_num_tokens * num_dp}")

        for expert in range(num_local_experts):
            # Indexing expert_num_tokens doesn't work w/cudagraphs or inductor
            if (torch.compiler.is_compiling()
                    or torch.cuda.is_current_stream_capturing()):
                num = max_num_tokens * num_dp
            else:
                num = int(expert_num_tokens[expert].item())
            tmp = _resize_cache(workspace2, (num, N))
            input = hidden_states[expert, :num, :] @ w1[expert].transpose(0, 1)
            self.activation(activation, tmp, input)
            out[expert, :num, :] = tmp @ w2[expert].transpose(0, 1)

        return out


class BatchedTritonExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    A Triton based MoE expert class that operates on expert batched format,
    i.e. E x max_num_tokens x K.  This is the format that the pplx
    dispatch/combine kernels use.
    """

    def __init__(
        self,
        max_num_tokens: Optional[int] = None,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        block_shape: Optional[list[int]] = None,
        world_size: int = 1,
        dp_size: int = 1,
    ):
        super().__init__()
        self.use_fp8_w8a8 = use_fp8_w8a8
        self.use_int8_w8a8 = use_int8_w8a8
        self.use_int4_w4a16 = use_int4_w4a16
        self.use_int8_w8a16 = use_int8_w8a16
        self.block_shape = block_shape
        self.max_num_tokens = max_num_tokens
        assert not use_int8_w8a8, "NYI"
        assert not use_int4_w4a16, "NYI"
        self.world_size = world_size
        self.dp_size = dp_size

    def workspace_shapes(
        self,
        a: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
    ) -> tuple[int, int, torch.dtype]:
        assert a.dim() == 2
        num_dp = self.world_size // self.dp_size
        max_num_tokens = a.size(
            0) if self.max_num_tokens is None else self.max_num_tokens
        workspace13 = num_experts * max_num_tokens * num_dp * max(K, N)
        workspace2 = num_experts * max_num_tokens * num_dp * (N // 2)
        return (workspace13, workspace2, a.dtype)

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
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
        expert_num_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Check constraints.
        if self.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), (
                "Hidden size mismatch")
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} "
                f"!= {w1.size(2)}")

        assert hidden_states.is_contiguous(
        ), "Hidden_states must be contiguous"
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn
        ]

        # TODO: num_tokens -> max_num_tokens?
        E, num_tokens, N, K, top_k_num = mk._moe_problem_size(
            hidden_states, w1, w2, topk_ids)

        assert w1.size(0) == E
        assert w2.size(0) == E

        config_dtype = get_config_dtype_str(use_fp8_w8a8=self.use_fp8_w8a8,
                                            use_int8_w8a16=self.use_int8_w8a16,
                                            use_int4_w4a16=self.use_int4_w4a16,
                                            dtype=hidden_states.dtype)

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            config_dtype,
            num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif hidden_states.dtype == torch.float8_e4m3fn:
            compute_type = tl.bfloat16
        else:
            raise ValueError(
                f"Unsupported compute_type: {hidden_states.dtype}")

        #print(f"shape: E={E}, M={num_tokens}, N={N}, K={K}, top_k={top_k_num}")
        # We can reuse the memory between these because by the time we need
        # cache3, we're done with cache1
        intermediate_cache1 = _resize_cache(workspace13, (E, num_tokens, N))
        intermediate_cache2 = _resize_cache(workspace2,
                                            (E, num_tokens, N // 2))
        intermediate_cache3 = _resize_cache(workspace13, (E, num_tokens, K))

        # MM1
        invoke_moe_batched_triton_kernel(A=hidden_states,
                                         B=w1,
                                         C=intermediate_cache1,
                                         expert_num_tokens=expert_num_tokens,
                                         compute_type=compute_type,
                                         A_scale=a1q_scale,
                                         B_scale=w1_scale,
                                         B_zp=w1_zp,
                                         use_fp8_w8a8=self.use_fp8_w8a8,
                                         use_int8_w8a16=self.use_int8_w8a16,
                                         use_int4_w4a16=self.use_int4_w4a16,
                                         config=config,
                                         block_shape=self.block_shape)

        # TODO: would be nice to use expert_num_tokens here to reduce
        # garbage compute
        self.activation(activation, intermediate_cache2.view(-1, N // 2),
                        intermediate_cache1.view(-1, N))

        #qintermediate_cache2 = intermediate_cache2
        a2q_scale = a2_scale
        # TODO (varun) : support w8a8
        assert not self.use_fp8_w8a8
        #if self.use_fp8_w8a8:
        #    qintermediate_cache2, a2q_scale = _fp8_quantize(
        #        intermediate_cache2, a2_scale, self.block_shape)

        invoke_moe_batched_triton_kernel(A=intermediate_cache2,
                                         B=w2,
                                         C=intermediate_cache3,
                                         expert_num_tokens=expert_num_tokens,
                                         compute_type=compute_type,
                                         A_scale=a2q_scale,
                                         B_scale=w2_scale,
                                         B_zp=w2_zp,
                                         use_fp8_w8a8=self.use_fp8_w8a8,
                                         use_int8_w8a16=self.use_int8_w8a16,
                                         use_int4_w4a16=self.use_int4_w4a16,
                                         config=config,
                                         block_shape=self.block_shape)

        return intermediate_cache3
