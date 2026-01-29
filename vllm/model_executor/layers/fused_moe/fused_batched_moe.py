# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused batched MoE kernel."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNaiveBatched,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
    normalize_batched_scales_shape,
    normalize_scales_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import group_broadcast
from vllm.triton_utils import tl, triton


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
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # Offsets and masks
    offs_m,
    offs_n,
    offs_bn,
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
    use_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)

    if use_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + expert_id * stride_bse + offs_n[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = b_scale_ptr + offs_bsn * stride_bsn

        # per act token
        elif per_act_token_quant:
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=mask_m, other=0.0)[:, None]

            b_scale_ptrs = b_scale_ptr + offs_bn[None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)

        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # We accumulate along the K dimension.
        if use_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=mask_m, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                # acc used to enable fp8_fast_accum
                accumulator = tl.dot(a, b, acc=accumulator)
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
    a_ptr,  # [max_tokens, K]
    b_ptr,  # [K, N]
    c_ptr,  # [max_tokens, N]
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
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # offsets
    offs_bn,
    # Blockwise quantization data
    group_n,
    group_k,
    # Quantization schemes
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) % N
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    # Make grids of a + b pointers
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
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Offsets and masks
        offs_m,
        offs_n,
        offs_bn,
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
        use_int8_w8a16,
        per_act_token_quant,
    )

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
    stride_ae: tl.int64,
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_be: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_ce: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # Blockwise quantization data
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Quantization schemes
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # Early exit
        return

    # axis 1 is M_blocks * N_blocks
    pid_mn = tl.program_id(axis=1)
    # num_pid_m = tl.cdiv(max_num_tokens, BLOCK_M)
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
    c_ptr = (
        c_ptr
        + expert_id * stride_ce
        + cta_m_start * stride_cm
        + cta_n_start * stride_cn
    )

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N

    if use_fp8_w8a8:
        a_scale_ptr = a_scale_ptr + expert_id * stride_ase
        b_scale_ptr = b_scale_ptr + expert_id * stride_bse

        # block-wise
        if group_k > 0 and group_n > 0 or per_act_token_quant:
            a_scale_ptr = a_scale_ptr + cta_m_start * stride_asm

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
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # offsets
        offs_bn,
        # Blockwise quantization data
        group_n,
        group_k,
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
        # Kernel config
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )


def invoke_moe_batched_triton_kernel(
    A: torch.Tensor,  # [E, max_tokens, K]
    B: torch.Tensor,  # [E, N, K]
    C: torch.Tensor,  # [E, max_tokens, N]
    expert_num_tokens: torch.Tensor,  # [E]
    compute_type: tl.dtype,
    # Quantization data
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor,
    # Quantization schemes
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    config: dict[str, int],
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
):
    assert not use_int4_w4a16
    max_num_tokens = A.size(1)
    K = A.size(2)
    N = C.size(2)

    BLOCK_M = config["BLOCK_SIZE_M"]
    BLOCK_N = config["BLOCK_SIZE_N"]
    BLOCK_K = config["BLOCK_SIZE_K"]

    grid = (
        expert_num_tokens.size(0),
        triton.cdiv(max_num_tokens, BLOCK_M) * triton.cdiv(B.size(1), BLOCK_N),
    )

    A_scale = normalize_batched_scales_shape(A_scale, expert_num_tokens.shape[0])

    if B_scale is not None and B_scale.ndim == 1:
        assert B_scale.numel() == expert_num_tokens.shape[0]
        B_scale = B_scale.view(-1, 1, 1)

    assert A_scale is None or A_scale.ndim == 3, (
        f"{0 if A_scale is None else A_scale.shape}"
    )
    assert B_scale is None or B_scale.ndim == 1 or B_scale.ndim == 3, (
        f"{0 if B_scale is None else B_scale.shape}"
    )

    if B_scale is not None:
        if B_scale.ndim == 1:
            stride_bse = 1
            stride_bsk = 0
            stride_bsn = 0
        else:
            stride_bse = B_scale.stride(0)
            stride_bsk = B_scale.stride(2)
            stride_bsn = B_scale.stride(1)

    else:
        stride_bse = 0
        stride_bsk = 0
        stride_bsn = 0

    if A_scale is not None:
        stride_ase = A_scale.stride(0)
        stride_asm = A_scale.stride(1)
        stride_ask = A_scale.stride(2)
    else:
        stride_ase = 0
        stride_asm = 0
        stride_ask = 0

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
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Blockwise quantization data
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
        # Kernel config
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


class BatchedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    A reference prepare/finalize class that reorganizes the tokens into
    expert batched format, i.e. E x max_num_tokens x K.  This is the format
    that the PPLX dispatch/combine kernels use.
    """

    def __init__(
        self,
        max_num_tokens: int,
        num_local_experts: int,
        num_dispatchers: int,
        rank: int,
    ):
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts
        self.rank = rank
        self.num_dispatchers_ = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_num_tokens

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        assert a1.dim() == 2
        assert topk_ids.dim() == 2
        assert topk_ids.size(0) == a1.size(0)

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1.mul_(topk_weights.to(a1.dtype))

        num_tokens, hidden_dim = a1.size()
        topk = topk_ids.size(1)

        tokens_per_expert = torch.zeros(num_experts, dtype=torch.int, device=a1.device)

        num_local_experts = self.num_local_experts

        if quant_config.quant_dtype is None:
            b_type = a1.dtype
        else:
            b_type = quant_config.quant_dtype

        b_a1 = torch.zeros(
            (num_local_experts, self.max_num_tokens, hidden_dim),
            dtype=b_type,
            device=a1.device,
        )

        if quant_config.is_quantized:
            scale_shape = quant_config.batched_scale_shape(
                num_local_experts, self.max_num_tokens, hidden_dim
            )

            b_a1_scale = torch.empty(scale_shape, dtype=torch.float32, device=a1.device)
        else:
            assert quant_config.a1_scale is None
            b_a1_scale = None

        first_expert = num_local_experts * self.rank
        last_expert = first_expert + num_local_experts

        a1_scale = normalize_scales_shape(quant_config.a1_scale)

        for expert_id in range(first_expert, last_expert):
            topks = torch.any(topk_ids == expert_id, dim=1).flatten()
            rows = torch.count_nonzero(topks.flatten())
            if rows == 0:
                continue
            idx = expert_id - first_expert
            tokens_per_expert[idx] = rows
            rhs = a1[: topks.numel()][topks]
            if quant_config.quant_dtype is not None:
                if a1_scale is not None:
                    if quant_config.is_per_act_token:
                        rhs_a1_scale = a1_scale[: topks.numel()][topks]
                    else:
                        rhs_a1_scale = a1_scale
                else:
                    rhs_a1_scale = None
                b_a1[idx, :rows, :], b_s = moe_kernel_quantize_input(
                    rhs,
                    rhs_a1_scale,
                    quant_config.quant_dtype,
                    quant_config.per_act_token_quant,
                    quant_config.block_shape,
                )
                assert b_s is not None
                if quant_config.is_per_act_token:
                    b_a1_scale[idx, :rows] = b_s[:rows]
                else:
                    b_a1_scale[idx, : b_s.shape[0]] = b_s
            else:
                b_a1[idx, :rows, :] = rhs

        assert b_a1_scale is None or b_a1_scale.ndim == 3

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=tokens_per_expert, expert_num_tokens_cpu=None
        )

        return b_a1, b_a1_scale, expert_tokens_meta, None, None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceNaiveBatched(self.rank)
        weight_and_reduce_impl.apply(
            output=output,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


class NaiveBatchedExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    A reference MoE expert class that operates on expert batched format,
    i.e. E x max_num_tokens x K.  This is the format that the pplx
    dispatch/combine kernels use.
    """

    def __init__(
        self,
        max_num_tokens: int,
        num_dispatchers: int,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(quant_config)
        assert not self.quant_config.use_int8_w8a8, "NYI"
        assert not self.quant_config.use_int8_w8a16, "NYI"
        assert not self.quant_config.use_int4_w4a16, "NYI"
        assert self.quant_config.ocp_mx_scheme is None, "NYI"
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.BatchedExperts,
            mk.FusedMoEActivationFormat.BatchedExperts,
        )

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

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
        num_dp = self.num_dispatchers
        num_experts = local_num_experts
        workspace13 = (num_experts, self.max_num_tokens * num_dp, K)
        workspace2 = (self.max_num_tokens * num_dp, N)
        output = workspace13
        return (workspace13, workspace2, output)

    def dequant(self, t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        assert self.quant_config.is_quantized
        f32 = torch.float32
        if self.quant_config.is_per_act_token or self.quant_config.is_per_tensor:
            return t.to(f32) * scale
        else:
            return t.to(f32) * group_broadcast(scale, t.shape)

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
        assert hidden_states.dim() == 3
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        num_local_experts = w1.size(0)
        assert num_local_experts == w1.size(0), f"{num_local_experts} == {w1.size(0)}"

        N = w1.size(1) // 2

        for expert in range(num_local_experts):
            # Indexing expert_num_tokens doesn't work w/cudagraphs or inductor
            if (
                torch.compiler.is_compiling()
                or torch.cuda.is_current_stream_capturing()
            ):
                num = hidden_states.shape[1]
            else:
                num = int(expert_num_tokens[expert].item())

            if num == 0:
                continue

            tmp = _resize_cache(workspace2, (num, N))

            if self.quant_config.is_quantized:
                assert a1q_scale is not None and self.w1_scale is not None
                input = self.dequant(hidden_states[expert, :, :], a1q_scale[expert])
                w1_dq = self.dequant(w1[expert], self.w1_scale[expert])
                input = input[:num] @ w1_dq.transpose(0, 1)
            else:
                input = hidden_states[expert, :num, :] @ w1[expert].transpose(0, 1)

            self.activation(activation, tmp, input.to(tmp.dtype))

            if self.quant_config.is_quantized:
                assert self.w2_scale is not None
                w2_dq = self.dequant(w2[expert], self.w2_scale[expert])
            else:
                w2_dq = w2[expert]

            output[expert, :num, :] = tmp @ w2_dq.transpose(0, 1).to(tmp.dtype)


def batched_moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    num_tokens: int,
    E: int,
    N: int,
    expert_num_tokens: torch.Tensor,
    qtype: torch.dtype | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        # Note: this does a bunch of extra work because expert_num_tokens is
        # ignored but it does support torch.compile + cudagraphs.
        hidden_dim = A.size(-1)
        assert A_scale is None or A_scale.ndim <= 2, (
            f"{A_scale.shape if A_scale is not None else None}"
        )
        A_q, A_q_scale = moe_kernel_quantize_input(
            A.view(-1, hidden_dim), A_scale, qtype, per_act_token_quant, block_shape
        )
        A_q = A_q.view(E, -1, hidden_dim)
        A_q_scale = normalize_batched_scales_shape(A_q_scale, E)

        return A_q, A_q_scale
    elif qtype is None:
        return A, normalize_batched_scales_shape(A_scale, E)
    else:
        A_q = torch.empty_like(A, dtype=qtype)

        if per_act_token_quant:
            assert block_shape is None
            scale_shape = (E, num_tokens, 1)
        elif block_shape is not None:
            _, block_k = block_shape
            k_tiles = (A.shape[-1] + block_k - 1) // block_k
            scale_shape = (E, num_tokens, k_tiles)
        else:
            scale_shape = (E, 1, 1)

        A_q_scale = torch.zeros(scale_shape, dtype=torch.float32, device=A.device)

        num_experts = expert_num_tokens.numel()

        A_scale = normalize_batched_scales_shape(A_scale, num_experts)

        for e in range(E):
            num_tokens = int(expert_num_tokens[e].item())
            if num_tokens > 0:
                if A_scale is not None:
                    scales = A_scale[e, : min(num_tokens, A_scale.shape[1])]
                else:
                    scales = None
                A_q[e, :num_tokens], tmp_scale = moe_kernel_quantize_input(
                    A[e, :num_tokens],
                    scales,
                    qtype,
                    per_act_token_quant,
                    block_shape,
                )
                assert tmp_scale is not None
                A_q_scale[e, : tmp_scale.shape[0]] = tmp_scale

        return A_q, A_q_scale


class BatchedTritonExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    A Triton based MoE expert class that operates on expert batched format,
    i.e. E x max_num_tokens x K.  This is the format that the pplx
    dispatch/combine kernels use.
    """

    def __init__(
        self,
        max_num_tokens: int,
        num_dispatchers: int,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(quant_config)
        assert not self.quant_config.use_int8_w8a8, "NYI"
        assert not self.quant_config.use_int8_w8a16, "NYI"
        assert not self.quant_config.use_int4_w4a16, "NYI"
        assert self.quant_config.ocp_mx_scheme is None, "NYI"
        assert max_num_tokens > 0
        assert num_dispatchers > 0
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.BatchedExperts,
            mk.FusedMoEActivationFormat.BatchedExperts,
        )

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

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
        num_dp = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = self.max_num_tokens
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (num_experts, max_num_tokens * num_dp, max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dp, activation_out_dim)
        output = (num_experts, max_num_tokens * num_dp, K)
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
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
        ]
        assert expert_tokens_meta is not None

        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        E, max_num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        assert w1.size(0) == E
        assert w2.size(0) == E

        config_dtype = self.quant_config.config_name(hidden_states.dtype)

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            config_dtype,
            max_num_tokens,
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
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # We can reuse the memory between these because by the time we need
        # cache3, we're done with cache1
        intermediate_cache1 = _resize_cache(workspace13, (E, max_num_tokens, N))
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace2, (E, max_num_tokens, activation_out_dim)
        )

        # TODO(bnell): should this be done for any quantized type?
        if self.quant_config.use_fp8_w8a8:
            intermediate_cache1.fill_(0)

        a1q_scale = normalize_batched_scales_shape(a1q_scale, E)

        # MM1
        invoke_moe_batched_triton_kernel(
            A=hidden_states,
            B=w1,
            C=intermediate_cache1,
            expert_num_tokens=expert_num_tokens,
            compute_type=compute_type,
            A_scale=a1q_scale,
            B_scale=self.w1_scale,
            B_zp=self.w1_zp,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            config=config,
            per_act_token_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
        )

        intermediate_cache2.fill_(0)

        # TODO (bnell): use triton utility from batched deep gemm.
        self.activation(
            activation,
            intermediate_cache2.view(-1, activation_out_dim),
            intermediate_cache1.view(-1, N),
        )

        qintermediate_cache2, a2q_scale = batched_moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            max_num_tokens,
            E,
            N,
            expert_num_tokens,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
        )

        invoke_moe_batched_triton_kernel(
            A=qintermediate_cache2,
            B=w2,
            C=output,
            expert_num_tokens=expert_num_tokens,
            compute_type=compute_type,
            A_scale=a2q_scale,
            B_scale=self.w2_scale,
            B_zp=self.w2_zp,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            config=config,
            per_act_token_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
        )
