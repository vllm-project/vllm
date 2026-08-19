# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch._subclasses.fake_tensor import FakeTensor

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# Persistent-launch oversubscription factor: the fixed grid is
# multi_processor_count * this. Kept modest so idle programs (when the valid
# row count is small) cost little; the grid-stride loop covers larger inputs.
_PERSISTENT_BLOCKS_PER_SM = 4


@triton.jit
def moe_fused_mul_sum_kernel(
    inputs_ptr,
    topk_weights_ptr,
    outputs_ptr,
    top_ids_ptr,
    expert_map_ptr,
    num_tokens,
    stride_m,
    has_topk_ids: tl.constexpr,
    has_expert_map: tl.constexpr,
    top_k: tl.constexpr,
    size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_mask = offs_m < num_tokens

    if has_expert_map:
        # Skip worst-case padding rows (all top ids < 0) with no host sync.
        row_valid = tl.zeros((BLOCK_M,), dtype=tl.int32)
        for n in tl.static_range(top_k):
            idn = tl.load(top_ids_ptr + offs_m * top_k + n, mask=m_mask, other=-1)
            row_valid += (idn >= 0).to(tl.int32)
        if tl.sum(row_valid) == 0:
            return
        m_mask = m_mask & (row_valid > 0)

    k_mask = offs_k < size
    mask = m_mask[:, None] & k_mask[None, :]

    a_base = inputs_ptr + (offs_m * stride_m)[:, None] + offs_k[None, :]
    b_base = topk_weights_ptr + offs_m * top_k

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in tl.static_range(top_k):
        b_val = tl.load(b_base + n, mask=m_mask, other=0.0).to(tl.float32)
        if has_topk_ids:
            # -1 marks a slot the expert GEMM skipped (non-local expert under
            # EP, or an all2all padding row), so `inputs` was never written
            # there. Both the map lookup and the value load must be masked off:
            # indexing the map with -1 reads out of bounds.
            id_val = tl.load(top_ids_ptr + offs_m * top_k + n, mask=m_mask, other=-1)
            valid = id_val >= 0
            if has_expert_map:
                local_id = tl.load(
                    expert_map_ptr + tl.where(valid, id_val, 0),
                    mask=valid,
                    other=-1,
                )
                valid = valid & (local_id >= 0)
            row_mask = mask & valid[:, None]
        else:
            row_mask = mask
        a_vec = tl.load(
            a_base + n * size,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        acc += a_vec * b_val[:, None]

    out_ptrs = outputs_ptr + (offs_m * size)[:, None] + offs_k[None, :]
    tl.store(
        out_ptrs,
        acc.to(outputs_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def moe_fused_mul_sum_persistent_kernel(
    inputs_ptr,
    topk_weights_ptr,
    outputs_ptr,
    top_ids_ptr,
    expert_map_ptr,
    num_valid_tokens_ptr,
    num_tokens,
    stride_m,
    has_topk_ids: tl.constexpr,
    has_expert_map: tl.constexpr,
    top_k: tl.constexpr,
    size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
):
    # Persistent variant: the launch grid is a fixed function of the SM count
    # (see moe_fused_mul_sum), not of num_tokens, so it stays static under CUDA
    # graph capture. The real row count is read from device (no host sync) and
    # bounds the grid-stride loop, so worst-case padding rows past num_recv are
    # never iterated instead of being launched as empty CTAs that early-return.
    row_bound = tl.load(num_valid_tokens_ptr).to(tl.int32)
    row_bound = tl.minimum(row_bound, num_tokens)

    num_m_tiles = tl.cdiv(row_bound, BLOCK_M)
    total_tiles = num_m_tiles * NUM_K_TILES

    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    for tile_id in range(pid, total_tiles, num_pid):
        pid_m = tile_id // NUM_K_TILES
        pid_k = tile_id % NUM_K_TILES

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        m_mask = offs_m < row_bound
        k_mask = offs_k < size

        keep_m = m_mask
        do_work = True
        if has_expert_map:
            # Row is kept (and its output written) iff it has any top id >= 0.
            # This uses top_ids only -- a real row whose ids all map to non-local
            # experts still has row_present > 0 and must be zeroed, not skipped.
            row_present = tl.zeros((BLOCK_M,), dtype=tl.int32)
            for n in tl.static_range(top_k):
                idn = tl.load(top_ids_ptr + offs_m * top_k + n, mask=m_mask, other=-1)
                row_present += (idn >= 0).to(tl.int32)
            keep_m = m_mask & (row_present > 0)
            do_work = tl.sum(row_present) > 0

        if do_work:
            store_mask = keep_m[:, None] & k_mask[None, :]
            a_row = inputs_ptr + (offs_m * stride_m)[:, None] + offs_k[None, :]
            b_base = topk_weights_ptr + offs_m * top_k
            acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

            for n in tl.static_range(top_k):
                b_val = tl.load(b_base + n, mask=m_mask, other=0.0).to(tl.float32)
                if has_topk_ids:
                    id_val = tl.load(
                        top_ids_ptr + offs_m * top_k + n, mask=m_mask, other=-1
                    )
                    valid = id_val >= 0
                    if has_expert_map:
                        local_id = tl.load(
                            expert_map_ptr + tl.where(valid, id_val, 0),
                            mask=valid,
                            other=-1,
                        )
                        valid = valid & (local_id >= 0)
                    row_mask = store_mask & valid[:, None]
                else:
                    row_mask = store_mask
                a_vec = tl.load(a_row + n * size, mask=row_mask, other=0.0).to(
                    tl.float32
                )
                acc += a_vec * b_val[:, None]

            out_ptrs = outputs_ptr + (offs_m * size)[:, None] + offs_k[None, :]
            tl.store(
                out_ptrs,
                acc.to(outputs_ptr.dtype.element_ty),
                mask=store_mask,
            )


def _heuristic_config(
    num_tokens: int,
    top_k: int,
    size: int,
    element_size: int,
):
    is_fp32 = element_size > 2
    is_sm90_plus = current_platform.has_device_capability(90)
    is_sm80_before = not current_platform.has_device_capability(80)

    if current_platform.has_device_capability(90):
        # SM90/SM100+: prefer small tiles + many CTAs.
        if is_fp32:
            BLOCK_M = 1 if num_tokens <= 4 else 2
        else:
            if num_tokens <= 4:
                BLOCK_M = 1
            elif num_tokens <= 128:
                BLOCK_M = 2
            else:
                BLOCK_M = 4
    elif is_fp32:
        if num_tokens <= 4:
            BLOCK_M = 1
        elif num_tokens <= 32:
            BLOCK_M = 2
        elif num_tokens <= 128:
            BLOCK_M = 4
        else:
            BLOCK_M = 4
    else:
        if num_tokens <= 4:
            BLOCK_M = 1
        elif num_tokens <= 32:
            BLOCK_M = 2
        elif num_tokens <= 128:
            BLOCK_M = 4
        elif num_tokens <= 1024:
            BLOCK_M = 16
        else:
            BLOCK_M = 8

    if is_fp32:
        max_block_k = 256
    elif is_sm80_before or is_sm90_plus:
        max_block_k = 512
    else:
        max_block_k = 1024
    BLOCK_K = min(triton.next_power_of_2(size), max_block_k)
    BLOCK_K = max(BLOCK_K, 256)

    total = BLOCK_M * BLOCK_K
    if is_fp32:
        num_warps = max(8, min(16, total // 64))
    else:
        num_warps = max(4, min(16, total // 256))

    if is_sm80_before:
        num_warps = min(num_warps, 8)
        num_stages = 2
    elif is_sm90_plus:
        num_warps = min(num_warps, 8)
        num_stages = 4 if total <= 2048 else 2
    else:
        num_stages = 4 if total <= 2048 else 2

    return BLOCK_M, BLOCK_K, num_warps, num_stages


def moe_fused_mul_sum(
    inputs: torch.Tensor,
    topk_weights: torch.Tensor,
    outputs: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    num_valid_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused kernel for MoE (Mixture of Experts) to perform weighted summation
    of expert outputs.

    Args:
        inputs: The output from experts.
            Shape: (num_tokens, top_k, hidden_size).
        topk_weights: The weights assigned to each expert for each token.
            Shape: (num_tokens, top_k).
        outputs: Optional pre-allocated output tensor.
            Shape: (num_tokens, hidden_size).
        topk_ids: Optional indices of the top-k experts. Shape:
            (num_tokens, top_k). A value of -1 marks a slot the expert GEMM
            skipped; those slots are excluded from the sum. Required when
            `expert_map` is provided.
        expert_map: Optional mapping for Expert Parallelism. A value < 0
            indicates an invalid token/expert pair that will be skipped. When
            provided, rows with all top ids < 0 (worst-case padding) are skipped
            and their output rows left untouched.
        num_valid_tokens: Optional device scalar (1-element tensor) holding the
            number of real token rows (e.g. num_recv for a decode dispatch).
            When provided, a persistent kernel with a fixed, CUDA-graph-safe grid
            is launched and only rows [0, num_valid_tokens) are processed; the
            padding tail is never iterated. Pass the token count, not
            token*top_k.

    Returns:
        The fused weighted sum of expert outputs.
        Shape: (num_tokens, hidden_size).
    """
    assert inputs.ndim == 3
    assert topk_weights.ndim == 2
    assert inputs.is_contiguous()
    assert topk_weights.is_contiguous()
    assert inputs.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert topk_weights.dtype in (torch.float32, torch.float16, torch.bfloat16)

    num_tokens, top_k, size = inputs.shape
    output_shape = (num_tokens, size)
    if outputs is None:
        outputs = torch.empty(output_shape, dtype=inputs.dtype, device=inputs.device)

    assert outputs.shape == output_shape
    assert topk_weights.shape == (num_tokens, top_k)
    assert expert_map is None or topk_ids is not None, (
        "topk_ids is required to interpret expert_map"
    )

    if not isinstance(inputs, FakeTensor):
        BLOCK_M, BLOCK_K, num_warps, num_stages = _heuristic_config(
            num_tokens,
            top_k,
            size,
            inputs.element_size(),
        )
        if num_valid_tokens is not None:
            # Occupancy is register-limited by the fp32 accumulator width, not by
            # pipelining. With BLOCK_K=512 / 256 threads the acc tile alone costs
            # 8 fp32/thread (+ the per-n a_vec load), pushing ~79 regs/thread ->
            # 3 blocks/SM (past the 64-reg cliff). Halving BLOCK_K to 256 halves
            # both to 4 fp32/thread -> ~63 regs -> 4 blocks/SM, and doubles the
            # k-tile count so the fixed grid is better filled when num_recv (the
            # real row count) is small. num_stages barely affects this masked,
            # no-MMA reduce, so just cap it to keep buffers off the reg file.
            persistent_block_k = min(BLOCK_K, 256)
            num_k_tiles = triton.cdiv(size, persistent_block_k)
            num_sms = torch.cuda.get_device_properties(
                inputs.device
            ).multi_processor_count
            max_tiles = triton.cdiv(num_tokens, BLOCK_M) * num_k_tiles
            grid: tuple[int, ...] = (
                min(num_sms * _PERSISTENT_BLOCKS_PER_SM, max_tiles),
            )
            persistent_num_stages = min(num_stages, 2)
            moe_fused_mul_sum_persistent_kernel[grid](
                inputs,
                topk_weights,
                outputs,
                topk_ids,
                expert_map,
                num_valid_tokens,
                num_tokens,
                top_k * size,
                topk_ids is not None,
                expert_map is not None,
                top_k,
                size,
                BLOCK_M,
                persistent_block_k,
                num_k_tiles,
                num_warps=num_warps,
                num_stages=persistent_num_stages,
            )
        else:
            grid = (triton.cdiv(size, BLOCK_K), triton.cdiv(num_tokens, BLOCK_M))
            moe_fused_mul_sum_kernel[grid](
                inputs,
                topk_weights,
                outputs,
                topk_ids,
                expert_map,
                num_tokens,
                top_k * size,
                topk_ids is not None,
                expert_map is not None,
                top_k,
                size,
                BLOCK_M,
                BLOCK_K,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    return outputs
