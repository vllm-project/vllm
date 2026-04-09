import torch
import triton
import triton.language as tl
from torch._subclasses.fake_tensor import FakeTensor


@triton.jit
def moe_fused_mul_sum_kernel(
    inputs_ptr,
    topk_weights_ptr,
    outputs_ptr,
    top_ids_ptr,
    expert_map_ptr,
    num_tokens,
    stride_m,
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
    k_mask = offs_k < size
    mask = m_mask[:, None] & k_mask[None, :]

    a_base = inputs_ptr + (offs_m * stride_m)[:, None] + offs_k[None, :]
    b_base = topk_weights_ptr + offs_m * top_k

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in tl.static_range(top_k):
        b_val = tl.load(b_base + n, mask=m_mask, other=0.0).to(tl.float32)
        if has_expert_map:
            id_val = tl.load(top_ids_ptr + offs_m * top_k + n, mask=m_mask, other=0)
            expert_mask = tl.load(expert_map_ptr + id_val) >= 0
            a_vec = tl.load(
                a_base + n * size,
                mask=mask & expert_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            a_vec = tl.load(
                a_base + n * size,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        acc += a_vec * b_val[:, None]

    out_ptrs = outputs_ptr + (offs_m * size)[:, None] + offs_k[None, :]
    tl.store(
        out_ptrs,
        acc.to(outputs_ptr.dtype.element_ty),
        mask=mask,
    )


def _get_sm_version() -> int:
    return torch.cuda.get_device_capability()[0]


def _heuristic_config(
    num_tokens: int,
    top_k: int,
    size: int,
    element_size: int,
    sm_major: int,
):
    is_fp32 = element_size > 2
    is_sm90_plus = sm_major >= 9

    if is_sm90_plus:
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
    elif sm_major <= 7:
        max_block_k = 512
    elif is_sm90_plus:
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

    if sm_major <= 7:
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
) -> torch.Tensor:
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

    if not isinstance(inputs, FakeTensor):
        sm_major = _get_sm_version()
        BLOCK_M, BLOCK_K, num_warps, num_stages = _heuristic_config(
            num_tokens,
            top_k,
            size,
            inputs.element_size(),
            sm_major,
        )
        grid = (triton.cdiv(size, BLOCK_K), triton.cdiv(num_tokens, BLOCK_M))
        moe_fused_mul_sum_kernel[grid](
            inputs,
            topk_weights,
            outputs,
            topk_ids,
            expert_map,
            num_tokens,
            top_k * size,
            expert_map is not None,
            top_k,
            size,
            BLOCK_M,
            BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return outputs
