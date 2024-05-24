import triton
import triton.language as tl
import torch


@triton.jit
def _sgmv_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    b_seq_start_loc,
    seq_lens,
    lora_indices,
    xm_stride,  # hidden_size
    xk_stride,  # 1
    l0_stride,  # hidden_size*max_rank
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_sk = tl.program_id(axis=1)
    cur_batch = tl.program_id(axis=2)
    cta_n_num = tl.cdiv(N, BLOCK_N)
    pid_m = pid // cta_n_num
    pid_n = pid % cta_n_num

    M = tl.load(seq_lens + cur_batch)
    if pid_m * BLOCK_M > M:
        return
    cur_seq_start = tl.load(b_seq_start_loc + cur_batch)
    offset_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    lora_index = tl.load(lora_indices + cur_batch)
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride +
             offset_k[None, :] * xk_stride)
    b_ptr = (lora_ptr + l0_stride * lora_index + rbn[None, :] * lora_k_stride +
             offset_k[:, None] * lora_n_stride)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(a_ptr)
            b = tl.load(b_ptr)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(a_ptr, mask=offset_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptr, mask=offset_k[:, None] < k_remaining, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptr += BLOCK_K * SPLIT_K * xk_stride
        b_ptr += BLOCK_K * SPLIT_K * lora_n_stride
    offset_cm = cur_seq_start + tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    c_ptr = (out_ptr + offset_cm[:, None] * cm_stride +
             offset_cn[None, :] * cn_stride)

    c_mask = (offset_cm[:, None] <
              (cur_seq_start + M)) & (offset_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask)


@torch.inference_mode()
def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batchs: int,
    max_seq_length: int,
):
    """_summary_

    Args:
        inputs (torch.Tensor): _description_
        lora_a_weights (torch.Tensor): _description_
        output_tensor (torch.Tensor): _description_
        b_seq_start_loc (torch.Tensor): _description_
        seq_len_tensor (torch.Tensor): _description_
        lora_indices_tensor (torch.Tensor): _description_
        batchs (int): _description_
        max_seq_length (int): _description_
    """
    _, N, K = lora_a_weights.shape  # K=hidden_size,N=rank
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    SPLIT_K = 8
    EVEN_K = K % (SPLIT_K * BLOCK_K) == 0

    grid = [
        triton.cdiv(max_seq_length, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        SPLIT_K,
        batchs,
    ]
    _sgmv_shrink_kernel[grid](
        inputs,
        lora_a_weights,
        output_tensor,
        N,
        K,
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        inputs.stride(0),
        inputs.stride(1),
        lora_a_weights.stride(0),
        lora_a_weights.stride(1),
        lora_a_weights.stride(2),
        output_tensor.stride(0),
        output_tensor.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
    )
    return
