# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _oscar_dequant_kernel(
    cache_ptr,  # uint8 [..., SLOT_SIZE]
    out_k_ptr,  # float16 [seq_len, Hk, D]
    out_v_ptr,  # float16 [seq_len, Hk, D]
    Pi_half_ptr,  # float16 [D, D]
    slot_mapping_ptr,  # int64 [seq_len]
    stride_cache_slot,
    stride_cache_head,
    stride_out_seq,
    stride_out_head,
    stride_out_dim,
    stride_pi_0,
    stride_pi_1,
    seq_len,
    Hk,
    D: tl.constexpr,
    BLOCK_QUARTER: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)

    if pid_seq >= seq_len or pid_head >= Hk:
        return

    slot_idx = tl.load(slot_mapping_ptr + pid_seq)
    if slot_idx < 0:
        return

    cache_base = slot_idx * stride_cache_slot + pid_head * stride_cache_head
    out_base = pid_seq * stride_out_seq + pid_head * stride_out_head

    # Cache Layout per slot:
    # k_int2:   D // 4 bytes  (4 values per byte)
    # k_sz:     4 bytes       (scale fp16 + zero fp16)
    # v_int2:   D // 4 bytes
    # v_sz:     4 bytes       (scale fp16 + zero fp16)
    k_q_bytes = D // 4
    sz_bytes = 4  # 2 × fp16 (scale + zero)

    k_aligned_size = (k_q_bytes + sz_bytes + 15) // 16 * 16

    k_start = 0
    ksz_start = k_q_bytes
    v_start = k_aligned_size
    vsz_start = v_start + k_q_bytes

    dim_offs_q = tl.arange(0, BLOCK_QUARTER)

    # ---------------- K ---------------- #
    k_packed_ptr = cache_ptr + cache_base + k_start + dim_offs_q
    k_packed = tl.load(k_packed_ptr).to(tl.int32)

    k_scale_ptr = tl.cast(
        cache_ptr + cache_base + ksz_start, tl.pointer_type(tl.float16)
    )
    k_zero_ptr = tl.cast(
        cache_ptr + cache_base + ksz_start + 2, tl.pointer_type(tl.float16)
    )
    k_scale = tl.load(k_scale_ptr).to(tl.float32)
    k_zero = tl.load(k_zero_ptr).to(tl.float32)

    k_q0 = (k_packed & 0x3).to(tl.float32)
    k_q1 = ((k_packed >> 2) & 0x3).to(tl.float32)
    k_q2 = ((k_packed >> 4) & 0x3).to(tl.float32)
    k_q3 = ((k_packed >> 6) & 0x3).to(tl.float32)

    k_v0 = (k_q0 - k_zero) * k_scale
    k_v1 = (k_q1 - k_zero) * k_scale
    k_v2 = (k_q2 - k_zero) * k_scale
    k_v3 = (k_q3 - k_zero) * k_scale

    # Inverse of the store packing layout:
    # store: r = reshape(row, (4, Q)) -> permute(1,0) -> (Q, 4)
    #        -> reshape(Q, 2, 2) -> split(even/odd on last dim)
    # v0[i] = row[i],       v1[i] = row[Q+i]
    # v2[i] = row[2Q+i],    v3[i] = row[3Q+i]
    # packed bits: q0|q1<<2|q2<<4|q3<<6
    # => q0 (bits 0-1) -> row[dim_offs_q]
    #    q1 (bits 2-3) -> row[Q  + dim_offs_q]
    #    q2 (bits 4-5) -> row[2Q + dim_offs_q]
    #    q3 (bits 6-7) -> row[3Q + dim_offs_q]
    d_offs = tl.arange(0, D)

    out_k = tl.zeros([D], dtype=tl.float32)
    out_offs_0 = dim_offs_q  # q0 bits 0-1 -> row[i]
    out_offs_1 = dim_offs_q + BLOCK_QUARTER  # q1 bits 2-3 -> row[Q+i]
    out_offs_2 = dim_offs_q + 2 * BLOCK_QUARTER  # q2 bits 4-5 -> row[2Q+i]
    out_offs_3 = dim_offs_q + 3 * BLOCK_QUARTER  # q3 bits 6-7 -> row[3Q+i]

    out_k = tl.where(d_offs[None, :] == out_offs_0[:, None], k_v0[:, None], out_k)
    out_k = tl.where(d_offs[None, :] == out_offs_1[:, None], k_v1[:, None], out_k)
    out_k = tl.where(d_offs[None, :] == out_offs_2[:, None], k_v2[:, None], out_k)
    out_k = tl.where(d_offs[None, :] == out_offs_3[:, None], k_v3[:, None], out_k)
    out_k_sum = tl.sum(out_k, axis=0)  # scatter-sum over BLOCK_QUARTER -> [D]

    # Inverse rotation K: out_k_sum @ Pi_half
    pi_r = tl.arange(0, D)
    pi_c = tl.arange(0, D)
    pi_offs = pi_r[:, None] * stride_pi_0 + pi_c[None, :] * stride_pi_1
    Pi_half = tl.load(Pi_half_ptr + pi_offs)
    k_unrot = tl.dot(out_k_sum[None, :].to(tl.float16), Pi_half, out_dtype=tl.float32)
    k_unrot_1d = tl.reshape(k_unrot, [D])

    out_k_ptr_base = out_k_ptr + out_base
    tl.store(out_k_ptr_base + d_offs, k_unrot_1d.to(tl.float16))

    # ---------------- V ---------------- #
    v_packed_ptr = cache_ptr + cache_base + v_start + dim_offs_q
    v_packed = tl.load(v_packed_ptr).to(tl.int32)

    v_scale_ptr = tl.cast(
        cache_ptr + cache_base + vsz_start, tl.pointer_type(tl.float16)
    )
    v_zero_ptr = tl.cast(
        cache_ptr + cache_base + vsz_start + 2, tl.pointer_type(tl.float16)
    )
    v_scale = tl.load(v_scale_ptr).to(tl.float32)
    v_zero = tl.load(v_zero_ptr).to(tl.float32)

    v_q0 = (v_packed & 0x3).to(tl.float32)
    v_q1 = ((v_packed >> 2) & 0x3).to(tl.float32)
    v_q2 = ((v_packed >> 4) & 0x3).to(tl.float32)
    v_q3 = ((v_packed >> 6) & 0x3).to(tl.float32)

    v_v0 = (v_q0 - v_zero) * v_scale
    v_v1 = (v_q1 - v_zero) * v_scale
    v_v2 = (v_q2 - v_zero) * v_scale
    v_v3 = (v_q3 - v_zero) * v_scale

    out_v = tl.zeros([D], dtype=tl.float32)
    out_v = tl.where(d_offs[None, :] == out_offs_0[:, None], v_v0[:, None], out_v)
    out_v = tl.where(d_offs[None, :] == out_offs_1[:, None], v_v1[:, None], out_v)
    out_v = tl.where(d_offs[None, :] == out_offs_2[:, None], v_v2[:, None], out_v)
    out_v = tl.where(d_offs[None, :] == out_offs_3[:, None], v_v3[:, None], out_v)
    out_v_sum = tl.sum(out_v, axis=0)  # sum over BLOCK_QUARTER -> [D]

    out_v_ptr_base = out_v_ptr + out_base
    tl.store(out_v_ptr_base + d_offs, out_v_sum.to(tl.float16))


def dequant_oscar_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    Pi_half: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = slot_mapping.shape[0]
    Hk = kv_cache.shape[2]

    # Flatten cache to slots
    cache_flat = kv_cache.view(-1, Hk, kv_cache.shape[-1])

    # The kernel uses float16 arithmetic internally — ensure Pi_half is fp16
    Pi_half_fp16 = Pi_half.to(torch.float16)

    out_k = torch.empty(
        (seq_len, Hk, head_dim), dtype=torch.float16, device=kv_cache.device
    )
    out_v = torch.empty(
        (seq_len, Hk, head_dim), dtype=torch.float16, device=kv_cache.device
    )

    if seq_len == 0:
        return out_k, out_v

    grid = (seq_len, Hk)
    _oscar_dequant_kernel[grid](
        cache_flat,
        out_k,
        out_v,
        Pi_half_fp16,
        slot_mapping,
        cache_flat.stride(0),
        cache_flat.stride(1),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        Pi_half_fp16.stride(0),
        Pi_half_fp16.stride(1),
        seq_len,
        Hk,
        D=head_dim,
        BLOCK_QUARTER=head_dim // 4,
    )
    return out_k, out_v
