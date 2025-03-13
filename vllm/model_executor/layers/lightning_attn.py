import torch
import triton
import triton.language as tl
from einops import rearrange

@triton.jit
def _fwd_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK
    off_block = off % NUM_BLOCK
    off_cblock = tl.program_id(1)

    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    cblock_offset = off_cblock * CBLOCK
    q_cblock_offset = cblock_offset * d
    o_cblock_offset = cblock_offset * e

    Q_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + q_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    K_trans_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    i = off_cblock
    q_index = tl.arange(0, CBLOCK) + i * CBLOCK

    q = tl.load(Q_block_ptr, mask=block_offset + q_index[:, None] < n, other=0.0).to(
        tl.float32
    )

    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)
    # none diag

    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = q_index[:, None] - kv_index[None, :]
        s_index = s * diff
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        decay = tl.exp(s_index)

        k_trans = tl.load(
            K_trans_block_ptr, 
            mask=block_offset + kv_index[None, :] < n, 
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr, 
            mask=block_offset + kv_index[:, None] < n, 
            other=0.0,
        ).to(tl.float32)

        qk = tl.dot(q, k_trans) * decay

        qkv += tl.dot(qk, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    tl.store(
        O_block_ptr,
        qkv.to(O_block_ptr.dtype.element_ty),
        mask=block_offset + q_index[:, None] < n,
    )


@triton.jit
def _fwd_kv_parallel(
    K,
    V,
    K_decay,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    # NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    # off_de = tl.program_id(2)

    off_h = off_bh % h
    # off_d = off_de // NUM_FBLOCK
    # off_e = off_de % NUM_FBLOCK

    block_offset = off_block * BLOCK

    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * NUM_BLOCK * d * e
    # d_offset = off_d * D_FBLOCK
    # e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    K_trans_block_ptr = (
        K
        + k_offset
        + k_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d  # d x c
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e  # c x d
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + kv_block_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    k_decay_ptr = (
        K_decay
        + off_h * BLOCK
        + tl.arange(0, CBLOCK)[None, :]
    )

    # compute block array
    kv_index = tl.arange(0, CBLOCK)

    # c_array = tl.arange(0, CBLOCK) + 1
    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)

    if off_block == NUM_BLOCK - 1:
        split_n = n - (NUM_BLOCK - 1) * BLOCK
    else:
        split_n = BLOCK
    left_shift = tl.cdiv(split_n, CBLOCK) * CBLOCK - split_n
    num_blocks = min(tl.cdiv(split_n, CBLOCK), NUM_CBLOCK)
    k_decay_ptr += (NUM_CBLOCK - num_blocks) * CBLOCK
    for j in range(num_blocks):
        # right align k, v with CBLOCK
        left_bound = (1 - j) * left_shift
        k_trans = tl.load(
            K_trans_block_ptr - left_shift * d, 
            mask=kv_index[None, :] >= left_bound, 
            other=0.0
        )
        v = tl.load(
            V_block_ptr - left_shift * d,
            mask=kv_index[:, None] >= left_bound, 
            other=0.0
        )

        k_decay = tl.load(k_decay_ptr)
        kv += tl.dot(k_trans * k_decay, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e
        k_decay_ptr += CBLOCK

    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _fwd_kv_reduce(
    K,
    V,
    S,
    KV,
    KV_HISTORY,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    # NUM_FBLOCK: tl.constexpr,
    # CBLOCK: tl.constexpr,
    # NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    # off_d = tl.program_id(1)
    # off_e = tl.program_id(2)

    kv_offset = off_bh * NUM_BLOCK * d * e
    # d_offset = off_d * D_FBLOCK
    # e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    KV_block_ptr = (
        KV
        + kv_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    # Initialize kv from KV_HISTORY
    kv_history_offset = off_bh * d * e
    KV_HISTORY_block_ptr = (
        KV_HISTORY + kv_history_offset +
        tl.arange(0, D_FBLOCK)[:, None] * e +
        tl.arange(0, E_FBLOCK)[None, :]
    )
    # compute block array
    # last step
    kv_pre = tl.load(KV_HISTORY_block_ptr).to(tl.float32)
    for i in range (NUM_BLOCK):
        block_size = min(n - i * BLOCK, BLOCK)
        block_decay = tl.exp(-s.to(tl.float32) * block_size)

        kv_cur = tl.load(KV_block_ptr).to(tl.float32)
        tl.store(KV_block_ptr, kv_pre.to(KV_block_ptr.dtype.element_ty))

        kv_pre = block_decay * kv_pre + kv_cur
        KV_block_ptr += d * e
    tl.store(KV_HISTORY_block_ptr, kv_pre)


@triton.jit
def _fwd_none_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    off_e = tl.program_id(2)

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    e_offset = off_e * E_FBLOCK
    block_offset = n_offset + c_offset
    

    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset

    kv_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e + e_offset

    Q_block_ptr = (
        Q
        + q_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + tl.arange(0, d)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    c_array = tl.arange(0, CBLOCK)

    kv = tl.load(KV_block_ptr).to(tl.float32)
    q_index = block_offset + tl.arange(0, CBLOCK)
    q = tl.load(Q_block_ptr, mask=q_index[:, None] < n, other=0.).to(tl.float32)
    
    q_decay = tl.exp(-s.to(tl.float32) * (off_c * CBLOCK + c_array[:, None]))
    qkv_none_diag = tl.dot(q, kv) * q_decay
    
    qkv_diag = tl.load(O_block_ptr, mask=q_index[:, None] < n, other=0.).to(tl.float32)

    qkv = qkv_diag + qkv_none_diag

    tl.store(O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty), mask=q_index[:, None] < n)

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s, kv_history):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError(
                "Flash attention currently only supported for compute capability >= 80"
            )
        # shape constraints
        b, h, n, d = q.shape
        e = v.shape[-1]
        # right
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        BLOCK = 256
        NUM_BLOCK = triton.cdiv(n, BLOCK)

        CBLOCK = 64
        CBLOCK = 32
        NUM_CBLOCK = BLOCK // CBLOCK; assert BLOCK % CBLOCK == 0, "BLOCK must be a multiple of CBLOCK"

        array = torch.arange(0, BLOCK, device=q.device) + 1
        k_decay = torch.exp(-s * (BLOCK - array.reshape(1, -1)))

        grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
        _fwd_diag_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        NUM_FBLOCK = 1
        D_FBLOCK = d // NUM_FBLOCK; assert d % NUM_FBLOCK == 0
        E_FBLOCK = e // NUM_FBLOCK; assert e % NUM_FBLOCK == 0
        
        CBLOCK = 64
        NUM_CBLOCK = BLOCK // CBLOCK; assert BLOCK % CBLOCK == 0, "BLOCK must be a multiple of CBLOCK"

        kv = torch.empty(
            (b, h, NUM_BLOCK, d, e), dtype=torch.float32, device=q.device
        )
        grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
        _fwd_kv_parallel[grid](
            k,
            v,
            k_decay,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)
        _fwd_kv_reduce[grid](
            k,
            v,
            s,
            kv,
            kv_history,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
        _fwd_none_diag_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        ctx.save_for_backward(q, k, v, s, kv)
        ctx.BLOCK = BLOCK

        return o, torch.cat([kv, kv_history.unsqueeze(2)], dim=2)

lightning_attention_ = _attention.apply

def lightning_attention(q, k, v, ed, block_size=256, kv_history=None):
    d = q.shape[-1]
    e = v.shape[-1]
    m = 128 if d >= 128 else 64
    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)
    n = len(arr)
    output = 0
    if kv_history is None:
        kv_history = torch.zeros((q.shape[0], q.shape[1], d, e), dtype=torch.float32, device=q.device)
    else:
        # make sure run in functional programming style
        kv_history = kv_history.clone().contiguous()

    for i in range(n - 1):
        s = arr[i]
        e = arr[i + 1]
        q1 = q[..., s:e]  # .contiguous()
        k1 = k[..., s:e]  # .contiguous()
        # print(output.shape)
        o, kv = lightning_attention_(q1, k1, v, ed, kv_history)
        output = output + o
    return output, kv

def lightning_attention2_parallel(q, k, v, ed, block_size=256, kv_history=None):
    return lightning_attention(q, k, v, ed, block_size, kv_history)

@triton.jit
def _linear_attn_decode_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr,      # [B, H, 1, D]  
    kv_cache_ptr,             # [B, H, D, D] 
    slope_rate, 
    slot_idx,
    output_ptr,               # [B, H, 1, D]
    B, H, 
    D: tl.constexpr,
    # Matrix dimensions
    qkv_b_stride, qkv_h_stride,
    cache_b_stride, cache_h_stride, cache_d0_stride, cache_d1_stride,
    BLOCK_SIZE: tl.constexpr,
):

    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    slot_id = tl.load(slot_idx + pid_b)

    # return when padding
    if slot_id == -1:
        return
    
    batch_id = pid_b
    head_id = pid_h
    
    ratio = tl.load(slope_rate + pid_h)


    qk_d_offsets = tl.arange(0, D)
    v_d_offsets = tl.arange(0, BLOCK_SIZE) + pid_d * BLOCK_SIZE
    cache_d_offsets = qk_d_offsets[:, None] * cache_d0_stride + v_d_offsets[None, :] * cache_d1_stride

    q_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    k_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    v_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride

    # cache_offset = batch_id * cache_b_stride + head_id * cache_h_stride
    cache_offset = slot_id * cache_b_stride + head_id * cache_h_stride

    qk_mask = qk_d_offsets < D
    v_mask = v_d_offsets < D 
    # load data to shm
    q = tl.load(q_ptr + q_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    k = tl.load(k_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    v = tl.load(v_ptr + v_offset + v_d_offsets, mask=v_mask, other=0.0)
    
    kv_outer = k[:, None] * v[None, :]  # [D, BLOCK_SIZE]
    kv_mask = qk_mask[:, None] & v_mask[None, :]

    # compute decay
    ratio = tl.exp(-ratio)
    # load kv_cache
    kv_ptr = kv_cache_ptr + cache_offset + cache_d_offsets
    kv_cache_old = tl.load(kv_ptr, mask=kv_mask, other=0.0)
    kv_outer = kv_outer + ratio * kv_cache_old

    output = q[:, None].to(tl.float32) * kv_outer
    output = tl.sum(output, axis=0)

    tl.store(kv_ptr, kv_outer, mask=kv_mask)
    tl.store(output_ptr + q_offset + v_d_offsets, output, mask=v_mask)



def linear_decode_forward_triton(
    q: torch.Tensor,      # [B, H, 1, D] 
    k: torch.Tensor,      # [B, H, 1, D]
    v: torch.Tensor,      # [B, H, 1, D] 
    kv_caches: torch.Tensor,  # [B, H, D, D]
    slope_rate: torch.Tensor,  # float
    slot_idx: torch.Tensor,
    BLOCK_SIZE: int = 32,
) -> torch.Tensor:
    
    B, H, _, D = q.shape
    assert k.shape == (B, H, 1, D)
    assert v.shape == (B, H, 1, D)
    
    output = torch.empty_like(q)
    
    grid = (B, H, D // BLOCK_SIZE)

    qkv_b_stride = q.stride(0)
    qkv_h_stride = q.stride(1)

    cache_b_stride = kv_caches.stride(0)
    cache_h_stride = kv_caches.stride(1)
    cache_d0_stride = kv_caches.stride(2)
    cache_d1_stride = kv_caches.stride(3)
    
    # launch kernel
    _linear_attn_decode_kernel[grid](
        q, k, v,
        kv_caches, 
        slope_rate,
        slot_idx,
        output,
        B, H, D,
        qkv_b_stride, qkv_h_stride,
        cache_b_stride, cache_h_stride,cache_d0_stride, cache_d1_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    output = rearrange(output, "b h n d -> b n (h d)")
    return output.squeeze(1).contiguous()
