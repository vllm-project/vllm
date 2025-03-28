# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl
from einops import rearrange


@triton.jit
def _fwd_diag_kernel(Q, K, V, Out, S, b: tl.constexpr, h: tl.constexpr, n,
                     d: tl.constexpr, e: tl.constexpr, BLOCK: tl.constexpr,
                     NUM_BLOCK, CBLOCK: tl.constexpr):
    # This kernel computes the diagonal blocks of the attention matrix
    # Each diagonal block represents attention
    # where queries attend to keys in the same block
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK  # batch-head index
    off_block = off % NUM_BLOCK  # block index within the sequence
    off_cblock = tl.program_id(1)  # sub-block index within a block

    off_h = off_bh % h  # head index

    # Calculate base offsets for the current batch and head
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    # Calculate offsets for the current block
    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    # Calculate offsets for the current sub-block
    cblock_offset = off_cblock * CBLOCK
    q_cblock_offset = cblock_offset * d
    o_cblock_offset = cblock_offset * e

    # Calculate pointers to the query, key, value, and output tensors
    Q_block_ptr = (Q + qk_offset + qk_block_offset + q_cblock_offset +
                   tl.arange(0, CBLOCK)[:, None] * d +
                   tl.arange(0, d)[None, :])
    K_trans_block_ptr = (K + qk_offset + qk_block_offset +
                         tl.arange(0, CBLOCK)[None, :] * d +
                         tl.arange(0, d)[:, None])
    V_block_ptr = (V + v_offset + v_block_offset +
                   tl.arange(0, CBLOCK)[:, None] * e +
                   tl.arange(0, e)[None, :])
    O_block_ptr = (Out + o_offset + o_block_offset + o_cblock_offset +
                   tl.arange(0, CBLOCK)[:, None] * e +
                   tl.arange(0, e)[None, :])

    # Load the decay rate for the current head
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    i = off_cblock
    q_index = tl.arange(0, CBLOCK) + i * CBLOCK

    # Load query values
    q = tl.load(Q_block_ptr,
                mask=block_offset + q_index[:, None] < n,
                other=0.0).to(tl.float32)

    # Initialize output accumulator
    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)

    # Process all sub-blocks up to and
    # including the current one (causal attention)
    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = q_index[:, None] - kv_index[None, :]
        s_index = s * diff
        # Apply causal mask: only attend to positions before the current one
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        decay = tl.exp(s_index)

        # Load key and value
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

        # Compute attention scores and apply decay
        qk = tl.dot(q, k_trans) * decay

        # Compute weighted values and accumulate
        qkv += tl.dot(qk, v)

        # Move to the next sub-block
        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    # Store the result
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
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    # This kernel computes the key-value outer
    # products for each block in parallel
    off_bh = tl.program_id(0)  # batch-head index
    off_block = tl.program_id(1)  # block index

    off_h = off_bh % h  # head index

    block_offset = off_block * BLOCK

    # Calculate offsets for the current block
    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    # Calculate base offsets for the current batch and head
    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * NUM_BLOCK * d * e

    # Calculate pointers to the key, value, and key-value tensors
    K_trans_block_ptr = (K + k_offset + k_block_offset +
                         tl.arange(0, CBLOCK)[None, :] * d +
                         tl.arange(0, D_FBLOCK)[:, None])
    V_block_ptr = (V + v_offset + v_block_offset +
                   tl.arange(0, CBLOCK)[:, None] * e +
                   tl.arange(0, E_FBLOCK)[None, :])
    KV_block_ptr = (KV + kv_offset + kv_block_offset +
                    tl.arange(0, D_FBLOCK)[:, None] * e +
                    tl.arange(0, E_FBLOCK)[None, :])

    # Load the decay factors for the current head and block
    k_decay_ptr = (K_decay + off_h * BLOCK + tl.arange(0, CBLOCK)[None, :])

    kv_index = tl.arange(0, CBLOCK)

    # Initialize the key-value outer product accumulator
    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)

    # Handle the last block which might be smaller than BLOCK
    if off_block == NUM_BLOCK - 1:
        split_n = n - (NUM_BLOCK - 1) * BLOCK
    else:
        split_n = BLOCK
    left_shift = tl.cdiv(split_n, CBLOCK) * CBLOCK - split_n
    num_blocks = min(tl.cdiv(split_n, CBLOCK), NUM_CBLOCK)
    k_decay_ptr += (NUM_CBLOCK - num_blocks) * CBLOCK

    # Process all sub-blocks in the current block
    for j in range(num_blocks):
        left_bound = (1 - j) * left_shift
        # Load key and value, handling boundary conditions
        k_trans = tl.load(K_trans_block_ptr - left_shift * d,
                          mask=kv_index[None, :] >= left_bound,
                          other=0.0)
        v = tl.load(V_block_ptr - left_shift * e,
                    mask=kv_index[:, None] >= left_bound,
                    other=0.0)

        # Load decay factor and compute weighted key-value outer product
        k_decay = tl.load(k_decay_ptr)
        kv += tl.dot(k_trans * k_decay, v)

        # Move to the next sub-block
        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e
        k_decay_ptr += CBLOCK

    # Store the result
    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _fwd_kv_reduce(S, KV, KV_HISTORY, b: tl.constexpr, h: tl.constexpr, n,
                   d: tl.constexpr, e: tl.constexpr, BLOCK: tl.constexpr,
                   NUM_BLOCK, D_FBLOCK: tl.constexpr, E_FBLOCK: tl.constexpr):
    # This kernel reduces the key-value outer products
    # across blocks and updates the KV history
    off_bh = tl.program_id(0)  # batch-head index
    off_h = off_bh % h  # head index

    kv_offset = off_bh * NUM_BLOCK * d * e

    # Calculate pointer to the key-value tensor
    KV_block_ptr = (KV + kv_offset + tl.arange(0, D_FBLOCK)[:, None] * e +
                    tl.arange(0, E_FBLOCK)[None, :])

    # Load the decay rate for the current head
    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    # Calculate pointer to the key-value history tensor
    kv_history_offset = off_bh * d * e
    KV_HISTORY_block_ptr = (KV_HISTORY + kv_history_offset +
                            tl.arange(0, D_FBLOCK)[:, None] * e +
                            tl.arange(0, E_FBLOCK)[None, :])

    # Load the previous key-value history
    kv_pre = tl.load(KV_HISTORY_block_ptr).to(tl.float32)

    # Process all blocks in reverse order to compute the prefix sum
    for i in range(NUM_BLOCK):
        block_size = min(n - i * BLOCK, BLOCK)
        # Compute decay factor for the current block
        block_decay = tl.exp(-s.to(tl.float32) * block_size)

        # Load the current key-value outer product
        kv_cur = tl.load(KV_block_ptr).to(tl.float32)
        # Store the previous key-value history to the current block
        tl.store(KV_block_ptr, kv_pre.to(KV_block_ptr.dtype.element_ty))

        # Update the key-value history with the current block
        kv_pre = block_decay * kv_pre + kv_cur
        KV_block_ptr += d * e

    # Store the updated key-value history
    tl.store(KV_HISTORY_block_ptr, kv_pre)


@triton.jit
def _fwd_none_diag_kernel(
    Q,
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
    E_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    # This kernel computes the non-diagonal blocks of the attention matrix
    # Each non-diagonal block represents attention
    # where queries attend to keys in different blocks
    off_bh = tl.program_id(0)  # batch-head index
    off_h = off_bh % h  # head index

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK  # block index
    off_c = off_nc % NUM_CBLOCK  # sub-block index
    off_e = tl.program_id(2)  # output feature block index

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    e_offset = off_e * E_FBLOCK
    block_offset = n_offset + c_offset

    # Calculate offsets for the current batch, head, and block
    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset
    kv_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e + e_offset

    # Calculate pointers to the query, output, and key-value tensors
    Q_block_ptr = (Q + q_offset + tl.arange(0, CBLOCK)[:, None] * d +
                   tl.arange(0, d)[None, :])
    O_block_ptr = (Out + o_offset + tl.arange(0, CBLOCK)[:, None] * e +
                   tl.arange(0, E_FBLOCK)[None, :])
    KV_block_ptr = (KV + kv_offset + tl.arange(0, d)[:, None] * e +
                    tl.arange(0, E_FBLOCK)[None, :])

    # Load the decay rate for the current head
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    c_array = tl.arange(0, CBLOCK)

    # Load the key-value outer product for the current block
    kv = tl.load(KV_block_ptr).to(tl.float32)
    q_index = block_offset + tl.arange(0, CBLOCK)

    # Load query values
    q = tl.load(Q_block_ptr, mask=q_index[:, None] < n,
                other=0.).to(tl.float32)

    # Compute decay factors for the current sub-block
    q_decay = tl.exp(-s.to(tl.float32) * (off_c * CBLOCK + c_array[:, None]))

    # Compute non-diagonal attention output
    qkv_none_diag = tl.dot(q, kv) * q_decay

    # Load diagonal attention output (computed by _fwd_diag_kernel)
    qkv_diag = tl.load(O_block_ptr, mask=q_index[:, None] < n,
                       other=0.).to(tl.float32)

    # Combine diagonal and non-diagonal attention outputs
    qkv = qkv_diag + qkv_none_diag

    # Store the result
    tl.store(O_block_ptr,
             qkv.to(O_block_ptr.dtype.element_ty),
             mask=q_index[:, None] < n)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, s, kv_history):
        # Forward pass of the lightning attention algorithm
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()

        # Check CUDA compute capability
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported",
                               "for compute capability >= 80")

        # Get input dimensions
        b, h, n, d = q.shape
        e = v.shape[-1]

        # Initialize output tensor
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        # Set block sizes
        BLOCK = 256
        NUM_BLOCK = triton.cdiv(n, BLOCK)

        CBLOCK = 32
        NUM_CBLOCK = BLOCK // CBLOCK
        assert BLOCK % CBLOCK == 0, "BLOCK must be a multiple of CBLOCK"

        # Compute decay factors for keys
        array = torch.arange(0, BLOCK, device=q.device) + 1
        k_decay = torch.exp(-s * (BLOCK - array.reshape(1, -1)))

        # Step 1: Compute diagonal blocks of attention
        grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
        _fwd_diag_kernel[grid](q,
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
                               CBLOCK=CBLOCK)

        # Set feature block sizes
        NUM_FBLOCK = 1
        D_FBLOCK = d // NUM_FBLOCK
        assert d % NUM_FBLOCK == 0
        E_FBLOCK = e // NUM_FBLOCK
        assert e % NUM_FBLOCK == 0

        CBLOCK = 64
        NUM_CBLOCK = BLOCK // CBLOCK
        assert BLOCK % CBLOCK == 0, "BLOCK must be a multiple of CBLOCK"

        # Step 2: Compute key-value outer products for each block in parallel
        kv = torch.empty((b, h, NUM_BLOCK, d, e),
                         dtype=torch.float32,
                         device=q.device)
        grid = (b * h, NUM_BLOCK)
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

        # Step 3: Reduce key-value outer products
        # across blocks and update KV history
        grid = (b * h, NUM_FBLOCK)
        _fwd_kv_reduce[grid](s,
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
                             E_FBLOCK=E_FBLOCK)

        # Step 4: Compute non-diagonal blocks of attention
        grid = (b * h, NUM_BLOCK * NUM_CBLOCK)
        _fwd_none_diag_kernel[grid](
            q,
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
            E_FBLOCK=E_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v, s, kv)
        ctx.BLOCK = BLOCK

        return o, torch.cat([kv, kv_history.unsqueeze(2)], dim=2)


# Apply the lightning attention function
lightning_attention_ = _attention.apply


def lightning_attention(q, k, v, ed, block_size=256, kv_history=None):
    """
    Apply lightning attention algorithm 
    to compute attention efficiently.
    
    Args:
        q: Query tensor of shape [batch, heads, seq_len, dim]
        k: Key tensor of shape [batch, heads, seq_len, dim]
        v: Value tensor of shape [batch, heads, seq_len, dim_v]
        ed: Decay rate tensor of shape [heads]
        block_size: Size of blocks for block-sparse attention
        kv_history: Optional key-value history from previous computations
        
    Returns:
        output: Attention output
        kv: Updated key-value history
    """
    d = q.shape[-1]
    e = v.shape[-1]

    if ed.dim() == 1:
        ed = ed.view(1, -1, 1, 1)

    # Split the computation into chunks for better parallelism
    m = 128 if d >= 128 else 64
    assert d % m == 0, f"Dimension d ({d}) must be divisible by m ({m})"
    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)
    n = len(arr)
    output = 0

    # Initialize or clone key-value history
    if kv_history is None:
        kv_history = torch.zeros((q.shape[0], q.shape[1], d, e),
                                 dtype=torch.float32,
                                 device=q.device)
    else:
        kv_history = kv_history.clone().contiguous()

    # Process each chunk and accumulate results
    for i in range(n - 1):
        s = arr[i]
        e = arr[i + 1]
        q1 = q[..., s:e]
        k1 = k[..., s:e]
        o, kv = lightning_attention_(q1, k1, v, ed, kv_history)
        output = output + o
    return output, kv


@triton.jit
def _linear_attn_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_cache_ptr,
    slope_rate,
    slot_idx,
    output_ptr,
    D: tl.constexpr,
    qkv_b_stride,
    qkv_h_stride,
    cache_b_stride,
    cache_h_stride,
    cache_d0_stride,
    cache_d1_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for linear attention decoding with KV cache.
    
    This kernel computes attention for a single token using the KV cache.
    """
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # head index
    pid_d = tl.program_id(2)  # dimension block index

    # Load slot index for the current batch
    slot_id = tl.load(slot_idx + pid_b)

    # Skip if slot_id is -1 (padding)
    if slot_id == -1:
        return

    batch_id = pid_b
    head_id = pid_h

    # Load decay rate for the current head
    ratio = tl.load(slope_rate + pid_h)

    # Calculate offsets for dimensions
    qk_d_offsets = tl.arange(0, D)
    v_d_offsets = tl.arange(0, BLOCK_SIZE) + pid_d * BLOCK_SIZE
    cache_d_offsets = qk_d_offsets[:, None] * cache_d0_stride + v_d_offsets[
        None, :] * cache_d1_stride

    # Calculate offsets for the current batch and head
    q_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    k_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    v_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride

    cache_offset = slot_id * cache_b_stride + head_id * cache_h_stride

    # Create masks for loading tensors
    qk_mask = qk_d_offsets < D
    v_mask = v_d_offsets < D

    # Load query, key, and value tensors
    q = tl.load(q_ptr + q_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    k = tl.load(k_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    v = tl.load(v_ptr + v_offset + v_d_offsets, mask=v_mask, other=0.0)

    # Compute key-value outer product
    kv_outer = k[:, None] * v[None, :]
    kv_mask = qk_mask[:, None] & v_mask[None, :]

    # Apply decay to previous KV cache
    ratio = tl.exp(-ratio)
    kv_ptr = kv_cache_ptr + cache_offset + cache_d_offsets
    kv_cache_old = tl.load(kv_ptr, mask=kv_mask, other=0.0)
    kv_outer = kv_outer + ratio * kv_cache_old

    # Compute attention output
    output = q[:, None].to(tl.float32) * kv_outer
    output = tl.sum(output, axis=0)

    # Update KV cache and store output
    tl.store(kv_ptr, kv_outer, mask=kv_mask)
    tl.store(output_ptr + q_offset + v_d_offsets, output, mask=v_mask)


def linear_decode_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_caches: torch.Tensor,
    slope_rate: torch.Tensor,
    slot_idx: torch.Tensor,
    BLOCK_SIZE: int = 32,
) -> torch.Tensor:
    """
    Perform linear attention decoding using Triton kernels.
    
    Args:
        q: Query tensor of shape [B, H, 1, D]
        k: Key tensor of shape [B, H, 1, D]
        v: Value tensor of shape [B, H, 1, D]
        kv_caches: Key-value cache tensor
        slope_rate: Decay rate tensor
        slot_idx: Slot indices for batches
        BLOCK_SIZE: Size of blocks for processing
        
    Returns:
        output: Attention output tensor
    """
    B, H, _, D = q.shape
    assert k.shape == (B, H, 1, D)
    assert v.shape == (B, H, 1, D)

    # Initialize output tensor
    output = torch.empty_like(q)

    # Set grid dimensions for the kernel
    grid = (B, H, D // BLOCK_SIZE)

    # Calculate strides for tensors
    qkv_b_stride = q.stride(0)
    qkv_h_stride = q.stride(1)

    cache_b_stride = kv_caches.stride(0)
    cache_h_stride = kv_caches.stride(1)
    cache_d0_stride = kv_caches.stride(2)
    cache_d1_stride = kv_caches.stride(3)

    # Launch the kernel
    _linear_attn_decode_kernel[grid](
        q,
        k,
        v,
        kv_caches,
        slope_rate,
        slot_idx,
        output,
        D,
        qkv_b_stride,
        qkv_h_stride,
        cache_b_stride,
        cache_h_stride,
        cache_d0_stride,
        cache_d1_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape output and return
    output = rearrange(output, "b h n d -> b n (h d)")
    return output.squeeze(1).contiguous()
