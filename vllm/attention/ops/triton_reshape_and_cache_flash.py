import triton
import triton.language as tl
import torch
import math

@triton.jit
def reshape_and_cache_flash_kernel(
    key_ptr, value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr, n, kv_dt,
    block_stride, key_stride, value_stride, num_heads, head_size, block_size: tl.constexpr,
    k_scale, v_scale
):
    
    # Get the block index
    token_idx = tl.program_id(0)
    # Compute the offset for this block
    #offset = block_id * block_size + tl.arange(0, block_size)
    # Mask to handle out-of-bounds
    #mask = offset < n
    
    #token_idx = block_id * block_size + tl.arange(0, block_size)
      
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
    # Handle padding (-1) as early return
    if slot_idx < 0:
        return
     
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    #n = num_heads * head_size
    mask = token_idx < n
    
    #i = block_offset + token_idx
    src_key_idx = token_idx * key_stride + n
    src_value_idx = token_idx * value_stride + n
    head_idx = n // head_size
    head_offset = n % head_size

    tgt_key_value_idx = (block_idx * block_stride +
                        block_offset * num_heads * head_size +
                        head_idx * head_size + head_offset)

   
    if (kv_dt == 1):
        tgt_key = tl.load(key_ptr + src_key_idx, mask=mask)
        tgt_value = tl.load(value_ptr + src_value_idx, mask=mask)
        scaled_key = tgt_key * k_scale
        scaled_value = tgt_value * v_scale
        tl.store(key_cache_ptr + tgt_key_value_idx, scaled_key, mask=mask)
        tl.store(value_cache_ptr + tgt_key_value_idx, scaled_value, mask=mask)
    else:
        tgt_key = tl.load(key_ptr + src_key_idx, mask=mask)
        tgt_value = tl.load(value_ptr + src_value_idx, mask=mask)
        tl.store(key_cache_ptr + tgt_key_value_idx, tgt_key, mask=mask)
        tl.store(value_cache_ptr + tgt_key_value_idx, tgt_value, mask=mask)
    
    
# Driver function to invoke the Triton kernel
def reshape_and_cache_flash(
    key, value, key_cache, value_cache, slot_mapping,
        kv_cache_dtype, k_scale, v_scale
):
    num_tokens = int(key.size(0))
    num_heads = int(key.size(1))
    head_size = int(key.size(2))
    block_size = int(key_cache.size(1))
    block_stride = int(key_cache.stride(0))
    key_stride = int(key.stride(0))
    value_stride = int(value.stride(0))
    
    if kv_cache_dtype == "fp8":
        kv_dt = 1
    else:
        kv_dt = 0
    
    n = slot_mapping.numel()

    # Launch Triton kernel
    grid = (num_tokens,)
    reshape_and_cache_flash_kernel[grid](
        key_ptr=key, value_ptr=value, key_cache_ptr=key_cache,
        value_cache_ptr=value_cache, slot_mapping_ptr=slot_mapping, n=n, kv_dt=kv_dt,
        block_stride=block_stride, key_stride=key_stride,
        value_stride=value_stride, num_heads=num_heads, head_size=head_size,
        block_size=block_size, 
         k_scale=k_scale, v_scale=v_scale
    )