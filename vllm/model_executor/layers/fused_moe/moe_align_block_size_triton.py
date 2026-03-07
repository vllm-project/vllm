
import torch
import triton
import triton.language as tl
from vllm.utils.math_utils import round_up

@triton.jit
def _moe_align_block_size_scatter_kernel(
    topk_ids_ptr,
    sorted_ids_ptr,
    expert_offsets_ptr,
    expert_counters_ptr,
    total_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each thread processes one token assignment
    # Global thread index
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_tokens

    # Load expert assignment for this token (flattened index)
    expert_idx = tl.load(topk_ids_ptr + idx, mask=mask, other=-1)

    # If valid assignment
    # We use atomic add to get the rank of this token within its expert
    # We need to handle the case where expert_idx is -1 (shouldn't happen if input is valid?)
    # But let's assume valid experts >= 0
    
    # We can't branch easily in Triton for atomics on different addresses in vector mode?
    # Actually we can, but masked.
    
    # Compute address for counter
    counter_ptr = expert_counters_ptr + expert_idx
    
    # Atomic add returns the old value (rank)
    # Note: mask must be boolean. expert_idx != -1 check.
    valid_mask = mask & (expert_idx >= 0)
    
    # Triton atomic_add supports tensor pointers/masks
    rank = tl.atomic_add(counter_ptr, 1, mask=valid_mask)
    
    # Load offset for this expert
    offset_ptr = expert_offsets_ptr + expert_idx
    start_offset = tl.load(offset_ptr, mask=valid_mask, other=0)
    
    # Compute destination index
    dest_idx = start_offset + rank
    
    # Write the original token index (which is just 'idx') to sorted_ids
    tl.store(sorted_ids_ptr + dest_idx, idx, mask=valid_mask)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton implementation of moe_align_block_size.
    """
    device = topk_ids.device
    num_tokens_flat = topk_ids.numel()
    
    # Flatten topk_ids for processing
    flat_topk_ids = topk_ids.view(-1)
    
    # Step 1: Count tokens per expert (Histogram)
    # This is fast in PyTorch
    if num_tokens_flat > 0:
        expert_counts = torch.bincount(flat_topk_ids, minlength=num_experts).int()
    else:
        expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        
    # Handle ignore_invalid_experts logic?
    # Docstring: "When True, all invalid expert_ids... will be ignored... no -1 in expert_ids"
    # If ignore_invalid_experts is True, we need to know which are invalid.
    # Usually this depends on expert_map.
    # If expert_map is provided, experts mapping to -1 are invalid?
    # "expert_map... maps... to local index space... If not in current shard... set to -1"
    # So if ignore_invalid_experts is True, we should zero out counts for experts that map to -1.
    
    valid_experts_mask = None
    if ignore_invalid_experts and expert_map is not None:
        # expert_map is [num_experts]
        valid_experts_mask = expert_map != -1
        # Zero out counts for invalid experts
        expert_counts = expert_counts * valid_experts_mask.int()

    # Step 2: Compute Offsets and Padding
    # Each expert needs to be padded to block_size
    padded_counts = (expert_counts + block_size - 1) // block_size * block_size
    
    # Compute offsets for each expert in the sorted_ids array
    expert_offsets = torch.cumsum(padded_counts, dim=0) - padded_counts
    
    # Total size
    total_tokens_post_pad = padded_counts.sum().item()
    
    # Prepare Output Tensors
    sorted_ids = torch.empty(total_tokens_post_pad, dtype=torch.int32, device=device)
    # Fill with a safe default (e.g. num_tokens_flat) to indicate padding?
    # The C++ op behavior for padding is implicit/undefined (or handled by fused_moe).
    # We can fill with num_tokens_flat to be safe (if fused_moe checks).
    # Or just leave uninitialized (fastest).
    # Let's fill with -1 or num_tokens_flat just in case.
    # sorted_ids.fill_(num_tokens_flat) # Optional safety
    
    # Prepare expert_ids output
    # Number of blocks
    num_m_blocks = total_tokens_post_pad // block_size
    expert_ids = torch.empty(num_m_blocks, dtype=torch.int32, device=device)
    
    # Fill expert_ids
    # We can do this on CPU or GPU. Since num_experts is small, GPU loop or repeat_interleave is fine.
    # We need to construct expert_ids based on padded_counts.
    # expert i has padded_counts[i] / block_size blocks.
    blocks_per_expert = padded_counts // block_size
    
    # We need to expand this.
    # mapped_experts = expert_map if expert_map is not None else arange
    if expert_map is not None:
        target_experts = expert_map
    else:
        target_experts = torch.arange(num_experts, device=device, dtype=torch.int32)
        
    # If ignore_invalid_experts is True, we skipped invalid experts in counts, so they have 0 blocks.
    # So repeating them 0 times is correct.
    # BUT, if expert_map maps to -1, and ignore_invalid_experts is False, we still have blocks.
    # And we should write -1 to expert_ids for those blocks.
    # target_experts already has -1.
    
    expert_ids = torch.repeat_interleave(target_experts, blocks_per_expert)
    
    # Step 3: Scatter
    # We need a counter array for atomics
    expert_counters = torch.zeros(num_experts, dtype=torch.int32, device=device)
    
    # Launch Triton Kernel
    # We launch with num_tokens_flat
    grid = lambda META: (triton.cdiv(num_tokens_flat, META['BLOCK_SIZE']),)
    
    _moe_align_block_size_scatter_kernel[grid](
        flat_topk_ids,
        sorted_ids,
        expert_offsets,
        expert_counters,
        num_tokens_flat,
        num_experts,
        BLOCK_SIZE=1024,
    )
    
    # Handle padding of the sorted_ids tensor itself if requested
    if pad_sorted_ids:
        # The function signature says "pad sorted_token_ids length should be padded to a multiple of block_size"
        # Our total_tokens_post_pad is ALREADY a multiple of block_size (sum of multiples).
        # So this flag might be redundant or referring to something else?
        # Re-reading docstring: "pad_sorted_ids: ... sorted_token_ids length should be padded to a multiple of block_size"
        # "max_num_tokens_padded = ... round_up(..., block_size)"
        # Since we calculated total based on padded_counts, it is already aligned.
        pass

    # Return
    return sorted_ids, expert_ids, torch.tensor([total_tokens_post_pad], dtype=torch.int32, device=device)
