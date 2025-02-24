import torch
import triton
import triton.language as tl
import unittest
from typing import Optional
from vllm.vllm_flash_attn import flash_attn_varlen_func

# === Use your provided merge_attn_states implementation exactly ===

def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)

    merge_attn_states_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        head_size,
        padded_head_size,
        output_lse is not None,
    )


@triton.jit
def merge_attn_states_kernel(
    output,        # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,    # [NUM_HEADS, NUM_TOKENS]
    prefix_output, # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,    # [NUM_HEADS, NUM_TOKENS]
    suffix_output, # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,    # [NUM_HEADS, NUM_TOKENS]
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    # Load lse values for this token & head.
    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)

    # Determine validity (for causal masking, masked positions will be -∞, which is not finite).
    p_valid = tl.isfinite(p_lse)
    s_valid = tl.isfinite(s_lse)
    both_valid = p_valid & s_valid
    only_p = p_valid & (~s_valid)
    only_s = s_valid & (~p_valid)

    # Compute merged candidate only if both sides are valid.
    max_lse = tl.maximum(p_lse, s_lse)
    p_shift = p_lse - max_lse
    s_shift = s_lse - max_lse
    out_se = tl.exp(p_shift) + tl.exp(s_shift)

    merged_lse_candidate = tl.log(out_se) + max_lse
    # If both are valid, merge; otherwise, choose the valid side.
    merged_lse = tl.where(both_valid, merged_lse_candidate, tl.where(only_p, p_lse, s_lse))

    # Optionally store merged lse.
    if OUTPUT_LSE:
        tl.store(output_lse + head_idx * num_tokens + token_idx, merged_lse)

    # Load the attention outputs.
    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    base_offset = token_idx * num_heads * HEAD_SIZE
    p_out = tl.load(prefix_output + base_offset + head_idx * HEAD_SIZE + head_arange,
                    mask=head_mask)
    s_out = tl.load(suffix_output + base_offset + head_idx * HEAD_SIZE + head_arange,
                    mask=head_mask)

    # Compute candidate merged output if both sides valid.
    p_scale = tl.exp(p_shift) / out_se
    s_scale = tl.exp(s_shift) / out_se
    merged_output_candidate = p_out * p_scale + s_out * s_scale
    merged_output = tl.where(both_valid, merged_output_candidate,
                             tl.where(only_p, p_out, s_out))

    tl.store(output + base_offset + head_idx * HEAD_SIZE + head_arange,
             merged_output,
             mask=head_mask)


# === Single test: iterative merge (via multiple flash_attn_varlen_func calls) 
#      vs. a single unchunked call. We transpose the softmax lse outputs 
#      because FlashAttention returns them as [NUM_TOKENS, NUM_HEADS],
#      but our merge kernel expects [NUM_HEADS, NUM_TOKENS]. ===

class TestFlashAttnMerge(unittest.TestCase):
    def test_flash_attn_merge(self):
        torch.manual_seed(0)
        device = "cuda"
        # Dimensions:
        num_tokens = 16         # number of query tokens
        num_heads = 4
        HEAD_SIZE = 8
        chunk_max_seq_len = 16  # keys/values length per chunk
        num_chunks = 3
        max_query_len = num_tokens  # for simplicity
        softmax_scale = 1.0

        # Create a fixed query tensor in fp16.
        q = torch.randn(num_tokens, num_heads, HEAD_SIZE, device=device, dtype=torch.float16)
        cu_seqlens_q = torch.tensor([0, num_tokens], device=device, dtype=torch.int32)

        # Compute chunked attention outputs.
        # (Note: flash_attn_varlen_func returns softmax_lse in shape [NUM_TOKENS, NUM_HEADS],
        # so we transpose it to [NUM_HEADS, NUM_TOKENS] for merging.)
        chunks_output = []
        chunks_lse = []
        chunks_k = []
        chunks_v = []
        for _ in range(num_chunks):
            chunk_k = torch.randn(chunk_max_seq_len, num_heads, HEAD_SIZE, device=device, dtype=torch.float16)
            chunk_v = torch.randn(chunk_max_seq_len, num_heads, HEAD_SIZE, device=device, dtype=torch.float16)
            chunks_k.append(chunk_k)
            chunks_v.append(chunk_v)
            cu_seqlens_k = torch.tensor([0, chunk_max_seq_len], device=device, dtype=torch.int32)
            attn_output, attn_softmax_lse = flash_attn_varlen_func(
                q=q,
                k=chunk_k,
                v=chunk_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=chunk_max_seq_len,
                softmax_scale=softmax_scale,
                causal=True,
                return_softmax_lse=True,
                fa_version=3,
            )
            chunks_output.append(attn_output)
            # Transpose lse from [num_tokens, num_heads] to [num_heads, num_tokens]
            chunks_lse.append(attn_softmax_lse.transpose(0, 1).contiguous())

        # Iteratively merge the chunk outputs.
        # Allocate temporary tensor for merged lse with shape [num_heads, num_tokens].
        merged_output = chunks_output[0].clone()
        merged_lse = chunks_lse[0].clone()
        for i in range(1, num_chunks):
            tmp_output = torch.empty_like(merged_output)
            tmp_lse = torch.empty_like(merged_lse)
            merge_attn_states(
                tmp_output,
                merged_output,
                merged_lse,
                chunks_output[i],
                chunks_lse[i],
                tmp_lse,
            )
            merged_output = tmp_output
            merged_lse = tmp_lse

        # Unchunked version: concatenate keys and values and call flash_attn_varlen_func once.
        full_k = torch.cat(chunks_k, dim=0)  # shape: (num_chunks*chunk_max_seq_len, num_heads, HEAD_SIZE)
        full_v = torch.cat(chunks_v, dim=0)
        total_seq_len = num_chunks * chunk_max_seq_len
        cu_seqlens_k_full = torch.tensor([0, total_seq_len], device=device, dtype=torch.int32)
        attn_output_full, attn_softmax_lse_full = flash_attn_varlen_func(
            q=q,
            k=full_k,
            v=full_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k_full,
            max_seqlen_q=max_query_len,
            max_seqlen_k=total_seq_len,
            softmax_scale=softmax_scale,
            causal=True,
            return_softmax_lse=True,
            fa_version=3,
        )
        # Transpose the full lse to [num_heads, num_tokens] for comparison.
        attn_softmax_lse_full = attn_softmax_lse_full.transpose(0, 1).contiguous()

        # Compare the merged (iterative) result with the unchunked result.
        # (fp16 numerics are less precise, so we use a looser tolerance.)
        torch.testing.assert_close(merged_output, attn_output_full, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse, attn_softmax_lse_full, atol=1e-3, rtol=1e-3)

if __name__ == '__main__':
    unittest.main()
