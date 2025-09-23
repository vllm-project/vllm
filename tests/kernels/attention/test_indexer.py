import random

import torch

from vllm.utils.tile_lang_kernels import act_quant, fp8_index
from vllm import _custom_ops as ops


def ref_compute_logits_fp8(q, kv, weights, mask, block_size):
    q_fp8, q_scale = act_quant(q, block_size, "ue8m0")
    k_fp8, k_scale = act_quant(kv, block_size, "ue8m0")
    
    weights = weights.unsqueeze(-1) * q_scale
    weights = weights * (128**(-0.5))
    index_score = fp8_index(
        q_fp8.contiguous(), weights,
        k_fp8.contiguous(),
        k_scale.contiguous())
    if mask is not None:
        index_score += mask
    return index_score

def ref_indexer(seq_len, q, kv, weights, block_size, topk):
    B = seq_len.shape[0]
    varlen_logits = []
    
    for i in range(B):
        S = seq_len[i]
        q_s = q[i][:S].contiguous().unsqueeze(0)
        kv_s = kv[i][:S].contiguous().unsqueeze(0)
        weights_s = weights[i][:S].contiguous().unsqueeze(0)
        mask = torch.full(
            (S, S), float("-inf"),
            device="cuda").triu_(1)
        logits = ref_compute_logits_fp8(q_s, kv_s, weights_s, mask, block_size)
        varlen_logits.append(logits)
    # topk_indices = index_score.topk(topk,
    #                                 dim=-1)[1]
    return varlen_logits

def kv_spans_from_batches(start_seq_loc: torch.Tensor,
                          seq_len_per_batch: torch.Tensor):
    """
    Args:
      start_seq_loc: 1D long tensor [B+1], cumulative counts of selected tokens per batch.
                     Example: [0, 2, 4, 7] -> batch sizes (selected) [2, 2, 3], N=7 tokens total.
      seq_len_per_batch: 1D long tensor [B], full sequence length (KV length) of each batch.
                         Example: [5, 9, 4].

    Returns:
      start_tensor: 1D long tensor [N], start offset in the concatenated KV cache for each token's batch.
      end_location: 1D long tensor [N], **exclusive** end = start + token's local position.
                    (So the attended KV slice is kv[start:end].)

    Assumes each batch contributes its full `seq_len_per_batch[i]` keys to the KV cache, and
    the selected tokens within a batch are the **last** `counts[i]` positions of that sequence.
    """
    q = start_seq_loc.to(dtype=torch.long)
    L = seq_len_per_batch.to(dtype=torch.long, device=q.device)
    assert q.dim() == 1 and L.dim() == 1
    assert q.numel() == L.numel() + 1, "start_seq_loc must have length B+1"

    # Selected tokens per batch and totals
    counts = q[1:] - q[:-1]                  # [B]
    N = int(q[-1].item())                    # total selected tokens
    B = L.numel()
    device = L.device

    if N == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device))

    # KV start offsets per batch in the concatenated KV cache
    kv_starts_per_batch = torch.cumsum(L, dim=0) - L          # [B]

    # For each selected token, which batch does it belong to?
    batch_id = torch.repeat_interleave(torch.arange(B, device=device), counts)  # [N]

    # Map batch KV start to each token
    start_tensor = kv_starts_per_batch[batch_id]              # [N]

    # End-align local positions inside each batch:
    # local_pos = L[b] - counts[b] + (1..counts[b])  for each batch b
    L_expand    = torch.repeat_interleave(L, counts)          # [N]
    m_expand    = torch.repeat_interleave(counts, counts)     # [N]
    # position within the selected block: 1..counts[b]
    pos_within  = (torch.arange(N, device=device, dtype=torch.long)
                   - torch.repeat_interleave(q[:-1], counts) + 1)

    local_pos   = L_expand - m_expand + pos_within            # [N], 1-based
    end_location = start_tensor + local_pos                   # exclusive end

    return start_tensor, end_location

def ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    k = kv
    q = q.float()
    k = k.float()

    seq_len_kv = kv.shape[0]
    mask_lo = (torch.arange(0, seq_len_kv, device="cuda")[None, :]
               >= cu_seqlen_ks[:, None])
    mask_hi = (torch.arange(0, seq_len_kv, device="cuda")[None, :]
               < cu_seqlen_ke[:, None])
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    cost = mask.sum()
    return logits, cost

def torch_indexer(seq_len, q, kv, weights, block_size, topk):
    NUM_BLOCKS = 8
    BLOCK_SIZE = 32

    B = seq_len.shape[0]
    concat_q = []
    concat_kv = []
    concat_weights = []
    total_slots = NUM_BLOCKS * BLOCK_SIZE
    head_dim = kv.shape[-1]
    max_num_block_per_batch = torch.max(seq_len)
    block_table = torch.empty((B, max_num_block_per_batch),
                              dtype=torch.int32,
                              device="cuda")

    for i in range(B):
        S = seq_len[i]
        q_s = q[i][:S].contiguous()
        kv_s = kv[i][:S].contiguous()
        weight_s = weights[i][:S].contiguous()
        concat_q.append(q_s)
        concat_kv.append(kv_s)
        concat_weights.append(weight_s)
    
    q = torch.cat(concat_q, dim=0)
    kv = torch.cat(concat_kv, dim=0)
    weights = torch.cat(concat_weights, dim=0)

    # write to kv cache based on slot mapping
    entry_size = head_dim * 2
    num_tokens = q.size(0)
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst,
                                dtype=torch.long,
                                device="cuda")
    kv_cache = torch.zeros(
        NUM_BLOCKS,
        BLOCK_SIZE,
        entry_size,
        dtype=torch.bfloat16,
        device="cuda"
    )
    scale = torch.tensor(1, dtype=torch.float32, device="cuda")
    ops.concat_and_cache_mla(
        kv, 
        kv.clone(), 
        kv_cache, 
        slot_mapping,
        "auto", 
        scale
    )

    current_index = 0
    for i in range(B):
        S = seq_len[i]
        block_table[i][:S] = slot_mapping[current_index: current_index + S]
        current_index += S
    
    weights = weights * (128**(-0.5))
    query_start_loc = torch.empty((B + 1), device="cuda")
    query_start_loc[0] = 0
    query_start_loc[1:] = seq_len.cumsum(dim=0).to(dtype=torch.int32)

    kv_gathered = kv_cache.view(-1, entry_size)[slot_mapping][..., :head_dim]
    torch.testing.assert_close(kv, kv_gathered)

    cu_seqlen_ks, cu_seqlen_ke = kv_spans_from_batches(query_start_loc, seq_len)

    logits, _ = ref_fp8_mqa_logits(
        q, 
        kv_gathered, 
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke
    )
    topk_indices = logits.topk(topk, dim=-1)[1]
    mask_lo = topk_indices >= cu_seqlen_ks[:, None]
    mask_hi = topk_indices < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi
    topk_indices = topk_indices.masked_fill(~mask, -1)
    return logits

def test_paged_indexer_python():
    B = 2
    S = 128
    SKV = S
    H = 64
    HKV = 1
    D = 128
    block_size = 128
    topk = 64
    device = "cuda"
    seq_len = torch.randint(low=64, high=S, size=(B,))
    
    q = torch.randn(B, S, H, D, device="cuda",
                    dtype=torch.bfloat16)
    kv = torch.randn(B, SKV, D, device="cuda",
                     dtype=torch.bfloat16)
    weights = torch.randn(B, S, H, device=device, dtype=torch.float32) * H**-0.5

    ref_indices = ref_indexer(seq_len, q, kv, weights, block_size, topk)
    torch_indices = torch_indexer(seq_len, q, kv, weights, block_size, topk)
    import pdb; pdb.set_trace()
    print(ref_indices)


if __name__ == "__main__":
    test_paged_indexer_python()
