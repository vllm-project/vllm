import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_apply_penalty(logits, presence_penalty, freqency_penalty,
                              repetition_penalty, p_token_ids, p_token_counts,
                              p_cumsum_seq_len, stride_logit_b,
                              block_p: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_freqency = tl.load(freqency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_repetition = tl.load(repetition_penalty + cur_batch)

    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)

    cur_batch_id_offset = cur_batch_start_index + tl.arange(0, block_p)
    batch_ids = tl.load(p_token_ids + cur_batch_id_offset,
                        mask=cur_batch_id_offset < cur_batch_end_index,
                        other=0)
    batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset,
                              mask=cur_batch_id_offset < cur_batch_end_index,
                              other=0)

    row_start_ptr = logits + cur_batch * stride_logit_b
    cur_offset = row_start_ptr + batch_ids
    cur_logits = tl.load(cur_offset,
                         mask=cur_batch_id_offset < cur_batch_end_index,
                         other=0.0)
    rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition,
                          cur_logits * cur_repetition)
    freq_logits = rep_logits - batch_ids_count * cur_freqency
    pre_logits = freq_logits - cur_presence
    output_ptr = logits + cur_batch * stride_logit_b + batch_ids
    tl.store(output_ptr,
             pre_logits,
             mask=cur_batch_id_offset < cur_batch_end_index)


@torch.no_grad()
def apply_penalty(logits, presence_penalty, freqency_penalty,
                  repetition_penalty, p_token_ids, p_token_counts,
                  p_cumsum_seq_len, p_max_len_in_batch):
    if not logits.is_contiguous():
        logits = logits.contiguous()
    block = triton.next_power_of_2(p_max_len_in_batch)
    if block <= 512:
        block = 512
    elif block <= 1024:
        block = 1024
    num_warps = 8
    _fwd_kernel_apply_penalty[(logits.shape[0], )](logits,
                                                   presence_penalty,
                                                   freqency_penalty,
                                                   repetition_penalty,
                                                   p_token_ids,
                                                   p_token_counts,
                                                   p_cumsum_seq_len,
                                                   logits.stride(0),
                                                   num_warps=num_warps,
                                                   block_p=block)
