# SPDX-License-Identifier: Apache-2.0
 
from typing import Optional
 
import torch
 
 
def get_aiter_mla_metadata(max_batch_size: int, block_size: int,
                           device: torch.device) -> tuple[torch.Tensor, ...]:
 
    paged_kv_indptr_tensor = torch.zeros(max_batch_size + 1,
                                         dtype=torch.int32,
                                         device=device)
    paged_kv_last_page_lens_tensor = torch.full((max_batch_size, ),
                                                block_size,
                                                dtype=torch.int32)
    return paged_kv_indptr_tensor, paged_kv_last_page_lens_tensor
 
 
def aiter_mla_decode_fwd(
    q,
    kv_buffer,
    o,
    sm_scale,
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    kv_last_page_lens: Optional[torch.Tensor] = None,
    logit_cap: float = 0.0,
):
    from aiter.mla import mla_decode_fwd
 
    mla_decode_fwd(q,
                   kv_buffer.view(-1, 1, 1, q.shape[-1]),
                   o,
                   kv_indptr,
                   kv_indices,
                   kv_last_page_lens,
                   sm_scale=sm_scale,
                   logit_cap=logit_cap)