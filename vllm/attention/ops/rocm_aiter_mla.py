# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op, is_torch_equal_or_newer


def get_aiter_mla_metadata(max_batch_size: int, block_size: int,
                           max_block_per_batch: int,
                           device: torch.device) -> tuple[torch.Tensor, ...]:
    paged_kv_indices = torch.zeros(max_batch_size * max_block_per_batch,
                                   dtype=torch.int32,
                                   device=device)
    paged_kv_indptr = torch.zeros(max_batch_size + 1,
                                  dtype=torch.int32,
                                  device=device)
    paged_kv_last_page_lens = torch.full((max_batch_size, ),
                                         block_size,
                                         dtype=torch.int32)
    qo_indptr = torch.zeros(max_batch_size + 1, dtype=torch.int, device=device)
    return paged_kv_indices, paged_kv_indptr, paged_kv_last_page_lens, qo_indptr


def aiter_mla_decode_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    sm_scale: float,
    qo_indptr: torch.Tensor,
    max_seqlen_qo: int,
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    kv_last_page_lens: Optional[torch.Tensor] = None,
    logit_cap: float = 0.0,
):

    torch.ops.vllm.rocm_aiter_mla_decode_fwd(q,
                                             kv_buffer.view(
                                                 -1, 1, 1, q.shape[-1]),
                                             o,
                                             qo_indptr,
                                             max_seqlen_qo,
                                             kv_indptr,
                                             kv_indices,
                                             kv_last_page_lens,
                                             sm_scale=sm_scale,
                                             logit_cap=logit_cap)


def mla_decode_fwd_impl(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_seqlen_qo: int,
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    kv_last_page_lens: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    logit_cap: float = 0.0,
) -> None:
    from aiter.mla import mla_decode_fwd

    mla_decode_fwd(q,
                   kv_buffer.view(-1, 1, 1, q.shape[-1]),
                   o,
                   qo_indptr,
                   kv_indptr,
                   kv_indices,
                   kv_last_page_lens,
                   max_seqlen_qo,
                   sm_scale=sm_scale,
                   logit_cap=logit_cap)


def mla_decode_fwd_fake(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_seqlen_qo: int,
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    kv_last_page_lens: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    logit_cap: float = 0.0,
) -> None:
    pass


if current_platform.is_rocm():
    if is_torch_equal_or_newer("2.7.0"):
        tags = ()
    else:
        tags = (torch.Tag.needs_fixed_stride_order, ),
    direct_register_custom_op(op_name="rocm_aiter_mla_decode_fwd",
                              op_func=mla_decode_fwd_impl,
                              mutates_args=["o"],
                              fake_impl=mla_decode_fwd_fake,
                              tags=tags)
