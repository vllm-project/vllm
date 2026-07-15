# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.parallel_state import get_pcp_group


def allgather_padded_token_tensors(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    pcp_group = get_pcp_group()
    return tuple(
        pcp_group.all_gather(
            tensor if tensor.is_contiguous() else tensor.contiguous(),
            dim=0,
        )
        for tensor in tensors
    )


def maybe_gather_mla_latent_cache_inputs(
    attn_metadata: object | None,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not use_pcp or attn_metadata is None:
        return kv_c_normed, k_pe, slot_mapping
    assert slot_mapping is not None
    num_tokens = kv_c_normed.shape[0]
    k_pe_flat = k_pe.reshape(num_tokens, -1)
    gathered_kv_c, gathered_k_pe_flat = allgather_padded_token_tensors(
        (kv_c_normed, k_pe_flat)
    )
    gathered_k_pe = gathered_k_pe_flat.view(-1, *k_pe.shape[1:])
    return gathered_kv_c, gathered_k_pe, slot_mapping[: gathered_kv_c.shape[0]]


def maybe_gather_indexer_k(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_pcp:
        return k, slot_mapping
    (gathered_k,) = allgather_padded_token_tensors((k,))
    return gathered_k, slot_mapping[: gathered_k.shape[0]]
