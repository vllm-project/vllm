from typing import Tuple

import torch


def expand_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_queries_per_kv = num_heads // num_kv_heads
    query = query.view(query.shape[0], num_kv_heads, num_queries_per_kv,
                       query.shape[-1])
    key = key[:, :, None, :].expand(key.shape[0], num_kv_heads,
                                    num_queries_per_kv, key.shape[-1])
    value = value[:, :, None, :].expand(value.shape[0], num_kv_heads,
                                        num_queries_per_kv, value.shape[-1])
    return query, key, value
