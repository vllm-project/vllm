# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A test-only attention backend that verifies trees: TRITON_ATTN, with every
row of a branching verify plan recomputed in PyTorch over its request's context,
itself and its ancestors along ``CommonAttentionMetadata.tree_parents``. It reads
the batch layout on the host, so it runs eager only."""

import torch

from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

from .reference import tree_ancestors


class TreeAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        parents = common_attn_metadata.tree_parents
        metadata.tree_ancestors = (
            None
            if parents is None
            else tree_ancestors(parents, common_attn_metadata.query_start_loc)
        )
        return metadata


class TreeAttentionImpl(TritonAttentionImpl):
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, **kw):
        output = super().forward(
            layer, query, key, value, kv_cache, attn_metadata, output, **kw
        )
        ancestors = getattr(attn_metadata, "tree_ancestors", None)
        if ancestors is not None and any(sees is not None for sees in ancestors):
            key_cache, value_cache = kv_cache.split(self.head_size, dim=-1)
            tree_attention(
                query, key_cache, value_cache, attn_metadata, self.scale, output
            )
        return output


class TreeAttentionBackend(TritonAttentionBackend):
    serves_trees = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[TreeAttentionImpl]:
        return TreeAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TreeAttentionMetadataBuilder]:
        return TreeAttentionMetadataBuilder


def tree_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: TritonAttentionMetadata,
    scale: float,
    output: torch.Tensor,
) -> None:
    """Overwrite the output of each branching request: query row ``i`` attends
    to the context and the query rows ``j`` with ``ancestors[i, j]`` set."""
    # A readable reference, not a fast path: it reads the layout on the host.
    with gpu_sync_allowed():
        query_start_loc = metadata.query_start_loc.tolist()
        seq_lens = metadata.seq_lens.tolist()
    ancestors = metadata.tree_ancestors
    block_size = key_cache.shape[2]
    for b in range(len(query_start_loc) - 1):
        start, end = query_start_loc[b], query_start_loc[b + 1]
        num_rows, seq_len = end - start, seq_lens[b]
        sees = ancestors[b]
        if sees is None:
            continue
        pos = torch.arange(seq_len, device=query.device)
        blocks = metadata.block_table[b, pos // block_size]
        k = key_cache[blocks, :, pos % block_size].float()  # [seq_len, kv_heads, hs]
        v = value_cache[blocks, :, pos % block_size].float()
        visible = torch.ones(num_rows, seq_len, dtype=torch.bool, device=query.device)
        visible[:, seq_len - num_rows :] = sees
        q = query[start:end].float()
        group = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * scale
        scores.masked_fill_(~visible, float("-inf"))
        out = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v)
        output[start:end] = out.reshape(output[start:end].shape).to(output.dtype)
