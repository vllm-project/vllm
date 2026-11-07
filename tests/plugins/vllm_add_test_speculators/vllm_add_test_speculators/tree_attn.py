# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A test-only attention backend that verifies trees: TRITON_ATTN, with the
rows of each branching request recomputed in PyTorch over its context, itself
and its ancestors along ``CommonAttentionMetadata.tree_parents``. It reads
nothing on the host at forward time, so FULL CUDA graphs capture it and each
replay reads that step's parents."""

import torch

from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

from . import EAGER_UNIFORM_VERIFY, UNIFORM_VERIFY, VerifySteps
from .reference import tree_window


class TreeAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    _steps = 0

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # A verify plan has at most this many rows (the round runner's max_rows).
        self.tree_width = 1 + vllm_config.num_speculative_tokens

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata.tree_parents = common_attn_metadata.tree_parents
        metadata.tree_width = self.tree_width
        # Every request verifies all its rows: the shape FULL CUDA graphs
        # capture. The host copy of query_start_loc needs no sync.
        num_reqs = common_attn_metadata.num_reqs
        query_lens = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1].diff()
        metadata.uniform_verify = None
        if VerifySteps.topology is not None and bool(
            (query_lens == self.tree_width).all()
        ):
            TreeAttentionMetadataBuilder._steps += 1
            metadata.uniform_verify = TreeAttentionMetadataBuilder._steps
            print(UNIFORM_VERIFY, VerifySteps.topology, flush=True)
        metadata.topology = VerifySteps.topology
        return metadata

    def build_for_cudagraph_capture(self, common_attn_metadata):
        metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        metadata.uniform_verify = None
        return metadata


class TreeAttentionImpl(TritonAttentionImpl):
    _printed_step = None

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, **kw):
        output = super().forward(
            layer, query, key, value, kv_cache, attn_metadata, output, **kw
        )
        if getattr(attn_metadata, "tree_parents", None) is None:
            return output
        step = attn_metadata.uniform_verify
        if step is not None and step != TreeAttentionImpl._printed_step:
            # A FULL CUDA graph replay runs no Python: the tests read this line.
            print(EAGER_UNIFORM_VERIFY, attn_metadata.topology, flush=True)
            TreeAttentionImpl._printed_step = step
        key_cache, value_cache = kv_cache.split(self.head_size, dim=-1)
        tree_attention(query, key_cache, value_cache, attn_metadata, self.scale, output)
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
    """Overwrite the last ``tree_width`` query rows of each branching request:
    a row attends to the request's context and to the rows it sees in
    :func:`tree_window`. A readable reference over every key position the
    block table holds, with static shapes; not a fast path."""
    tokens, valid, sees = tree_window(
        metadata.tree_parents, metadata.query_start_loc, metadata.tree_width
    )
    num_reqs, width = tokens.shape
    causal = torch.ones(width, width, dtype=torch.bool, device=query.device).tril()
    branching = ((sees != causal) & valid[..., None]).flatten(1).any(dim=1)
    block_table = metadata.block_table[:num_reqs]
    block_size = key_cache.shape[2]
    pos = torch.arange(block_table.shape[1] * block_size, device=query.device)
    blocks = block_table[:, pos // block_size]
    k = key_cache[blocks, :, pos % block_size].float()  # [R, L, kv_heads, hs]
    v = value_cache[blocks, :, pos % block_size].float()
    # Window column j is key position seq_len - width + j.
    col = pos - (metadata.seq_lens[:num_reqs, None] - width)  # [R, L]
    in_window = (col >= 0) & (col < width)
    index = col.clamp(0, width - 1)[:, None].expand(-1, width, -1)
    visible = (col < 0)[:, None] | (in_window[:, None] & sees.gather(2, index))
    q = query[tokens.clamp(min=0)].float()  # [R, W, heads, hs]
    q = q.unflatten(2, (k.shape[2], -1))  # [R, W, kv_heads, group, hs]
    scores = torch.einsum("rwkgd,rlkd->rkgwl", q, k) * scale
    scores.masked_fill_(~visible[:, None, None], float("-inf"))
    out = torch.einsum("rkgwl,rlkd->rwkgd", scores.softmax(dim=-1), v)
    # Rows to keep go to a scratch row past the end of the output.
    rows = torch.where(valid & branching[:, None], tokens, output.shape[0])
    scratch = torch.cat([output, output[:1]])
    out = out.reshape(num_reqs * width, *output.shape[1:]).to(output.dtype)
    scratch.index_copy_(0, rows.flatten(), out)
    output.copy_(scratch[:-1])
