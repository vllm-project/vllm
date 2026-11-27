# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm sparse MLA backends and causal window indexing for HYV4 Flash.

The AITER sparse path reads each row's first `paged_kv_indptr` delta entries
and maps an invalid index to slot 0 rather than skipping it, so every indexer
must pack its live indices to the front of the row, and the builder must count
exactly the live entries of each row.
"""

from typing import Any

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
    ROCMAiterMLASparseMetadataBuilder,
)


@triton.jit
def _live_counts_kernel(
    seq_lens_ptr,
    query_start_loc_ptr,
    out_ptr,
    WINDOW: tl.constexpr,
    RATIO: tl.constexpr,
    GROUP_BUDGET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(query_start_loc_ptr + request)
    end = tl.load(query_start_loc_ptr + request + 1)
    mask = start + offsets < end
    seq_len = tl.load(seq_lens_ptr + request)
    logical = seq_len - (end - start) + offsets
    if WINDOW > 0:
        count = tl.minimum(logical + 1, WINDOW)
    else:
        groups = tl.minimum(logical // RATIO, GROUP_BUDGET)
        count = groups * RATIO + logical % RATIO + 1
    count = tl.where((seq_len > 0) & (logical >= 0), count, 0)
    tl.store(out_ptr + start + offsets, count, mask=mask)


def live_counts(
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_tokens: int,
    max_query_len: int,
    window: int,
    ratio: int,
    group_budget: int,
) -> torch.Tensor:
    """Live top-k entries per query token, given its logical position.

    A window row holds `min(pos + 1, window)` entries. A compressed row holds
    up to `group_budget` whole history groups plus the `pos % ratio + 1`
    tokens of its open group, which is `pos + 1` until the budget saturates.
    """
    out = torch.zeros(num_tokens, dtype=torch.int32, device=seq_lens.device)
    num_reqs = query_start_loc.numel() - 1
    if num_tokens == 0 or num_reqs == 0:
        return out
    block = 64
    _live_counts_kernel[(num_reqs, triton.cdiv(max_query_len, block))](
        seq_lens,
        query_start_loc,
        out,
        WINDOW=window,
        RATIO=ratio,
        GROUP_BUDGET=group_budget,
        BLOCK=block,
    )
    return out


class FlashSparseMetadataBuilder(ROCMAiterMLASparseMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        layer = vllm_config.compilation_config.static_forward_context[layer_names[0]]
        assert layer.topk_buffer_width is not None
        self.topk_tokens = layer.topk_buffer_width
        config = vllm_config.model_config.hf_text_config
        sliding = getattr(layer, "sliding_window", None)
        self.window = int(sliding) if sliding else 0
        self.ratio = config.compressed_indexer_compress_ratio
        self.group_budget = config.compressed_indexer_group_topk - 1
        if self.window:
            assert self.window <= self.topk_tokens
        else:
            assert (self.group_budget + 1) * self.ratio <= self.topk_tokens

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        cm = common_attn_metadata
        num_reqs = metadata.num_reqs
        num_tokens = int(cm.query_start_loc_cpu[num_reqs].item())
        counts = live_counts(
            cm.seq_lens[:num_reqs],
            cm.query_start_loc[: num_reqs + 1],
            num_tokens,
            cm.max_query_len,
            self.window,
            self.ratio,
            self.group_budget,
        )
        torch.cumsum(counts, dim=0, out=self.paged_kv_indptr[1 : num_tokens + 1])
        tail = self.paged_kv_indptr[num_tokens + 1 :]
        tail.copy_(self.paged_kv_indptr[num_tokens].expand_as(tail))
        return metadata


class FlashSparseBackend(ROCMAiterMLASparseBackend):
    @staticmethod
    def get_builder_cls():
        return FlashSparseMetadataBuilder

    @classmethod
    def supports_pcp(cls):
        return False

    @classmethod
    def supports_dcp(cls):
        return False


class FlashSWABackend(FlashSparseBackend):
    @classmethod
    def supports_sliding_window(cls):
        return True


def window_indices(positions: torch.Tensor, window: int, width: int) -> torch.Tensor:
    """Causal window indices packed to the front of each row."""
    if window < 1 or width < window:
        raise ValueError("SWA index width must cover its causal window")
    columns = torch.arange(width, device=positions.device)
    first = (positions - window + 1).clamp_min(0)
    count = (positions + 1).clamp(0, window)
    indices = first[:, None] + columns
    live = (positions[:, None] >= 0) & (columns < count[:, None])
    return torch.where(live, indices, -1).int()


class FlashWindowIndexer(nn.Module):
    def __init__(self, window: int, topk_indices_buffer, attention_prefix: str):
        super().__init__()
        self.window = window
        self.attention_prefix = attention_prefix
        # Logical-to-physical remapping uses 128-column tiles.
        width = round_up(window, 128)
        if window < 1 or topk_indices_buffer.shape[1] < width:
            raise ValueError("SWA index width must cover its causal window")
        rows = topk_indices_buffer.shape[0]
        # Reuse the shared scratch storage with a packed, window-sized stride.
        self.topk_indices_buffer = topk_indices_buffer.view(-1)[: rows * width].view(
            rows, width
        )

    def forward(self, hidden_states, q_c, positions, rope=None):
        output = self.topk_indices_buffer[: hidden_states.shape[0]]
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return output.fill_(-1)
        if not isinstance(metadata, dict):
            raise TypeError("Flash SWA requires per-layer metadata")
        meta: Any = metadata[self.attention_prefix]
        if meta.num_actual_tokens == 0 or meta.seq_lens.numel() == 0:
            return output.fill_(-1)
        rows = torch.arange(output.shape[0], device=output.device)
        starts = meta.query_start_loc
        requests = torch.searchsorted(starts[1:], rows, right=True)
        requests = requests.clamp_max(starts.numel() - 2).long()
        logical = meta.seq_lens[requests] - (starts[requests + 1] - starts[requests])
        logical = logical + rows - starts[requests]
        live = (rows < meta.num_actual_tokens) & (rows < starts[-1])
        output.copy_(
            window_indices(torch.where(live, logical, -1), self.window, output.shape[1])
        )
        return output
