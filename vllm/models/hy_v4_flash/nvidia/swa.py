# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal window indexing over engine-managed sliding MLA pages."""

from typing import Any

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.models.hy_v4.nvidia.flashmla_sparse import HYV4FlashMLASparseBackend
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadataBuilder


class FlashSparseMetadataBuilder(FlashMLASparseMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        layer = vllm_config.compilation_config.static_forward_context[layer_names[0]]
        assert layer.topk_buffer_width is not None
        self.topk_tokens = layer.topk_buffer_width
        self.topk_tokens_tensor.fill_(self.topk_tokens)
        # The direct paged kernel never gathers evicted SWA history.
        self.fp8_use_mixed_batch = True


class FlashSparseBackend(HYV4FlashMLASparseBackend):
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
    if window < 1 or width < window:
        raise ValueError("SWA index width must cover its causal window")
    columns = torch.arange(width, device=positions.device)
    indices = positions[:, None] - window + 1 + columns
    live = (positions[:, None] >= 0) & (columns < window) & (indices >= 0)
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
