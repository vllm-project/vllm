# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal window indexing over engine-managed sliding MLA pages."""

from typing import Any

from torch import nn

from vllm.forward_context import get_forward_context
from vllm.models.hy_v4.nvidia.flashmla_sparse import HYV4FlashMLASparseBackend
from vllm.triton_utils import tl, triton
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


@triton.jit(do_not_specialize=["num_requests", "num_tokens"])
def _window_indices_kernel(
    Out,
    Starts,
    SeqLens,
    num_requests,
    num_tokens,
    OUT_ROW,
    WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    WIDTH_P2: tl.constexpr,
):
    row = tl.program_id(0)
    lo = 0
    hi = num_requests
    while lo < hi:
        mid = (lo + hi) // 2
        right = tl.load(Starts + mid + 1) <= row
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    request = tl.minimum(lo, num_requests - 1)
    start = tl.load(Starts + request)
    end = tl.load(Starts + request + 1)
    position = tl.load(SeqLens + request) - (end - start) + row - start
    live = (row < num_tokens) & (row < tl.load(Starts + num_requests))
    position = tl.where(live, position, -1)
    column = tl.arange(0, WIDTH_P2)
    index = position - WINDOW + 1 + column
    valid = (position >= 0) & (column < WINDOW) & (index >= 0)
    tl.store(
        Out + row.to(tl.int64) * OUT_ROW + column,
        tl.where(valid, index, -1),
        column < WIDTH,
    )


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
        num_requests = meta.query_start_loc.numel() - 1
        _window_indices_kernel[(output.shape[0],)](
            output,
            meta.query_start_loc,
            meta.seq_lens,
            num_requests,
            meta.num_actual_tokens,
            output.stride(0),
            self.window,
            output.shape[1],
            triton.next_power_of_2(output.shape[1]),
        )
        return output
