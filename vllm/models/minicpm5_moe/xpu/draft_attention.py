# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse native paged SplitKV for each non-causal draft query."""

from collections import Counter

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv


@triton.jit
def _expand_draft_pages(
    query_start,
    lengths,
    pages,
    out_lengths,
    out_pages,
    epochs,
    num_reqs: tl.constexpr,
    source_stride: tl.constexpr,
    source_columns: tl.constexpr,
    output_columns: tl.constexpr,
    block_size: tl.constexpr,
    REQS: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.arange(0, REQS)
    start = tl.load(query_start + req, req < num_reqs, other=0)
    end = tl.load(query_start + req + 1, req < num_reqs, other=0)
    owner = tl.max(tl.where((req < num_reqs) & (start <= row) & (row < end), req, -1))
    length = tl.load(lengths + tl.maximum(owner, 0), owner >= 0, other=0)
    tl.store(out_lengths + row, length)
    col = tl.arange(0, COLS)
    page = tl.load(
        pages + tl.maximum(owner, 0) * source_stride + col,
        (owner >= 0) & (col < source_columns) & (col < tl.cdiv(length, block_size)),
        other=0,
    )
    tl.store(out_pages + row * output_columns + col, page, col < output_columns)
    if row == 0:
        tl.atomic_add(epochs, 1, sem="relaxed")


class MiniCPM5DraftAttention:
    def __init__(
        self, kernel, max_context_len, max_requests, queries, heads, enabled=True
    ):
        self.kernel = kernel
        self.max_context_len = max_context_len
        self.max_requests = max_requests
        self.queries = queries
        self.heads = heads
        self.max_rows = max_requests * queries
        self.enabled = enabled
        self.hits = Counter()
        self.buffers = None

    def eligible(self, arguments):
        q, k, v = (arguments[name] for name in ("q", "k", "v"))
        starts = arguments.get("cu_seqlens_q")
        lengths = arguments.get("seqused_k")
        pages = arguments.get("block_table")
        return (
            self.enabled
            and q.device.type == "xpu"
            and q.dtype in (torch.float16, torch.bfloat16)
            and q.dtype == k.dtype == v.dtype
            and q.ndim == 3
            and 0 < q.shape[0] <= self.max_rows
            and q.shape[1:] == (self.heads, 128)
            and q.stride(1) == 128
            and q.stride(2) == 1
            and k.ndim == v.ndim == 4
            and k.shape == v.shape
            and k.shape[1] % 64 == 0
            and k.shape[2] in (1, 2)
            and k.shape[3] == 128
            and k.stride(3) == v.stride(3) == 1
            and arguments.get("causal") is False
            and 1 < arguments["max_seqlen_q"] <= self.queries
            and 0 < arguments["max_seqlen_k"] <= self.max_context_len
            and starts is not None
            and starts.dtype == torch.int32
            and starts.ndim == 1
            and starts.stride(0) == 1
            and 2 <= starts.numel() <= self.max_requests + 1
            and lengths is not None
            and lengths.dtype == torch.int32
            and lengths.ndim == 1
            and lengths.stride(0) == 1
            and lengths.numel() == starts.numel() - 1
            and pages is not None
            and pages.dtype == torch.int32
            and pages.ndim == 2
            and pages.shape[0] >= lengths.numel()
            and 0 < pages.shape[1] <= cdiv(self.max_context_len, k.shape[1])
            and pages.stride(1) == 1
            and all(t.device == q.device for t in (k, v, starts, lengths, pages))
            and not arguments.get("return_softmax_lse")
            and not arguments.get("return_attn_probs")
            and not arguments.get("dropout_p")
            and not arguments.get("softcap")
            and arguments.get("window_size") in (None, [-1, -1], (-1, -1))
            and all(
                arguments.get(name) is None
                for name in (
                    "cu_seqlens_k",
                    "alibi_slopes",
                    "q_v",
                    "q_descale",
                    "scheduler_metadata",
                    "s_aux",
                    "mask_mod",
                    "aux_tensors",
                    "dynamic_causal",
                )
            )
        )

    def _allocate(self, q, block_size):
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError(
                "MiniCPM5 draft KV buffers must be warmed before capture"
            )
        device, rows = q.device, self.max_rows
        columns = cdiv(self.max_context_len, block_size)
        self.buffers = {
            "block_size": block_size,
            "lengths": torch.empty(rows, dtype=torch.int32, device=device),
            "pages": torch.empty((rows, columns), dtype=torch.int32, device=device),
            "starts": torch.arange(rows + 1, dtype=torch.int32, device=device),
            "epochs": torch.zeros(1, dtype=torch.int32, device=device),
        }

    def __call__(self, *args, **arguments):
        if args or not self.eligible(arguments):
            return self.kernel(*args, **arguments)
        q, k, v = (arguments[name] for name in ("q", "k", "v"))
        if self.buffers is None:
            self._allocate(q, k.shape[1])
        buffers = self.buffers
        if buffers["block_size"] != k.shape[1]:
            raise RuntimeError("MiniCPM5 draft KV block size changed after warmup")
        rows = q.shape[0]
        pages = arguments["block_table"]
        num_reqs = arguments["seqused_k"].numel()
        columns = buffers["pages"].shape[1]
        _expand_draft_pages[(rows,)](
            arguments["cu_seqlens_q"],
            arguments["seqused_k"],
            pages,
            buffers["lengths"],
            buffers["pages"],
            buffers["epochs"],
            num_reqs,
            pages.stride(0),
            pages.shape[1],
            columns,
            k.shape[1],
            triton.next_power_of_2(num_reqs),
            triton.next_power_of_2(columns),
        )
        # Every query keeps its owner's full non-causal context, including all
        # query-block K/V written by the ordinary attention cache update.
        # Single-query sequences select the native paged decode SplitKV path.
        result = self.kernel(
            **dict(
                arguments,
                cu_seqlens_q=buffers["starts"][: rows + 1],
                max_seqlen_q=1,
                seqused_k=buffers["lengths"][:rows],
                block_table=buffers["pages"][:rows],
            )
        )
        self.hits[rows] += 1
        return result


def configure_draft_attention(model, config):
    options = config.additional_config or {}
    enabled = options.get("minicpm5_dspark_split_kv", False)
    if type(enabled) is not bool:
        raise ValueError("minicpm5_dspark_split_kv must be a boolean")
    speculative = config.speculative_config
    if not enabled or speculative is None or speculative.method != "dspark":
        return
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    queries = speculative.num_speculative_tokens
    if not 2 <= queries <= 7:
        return
    for layer in model.model.layers:
        attention = layer.self_attn
        impl = attention.attn.impl
        if (
            type(impl) is FlashAttentionImpl
            and impl.vllm_flash_attn_version == 2
            and attention.causal is False
            and not isinstance(impl.dense_attention_kernel, MiniCPM5DraftAttention)
        ):
            impl.dense_attention_kernel = MiniCPM5DraftAttention(
                impl.dense_attention_kernel,
                config.model_config.max_model_len,
                config.scheduler_config.max_num_seqs,
                queries,
                impl.num_heads,
            )
