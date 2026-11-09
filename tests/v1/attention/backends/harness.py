# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-agnostic pieces of the attention backend suites.

- ``PagedBatch``: a reordered (decode, extend, prefill) batch with a unique
  block per page, so a test can say which cache slots each request reads.
- ``ref_attention``: float32 reference over logical (unpaged) K/V.
- ``KernelSpy``: records which kernels a forward actually called.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_prefills_and_extends


@dataclass
class PagedBatch:
    """Requests in the order the runner's reorder pass produces them."""

    query_lens: list[int]
    seq_lens: list[int]
    block_size: int
    num_blocks: int
    common: CommonAttentionMetadata
    # Cache slot of every position 0..seq_len-1, per request.
    slots: list[torch.Tensor]
    # Cache slot of every query token, in token order.
    query_slots: torch.Tensor

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def num_slots(self) -> int:
        return self.num_blocks * self.block_size


def blocks_needed(seq_lens: list[int], block_size: int) -> int:
    return sum(math.ceil(s / block_size) for s in seq_lens)


def make_paged_batch(
    query_lens: list[int],
    seq_lens: list[int],
    block_size: int,
    device: torch.device,
    *,
    num_blocks: int | None = None,
    num_padded_tokens: int | None = None,
    causal: bool = True,
    seed: int = 0,
) -> PagedBatch:
    """Build metadata whose blocks are a random permutation (no aliasing).

    Block-table rows are padded with block 0, as the model runner does.
    ``num_padded_tokens`` > sum(query_lens) leaves trailing padding slots of
    -1, as a CUDA-graph-padded batch does. A request may have no query tokens:
    seq_len 0 is a FULL CUDA graph padding row, seq_len > 0 a request whose KV
    an async connector is still loading.
    """
    assert len(query_lens) == len(seq_lens)
    assert all(0 <= q <= s for q, s in zip(query_lens, seq_lens))
    assert any(q > 0 for q in query_lens)
    gen = torch.Generator().manual_seed(seed)
    needed = blocks_needed(seq_lens, block_size)
    num_blocks = num_blocks or needed + 1
    assert num_blocks >= needed + 1
    # Block 0 is reserved as the pad value and never holds a request's page.
    perm = torch.randperm(num_blocks - 1, generator=gen)[:needed] + 1
    max_blocks = max(1, *(math.ceil(s / block_size) for s in seq_lens))
    block_table = torch.zeros(len(seq_lens), max_blocks, dtype=torch.int32)
    slots, query_slots = [], []
    cursor = 0
    for i, (q, s) in enumerate(zip(query_lens, seq_lens)):
        n = math.ceil(s / block_size)
        block_table[i, :n] = perm[cursor : cursor + n].to(torch.int32)
        cursor += n
        pos = torch.arange(s)
        req_slots = block_table[i, pos // block_size].long() * block_size + (
            pos % block_size
        )
        slots.append(req_slots.to(device))
        query_slots.append(req_slots[s - q :])
    query_slots_t = torch.cat(query_slots)
    num_tokens = sum(query_lens)
    slot_mapping = query_slots_t.clone()
    if num_padded_tokens is not None and num_padded_tokens > num_tokens:
        slot_mapping = torch.cat(
            [slot_mapping, torch.full((num_padded_tokens - num_tokens,), -1)]
        )
    qsl = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens_t.to(device),
        seq_lens_cpu_upper_bound=seq_lens_t,
        num_reqs=len(query_lens),
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table.to(device),
        slot_mapping=slot_mapping.to(device),
        causal=causal,
    )
    return PagedBatch(
        query_lens=list(query_lens),
        seq_lens=list(seq_lens),
        block_size=block_size,
        num_blocks=num_blocks,
        common=common,
        slots=slots,
        query_slots=query_slots_t.to(device),
    )


def phase_counts(common: CommonAttentionMetadata, threshold: int = 1):
    """(num_decodes, num_extends, num_prefills) as the builder will split."""
    d, e, p, *_ = split_decodes_prefills_and_extends(common, threshold)
    return d, e, p


def ref_attention(
    query: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    batch: PagedBatch,
    scale: float,
    *,
    sliding_window: int | None = None,
    causal: bool = True,
    logits_soft_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    q_chunk: int = 512,
) -> torch.Tensor:
    """Float32 attention over the logical K/V each request can see.

    Args:
        query: [num_tokens, H, D].
        k_all, v_all: [num_slots, KV, D], indexed by cache slot.
        sliding_window: vLLM semantics, a query at position p sees keys at
            positions > p - sliding_window.
        sinks: [H] per-head sink logits (an extra softmax column, no value).
        alibi_slopes: [H]; bias = slope * (k_pos - q_pos).

    """
    num_heads = query.shape[1]
    out = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    offset = 0
    for q_len, seq_len, slots in zip(batch.query_lens, batch.seq_lens, batch.slots):
        q_all = query[offset : offset + q_len].float()
        k = k_all[slots].float()
        v = v_all[slots].float()
        group = num_heads // k.shape[1]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        k_pos = torch.arange(seq_len, device=query.device)
        for start in range(0, q_len, q_chunk):
            q = q_all[start : start + q_chunk]
            q_pos = torch.arange(
                seq_len - q_len + start,
                seq_len - q_len + start + q.shape[0],
                device=query.device,
            )
            logits = torch.einsum("qhd,khd->hqk", q, k) * scale
            if logits_soft_cap:
                logits = logits_soft_cap * torch.tanh(logits / logits_soft_cap)
            rel = k_pos[None, :] - q_pos[:, None]
            if alibi_slopes is not None:
                logits = logits + alibi_slopes.float()[:, None, None] * rel[None]
            mask = torch.ones_like(rel, dtype=torch.bool)
            if causal:
                mask &= rel <= 0
            if sliding_window is not None:
                mask &= rel > -sliding_window
            logits = logits.masked_fill(~mask[None], float("-inf"))
            if sinks is not None:
                sink_col = sinks.float()[:, None, None].expand(-1, q.shape[0], 1)
                probs = torch.softmax(torch.cat([logits, sink_col], dim=-1), -1)
                probs = probs[..., :-1]
            else:
                probs = torch.softmax(logits, dim=-1)
            out[offset + start : offset + start + q.shape[0]] = torch.einsum(
                "hqk,khd->qhd", probs, v
            )
        offset += q_len
    return out


@dataclass
class KernelSpy:
    """Wraps kernels so a test can assert which ones a forward called."""

    monkeypatch: pytest.MonkeyPatch
    calls: list[tuple[str, tuple, dict[str, Any]]] = field(default_factory=list)

    def watch(self, owner: Any, attr: str, name: str | None = None) -> None:
        original: Callable = getattr(owner, attr)
        label = name or attr

        def wrapper(*args, **kwargs):
            self.calls.append((label, args, kwargs))
            return original(*args, **kwargs)

        self.monkeypatch.setattr(owner, attr, wrapper)

    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]

    def kwargs_of(self, name: str) -> list[dict[str, Any]]:
        return [kw for n, _, kw in self.calls if n == name]

    def args_of(self, name: str) -> list[tuple]:
        return [args for n, args, _ in self.calls if n == name]

    def reset(self) -> None:
        self.calls.clear()
