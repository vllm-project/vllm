# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefixLM ranges as causal and bidirectional varlen attention rows.

A PrefixLM range `[start, end)` of a request, a multimodal item or a
bidirectional attention span, lets each of its tokens attend to every token of
the request below `end`; the request's other tokens stay causal. On a layer
without a sliding window, both kinds of token can be computed by a stock varlen
attention kernel, with no mask function:

* the scheduled tokens of a range attend non-causally to the keys `[0, end)`;
* each run of other scheduled tokens attends causally (bottom-right aligned) to
  the keys up to the run's own end.

`PrefixLMRowsBuilder` splits each request that prefills a range in this step
into consecutive rows, one per range and one per causal run, in token order.
Every other request keeps one causal row. A row reads its request's block table
row, so the packed queries, the outputs and the KV cache are unchanged. The
rows run in one varlen call with a per-row causal flag (`row_causal`), so
attention costs what the mask requires and plain requests in the same step
keep causal tile skipping.

The call takes the step's `max_num_splits`, as the backend's other
FlashAttention calls do. With FlashAttention 4 that is 0, so the kernel chooses
how many ways to split the rows' keys.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from vllm.utils.torch_utils import PIN_MEMORY


@dataclass
class PrefixLMRows:
    """Varlen rows for a step in which some request prefills a PrefixLM range.

    The tensors are on the device and have one entry per row, except
    `cu_seqlens_q`, which has `num_rows + 1`.
    """

    num_rows: int
    cu_seqlens_q: torch.Tensor
    """Row start offsets into the packed queries (int32)."""
    seqused_k: torch.Tensor
    """Keys each row attends over: `[0, seqused_k)` (int32)."""
    block_table: torch.Tensor
    """`(num_rows, max_blocks)`: each row's request's block table row (int32)."""
    row_req: torch.Tensor
    """Request index of each row (int32), to gather another block table."""
    row_causal: torch.Tensor
    """1 for causal rows, 0 for rows that cover a range (int32)."""
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass
class PrefixLMRowsCPU:
    """Host-side result of `compute_prefix_lm_rows` (int32 arrays)."""

    row_starts: np.ndarray
    """Offset of each row's first query token in the packed queries."""
    row_lens: np.ndarray
    row_req: np.ndarray
    """Request index of each row."""
    row_seqused_k: np.ndarray
    """Exact key count of the rows of requests that prefill a range; -1 for
    the other requests' rows, which then use the request's own `seq_lens`."""
    row_bidirectional: np.ndarray
    """True for rows that cover a range."""


def compute_prefix_lm_rows(
    ranges: Mapping[int, Sequence[tuple[int, int]]],
    query_start_loc: np.ndarray,
    seq_lens: np.ndarray,
) -> PrefixLMRowsCPU | None:
    """Split the requests that prefill a PrefixLM range into attention rows.

    Args:
        ranges: Request index to its disjoint, inclusive `(start, end)`
            absolute ranges, in any order
            (`CommonAttentionMetadata.mm_req_doc_ranges`). Requests without
            ranges may be absent.
        query_start_loc: `(num_reqs + 1,)` offsets of each request's
            scheduled tokens in the packed queries.
        seq_lens: `(num_reqs,)` tokens of each request after this step. Read
            for requests with ranges, and must be exact for those that
            prefill one. An upper bound for a decode only moves its query
            further past its ranges.

    Returns:
        None when no scheduled query sees a later key of its range (the
        caller keeps its stock path), else the rows. Also None when a range
        that a scheduled query needs runs past the scheduled tokens (a chunked
        multimodal item, say): its tokens need keys that are not computed
        yet, which rows cannot express, so the caller keeps the mask path.
        The scheduler never ends a chunk inside a bidirectional span, so
        spans never take that exit.

    """
    num_reqs = len(query_start_loc) - 1
    starts: list[np.ndarray] = []
    lens: list[np.ndarray] = []
    reqs: list[np.ndarray] = []
    ks: list[np.ndarray] = []
    bidir: list[np.ndarray] = []

    def plain(lo: int, hi: int) -> None:
        # Requests [lo, hi) keep one causal row each, read on the device.
        if hi <= lo:
            return
        starts.append(query_start_loc[lo:hi].astype(np.int32))
        lens.append(np.diff(query_start_loc[lo : hi + 1]).astype(np.int32))
        reqs.append(np.arange(lo, hi, dtype=np.int32))
        ks.append(np.full(hi - lo, -1, dtype=np.int32))
        bidir.append(np.zeros(hi - lo, dtype=bool))

    prev = 0
    for req in sorted(r for r, rs in ranges.items() if rs and r < num_reqs):
        q0 = int(query_start_loc[req])
        seq_len = int(seq_lens[req])
        context_len = seq_len - (int(query_start_loc[req + 1]) - q0)
        cursor = context_len
        rows: list[tuple[int, int, bool]] = []
        for first, last in sorted(ranges[req]):
            start, end = first, last + 1
            # Only a scheduled query that sees a later key of its range needs a
            # bidirectional row. A range's last token attends causally, so a
            # one-token range or a computed range up to its last token needs
            # none.
            if max(start, cursor) >= end - 1:
                continue
            if start >= seq_len:
                break  # scheduled in a later step
            if end > seq_len:
                return None
            if start > cursor:
                rows.append((cursor, start, False))
            rows.append((max(start, cursor), end, True))
            cursor = end
        if not rows:
            continue  # stays a plain row
        if cursor < seq_len:
            rows.append((cursor, seq_len, False))
        plain(prev, req)
        prev = req + 1
        arr = np.asarray(rows, dtype=np.int64)
        starts.append((q0 + arr[:, 0] - context_len).astype(np.int32))
        lens.append((arr[:, 1] - arr[:, 0]).astype(np.int32))
        reqs.append(np.full(len(rows), req, dtype=np.int32))
        ks.append(arr[:, 1].astype(np.int32))
        bidir.append(arr[:, 2].astype(bool))
    if not bidir:
        return None
    plain(prev, num_reqs)
    return PrefixLMRowsCPU(
        row_starts=np.concatenate(starts),
        row_lens=np.concatenate(lens),
        row_req=np.concatenate(reqs),
        row_seqused_k=np.concatenate(ks),
        row_bidirectional=np.concatenate(bidir),
    )


class PrefixLMRowsBuilder:
    """Builds `PrefixLMRows` once per step for one attention group."""

    def __init__(self, device: torch.device):
        self.device = device

    def build(
        self,
        ranges: Mapping[int, Sequence[tuple[int, int]]],
        query_start_loc_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        max_seq_len: int,
    ) -> PrefixLMRows | None:
        """Device rows for this step, or None if it needs none.

        `ranges` are inclusive, as in `compute_prefix_lm_rows`.
        `seq_lens_cpu` must be exact for requests that prefill a range (an
        upper bound is fine elsewhere); `seq_lens` is the exact device copy,
        which the other requests' rows use.
        """
        rows = compute_prefix_lm_rows(
            ranges, query_start_loc_cpu.numpy(), seq_lens_cpu.numpy()
        )
        if rows is None:
            return None
        return self._device_rows(
            rows, query_start_loc_cpu, seq_lens, block_table, max_seq_len
        )

    def _device_rows(
        self,
        rows: PrefixLMRowsCPU,
        query_start_loc_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        max_seq_len: int,
    ) -> PrefixLMRows:
        """Copy the host rows to the device."""
        n = len(rows.row_lens)
        # A fresh pinned tensor per step: the host allocator does not reuse it
        # before its copy has run, which a buffer kept across steps would need.
        staging = torch.empty(4 * n + 1, dtype=torch.int32, pin_memory=PIN_MEMORY)
        host = staging.numpy()
        host[:n] = rows.row_starts
        host[n] = int(query_start_loc_cpu[-1])
        host[n + 1 : 2 * n + 1] = rows.row_req
        host[2 * n + 1 : 3 * n + 1] = rows.row_seqused_k
        host[3 * n + 1 : 4 * n + 1] = ~rows.row_bidirectional
        dev = staging.to(self.device, non_blocking=True)
        row_req = dev[n + 1 : 2 * n + 1]
        k_exact = dev[2 * n + 1 : 3 * n + 1]
        has_plain = bool((rows.row_seqused_k < 0).any())
        max_seqlen_k = max(
            int(rows.row_seqused_k.max()), max_seq_len if has_plain else 0
        )
        return PrefixLMRows(
            num_rows=n,
            cu_seqlens_q=dev[: n + 1],
            seqused_k=torch.where(
                k_exact >= 0, k_exact, seq_lens.index_select(0, row_req)
            ).to(torch.int32),
            block_table=block_table.index_select(0, row_req),
            row_req=row_req,
            row_causal=dev[3 * n + 1 : 4 * n + 1],
            max_seqlen_q=int(rows.row_lens.max()),
            max_seqlen_k=max_seqlen_k,
        )
