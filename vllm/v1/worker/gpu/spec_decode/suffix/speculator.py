# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU suffix decoding speculator for the V2 model runner.

Extends the n-gram speculator with the two parts of SuffixDecoding
(https://arxiv.org/abs/2411.04975) that carry most of its acceptance gain:

* Variable-length matching: drafts continue the occurrence whose preceding
  tokens match the longest suffix of the request's history (up to
  ``suffix_decoding_max_tree_depth``), instead of a fixed n-gram window.
* Cross-request memory: responses of finished requests are appended to a
  device-resident token corpus. A hash index keyed on the last 2 and last 4
  tokens gives candidate positions, and each candidate is extended to the
  longest suffix match, so the index is only an entry point for suffix
  matching.

The propose path never synchronizes with the host. Ingestion reads token
history and lengths directly from ``RequestState`` and processes finished
requests in sorted id order; index updates are order-independent, so every
TP rank holds an identical corpus without any collective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.states import RequestState

# Odd 64-bit multiplier (golden ratio), as a signed int64.
_HASH_MUL = 0x9E3779B97F4A7C15 - (1 << 64)
# Corpus slots kept per hash bucket: the newest occurrence in each of the
# last _BUCKET_SLOTS documents that contain the bucket's key tokens. Entries
# pack (document sequence number << 32 | corpus position).
_BUCKET_SLOTS = 8
# Prompt tokens stored ahead of each response (longest hash key length - 1).
_PROMPT_CONTEXT = 3
# Adaptive verification: acceptance is tracked per (draft source, match-length
# bucket, position). Match lengths 1-6 get their own bucket, then 7-10, 11-20
# and 21+. Counts are halved every _STATS_HALF_LIFE steps so the estimate
# follows the workload.
_NUM_LEN_BUCKETS = tl.constexpr(9)
_STATS_HALF_LIFE = 1024


@triton.jit
def _len_bucket(L):
    return tl.where(L <= 6, L - 1, tl.where(L <= 10, 6, tl.where(L <= 20, 7, 8)))


@triton.jit
def _bucket(hv, NUM_BUCKETS: tl.constexpr):
    return ((hv >> 29) ^ hv) & (NUM_BUCKETS - 1)


@triton.jit
def _suffix_local_scan_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    idx_mapping_ptr,  # *int32  [B]  batch_idx -> req_state_idx
    total_len_ptr,  # *int32  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    scratch_ptr,  # *int64  [B, scratch_stride]  (output)
    scratch_stride,
    Lp1,  # int64 packing radix (> any position)
    MAX_DEPTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per (request, block): best (match_len, end_pos) packed into one int64.

    end_pos is the exclusive end of an earlier occurrence of the request's
    suffix in its own history; the draft continues from end_pos. Longer
    matches win, then later positions.
    """
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    seq_len = tl.load(total_len_ptr + req).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b)
    row = token_ids_ptr + req * token_ids_stride

    scratch_off = b * scratch_stride + blk
    # Blocks past the last candidate (an occurrence must leave a token).
    if blk * BLOCK >= seq_len:
        tl.store(scratch_ptr + scratch_off, tl.zeros((), tl.int64))
        return

    pos = blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    active = (pos >= 1) & (pos < seq_len) & (num_sampled > 0)
    match_len = tl.zeros([BLOCK], dtype=tl.int64)
    for t in range(MAX_DEPTH):
        # Request token t back from its end (newest first).
        tail_pos = seq_len - 1 - t
        tail_tok = tl.load(row + tail_pos, mask=tail_pos >= 0, other=-2)
        src_pos = pos - 1 - t
        ok = active & (src_pos >= 0) & (tail_pos >= 0)
        tok = tl.load(row + src_pos, mask=ok, other=-1)
        active = ok & (tok == tail_tok)
        match_len += active.to(tl.int64)

    score = tl.where(match_len > 0, match_len * Lp1 + pos, 0)
    tl.store(scratch_ptr + scratch_off, tl.max(score, axis=0))


@triton.jit
def _corpus_draft(
    row,
    seq_len,
    table_ptr,
    corpus_ptr,
    corpus_len,
    HASH_MUL: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    SLOTS: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    DEPTH_PO2: tl.constexpr,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
):
    """(match_len, draft, num_tokens) from the corpus for one request.

    Candidates are the slots of the request's 4-token and 2-token buckets.
    Each is verified and extended against the corpus, so hash collisions and
    entries for overwritten documents only cost a slot. The draft is then
    chosen by frequency, as in SuffixDecoding: for each candidate match
    length L, the candidates matching at least L tokens vote token by token
    (prefix-consistent; ties go to the newest candidate), and the chain with
    the highest sum of per-token probabilities wins (ties: longer L).
    """
    C: tl.constexpr = 2 * SLOTS
    c_iota = tl.arange(0, C).to(tl.int64)
    d_iota = tl.arange(0, DEPTH_PO2).to(tl.int64)
    k_iota = tl.arange(0, K_PO2).to(tl.int64)

    tail_pos = seq_len - 1 - d_iota
    tail = tl.load(
        row + tail_pos, mask=(d_iota < MAX_DEPTH) & (tail_pos >= 0), other=-2
    )
    # Hash the trailing tokens newest-first, as ingestion does.
    hv = tl.zeros((), tl.int64)
    for d in tl.static_range(4):
        tok = tl.load(row + seq_len - 1 - d, mask=seq_len - 1 - d >= 0, other=0)
        hv = (hv ^ tok.to(tl.int64)) * HASH_MUL
        if d == 1:
            bucket2 = _bucket(hv, NUM_BUCKETS)
    bucket4 = _bucket(hv, NUM_BUCKETS)
    is4 = c_iota < SLOTS
    key_len = tl.where(is4, 4, 2)
    slot = tl.where(
        is4, bucket4 * SLOTS + c_iota, (NUM_BUCKETS + bucket2) * SLOTS + c_iota - SLOTS
    )
    vals = tl.load(table_ptr + slot)
    cpos = vals & 0xFFFFFFFF

    src_pos = cpos[:, None] - 1 - d_iota[None, :]
    src = tl.load(
        corpus_ptr + src_pos,
        mask=(vals[:, None] >= 0) & (src_pos >= 0) & (d_iota[None, :] < MAX_DEPTH),
        other=-1,
    )
    miss = (src != tail[None, :]).to(tl.int32)
    c_len = tl.sum((tl.cumsum(miss, axis=1) == 0).to(tl.int64), axis=1)
    nxt = tl.load(corpus_ptr + cpos, mask=(vals >= 0) & (cpos < corpus_len), other=-1)
    ok = (vals >= 0) & (nxt >= 0) & (c_len >= key_len) & (seq_len >= key_len)
    # Both buckets often hold the same position; count it once.
    dup = (
        (cpos[:, None] == cpos[None, :])
        & ok[:, None]
        & (c_iota[:, None] < c_iota[None, :])
    )
    ok = ok & (tl.max(dup.to(tl.int32), axis=0) == 0)

    # Continuations, cut at the first document separator.
    cont_pos = cpos[:, None] + k_iota[None, :]
    cont = tl.load(
        corpus_ptr + cont_pos,
        mask=ok[:, None] & (k_iota[None, :] < K) & (cont_pos < corpus_len),
        other=-1,
    ).to(tl.int64)
    cont = tl.where(tl.cumsum((cont < 0).to(tl.int32), axis=1) == 0, cont, -1)

    # Candidates sharing a match length draft identical chains; score each
    # distinct length once.
    same_len = (
        (c_len[:, None] == c_len[None, :])
        & ok[:, None]
        & (c_iota[:, None] < c_iota[None, :])
    )
    first = ok & (tl.max(same_len.to(tl.int32), axis=0) == 0)

    best_score = tl.full((), -1.0, tl.float32)
    best_len = tl.zeros((), tl.int64)
    best_n = tl.zeros((), tl.int64)
    best_chain = tl.zeros([K_PO2], tl.int64)
    for i in range(C):
        sel = c_iota == i
        if tl.sum(tl.where(sel, first.to(tl.int32), 0), axis=0) > 0:
            L = tl.sum(tl.where(sel, c_len, 0), axis=0)
            alive = ok & (c_len >= L)
            go = L > 0
            cum = tl.full((), 1.0, tl.float32)
            score = tl.zeros((), tl.float32)
            n_emit = tl.zeros((), tl.int64)
            chain = tl.zeros([K_PO2], tl.int64)
            for d in tl.static_range(K):
                col = tl.sum(tl.where(k_iota[None, :] == d, cont, 0), axis=1)
                has = alive & (col >= 0)
                cnt = tl.sum(
                    ((col[:, None] == col[None, :]) & has[None, :]).to(tl.int32), axis=1
                )
                cnt = tl.where(has, cnt, 0)
                cmax = tl.max(cnt, axis=0)
                # Most frequent next token; ties go to the newest candidate.
                newest = tl.max(tl.where(has & (cnt == cmax), vals, -1), axis=0)
                next_tok = tl.sum(tl.where(has & (vals == newest), col, 0), axis=0)
                go = go & (cmax > 0)
                denom = tl.maximum(tl.sum(alive.to(tl.int32), axis=0), 1)
                cum = tl.where(
                    go, cum * cmax.to(tl.float32) / denom.to(tl.float32), cum
                )
                score = tl.where(go, score + cum, score)
                chain = tl.where((k_iota == d) & go, next_tok, chain)
                n_emit += go.to(tl.int64)
                alive = alive & has & (col == next_tok)
            take = (score > best_score) | ((score == best_score) & (best_len < L))
            best_score = tl.where(take, score, best_score)
            best_len = tl.where(take, L, best_len)
            best_n = tl.where(take, n_emit, best_n)
            best_chain = tl.where(take, chain, best_chain)
    return best_len, best_chain, best_n


@triton.jit
def _suffix_finalize_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    idx_mapping_ptr,  # *int32  [B]
    total_len_ptr,  # *int32  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    last_sampled_ptr,  # *int64  [max_num_reqs]
    scratch_ptr,  # *int64  [B, scratch_stride]
    scratch_stride,
    num_blocks,
    Lp1,
    corpus_ptr,  # *int32  [corpus_len]
    corpus_len,
    table_ptr,  # *int64  [2, NUM_BUCKETS, SLOTS]
    drafts_ptr,  # *int64  [B, K]  (output)
    num_valid_ptr,  # *int32  [B]  (output)
    stats_ptr,  # *int64  [2 * _NUM_LEN_BUCKETS, K, 2]  (accepted, observed)
    last_bucket_ptr,  # *int32  [max_num_reqs]  (output)
    last_valid_ptr,  # *int32  [max_num_reqs]  (output)
    conf_ptr,  # *fp32  [B, K]  (output)
    HAS_CORPUS: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    HASH_MUL: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    SLOTS: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    DEPTH_PO2: tl.constexpr,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
    NB_PO2: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    seq_len = tl.load(total_len_ptr + req).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b)
    last_tok = tl.load(last_sampled_ptr + req)
    row = token_ids_ptr + req * token_ids_stride

    # Best match in the request's own history.
    nb = tl.arange(0, NB_PO2).to(tl.int64)
    l_score = tl.max(
        tl.load(scratch_ptr + b * scratch_stride + nb, mask=nb < num_blocks, other=0),
        axis=0,
    )
    l_len = l_score // Lp1
    l_pos = l_score - l_len * Lp1

    k_iota = tl.arange(0, K_PO2).to(tl.int64)
    k_mask = k_iota < K
    idx = l_pos + k_iota
    draft_ok = k_mask & (idx < seq_len) & (l_len > 0)
    draft = tl.load(row + idx, mask=draft_ok, other=0).to(tl.int64)

    if HAS_CORPUS:
        g_len, g_draft, g_n = _corpus_draft(
            row,
            seq_len,
            table_ptr,
            corpus_ptr,
            corpus_len,
            HASH_MUL,
            NUM_BUCKETS,
            SLOTS,
            MAX_DEPTH,
            DEPTH_PO2,
            K,
            K_PO2,
        )
        # The longer match wins; ties go to the request's own history.
        use_corpus = g_len > l_len
        draft = tl.where(use_corpus, g_draft, draft)
        draft_ok = tl.where(use_corpus, k_iota < g_n, draft_ok)
        match_len = tl.where(use_corpus, g_len, l_len)
        source = use_corpus.to(tl.int64)
    else:
        match_len = l_len
        source = tl.zeros((), tl.int64)

    draft_ok = draft_ok & (num_sampled > 0)
    num_valid = tl.sum(draft_ok.to(tl.int32), axis=0)
    # Invalid slots fall back to the last sampled token; they are verified as
    # ordinary (rejectable) drafts, so the fill value only affects efficiency.
    out = tl.where(draft_ok, draft, last_tok)
    tl.store(drafts_ptr + b * K + k_iota, out, mask=k_mask)
    tl.store(num_valid_ptr + b, num_valid)

    if ADAPTIVE:
        # Per-position confidence for adaptive verification: the observed
        # acceptance rate for drafts like this one (Laplace-smoothed); filler
        # slots get 0 so they are trimmed first.
        bucket = source * _NUM_LEN_BUCKETS + _len_bucket(tl.maximum(match_len, 1))
        stat = stats_ptr + (bucket * K + k_iota) * 2
        accepted = tl.load(stat, mask=k_mask, other=0).to(tl.float32)
        observed = tl.load(stat + 1, mask=k_mask, other=0).to(tl.float32)
        conf = tl.where(draft_ok, (accepted + 1.0) / (observed + 2.0), 0.0)
        tl.store(conf_ptr + b * K + k_iota, conf, mask=k_mask)
        tl.store(
            last_bucket_ptr + req, tl.where(num_valid > 0, bucket, -1).to(tl.int32)
        )
        tl.store(last_valid_ptr + req, num_valid)


@triton.jit
def _suffix_observe_kernel(
    idx_mapping_ptr,  # *int32  [B]
    num_sampled_ptr,  # *int32  [B]
    num_rejected_ptr,  # *int32  [B]
    stats_ptr,  # *int64  [2 * _NUM_LEN_BUCKETS, K, 2]
    last_bucket_ptr,  # *int32  [max_num_reqs]
    last_valid_ptr,  # *int32  [max_num_reqs]
    K: tl.constexpr,
    K_PO2: tl.constexpr,
):
    """Fold the target's verdict on each request's last draft into the stats.

    Integer atomics are order-independent, so every TP rank keeps identical
    stats (and therefore identical confidences).
    """
    b = tl.program_id(0).to(tl.int64)
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    bucket = tl.load(last_bucket_ptr + req).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b).to(tl.int64)
    if (bucket < 0) | (num_sampled == 0):
        return
    num_accepted = num_sampled - 1
    num_verified = num_accepted + tl.load(num_rejected_ptr + b).to(tl.int64)
    num_valid = tl.load(last_valid_ptr + req).to(tl.int64)
    k_iota = tl.arange(0, K_PO2).to(tl.int64)
    # Accepted positions plus the first rejected one were observed.
    observed = (
        (k_iota < K)
        & (k_iota <= num_accepted)
        & (k_iota < num_verified)
        & (k_iota < num_valid)
    )
    stat = stats_ptr + (bucket * K + k_iota) * 2
    tl.atomic_add(stat, (k_iota < num_accepted).to(tl.int64), mask=observed)
    tl.atomic_add(stat + 1, tl.full([K_PO2], 1, tl.int64), mask=observed)
    tl.store(last_bucket_ptr + req, -1)


@triton.jit
def _suffix_ingest_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    docs_ptr,  # *int32  [F, 2]  (slot, start) of finished requests, sorted
    num_docs,
    total_len_ptr,  # *int32  [max_num_reqs]
    corpus_ptr,  # *int32  [corpus_len]
    corpus_len,
    head_ptr,  # *int64  [1]  next write offset
    doc_seq_ptr,  # *int64  [1]  documents ingested so far
    table_ptr,  # *int64  [2, NUM_BUCKETS, SLOTS]
    HASH_MUL: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Append finished responses to the corpus ring and index them.

    A document that does not fit before the end of the ring wraps to offset
    0, so no document straddles the boundary. Older documents are
    overwritten in FIFO order; their index entries become stale and are
    rejected by verification at lookup.
    """
    head = tl.load(head_ptr)
    doc_seq = tl.load(doc_seq_ptr)
    cap = tl.cast(corpus_len, tl.int64)
    offs = tl.arange(0, BLOCK).to(tl.int64)
    for f in range(num_docs):
        req = tl.load(docs_ptr + 2 * f).to(tl.int64)
        start = tl.load(docs_ptr + 2 * f + 1).to(tl.int64)
        end = tl.load(total_len_ptr + req).to(tl.int64)
        # Keep the newest cap - 1 tokens of an over-long response.
        start = tl.maximum(start, end - (cap - 1))
        n = tl.maximum(end - start, 0)
        head = tl.where(head + n + 1 > cap, 0, head)
        row = token_ids_ptr + req * token_ids_stride + start
        slot = doc_seq % SLOTS
        for off in range(0, n, BLOCK):
            j = off + offs
            m = j < n
            tok = tl.load(row + j, mask=m)
            tl.store(corpus_ptr + head + j, tok, mask=m)
            # Index each position j under its preceding 2 and 4 tokens, when
            # they lie in the document. Entries order by (document, position),
            # so atomic_max keeps the newest occurrence regardless of
            # execution order.
            val = (doc_seq << 32) | (head + j)
            hv = tl.zeros([BLOCK], tl.int64)
            for d in tl.static_range(4):
                t = tl.load(row + j - 1 - d, mask=m & (j - 1 - d >= 0), other=0)
                hv = (hv ^ t.to(tl.int64)) * HASH_MUL
                if d == 1:
                    tl.atomic_max(
                        table_ptr
                        + (NUM_BUCKETS + _bucket(hv, NUM_BUCKETS)) * SLOTS
                        + slot,
                        val,
                        mask=m & (j >= 2),
                    )
                if d == 3:
                    tl.atomic_max(
                        table_ptr + _bucket(hv, NUM_BUCKETS) * SLOTS + slot,
                        val,
                        mask=m & (j >= 4),
                    )
        tl.store(corpus_ptr + head + n, -1)
        head += n + 1
        doc_seq += 1
    tl.store(head_ptr, head)
    tl.store(doc_seq_ptr, doc_seq)


class SuffixSpeculator(BaseSpeculator):
    """V2-compatible GPU suffix decoding speculator."""

    supports_mm_inputs = False
    draft_logits = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        req_states: RequestState,
    ):
        if not HAS_TRITON:
            raise RuntimeError("suffix speculative decoding requires Triton.")
        spec = vllm_config.speculative_config
        assert spec is not None
        self.vllm_config = vllm_config
        self.device = device
        self.req_states = req_states
        self.num_speculative_steps: int = spec.num_speculative_tokens
        self.max_depth: int = spec.suffix_decoding_max_tree_depth
        self.max_num_reqs: int = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len: int = vllm_config.model_config.max_model_len

        L = self.max_model_len
        if L >= 1024:
            self.block_l = 256
        elif L >= 256:
            self.block_l = 128
        else:
            self.block_l = max(16, triton.next_power_of_2(max(L, 1)))
        self.n_blocks = triton.cdiv(L, self.block_l)
        self.scratch = torch.zeros(
            (self.max_num_reqs, self.n_blocks), dtype=torch.int64, device=device
        )

        # Corpus of finished responses separated by -1, plus two hash tables
        # keyed on the last 4 and last 2 tokens before each position. A
        # capacity of 0 disables it.
        self.corpus_len: int = (
            spec.suffix_decoding_corpus_tokens
            if spec.suffix_decoding_max_cached_requests != 0
            else 0
        )
        self.has_corpus = self.corpus_len > 0
        self.num_buckets = triton.next_power_of_2(max(16, self.corpus_len // 4))
        self.corpus = torch.full(
            (max(self.corpus_len, 1),), -1, dtype=torch.int32, device=device
        )
        self.table = torch.full(
            (2, self.num_buckets if self.has_corpus else 1, _BUCKET_SLOTS),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.corpus_head = torch.zeros(1, dtype=torch.int64, device=device)
        self.doc_seq = torch.zeros(1, dtype=torch.int64, device=device)

        # Batch-ordered draft output, scattered into RequestState.draft_tokens
        # by the model runner (same contract as the model-based speculators).
        self.drafts = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps),
            dtype=torch.int64,
            device=device,
        )
        # Number of real (non-filler) draft tokens per batch row.
        self.num_valid = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        # Adaptive verification (optional): per-slot confidences from
        # acceptance statistics learned online.
        self.enable_adaptive_verification = spec.enable_adaptive_verification
        self.draft_token_confidence_probs = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self.accept_stats = torch.zeros(
            (2 * _NUM_LEN_BUCKETS, self.num_speculative_steps, 2),
            dtype=torch.int64,
            device=device,
        )
        self.last_bucket = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=device
        )
        self.last_valid = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.num_steps = 0

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode

    def capture(self) -> None:
        return None

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Add finished responses to the corpus before their slots are freed."""
        if not self.has_corpus:
            return
        req_states = self.req_states
        # Sorted so every TP rank ingests documents in the same order. Starts
        # come from the host copy of prompt_len: the device-visible copy is
        # republished every step and may be reused for a new request before
        # this kernel runs.
        docs = []
        for req_id in sorted(finished_req_ids):
            slot = req_states.req_id_to_index.get(req_id)
            if slot is None:
                continue
            # Keep the prompt's last few tokens (typically the chat template's
            # assistant header) so the first response tokens are indexed by
            # the tokens that precede them.
            start = max(int(req_states.prompt_len.np[slot]) - _PROMPT_CONTEXT, 0)
            docs += [slot, start]
        if not docs:
            return
        docs_gpu = async_tensor_h2d(docs, self.device, torch.int32)
        token_ids = self.req_states.all_token_ids.gpu
        _suffix_ingest_kernel[(1,)](
            token_ids,
            token_ids.stride(0),
            docs_gpu,
            len(docs) // 2,
            req_states.total_len.gpu,
            self.corpus,
            self.corpus_len,
            self.corpus_head,
            self.doc_seq,
            self.table,
            HASH_MUL=_HASH_MUL,
            NUM_BUCKETS=self.num_buckets,
            SLOTS=_BUCKET_SLOTS,
            BLOCK=1024,
            num_warps=4,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: Any,
        slot_mappings: Any,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        if dummy_run:
            return self.drafts[:num_reqs]

        token_ids = self.req_states.all_token_ids.gpu
        total_len = self.req_states.total_len.gpu
        idx_mapping = input_batch.idx_mapping
        k_po2 = max(1, triton.next_power_of_2(self.num_speculative_steps))

        if self.enable_adaptive_verification:
            _suffix_observe_kernel[(num_reqs,)](
                idx_mapping,
                num_sampled,
                num_rejected,
                self.accept_stats,
                self.last_bucket,
                self.last_valid,
                K=self.num_speculative_steps,
                K_PO2=k_po2,
                num_warps=1,
            )
            self.num_steps += 1
            if self.num_steps % _STATS_HALF_LIFE == 0:
                self.accept_stats.bitwise_right_shift_(1)

        _suffix_local_scan_kernel[(num_reqs, self.n_blocks)](
            token_ids,
            token_ids.stride(0),
            idx_mapping,
            total_len,
            num_sampled,
            self.scratch,
            self.scratch.stride(0),
            self.max_model_len + 1,
            MAX_DEPTH=self.max_depth,
            BLOCK=self.block_l,
            num_warps=4,
        )
        _suffix_finalize_kernel[(num_reqs,)](
            token_ids,
            token_ids.stride(0),
            idx_mapping,
            total_len,
            num_sampled,
            last_sampled.view(-1),
            self.scratch,
            self.scratch.stride(0),
            self.n_blocks,
            self.max_model_len + 1,
            self.corpus,
            self.corpus.shape[0],
            self.table,
            self.drafts,
            self.num_valid,
            self.accept_stats,
            self.last_bucket,
            self.last_valid,
            self.draft_token_confidence_probs,
            HAS_CORPUS=self.has_corpus,
            ADAPTIVE=self.enable_adaptive_verification,
            HASH_MUL=_HASH_MUL,
            NUM_BUCKETS=self.num_buckets,
            SLOTS=_BUCKET_SLOTS,
            MAX_DEPTH=self.max_depth,
            DEPTH_PO2=triton.next_power_of_2(self.max_depth),
            K=self.num_speculative_steps,
            K_PO2=k_po2,
            NB_PO2=max(1, triton.next_power_of_2(self.n_blocks)),
            num_warps=4,
        )
        return self.drafts[:num_reqs]
