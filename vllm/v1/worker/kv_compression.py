# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KeyDiff KV cache compression (opt-in, prototype).

Two strategies, selected via ``CacheConfig.kv_compression_algorithm``:

- ``"full_replacement"``: retroactive compaction (kvpress
  CompressionRatioDecodingPress semantics) covering all phases: it runs
  every ``compression_interval`` logical tokens — during chunked prefill
  (block-wise iterative compression, as in the original KeyDiff paper)
  and during decode — plus unconditionally when prefill completes. Each
  compaction keeps the top ``int(logical_total * (1 - ratio))`` tokens.
  Pipeline per layer: gather -> KeyDiff score -> per-head top-k select ->
  scatter the survivors back into the leading slots of the request's
  cache. Every KV head keeps the same *count* of tokens but potentially a
  different *set*, so the single per-request sequence length stays valid.

- ``"filtering"``: retroactive compaction during the prefill phase (the
  vLLM equivalent of kvpress PrefillDecodingPress wrapping a retroactive
  KeyDiffPress — interval-based for prefill chunks, unconditional at
  prefill completion), then online per-head keep/skip decisions during
  decode — a faithful port of kvpress FilteringPress (with PaddedTensor
  semantics). The new token is always written to the cache before the
  forward pass (so the current step attends to it); after the forward
  pass every
  (layer, head) independently decides whether to keep it. Each head's
  accepted tokens stay packed at the front of that head's cache columns
  (accepting heads copy the new token's slice into their first free
  column), so different heads hold different token sets in the same cache
  slots. The shared physical length is ``max over (layer, head) lengths``;
  a cache column is freed (``num_kv_discarded`` grows) only when no head
  extended past it — kvpress's "shrink when the trailing column is all
  padding" rule. Heads behind the max attend over a few stale trailing
  columns, equivalent to FilteringPress with ``fill_padding=False``
  (validated as quality-neutral in the Phase-1 NIAH experiments).

Both strategies rely on the same coordinate shift, tracked per request as
``num_kv_discarded``: logical token positions (used for RoPE and by the
scheduler to track computation progress) minus the discarded count give
physical cache positions (used for slot mapping, attention seq_lens and
block allocation).

Scoring is KeyDiff (https://arxiv.org/abs/2504.15364): the score of a token
is the negative cosine similarity between its key and the per-head anchor
(the mean of the L2-normalized keys). Tokens whose keys are most similar to
the anchor are considered redundant.

Only supported with the FLASH_ATTN-style paged KV cache layout
``[2, num_blocks, block_size, num_kv_heads, head_size]`` and a single
full-attention KV cache group.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)


def keydiff_scores(keys: torch.Tensor) -> torch.Tensor:
    """Compute KeyDiff scores for a dense key tensor.

    Args:
        keys: [num_tokens, num_kv_heads, head_size]

    Returns:
        scores: [num_kv_heads, num_tokens]. Higher means more distinctive
        (more worth keeping).
    """
    anchor = F.normalize(keys, p=2, dim=-1).mean(dim=0, keepdim=True)
    scores = -F.cosine_similarity(keys, anchor, dim=-1)
    return scores.transpose(0, 1)


def gather_slots(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Gather entries from a paged cache [num_blocks, block_size, H, D]."""
    block_size = cache.shape[1]
    return cache[slots // block_size, slots % block_size]


def scatter_slots(
    cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor
) -> None:
    """Write dense entries into a paged cache [num_blocks, block_size, H, D]."""
    block_size = cache.shape[1]
    cache[slots // block_size, slots % block_size] = values


def _slots_for_positions(
    block_row: np.ndarray,
    block_size: int,
    num_positions: int,
    device: torch.device,
) -> torch.Tensor:
    """Physical slot ids for cache positions [0, num_positions)."""
    positions = np.arange(num_positions, dtype=np.int64)
    block_ids = block_row[positions // block_size].astype(np.int64)
    slots_np = block_ids * block_size + positions % block_size
    return torch.from_numpy(slots_np).to(device)


def compact_request_kv(
    kv_caches: list[torch.Tensor],
    block_row: np.ndarray,
    block_size: int,
    num_cached_tokens: int,
    n_kept: int,
) -> int:
    """Retroactive compaction of one request's cached KV entries.

    For each layer: gather all cached keys/values, score them with KeyDiff,
    keep the top ``n_kept`` per head, and scatter the survivors back into
    cache positions [0, n_kept). Slots beyond n_kept become stale and are
    overwritten by subsequently written tokens.

    ``n_kept`` is computed by the caller — mirroring kvpress it is
    ``max(1, int(logical_total_tokens * (1 - ratio)))``, i.e. a fraction of
    all tokens seen so far including previously discarded ones
    (CompressionRatioDecodingPress._resolve_target_size). The cache may
    hold composite per-head entries from earlier compactions; re-compaction
    is valid because scoring and selection are fully per-head.

    Returns the number of kept tokens.
    """
    n_kept = max(1, n_kept)
    if n_kept >= num_cached_tokens:
        return num_cached_tokens

    device = kv_caches[0].device
    src_slots = _slots_for_positions(block_row, block_size, num_cached_tokens, device)
    dst_slots = src_slots[:n_kept]

    for kv_cache in kv_caches:
        key_cache, value_cache = kv_cache.unbind(0)
        keys = gather_slots(key_cache, src_slots)  # [T, H, D]
        head_size = keys.shape[-1]
        scores = keydiff_scores(keys)  # [H, T]
        # Sort kept indices so surviving tokens stay in temporal order.
        kept_idx = scores.topk(n_kept, dim=-1).indices.sort(dim=-1).values  # [H, n]
        kept_idx = kept_idx.unsqueeze(-1).expand(-1, -1, head_size)  # [H, n, D]

        kept_keys = keys.transpose(0, 1).gather(1, kept_idx).transpose(0, 1)
        values = gather_slots(value_cache, src_slots)
        kept_values = values.transpose(0, 1).gather(1, kept_idx).transpose(0, 1)

        scatter_slots(key_cache, dst_slots, kept_keys.contiguous())
        scatter_slots(value_cache, dst_slots, kept_values.contiguous())

    return n_kept


def masked_keydiff_scores(keys: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """KeyDiff scores over per-head valid positions only.

    Equivalent to running :func:`keydiff_scores` per head on just that
    head's valid keys (the per-head anchor is the mean of the normalized
    *valid* keys), which is what kvpress FilteringPress does. Invalid
    positions get ``-inf``.

    Args:
        keys: [num_tokens, num_kv_heads, head_size]
        valid_mask: [num_kv_heads, num_tokens] boolean

    Returns:
        scores: [num_kv_heads, num_tokens], ``-inf`` at invalid positions.
    """
    normed = F.normalize(keys, p=2, dim=-1).transpose(0, 1)  # [H, T, D]
    mask = valid_mask.unsqueeze(-1)
    anchor = (normed * mask).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(
        min=1
    )  # [H, D]
    scores = -F.cosine_similarity(
        keys.transpose(0, 1), anchor.unsqueeze(1), dim=-1
    )  # [H, T]
    return scores.masked_fill(~valid_mask, float("-inf"))


def filtering_step(
    kv_caches: list[torch.Tensor],
    block_row: np.ndarray,
    block_size: int,
    lengths: torch.Tensor,
    num_cached_tokens: int,
    logical_total_tokens: int,
    compression_ratio: float,
) -> int:
    """Per-head online filtering of the newest decoded token.

    Faithful port of kvpress ``FilteringPress`` + ``PaddedTensor`` onto the
    paged cache. The newest token occupies the shared cache column
    ``num_cached_tokens - 1`` (written before the forward pass, so this
    step's attention saw it). Per (layer, head), independently:

    1. Score that head's valid tokens (its packed prefix ``[0, L)`` plus
       the new token) with KeyDiff.
    2. ``n_kept = round(logical_total_tokens * (1 - ratio))`` clamped to
       [1, num_cached_tokens]; threshold = n_kept-th highest score
       (``-inf`` if the head has fewer valid tokens than n_kept, i.e.
       always accept). ``logical_total_tokens`` counts every token seen so
       far, including previously skipped ones — this is what makes the
       realized compression ratio converge to the target (see
       05a_filtering_bias_analysis.md: cached tokens are self-selected
       high scorers, so the threshold must be calibrated as if skipped
       tokens were still competing).
    3. If accepted: copy the new token's K/V slice for this head into the
       head's first free column ``L`` (kvpress ``accept_last``) and
       increment ``L``. If rejected: nothing — the head's view never
       includes this token.

    The shared physical length is ``max(lengths)`` (kvpress ``shrink``
    keeps the buffer at the max valid length). The trailing column is
    freed — i.e. the caller must grow ``num_kv_discarded`` — only when no
    head extended past the previous max.

    Args:
        lengths: [num_layers, num_kv_heads] per-head valid lengths on the
            cache device. Mutated in place.
        num_cached_tokens: shared cache columns including the new token
            (= previous max length + 1).
        logical_total_tokens: tokens seen so far including skipped ones.

    Returns the new shared physical length (= ``lengths.max()``), in
    [num_cached_tokens - 1, num_cached_tokens].
    """
    device = kv_caches[0].device
    num_cols = num_cached_tokens
    slots = _slots_for_positions(block_row, block_size, num_cols, device)
    block_indices = slots // block_size
    block_offsets = slots % block_size

    n_kept = int(round(logical_total_tokens * (1.0 - compression_ratio)))
    n_kept = min(max(n_kept, 1), num_cols)

    cols = torch.arange(num_cols, device=device)

    for layer_idx, kv_cache in enumerate(kv_caches):
        key_cache, value_cache = kv_cache.unbind(0)
        layer_lengths = lengths[layer_idx]  # [H]

        keys = key_cache[block_indices, block_offsets]  # [T, H, D]
        # Valid = the head's packed prefix plus the new token (last column).
        valid = cols.unsqueeze(0) < layer_lengths.unsqueeze(1)  # [H, T]
        valid[:, -1] = True

        scores = masked_keydiff_scores(keys, valid)  # [H, T]
        # Threshold: n_kept-th highest score per head (-inf padding ranks
        # last, so heads with fewer than n_kept valid tokens always accept).
        threshold = scores.topk(n_kept, dim=-1).values[:, -1]  # [H]
        accepts = scores[:, -1] >= threshold  # [H]

        # kvpress PaddedTensor.accept_last: for accepting heads whose valid
        # prefix ends before the new token's column, copy the new token's
        # per-head K/V slice into the head's first free column. To stay
        # sync-free, rejected heads (and accepting heads already at the
        # last column) perform a harmless self-copy of the last column.
        head_indices = torch.arange(layer_lengths.shape[0], device=device)
        dst_cols = torch.where(accepts, layer_lengths, num_cols - 1)
        dst = slots[dst_cols]
        src_blk, src_off = block_indices[-1], block_offsets[-1]
        key_cache[dst // block_size, dst % block_size, head_indices] = key_cache[
            src_blk, src_off, head_indices
        ]
        value_cache[dst // block_size, dst % block_size, head_indices] = value_cache[
            src_blk, src_off, head_indices
        ]

        lengths[layer_idx] = layer_lengths + accepts.long()

    return int(lengths.max().item())


class KVCompressionManager:
    """Drives KV compression from the GPU model runner.

    The runner calls :meth:`run_post_forward` after each model forward pass.
    Decisions update ``CachedRequestState.num_kv_discarded`` (the runner's
    authoritative copy, used by ``_prepare_inputs`` to shift cache
    coordinates) and are returned so they can be reported to the scheduler
    via ``ModelRunnerOutput.kv_compression_discarded`` (used only for block
    allocation accounting).
    """

    def __init__(
        self,
        algorithm: str,
        compression_ratio: float,
        compression_interval: int = 512,
    ):
        assert algorithm in ("full_replacement", "filtering")
        assert 0.0 < compression_ratio < 1.0
        assert compression_interval >= 1
        self.algorithm = algorithm
        self.compression_ratio = compression_ratio
        self.compression_interval = compression_interval
        self.kv_caches: list[torch.Tensor] = []

    def bind_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        shared_kv_cache_layers: dict[str, str],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Collect the unique per-layer paged KV cache tensors."""
        if len(kv_cache_config.kv_cache_groups) != 1 or not isinstance(
            kv_cache_config.kv_cache_groups[0].kv_cache_spec, FullAttentionSpec
        ):
            raise ValueError(
                "KV compression requires a single full-attention KV cache "
                "group (no sliding window / hybrid / Mamba models)."
            )
        seen: set[int] = set()
        unique: list[torch.Tensor] = []
        for layer_name, kv_cache in kv_caches.items():
            if layer_name in shared_kv_cache_layers:
                # Reuses another layer's cache; skip to avoid double work.
                continue
            if id(kv_cache) in seen:
                continue
            if kv_cache.dim() != 5 or kv_cache.shape[0] != 2:
                raise ValueError(
                    "KV compression requires the FLASH_ATTN paged KV cache "
                    "layout [2, num_blocks, block_size, num_kv_heads, "
                    f"head_size]; got shape {tuple(kv_cache.shape)} for "
                    f"layer {layer_name}. Use the FLASH_ATTN attention "
                    "backend."
                )
            seen.add(id(kv_cache))
            unique.append(kv_cache)
        self.kv_caches = unique
        logger.info(
            "KV compression enabled: algorithm=%s, ratio=%.2f, %d layers",
            self.algorithm,
            self.compression_ratio,
            len(unique),
        )

    def run_post_forward(
        self,
        input_batch: "InputBatch",
        requests: dict[str, "CachedRequestState"],
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, int]:
        """Apply compression decisions after a forward pass.

        Returns {req_id: num newly discarded KV entries} for this step.
        """
        if not self.kv_caches:
            return {}

        block_table = input_batch.block_table[0]
        block_table_np = block_table.block_table.np
        block_size = block_table.block_size

        discarded: dict[str, int] = {}
        for req_index, req_id in enumerate(input_batch.req_ids):
            req_state = requests[req_id]
            num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
            num_computed_before = int(input_batch.num_computed_tokens_cpu[req_index])
            prompt_len = req_state.num_prompt_tokens
            block_row = block_table_np[req_index]

            logical_total = num_computed_before + num_scheduled
            num_cached = logical_total - req_state.num_kv_discarded
            prefill_completes = num_computed_before < prompt_len <= logical_total
            in_prefill = logical_total <= prompt_len
            compaction_due = (
                logical_total - req_state.kv_last_compaction_total
                >= self.compression_interval
            )

            num_discarded = 0
            if self.algorithm == "full_replacement":
                # Retroactive compaction (kvpress
                # CompressionRatioDecodingPress semantics, extended to
                # prefill chunks — block-wise iterative compression as in
                # the KeyDiff paper): triggered every
                # `compression_interval` logical tokens at any phase, and
                # unconditionally when prefill completes. The forward pass
                # of this step already attended to the pre-compaction
                # cache, so compaction only affects subsequent steps.
                if compaction_due or prefill_completes:
                    num_discarded = self._compact(
                        req_state, block_row, block_size, num_cached, logical_total
                    )
            elif not req_state.kv_compressed:
                # Filtering, prefill phase: retroactive compaction of
                # prompt chunks (interval-based) plus an unconditional
                # compaction when prefill completes — the vLLM equivalent
                # of kvpress PrefillDecodingPress wrapping a retroactive
                # KeyDiffPress for prefill.
                if prefill_completes or (in_prefill and compaction_due):
                    num_discarded = self._compact(
                        req_state, block_row, block_size, num_cached, logical_total
                    )
                if prefill_completes:
                    # Hand off to per-head online filtering for decode.
                    req_state.kv_compressed = True
            elif num_scheduled == 1 and num_computed_before >= prompt_len:
                # Filtering, decode phase: per-head keep/skip for the new
                # token, every step (kvpress FilteringPress).
                lengths = req_state.kv_filter_lengths
                if lengths is None:
                    # First filtered token: the cache is fully packed (the
                    # prefill compaction left it packed at the compacted
                    # length), so every (layer, head) starts at the
                    # pre-step length.
                    lengths = torch.full(
                        (len(self.kv_caches), self.kv_caches[0].shape[3]),
                        num_cached - 1,
                        dtype=torch.long,
                        device=self.kv_caches[0].device,
                    )
                    req_state.kv_filter_lengths = lengths

                new_shared_len = filtering_step(
                    self.kv_caches,
                    block_row,
                    block_size,
                    lengths,
                    num_cached,
                    logical_total,
                    self.compression_ratio,
                )
                # The trailing column is freed iff no head extended past the
                # previous shared length (kvpress: shrink when the last
                # column is all padding).
                num_discarded = num_cached - new_shared_len
                assert 0 <= num_discarded <= 1

            if num_discarded > 0:
                req_state.num_kv_discarded += num_discarded
                discarded[req_id] = num_discarded

        return discarded

    def _compact(
        self,
        req_state: "CachedRequestState",
        block_row: np.ndarray,
        block_size: int,
        num_cached: int,
        logical_total: int,
    ) -> int:
        """Run retroactive compaction; returns newly discarded entries."""
        # kvpress CompressionRatioDecodingPress: the target is a fraction
        # of all tokens seen so far (including previously discarded ones).
        n_kept_target = max(1, int(logical_total * (1.0 - self.compression_ratio)))
        n_kept = compact_request_kv(
            self.kv_caches,
            block_row,
            block_size,
            num_cached,
            n_kept_target,
        )
        req_state.kv_last_compaction_total = logical_total
        return num_cached - n_kept
