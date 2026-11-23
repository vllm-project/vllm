# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side shadow grammar manager for grammar-aware MRV2 draft proposals.

Maintains per-request xgrammar matchers synchronized with verified target
tokens so autoregressive drafters (EAGLE, EAGLE-3, MTP, standalone draft
models) apply the exact grammar bitmask *before* writing `DraftLogitsCache`
and sampling.

Key invariants:
1. Masking before `DraftLogitsCache` preserves lossless rejection sampling
   ($P(\text{output}=x) = p_M(x)$) for both greedy and probabilistic drafting.
2. A two-pass Triton bitmask kernel guards cropped-vocab drafters (EAGLE-3
   `d2t`): if the intersection of allowed grammar tokens and finite draft
   logits is empty, the row is left unmasked to avoid all-`-inf` `NaN`s, and
   the target verifier losslessly rejects the draft and resamples from $p_M$.
3. Deterministic fast-path: when a bitmask row allows a single token inside
   the draft vocabulary, the CPU advances the shadow matcher without waiting
   on a GPU-to-CPU sync of the sampled draft token.
4. At the end of `propose()`, all speculative draft advancements are rolled
   back so every matcher rests at the exact committed target state.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
from vllm.v1.structured_output.utils import allocate_token_bitmask

if TYPE_CHECKING:
    import xgrammar as xgr

    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import DraftGrammarInit
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)

_DRAFT_GRAMMAR_BLOCK_SIZE = 8192
_COMPILED_GRAMMAR_CACHE_LIMIT = 512
# When >= 25% of packed 32-bit bitmask words are all-ones (e.g. inside an open
# free-text JSON string value where ~96%+ of tokens are allowed vs ~0% on
# structural keys/enums/numbers), draft masking rarely improves acceptance
# enough to justify breaking FULL CUDA graphs / syncing draft tokens.
_OPEN_VOCAB_WORD_RATIO_NUM = 1
_OPEN_VOCAB_WORD_RATIO_DEN = 4


@triton.jit
def _check_draft_grammar_intersection_kernel(
    logits_ptr,
    logits_stride,
    req_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    has_valid_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Set `has_valid[i] = 1` iff row `req_indices[i]` has at least one finite
    logit whose grammar bitmask bit is 1."""
    mask_idx = tl.program_id(0)
    req_idx = tl.load(req_indices_ptr + mask_idx)
    block_id = tl.program_id(1)

    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + mask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
        other=0,
    )
    allowed = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) != 0
    allowed = allowed.reshape(BLOCK_SIZE)

    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = block_offset < vocab_size
    vals = tl.load(
        logits_ptr + req_idx * logits_stride + block_offset,
        mask=in_bounds,
        other=-float("inf"),
    ).to(tl.float32)
    is_finite = (vals > -float("inf")) & (vals < float("inf"))
    has_any = tl.sum((in_bounds & allowed & is_finite).to(tl.int32)) > 0
    if has_any:
        tl.atomic_or(has_valid_ptr + mask_idx, 1)


@triton.jit
def _apply_draft_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    req_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    has_valid_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Mask disallowed logits to `-inf` iff `has_valid[i] != 0`."""
    mask_idx = tl.program_id(0)
    if tl.load(has_valid_ptr + mask_idx) == 0:
        return
    req_idx = tl.load(req_indices_ptr + mask_idx)
    block_id = tl.program_id(1)

    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + mask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
        other=0,
    )
    disallowed = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    disallowed = disallowed.reshape(BLOCK_SIZE)

    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + req_idx * logits_stride + block_offset,
        -float("inf"),
        mask=disallowed & (block_offset < vocab_size),
    )


def apply_safe_draft_grammar_bitmask(
    logits: torch.Tensor,
    req_indices: torch.Tensor,
    bitmask: torch.Tensor,
    has_valid_buf: torch.Tensor,
    check_intersection: bool = True,
) -> torch.Tensor:
    """Apply `bitmask[i]` in-place to `logits[req_indices[i]]`, preserving any
    row whose allowed grammar tokens have no finite draft logits (e.g. when
    EAGLE-3's cropped `d2t` vocabulary excludes all grammar-valid tokens).

    When `check_intersection` is `False` (full-vocab drafters), skips the
    intersection pre-pass and masks in a single kernel launch.

    Returns the `[num_active]` int32 `has_valid` tensor (`1` if masked, `0` if
    left unmasked due to empty support).
    """
    num_active = req_indices.shape[0]
    if num_active == 0:
        return has_valid_buf[:0]
    vocab_size = logits.shape[-1]
    block_size = _DRAFT_GRAMMAR_BLOCK_SIZE
    grid = (num_active, triton.cdiv(vocab_size, block_size))
    has_valid = has_valid_buf[:num_active]
    if check_intersection:
        has_valid.zero_()
        _check_draft_grammar_intersection_kernel[grid](
            logits,
            logits.stride(0),
            req_indices,
            bitmask,
            bitmask.stride(0),
            has_valid,
            vocab_size,
            BLOCK_SIZE=block_size,
        )
    else:
        has_valid.fill_(1)
    _apply_draft_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        req_indices,
        bitmask,
        bitmask.stride(0),
        has_valid,
        vocab_size,
        BLOCK_SIZE=block_size,
    )
    return has_valid


def find_single_allowed_token(row_int32: np.ndarray, vocab_size: int) -> int | None:
    """Return the sole allowed token ID when a packed bitmask row has popcount
    == 1, or `None` otherwise."""
    words = row_int32.view(np.uint32)
    nz = np.flatnonzero(words)
    if nz.size != 1:
        return None
    word_idx = int(nz[0])
    word = int(words[word_idx])
    if (word_idx + 1) * 32 > vocab_size:
        valid_bits = vocab_size - word_idx * 32
        word &= (1 << valid_bits) - 1
    if word == 0 or (word & (word - 1)) != 0:
        return None
    token_id = word_idx * 32 + (word & -word).bit_length() - 1
    return token_id if token_id < vocab_size else None


@dataclass(slots=True)
class _DraftGrammarEntry:
    matcher: xgr.GrammarMatcher
    draft_advancements: int = 0
    diverged_in_propose: bool = False


class DraftGrammarManager:
    """Tracks per-request shadow xgrammar matchers for autoregressive drafting."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        draft_vocab_target_ids: frozenset[int] | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size
        self.device = device
        self.draft_vocab_target_ids = draft_vocab_target_ids

        self._tokenizer_info: xgr.TokenizerInfo | None = None
        self._full_vocab_tokenizer_info: xgr.TokenizerInfo | None = None
        self._compiled_cache: OrderedDict[tuple[str, bool], xgr.CompiledGrammar] = (
            OrderedDict()
        )
        self._entries: dict[str, _DraftGrammarEntry] = {}

        # Active state for the current `propose()` call.
        self._propose_entries: list[tuple[int, str, _DraftGrammarEntry]] = []
        self._step_forced_tokens: list[int | None] = []
        self._step_active_entries: list[tuple[int, _DraftGrammarEntry]] = []
        self._step_idx: int = 0
        self._num_spec_steps: int = 0
        # Deferred target-verified token sync overlapped with draft step 0.
        self._pending_sync: (
            tuple[Sequence[str], torch.Event, np.ndarray, np.ndarray] | None
        ) = None

        # Pre-allocated pinned CPU ring buffers (1 slot per speculative step)
        # to avoid per-step `cudaHostAlloc` / `cudaFreeHost` device stalls.
        self._bitmask_cpu_ring: list[torch.Tensor] = []
        self._req_indices_cpu_ring: list[torch.Tensor] = []
        self._bitmask_gpu = torch.zeros(
            (max_num_reqs, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self._req_indices_gpu = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._has_valid_gpu = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )

        # Lightweight counters for diagnostics and tests.
        self.num_forced_fast_path_steps: int = 0
        self.num_synced_draft_steps: int = 0

    def _ensure_ring_slot(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self._bitmask_cpu_ring) <= slot:
            bm = allocate_token_bitmask(self.max_num_reqs, self.vocab_size)
            if PIN_MEMORY:
                bm = bm.pin_memory()
            self._bitmask_cpu_ring.append(bm)
            ri = torch.zeros(
                self.max_num_reqs,
                dtype=torch.int32,
                device="cpu",
                pin_memory=PIN_MEMORY,
            )
            self._req_indices_cpu_ring.append(ri)
        return self._bitmask_cpu_ring[slot], self._req_indices_cpu_ring[slot]

    def _ensure_tokenizer_infos(self) -> None:
        if self._tokenizer_info is not None:
            return
        tokenizer = cached_tokenizer_from_config(self.vllm_config.model_config)
        model_config = self.vllm_config.model_config
        is_plamo3 = (
            model_config is not None
            and getattr(model_config.hf_config, "model_type", None) == "plamo3"
        )
        self._tokenizer_info, self._full_vocab_tokenizer_info = (
            XgrammarBackend.build_tokenizer_infos(
                tokenizer, self.vocab_size, is_plamo3=is_plamo3
            )
        )
        self._ensure_ring_slot(0)

    def _get_or_deserialize_compiled(
        self, init: DraftGrammarInit
    ) -> xgr.CompiledGrammar:
        cache_id = (init.cache_key, init.use_full_vocab)
        cached = self._compiled_cache.get(cache_id)
        if cached is not None:
            self._compiled_cache.move_to_end(cache_id)
            return cached
        self._ensure_tokenizer_infos()
        ti = (
            self._full_vocab_tokenizer_info
            if init.use_full_vocab
            else self._tokenizer_info
        )
        assert ti is not None
        ctx = xgr.CompiledGrammar.deserialize_json(init.serialized_grammar, ti)
        self._compiled_cache[cache_id] = ctx
        if len(self._compiled_cache) > _COMPILED_GRAMMAR_CACHE_LIMIT:
            self._compiled_cache.popitem(last=False)
        return ctx

    def register_inits(self, inits: Sequence[DraftGrammarInit]) -> None:
        """Instantiate shadow matchers for newly-active or resumed requests."""
        for init in inits:
            ctx = self._get_or_deserialize_compiled(init)
            matcher = xgr.GrammarMatcher(
                ctx,
                override_stop_tokens=(
                    list(init.stop_token_ids) if init.stop_token_ids else None
                ),
            )
            replay_ok = True
            for token_id in init.accepted_token_ids:
                if matcher.is_terminated():
                    break
                if not matcher.accept_token(token_id):
                    logger.warning(
                        "DraftGrammarManager failed to replay token %d for "
                        "request %s; disabling draft grammar for this request.",
                        token_id,
                        init.req_id,
                    )
                    replay_ok = False
                    break
            if replay_ok:
                self._entries[init.req_id] = _DraftGrammarEntry(matcher=matcher)

    def remove_request(self, req_id: str) -> None:
        self._entries.pop(req_id, None)

    def has_requests_in_batch(self, req_ids: Sequence[str]) -> bool:
        if not self._entries:
            return False
        return any(req_id in self._entries for req_id in req_ids)

    @property
    def has_active_requests(self) -> bool:
        return any(
            not entry.diverged_in_propose and not entry.matcher.is_terminated()
            for _, _, entry in self._propose_entries
        )

    def sync_verified_tokens(
        self,
        req_ids: Sequence[str],
        sampled_token_ids: np.ndarray,
        num_sampled_tokens: np.ndarray,
    ) -> None:
        """Advance shadow matchers with tokens just verified by the target."""
        if not self._entries:
            return
        for row_idx, req_id in enumerate(req_ids):
            entry = self._entries.get(req_id)
            if entry is None:
                continue
            n = int(num_sampled_tokens[row_idx])
            if n <= 0 or entry.matcher.is_terminated():
                continue
            row_tokens = sampled_token_ids[row_idx]
            for j in range(n):
                tok = int(row_tokens[j])
                if tok < 0:
                    continue
                if entry.matcher.is_terminated():
                    break
                if not entry.matcher.accept_token(tok):
                    logger.debug(
                        "DraftGrammarManager detached request %s after "
                        "unexpected token %d during verified sync.",
                        req_id,
                        tok,
                    )
                    self._entries.pop(req_id, None)
                    break

    def _flush_pending_sync(self) -> None:
        if self._pending_sync is None:
            return
        req_ids, copy_event, sampled_token_ids, num_sampled_tokens = self._pending_sync
        self._pending_sync = None
        copy_event.synchronize()
        self.sync_verified_tokens(req_ids, sampled_token_ids, num_sampled_tokens)
        # Drop entries detached or terminated by the newly verified tokens.
        if self._propose_entries:
            self._propose_entries = [
                (row_idx, req_id, entry)
                for row_idx, req_id, entry in self._propose_entries
                if req_id in self._entries and not entry.matcher.is_terminated()
            ]

    def begin_propose(
        self,
        req_ids: Sequence[str],
        num_spec_steps: int,
        pending_sync: tuple[torch.Event, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Select active non-terminated grammar requests for this propose().

        When `pending_sync` is supplied, synchronizing `copy_event` and
        advancing shadow matchers with target-verified tokens is deferred until
        step 0's `apply_draft_bitmask()`, overlapping the D2H wait with draft
        step 0's model forward and logits projection.
        """
        self._propose_entries = []
        self._step_forced_tokens = []
        self._step_active_entries = []
        self._step_idx = 0
        self._num_spec_steps = num_spec_steps
        self._pending_sync = None
        if not self._entries:
            return
        if pending_sync is not None and self.has_requests_in_batch(req_ids):
            copy_event, sampled_token_ids, num_sampled_tokens = pending_sync
            self._pending_sync = (
                req_ids,
                copy_event,
                sampled_token_ids,
                num_sampled_tokens,
            )
        if num_spec_steps <= 0:
            self._flush_pending_sync()
            return
        for row_idx, req_id in enumerate(req_ids):
            entry = self._entries.get(req_id)
            if entry is not None and not entry.matcher.is_terminated():
                entry.draft_advancements = 0
                entry.diverged_in_propose = False
                self._propose_entries.append((row_idx, req_id, entry))
        if not self._propose_entries:
            self._flush_pending_sync()

    def apply_draft_bitmask(self, logits: torch.Tensor) -> None:
        """Fill and apply the current step's grammar bitmask in-place to draft
        `logits` (`[num_reqs, vocab_size]`), BEFORE `DraftLogitsCache` write."""
        self._flush_pending_sync()
        if not self._propose_entries:
            return
        active: list[tuple[int, _DraftGrammarEntry]] = []
        for row_idx, _, entry in self._propose_entries:
            if not entry.diverged_in_propose and not entry.matcher.is_terminated():
                active.append((row_idx, entry))
        self._step_active_entries = active
        self._step_forced_tokens = []
        if not active:
            return

        bitmask_cpu, req_indices_cpu = self._ensure_ring_slot(self._step_idx)
        bitmask_np = bitmask_cpu.numpy()
        req_indices_np = req_indices_cpu.numpy()
        draft_vocab = self.draft_vocab_target_ids

        num_masked = 0
        masked_entries: list[tuple[int, _DraftGrammarEntry]] = []
        for row_idx, entry in active:
            needs_mask = entry.matcher.fill_next_token_bitmask(bitmask_cpu, num_masked)
            if not needs_mask:
                entry.diverged_in_propose = True
                continue
            row_bm = bitmask_np[num_masked]
            # Free-text fast-bypass: when almost the entire vocab is allowed
            # (e.g. inside an unconstrained JSON string value), skip draft
            # masking and per-step sync for the rest of this propose() call so
            # remaining decode steps can stay on FULL CUDA graphs. Target
            # verification still enforces the exact grammar losslessly.
            if (
                int(np.count_nonzero(row_bm == -1)) * _OPEN_VOCAB_WORD_RATIO_DEN
                >= row_bm.size * _OPEN_VOCAB_WORD_RATIO_NUM
            ):
                entry.diverged_in_propose = True
                continue
            req_indices_np[num_masked] = row_idx
            forced = find_single_allowed_token(row_bm, self.vocab_size)
            if (
                forced is not None
                and draft_vocab is not None
                and forced not in draft_vocab
            ):
                # Rare single token outside EAGLE-3 cropped vocab: the safe
                # kernel will leave logits unmasked, so we cannot predict it.
                forced = None
            masked_entries.append((row_idx, entry))
            self._step_forced_tokens.append(forced)
            num_masked += 1

        self._step_active_entries = masked_entries
        if num_masked == 0:
            return

        bitmask_gpu = self._bitmask_gpu[:num_masked].copy_(
            bitmask_cpu[:num_masked], non_blocking=True
        )
        req_indices_gpu = self._req_indices_gpu[:num_masked].copy_(
            req_indices_cpu[:num_masked], non_blocking=True
        )
        apply_safe_draft_grammar_bitmask(
            logits,
            req_indices_gpu,
            bitmask_gpu,
            self._has_valid_gpu,
            check_intersection=draft_vocab is not None,
        )

    def advance_after_draft_step(self, sampled_tokens: torch.Tensor) -> None:
        """Advance shadow matchers after sampling draft step `self._step_idx`.

        Skips GPU-to-CPU sync on the final draft step AND on any step where all
        active requests hit the deterministic single-token fast-path.
        """
        is_last_step = self._step_idx + 1 >= self._num_spec_steps
        self._step_idx += 1
        if is_last_step or not self._step_active_entries:
            return

        need_sync_rows: list[int] = []
        need_sync_entries: list[_DraftGrammarEntry] = []

        for (row_idx, entry), forced_tok in zip(
            self._step_active_entries, self._step_forced_tokens
        ):
            if forced_tok is not None:
                if entry.matcher.accept_token(forced_tok):
                    entry.draft_advancements += 1
                else:
                    entry.diverged_in_propose = True
            else:
                need_sync_rows.append(row_idx)
                need_sync_entries.append(entry)

        if not need_sync_rows:
            self.num_forced_fast_path_steps += 1
            return

        self.num_synced_draft_steps += 1
        sampled_cpu = sampled_tokens.cpu().tolist()
        for row_idx, entry in zip(need_sync_rows, need_sync_entries):
            tok = int(sampled_cpu[row_idx])
            if entry.matcher.is_terminated() or not entry.matcher.accept_token(tok):
                entry.diverged_in_propose = True
            else:
                entry.draft_advancements += 1

    def end_propose(self) -> None:
        """Roll back all speculative draft advancements to committed state."""
        self._flush_pending_sync()
        for _, _, entry in self._propose_entries:
            if entry.draft_advancements > 0:
                entry.matcher.rollback(entry.draft_advancements)
                entry.draft_advancements = 0
            entry.diverged_in_propose = False
        self._propose_entries = []
        self._step_active_entries = []
        self._step_forced_tokens = []
        self._step_idx = 0
        self._num_spec_steps = 0
