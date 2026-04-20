# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 GPU-native N-gram speculator.

Unlike V1's ``NgramProposerGPU`` (in ``vllm/v1/spec_decode/ngram_proposer_gpu.py``),
this speculator is designed to slot directly into the V2 runner's speculator
interface. It reuses the runner-owned ``RequestState.all_token_ids`` /
``RequestState.total_len`` tensors as the persistent token store and thus
does not maintain any shadow GPU state.

Key properties:
  * Fully vectorized ``unfold → broadcast-compare → argmax → gather`` algorithm
    — no CPU-GPU synchronization inside ``propose``.
  * ``-1`` no-match positions are rewritten on GPU to ``last_sampled_tokens``
    so that the downstream ``combine_sampled_and_draft_tokens`` kernel never
    writes an invalid token id into ``input_ids``.
  * A companion ``num_valid_draft_tokens`` tensor is emitted every step and
    propagated to the scheduler so that ``request.spec_token_ids`` can be
    correctly truncated before the next scheduling round.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.states import RequestState


@support_torch_compile()
class _NgramKernel(nn.Module):
    """Pure, stateless n-gram match kernel.

    Exposed as an ``nn.Module`` so that ``@support_torch_compile`` can hoist
    the whole body into a single Inductor-compiled region. The module has
    no parameters; it is compiled purely for operator fusion.
    """

    def __init__(self, min_n: int, max_n: int, k: int):
        super().__init__()
        assert 1 <= min_n <= max_n, (
            f"min_n must be in [1, max_n]; got min_n={min_n}, max_n={max_n}"
        )
        assert k >= 1
        self.min_n = min_n
        self.max_n = max_n
        self.k = k
        self.num_sizes = max_n - min_n + 1

    def forward(
        self,
        token_ids: torch.Tensor,  # [B, L] int32
        seq_lens: torch.Tensor,  # [B]   int32  (current total_len per req)
        valid_mask: torch.Tensor,  # [B]   bool (row eligible for n-gram lookup)
        last_sampled: torch.Tensor,  # [B]   int64 (fallback for -1 positions)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(draft_tokens[B, k] int64, num_valid_draft_tokens[B] int32)``.

        The ``draft_tokens`` tensor is guaranteed to contain only valid
        vocabulary IDs. Positions that correspond to no-match / out-of-range
        slots are filled with ``last_sampled`` so that writing them to
        ``input_ids`` later is safe. The companion ``num_valid_draft_tokens``
        encodes how many *leading contiguous* draft positions were actually
        derived from a real n-gram match.
        """
        B, L = token_ids.shape
        device = token_ids.device

        # --- (1) For each n-gram size in [min_n, max_n], find the earliest
        # prior occurrence of the trailing suffix. The outer `for` loop runs
        # at most `max_n - min_n + 1` times (typically <= 4), and is required
        # because `Tensor.unfold` needs a Python int window size. Each
        # iteration executes as one fused GPU pass.
        first_match_pos = torch.full(
            (B, self.num_sizes), -1, dtype=torch.long, device=device
        )
        batch_idx = torch.arange(B, device=device)

        for i, n in enumerate(range(self.min_n, self.max_n + 1)):
            # [B, L-n+1, n] view — O(1) cost (contiguous stride trick).
            windows = token_ids.unfold(1, n, 1)
            num_windows = windows.shape[1]

            # Gather the trailing n tokens from each sequence as the query.
            # `suffix_start` is clamped to 0 to keep the gather in-bounds; the
            # corresponding row is also filtered by `valid_mask` upstream.
            suffix_start = (seq_lens.long() - n).clamp(min=0)
            offsets = torch.arange(n, device=device)
            suffix_idx = suffix_start.unsqueeze(1) + offsets  # [B, n]
            suffix = torch.gather(token_ids, 1, suffix_idx)  # [B, n]

            # Element-wise equality → all(-1) → per-window boolean match.
            matches = (windows == suffix.unsqueeze(1)).all(dim=-1)  # [B, L-n+1]

            # A match is only actionable if at least one draft token follows.
            max_valid_pos = seq_lens.long() - n - 1  # may be negative
            window_pos = torch.arange(num_windows, device=device)
            matches = matches & (window_pos.unsqueeze(0) <= max_valid_pos.unsqueeze(1))

            # `argmax(int)` returns 0 on all-false rows; verify with a lookup.
            idx = matches.int().argmax(dim=1)  # [B]
            has_match = matches[batch_idx, idx]  # [B]
            first_match_pos[:, i] = torch.where(has_match, idx.long(), -1)

        # --- (2) Pick the LONGEST n-gram with a match (flip+argmax, no sync).
        # Equivalent to "last True from the right".
        best_i = (first_match_pos >= 0).int().flip(dims=[1]).argmax(dim=1)
        best_i = self.num_sizes - 1 - best_i  # flip index back
        best_pos = first_match_pos[batch_idx, best_i]  # [B]
        ngram_lens_table = torch.arange(
            self.min_n, self.max_n + 1, device=device, dtype=torch.long
        )
        best_n = ngram_lens_table[best_i]  # [B]
        has_any = best_pos >= 0  # [B] bool

        # --- (3) Gather k tokens starting right after the matched suffix.
        draft_start = torch.where(
            has_any,
            best_pos + best_n,
            torch.zeros_like(best_pos),
        )  # [B]
        k_range = torch.arange(self.k, device=device)  # [k]
        draft_idx = (draft_start.unsqueeze(1) + k_range).clamp_(0, L - 1)
        drafts = torch.gather(token_ids, 1, draft_idx).to(torch.int64)  # [B, k]

        # --- (4) Compute the leading-valid mask per row.
        tokens_available = (seq_lens.long() - draft_start).clamp_(min=0)  # [B]
        valid_positions = k_range.unsqueeze(0) < tokens_available.unsqueeze(1)
        row_valid = has_any & valid_mask  # [B]
        leading_valid_mask = valid_positions & row_valid.unsqueeze(1)  # [B, k]

        # --- (5) num_valid = length of the leading contiguous valid run.
        # cumsum counts valid positions so far; match against [1, 2, ..., k]
        # tells us whether each position is still in the leading run.
        cum_valid = leading_valid_mask.int().cumsum(dim=1)  # [B, k]
        positions = torch.arange(1, self.k + 1, device=device)  # [k]
        num_valid = (cum_valid == positions.unsqueeze(0)).int().sum(dim=1)  # [B]

        # --- (6) Safe-substitute invalid slots with last_sampled so that the
        # next step's input_ids are always in-vocabulary. `last_sampled` is
        # int64 already (matches RequestState.last_sampled_tokens dtype).
        safe_drafts = torch.where(
            leading_valid_mask,
            drafts,
            last_sampled.view(-1, 1).expand(B, self.k),
        )
        return safe_drafts, num_valid.to(torch.int32)


class NgramGPUSpeculator:
    """V2-compatible GPU n-gram speculator.

    Public surface mirrors ``EagleSpeculator`` (``load_model``, ``set_attn``,
    ``init_cudagraph_manager``, ``capture_model``, ``propose``). Unlike the
    Eagle path, this speculator:

      * has no neural model, no attention layers, no CUDA graph capture;
      * reuses the runner-owned ``RequestState`` persistent token store;
      * emits ``num_valid_draft_tokens`` as a secondary output so that the
        scheduler can trim per-request drafts that were effectively a miss.
    """

    # Consumed by the runner's spec-decode sample path — see
    # ``vllm/v1/worker/gpu/model_runner.py::sample``.
    supports_mm_inputs = False
    draft_logits = None  # probabilistic rejection sampling is not supported

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        spec = vllm_config.speculative_config
        assert spec is not None
        assert spec.prompt_lookup_min is not None, (
            "prompt_lookup_min must be configured for ngram_gpu"
        )
        assert spec.prompt_lookup_max is not None, (
            "prompt_lookup_max must be configured for ngram_gpu"
        )

        self.vllm_config = vllm_config
        self.device = device
        self.speculative_config = spec
        self.num_speculative_steps: int = spec.num_speculative_tokens

        self.min_n: int = spec.prompt_lookup_min
        self.max_n: int = spec.prompt_lookup_max

        self.max_num_reqs: int = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len: int = vllm_config.model_config.max_model_len

        self.kernel = (
            _NgramKernel(
                min_n=self.min_n,
                max_n=self.max_n,
                k=self.num_speculative_steps,
            )
            .to(device)
            .eval()
        )

        # Persistent scratch buffer — shaped [max_num_reqs] on device — used to
        # stash the last ``num_valid_draft_tokens`` tensor so that the runner
        # can forward it to ``DraftTokensHandler`` after ``propose`` returns.
        self.num_valid_draft_tokens: torch.Tensor = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        # Injected by the model runner once ``req_states`` is constructed.
        # Accessed inside ``propose``.
        self.req_states: RequestState | None = None

    # ------------------------------------------------------------------
    # V2 Speculator interface: intentional no-ops.
    # ------------------------------------------------------------------
    def load_model(self, target_model: nn.Module) -> None:
        """No weights to load — ngram is a data-only proposer."""
        pass

    def set_attn(self, *args: Any, **kwargs: Any) -> None:
        """No attention layers owned by this speculator."""
        pass

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """N-gram kernel is torch.compile-managed; no explicit CG capture."""
        pass

    def capture_model(self) -> None:
        """No graph capture phase required."""
        pass

    # ------------------------------------------------------------------
    # V2 Speculator interface: main entry point.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: Any,  # unused
        slot_mappings: Any,  # unused
        # [num_tokens, hidden_size] — unused by ngram, kept for signature parity
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size] — unused
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs] int32
        num_sampled: torch.Tensor,
        # [num_reqs] int32 — unused by ngram
        num_rejected: torch.Tensor,
        # [max_num_reqs, 1] int64
        last_sampled_tokens: torch.Tensor,
        # [max_num_reqs] int32 — unused
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs] — unused
        temperature: torch.Tensor,
        # [max_num_reqs] — unused
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        """Propose up to ``num_speculative_steps`` draft tokens per request.

        Returns a ``[num_reqs, num_speculative_steps]`` int64 tensor. Invalid
        or no-match positions are backfilled with ``last_sampled`` rather
        than ``-1`` so that downstream ``combine_sampled_and_draft_tokens``
        never writes out-of-vocab ids into ``input_ids``. Truth about how
        many of those drafts are "real" lives in
        ``self.num_valid_draft_tokens[:num_reqs]``.
        """
        assert self.req_states is not None, (
            "NgramGPUSpeculator.req_states was not injected by the model "
            "runner. Ensure model_runner sets `speculator.req_states = "
            "self.req_states` after RequestState is constructed."
        )

        num_reqs = input_batch.num_reqs
        # `idx_mapping` is [num_reqs] int32 on device; advanced indexing
        # requires int64 → one-time cast (negligible cost vs. the
        # O(B * L * num_sizes) compare).
        idx_mapping_long = input_batch.idx_mapping.long()

        # Persistent [max_num_reqs, max_model_len] store, UVA-backed.
        # Advanced indexing materialises a contiguous [num_reqs, L] view on
        # device — identical access pattern to V1's shadow tensor. No extra
        # H2D copies are issued.
        active_tokens: torch.Tensor = self.req_states.all_token_ids.gpu[
            idx_mapping_long
        ]
        active_seq_lens: torch.Tensor = self.req_states.total_len.gpu[idx_mapping_long]
        active_last_sampled: torch.Tensor = last_sampled_tokens.view(-1)[
            idx_mapping_long
        ]

        # A request can draft iff (a) at least one real token was just
        # sampled for it (otherwise we cannot trust its suffix) AND
        # (b) the sequence already contains min_n tokens for the lookup.
        valid_mask = (num_sampled > 0) & (active_seq_lens >= self.min_n)

        with set_forward_context(None, self.vllm_config):
            drafts, num_valid = self.kernel(
                active_tokens,
                active_seq_lens,
                valid_mask,
                active_last_sampled,
            )

        # Stash num_valid so the runner can forward it to DraftTokensHandler
        # after scatter. Note: this is a device tensor; D2H happens on a
        # side stream in the handler.
        self.num_valid_draft_tokens[:num_reqs].copy_(num_valid)
        # Zero out the tail slots to avoid leaking stale values from prior
        # steps in case the runner ever peeks beyond num_reqs.
        if num_reqs < self.max_num_reqs:
            self.num_valid_draft_tokens[num_reqs:].zero_()

        return drafts  # [num_reqs, num_speculative_steps] int64, no -1

    def get_num_valid_draft_tokens(self, num_reqs: int) -> torch.Tensor:
        """Return the last step's per-request valid draft counts.

        Sliced view of the internal buffer; safe to pass to an async D2H
        copy on a side stream.
        """
        return self.num_valid_draft_tokens[:num_reqs]
