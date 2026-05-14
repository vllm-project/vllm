# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side state for auxiliary mean-pool of prompt hidden states.

Supports generative requests opting in via
``SamplingParams.extra_args["return_embed"] = True``. When enabled,
the runner sums hidden states for prompt tokens and emits the
per-prompt mean alongside generation, without taking over the whole batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class _ReqState:
    """Per-request running sum for an opted-in generative request."""

    prompt_len: int
    # Number of prompt tokens whose hidden states have been added to
    # ``running_sum`` so far (across cached-prefix + computed-suffix).
    tokens_seen: int
    # fp32 GPU tensor of shape [hidden_size]; accumulates the prompt sum.
    running_sum: torch.Tensor
    # True once we've emitted the final pooled vector.
    finalized: bool = False


class AuxPoolState:
    """Per-block + per-request state for prompt mean-pool.

    The per-block store is a dense GPU tensor sized by the worker's KV
    cache block count, allocated eagerly at engine init when the server
    is started with --enable-return-embed. Every prefill step sums its
    tokens' hidden states into the matching block_aux_sums entry (via
    ``index_add_``); the very first slot of each block is zeroed first
    so a reused block_id doesn't carry stale data.

    Per-request state holds the running sum (initialised from cached-prefix
    block sums) and a tokens_seen counter; finalisation happens on the step
    that brings ``tokens_seen`` up to ``prompt_len``.
    """

    def __init__(self, block_size: int, hidden_size: int, device: torch.device):
        self._block_size = block_size
        self._hidden_size = hidden_size
        self._device = device
        # block_id -> [hidden_size] fp32 sum on GPU.
        # Allocated by enable(), called once kv_cache_config.num_blocks
        # is known.
        self._block_aux_sums: torch.Tensor | None = None
        self._req_states: dict[str, _ReqState] = {}
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self, num_blocks: int) -> None:
        """Allocate the per-block store. Idempotent."""
        if self._enabled:
            return
        assert num_blocks > 0, f"num_blocks must be > 0, got {num_blocks}"
        # Dense fp32 store. num_blocks × hidden_size × 4 bytes
        self._block_aux_sums = torch.zeros(
            (num_blocks, self._hidden_size),
            dtype=torch.float32,
            device=self._device,
        )
        self._enabled = True

    def has_request(self, req_id: str) -> bool:
        return req_id in self._req_states

    def cleanup(self, req_id: str) -> None:
        self._req_states.pop(req_id, None)

    def init_request(
        self,
        req_id: str,
        prompt_len: int,
        cached_block_ids: list[int],
        num_computed_tokens: int,
    ) -> None:
        """
        Initialize with a request's running sum for
        prefix-cache and chunked-prefill
        """
        assert self._block_aux_sums is not None
        running_sum = torch.zeros(
            self._hidden_size, dtype=torch.float32, device=self._device
        )
        if num_computed_tokens > 0 and cached_block_ids:
            # pre-calculated prefix blocks
            num_full_cached = num_computed_tokens // self._block_size
            if num_full_cached > 0:
                prefix_ids = torch.tensor(
                    cached_block_ids[:num_full_cached],
                    dtype=torch.long,
                    device=self._device,
                )
                running_sum.add_(
                    self._block_aux_sums.index_select(0, prefix_ids).sum(dim=0)
                )
        self._req_states[req_id] = _ReqState(
            prompt_len=prompt_len,
            tokens_seen=num_computed_tokens,
            running_sum=running_sum,
        )

    def update_block_sums(
        self,
        slot_mapping: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Accumulate per-token hidden states into per-block aux sums."""

        assert self._enabled
        assert self._block_aux_sums is not None

        # Empty step (e.g. a no-op synchronisation/dummy step): nothing to do.
        if slot_mapping.numel() == 0:
            return

        # Mask out padding slots (-1 from cudagraph padding).
        valid = slot_mapping >= 0
        if not valid.all():
            slot_mapping = slot_mapping[valid]
            hidden_states = hidden_states[valid]
            if slot_mapping.numel() == 0:
                return

        block_ids = slot_mapping // self._block_size
        slot_in_block = slot_mapping % self._block_size

        # Zero out aux for blocks where this step is writing slot 0 — i.e.
        # the block has been (re)allocated since its last full state.
        first_slot_mask = slot_in_block == 0
        if first_slot_mask.any():
            first_block_ids = block_ids[first_slot_mask]
            self._block_aux_sums.index_fill_(0, first_block_ids, 0.0)

        self._block_aux_sums.index_add_(0, block_ids, hidden_states.float())

    def update_request(
        self,
        req_id: str,
        prompt_token_slice: torch.Tensor,
    ) -> torch.Tensor | None:
        """Add this step's prompt-token hidden states to a request's sum.

        Returns the finalised pooled tensor (on GPU, shape [hidden_size])
        if this step completes the prompt; otherwise ``None``.
        Caller is responsible for the device->host copy and for
        ``cleanup(req_id)`` once the value has been consumed.
        """

        assert self._enabled

        state = self._req_states.get(req_id)

        assert state is not None and not state.finalized, (
            f"update_request called for {req_id!r} with no live state "
            f"(state={state}); pool_req_ids out of sync with _req_states."
        )

        if prompt_token_slice.shape[0] > 0:
            state.running_sum.add_(prompt_token_slice.sum(dim=0, dtype=torch.float32))
            state.tokens_seen += prompt_token_slice.shape[0]

        if state.tokens_seen >= state.prompt_len:
            state.finalized = True
            return state.running_sum / state.prompt_len

        return None
