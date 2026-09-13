# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-batch Mirostat v2 sampling state.

Mirostat (Basu et al., 2020, https://arxiv.org/abs/2007.14966) is a feedback
controller over the *surprise* (``-log2(p)``, i.e. per-token cross-entropy in
bits) of the sampled token. It keeps the running surprise near a target ``tau``
by maintaining a per-request threshold ``mu`` that is updated after every step:

    mu <- mu - eta * (observed_surprise - tau)

This resists both degeneration into repetition (surprise collapses toward 0)
and incoherence (surprise spikes), which is why it is effective against the
"infinite repetition" failure mode.

Only Mirostat v2 is implemented (``mirostat_mode == 2``): at each step every
token whose surprise exceeds ``mu`` is truncated (its logit set to ``-inf``),
then a token is sampled from the remaining distribution, then ``mu`` is updated
from the sampled token's surprise. The truncation only removes low-probability
tokens and always keeps the argmax, so it is argmax-invariant and composes with
the rest of the (random) sampling pipeline.

Performance: all per-step math runs on-device and the ``mu`` state persists as a
GPU tensor, so decoding never forces a host<->device synchronization (the naive
approach of reading surprise back to Python each step serializes the CPU/GPU
pipeline and drops decode throughput). Inactive rows carry ``mu = +inf`` and
``eta = 0``, which makes truncation a no-op (threshold ``-inf`` keeps every
token) and the ``mu`` update a no-op (delta scaled by ``eta = 0``), so a single
vectorized path covers the whole batch without per-request gathers -- this also
keeps shapes static for CUDA graph capture. The state is consumed inside
:class:`~vllm.v1.sample.sampler.Sampler` rather than as a ``LogitsProcessor``
because the ``mu`` update needs the sampled token, which only exists after
sampling. This mirrors
:class:`~vllm.v1.sample.thinking_budget_state.ThinkingBudgetStateHolder`.
"""

import math

import torch

from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality

# ``mirostat_mode`` value that selects Mirostat v2.
MIROSTAT_V2 = 2

_LN2 = math.log(2.0)


class MirostatStateHolder:
    """Tracks per-request Mirostat v2 ``mu`` state across decoding steps.

    ``mu``/``tau``/``eta`` are kept as device tensors indexed by batch-slot and
    stay in sync with the running batch via :meth:`sync_batch`. Inactive slots
    hold ``mu = +inf`` / ``eta = 0`` so the per-step math is a no-op for them.
    A holder with no tracked requests is a no-op, so a single instance can be
    created unconditionally per :class:`InputBatch`.
    """

    def __init__(self, max_num_reqs: int = 1, device: torch.device | str | None = None):
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.max_num_reqs = max_num_reqs
        # Slot state (device tensors). Inactive slot: mu=+inf, tau=0, eta=0.
        self.mu = torch.full(
            (max_num_reqs,), float("inf"), dtype=torch.float32, device=self.device
        )
        self.tau = torch.zeros((max_num_reqs,), dtype=torch.float32, device=self.device)
        self.eta = torch.zeros((max_num_reqs,), dtype=torch.float32, device=self.device)
        # CPU-side set of active slots: the gate + move bookkeeping never touch
        # the device (no sync).
        self._active: set[int] = set()
        # log-probs of the pre-truncation distribution, cached by
        # ``apply_to_logits`` and consumed by ``update_mu`` in the same step.
        self._cached_logprobs: torch.Tensor | None = None

    def has_tracked_requests(self) -> bool:
        """True when at least one request in the batch uses Mirostat."""
        return bool(self._active)

    def _set_inactive(self, index: int) -> None:
        self.mu[index] = float("inf")
        self.tau[index] = 0.0
        self.eta[index] = 0.0
        self._active.discard(index)

    def _set_active(self, index: int, tau: float, eta: float) -> None:
        self.tau[index] = tau
        self.eta[index] = eta
        # Initialize mu to 2*tau, as in the reference implementation.
        self.mu[index] = 2.0 * tau
        self._active.add(index)

    def sync_batch(self, batch_update: BatchUpdate | None) -> None:
        """Add/remove/move per-request ``mu`` state to match the batch."""
        if not batch_update:
            return

        for index in batch_update.removed:
            self._set_inactive(index)

        for index, params, _prompt_tok_ids, _output_tok_ids in batch_update.added:
            mode = getattr(params, "mirostat_mode", 0) or 0
            if mode == MIROSTAT_V2:
                self._set_active(
                    index, float(params.mirostat_tau), float(params.mirostat_eta)
                )
            else:
                self._set_inactive(index)

        for i1, i2, direction in batch_update.moved:
            a1 = i1 in self._active
            if direction == MoveDirectionality.SWAP:
                a2 = i2 in self._active
                for t in (self.mu, self.tau, self.eta):
                    tmp = t[i1].clone()
                    t[i1] = t[i2]
                    t[i2] = tmp
                self._active.discard(i1)
                self._active.discard(i2)
                if a2:
                    self._active.add(i1)
                if a1:
                    self._active.add(i2)
            else:
                self.mu[i2] = self.mu[i1]
                self.tau[i2] = self.tau[i1]
                self.eta[i2] = self.eta[i1]
                self._active.discard(i2)
                if a1:
                    self._active.add(i2)
                self._set_inactive(i1)

    def apply_to_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Truncate tokens with surprise greater than ``mu`` (Mirostat v2 step).

        For every tracked request tokens whose surprise ``-log2(p)`` exceeds that
        request's ``mu`` have their logit set to ``-inf``; the argmax is always
        kept so at least one token survives and the transform stays
        argmax-invariant. The pre-truncation log-probs are cached for
        :meth:`update_mu` so the surprise is only computed once per step.

        Args:
            logits: ``[num_rows, vocab_size]`` float logits (modified in place).

        Returns:
            The same ``logits`` tensor.
        """
        if not self._active:
            return logits

        n = logits.shape[0]
        logprobs = torch.log_softmax(logits, dim=-1)
        self._cached_logprobs = logprobs
        # Keep tokens with surprise <= mu, i.e. logprob >= -mu*ln2. Inactive rows
        # have mu=+inf -> threshold -inf -> keep everything (untouched).
        threshold = (-self.mu[:n] * _LN2).unsqueeze(-1)
        keep = logprobs >= threshold
        argmax = logits.argmax(dim=-1, keepdim=True)
        keep.scatter_(-1, argmax, True)
        logits.masked_fill_(~keep, float("-inf"))
        return logits

    def update_mu(self, sampled: torch.Tensor) -> None:
        """Update each request's ``mu`` from the sampled token's surprise.

        Fully on-device (no host sync). Uses the log-probs cached by
        :meth:`apply_to_logits`. Inactive rows have ``eta = 0`` so their update
        is a no-op and ``mu`` stays ``+inf``.

        Args:
            sampled: ``[num_rows]`` sampled token ids.
        """
        if not self._active or self._cached_logprobs is None:
            return
        logprobs = self._cached_logprobs
        n = logprobs.shape[0]
        tok = sampled[:n].long().unsqueeze(-1)
        sampled_logprob = logprobs.gather(-1, tok).squeeze(-1)
        # observed surprise in bits = -logprob / ln2.
        surprise = -sampled_logprob / _LN2
        self.mu[:n] = self.mu[:n] - self.eta[:n] * (surprise - self.tau[:n])
        self._cached_logprobs = None
