# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable policies for dynamically deciding the number of draft tokens.

A DraftLengthPolicy is called once per draft step via ``update()`` with no
GPU→CPU synchronisation.  All state is maintained on the GPU.  After the
draft loop completes, the caller initiates a single async D→H copy of
``k_valid_gpu`` and reads the result later (after all K steps have run), so
the K draft model forward passes pipeline without interruption.

Reducing from K blocking syncs (the old ``should_continue().item()`` style)
to 1 deferred sync is the key performance improvement over the previous design.

Example usage (see SpecDecodeBaseProposer.__init__)::

    from vllm.v1.spec_decode.draft_length_policy import (
        DraftLengthPolicy,
        ConfidenceThresholdPolicy,
        AlwaysContinuePolicy,
    )

    threshold = speculative_config.draft_confidence_threshold
    policy: DraftLengthPolicy = (
        ConfidenceThresholdPolicy(threshold, device, num_speculative_tokens)
        if threshold > 0
        else AlwaysContinuePolicy()
    )
"""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class DraftLengthPolicy(Protocol):
    """Protocol for GPU-async dynamic draft-length policies.

    ``update()`` is called once per draft step; it performs only GPU tensor
    operations and never causes a GPU→CPU synchronisation.  After all K steps
    the caller reads ``k_valid_gpu`` (a GPU int32 scalar) via an async D→H
    copy; ``k_valid_gpu is None`` signals "always use full K" (no-op policy).

    Args for update():
        step: Zero-based loop index of the draft token just appended
              (0 = first extra token after the seed).
        token_probs: 1-D float tensor ``[batch_size]`` holding the draft
                     model's top-1 softmax probability for each sequence.
    """

    def reset(self) -> None:
        """Reset GPU state at the start of each propose() call."""
        ...

    def update(self, step: int, token_probs: torch.Tensor) -> None:
        """Update GPU state after draft step ``step``.  No CPU sync."""
        ...

    @property
    def k_valid_gpu(self) -> torch.Tensor | None:
        """GPU int32 scalar: number of valid tokens (seed + drafted).
        ``None`` means "always full K" (AlwaysContinuePolicy)."""
        ...


class ConfidenceThresholdPolicy:
    """Mean-confidence early-exit policy (DISCO-style), GPU-async variant.

    All exit-detection state is maintained as GPU tensors updated by
    ``update()``.  No GPU→CPU sync occurs inside the draft loop.

    After all K draft steps, the caller copies ``k_valid_gpu`` to CPU via a
    single async D→H transfer, reducing from K blocking syncs to one
    deferred sync per decode step.

    ``k_valid`` starts at ``num_spec_tokens`` (no exit).  On the first step
    where ``mean(token_probs) < threshold``, it is frozen to ``step + 2``
    (seed token + all tokens generated up to and including that step).
    Subsequent steps leave ``k_valid`` unchanged (``_exited_gpu`` gate).

    Reference: arXiv:2405.04304 (DISCO).
    """

    def __init__(
        self, threshold: float, device: torch.device, num_spec_tokens: int
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                f"ConfidenceThresholdPolicy threshold must be in (0, 1], "
                f"got {threshold}"
            )
        self.threshold = threshold
        self.num_spec_tokens = num_spec_tokens

        # k_valid_gpu: GPU int32 scalar.  Default = num_spec_tokens (no exit).
        # frozen to first-exit step value via masked_fill_.
        self._k_valid_gpu = torch.tensor(
            num_spec_tokens, dtype=torch.int32, device=device
        )
        # _exited_gpu: GPU bool scalar gate — once True, k_valid is frozen.
        self._exited_gpu = torch.tensor(False, dtype=torch.bool, device=device)

    def reset(self) -> None:
        self._k_valid_gpu.fill_(self.num_spec_tokens)
        self._exited_gpu.fill_(False)

    def update(self, step: int, token_probs: torch.Tensor) -> None:
        """GPU-only update.  No CPU sync."""
        mean_conf_below = token_probs.mean() < self.threshold  # GPU bool scalar
        # Gate: only update on the first exit (prevents overwriting k_valid).
        first_exit = mean_conf_below & (~self._exited_gpu)
        # step + 2 = seed token + (step + 1) drafted tokens.
        self._k_valid_gpu.masked_fill_(first_exit, step + 2)
        self._exited_gpu.logical_or_(mean_conf_below)

    @property
    def k_valid_gpu(self) -> torch.Tensor:
        return self._k_valid_gpu


class AlwaysContinuePolicy:
    """No-op policy (DSL disabled).  Never exits early.

    All methods are stubs so the draft loop can call ``update()`` uniformly
    without a per-step ``if dsl_enabled`` guard.  ``k_valid_gpu`` returns
    ``None`` to signal "always full K" to the async-copy infrastructure.
    """

    def reset(self) -> None:
        pass

    def update(self, step: int, token_probs: torch.Tensor) -> None:
        pass

    @property
    def k_valid_gpu(self) -> None:
        return None
