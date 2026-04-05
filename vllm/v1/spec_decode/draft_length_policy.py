# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable policies for dynamically deciding the number of draft tokens.

Design (async early-exit loop, isolating while condition):

  * ``update()`` performs only GPU tensor ops then kicks off an async D→H copy
    of ``_exited_gpu`` (a single bool scalar) on a dedicated stream.
  * The loop checks ``policy.exited`` at the *start* of the *next* iteration.
    By then Python loop overhead is enough for the copy to complete, so
    ``event.synchronize()`` has near-zero additional latency.
  * Only ``k_valid`` draft-model forward passes actually run; the GPU tensor
    is zero-padded back to ``[B, K]`` after the loop so that
    ``_prepare_input_ids`` can keep its fixed ``num_speculative_tokens`` stride.

This reduces K blocking syncs (old ``should_continue().item()`` style) to at
most K−2 near-free async syncs, while actually skipping K−k_valid forward
passes.

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
    operations and never causes a GPU→CPU synchronisation.  The draft loop
    breaks after ``k_valid`` steps (fewer than K when confidence drops early),
    skipping the remaining forward passes.  ``k_valid_gpu is None`` signals
    "always use full K" (AlwaysContinuePolicy).

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

    @property
    def exited(self) -> bool:
        """Return the exit decision recorded at the previous ``update()``.

        Syncs the async D→H event recorded by ``update()``.  Should be called
        once per loop iteration, after any GPU setup work so the event is
        already signalled and the sync is effectively free."""
        ...


class ConfidenceThresholdPolicy:
    """Mean-confidence early-exit policy (DISCO-style), GPU-async variant.

    All exit-detection state is maintained as GPU tensors updated by
    ``update()``.  No GPU→CPU sync occurs inside the draft loop.

    The draft loop breaks after ``k_valid`` steps: ``update()`` kicks off an
    async 1-byte D→H copy of ``_exited_gpu`` after each step; the next
    iteration checks ``exited`` (near-free — copy completes during Python
    loop overhead) and breaks early, skipping up to ``K − k_valid`` forward
    passes.  ``k_valid_gpu`` is read by the caller after the loop to trim the
    CPU token list handed to the scheduler.

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

        # Async D→H infrastructure for per-step exit check.
        # _exited_cpu is pinned so the D→H transfer is DMA-accelerated.
        self._exited_cpu: torch.Tensor = torch.zeros(1, dtype=torch.bool).pin_memory()
        self._exit_copy_stream = torch.cuda.Stream(device=device)
        self._exit_event = torch.cuda.Event()

    def reset(self) -> None:
        self._k_valid_gpu.fill_(self.num_spec_tokens)
        self._exited_gpu.fill_(False)
        self._exited_cpu.fill_(False)

    def update(self, step: int, token_probs: torch.Tensor) -> None:
        """GPU tensor ops + async D→H copy of the exit flag.  No blocking sync.

        After all GPU ops complete on the default stream, a dedicated copy
        stream transfers ``_exited_gpu`` (1 byte) to the pinned ``_exited_cpu``
        buffer and records ``_exit_event``.  The event will be signalled by the
        time the caller checks ``self.exited`` at the start of the next loop
        iteration (Python loop overhead is sufficient for a 1-byte transfer).
        """
        mean_conf_below = token_probs.mean() < self.threshold  # GPU bool scalar
        # Gate: only update on the first exit (prevents overwriting k_valid).
        first_exit = mean_conf_below & (~self._exited_gpu)
        # step + 2 = seed token + (step + 1) drafted tokens.
        self._k_valid_gpu.masked_fill_(first_exit, step + 2)
        self._exited_gpu.logical_or_(mean_conf_below)
        # Kick off async D→H copy so the NEXT iteration can read the result
        # from CPU without blocking the GPU main stream.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._exit_copy_stream):
            self._exit_copy_stream.wait_stream(default_stream)
            self._exited_cpu.copy_(self._exited_gpu, non_blocking=True)
            self._exit_event.record()

    @property
    def exited(self) -> bool:
        """Sync the async D→H event and return the CPU exit flag.

        In practice the 1-byte transfer completes during Python loop overhead
        between iterations, so this sync has near-zero additional latency.
        """
        self._exit_event.synchronize()
        return bool(self._exited_cpu.item())  # plain CPU read, no GPU sync

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
    def exited(self) -> bool:
        return False

    @property
    def k_valid_gpu(self) -> None:
        return None
