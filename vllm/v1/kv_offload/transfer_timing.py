# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TransferTimingStats: rolling-window PCIe transfer rate estimator.

Used by the non-blocking load path (Strategy B) to decide whether waiting for
an in-flight CPU→GPU transfer is cheaper than falling back to prefix recompute.

The estimator bootstraps with a conservative default (0.003 ms/token ≈ 3 µs/
token, corresponding to PCIe Gen4 at ~40 GB/s for fp16 KV tokens) until at
least 10 observations have been collected.  After that it uses a true rolling
harmonic mean over the last *window_size* completed transfers.
"""
from collections import deque


class TransferTimingStats:
    """Rolling-window estimator of observed PCIe transfer time per token.

    Args:
        window_size: Number of recent (tokens, elapsed_ms) observations to
            retain.  Older observations are dropped automatically.
        default_ms_per_token: Conservative cold-start value used until
            ``min_observations`` samples have been collected.
        min_observations: Minimum samples before switching from the default
            to the observed average.
    """

    def __init__(
        self,
        window_size: int = 100,
        default_ms_per_token: float = 0.003,
        min_observations: int = 10,
    ) -> None:
        self._obs: deque[tuple[int, float]] = deque(maxlen=window_size)
        self.default_ms_per_token: float = default_ms_per_token
        self._min_observations = min_observations

    def record(self, tokens: int, elapsed_ms: float) -> None:
        """Record a completed transfer.

        Args:
            tokens: Number of KV tokens transferred.
            elapsed_ms: Wall-clock duration of the transfer in milliseconds
                (from CUDA event ``elapsed_time``).
        """
        if tokens > 0 and elapsed_ms >= 0.0:
            self._obs.append((tokens, elapsed_ms))

    @property
    def ms_per_token(self) -> float:
        """Current estimate of milliseconds per token over the rolling window.

        Falls back to *default_ms_per_token* until enough samples exist.
        """
        if len(self._obs) < self._min_observations:
            return self.default_ms_per_token
        total_tokens = sum(t for t, _ in self._obs)
        total_ms = sum(ms for _, ms in self._obs)
        if total_tokens == 0:
            return self.default_ms_per_token
        return total_ms / total_tokens

    def estimate_ms(self, tokens: int) -> float:
        """Estimate expected transfer time for *tokens* KV tokens."""
        return tokens * self.ms_per_token

    def __len__(self) -> int:
        return len(self._obs)
