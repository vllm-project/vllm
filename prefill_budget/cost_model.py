"""Quadratic prefill step-time model and the chunk policy that uses it.

cost(q, c) = a q^2 + b q c + g q + d

q is the new tokens in the chunk. c is the prefix already computed.
q^2 is attention inside the chunk, q c is attention against the prefix,
g q is the token-linear GEMMs, and d is fixed overhead.

Coefficients are fit in a scaled basis so least squares stays stable:
u = q / SCALE, v = c / SCALE, cost = a u^2 + b u v + g u + d.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCALE = 1000.0
WARMUP_CHUNKS = (128, 256, 512, 1024, 1536, 768, 384, 2048)


@dataclass
class CostFit:
    a: float
    b: float
    g: float
    d: float
    n: int

    def predict_ms(self, q: int, c: int) -> float:
        u = q / SCALE
        v = c / SCALE
        return float(self.a * u * u + self.b * u * v + self.g * u + self.d)

    def solve_q(self, c: int, budget_ms: float, q_min: int, q_max: int) -> int:
        """Largest q in [q_min, q_max] whose predicted cost stays within budget."""
        if q_max < q_min:
            return q_min
        if self.predict_ms(q_min, c) > budget_ms:
            return q_min
        if self.predict_ms(q_max, c) <= budget_ms:
            return q_max
        lo, hi = q_min, q_max
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.predict_ms(mid, c) <= budget_ms:
                lo = mid
            else:
                hi = mid - 1
        return lo


def fit_cost(samples: list[tuple[int, int, float]]) -> CostFit | None:
    """Fit non-negative coefficients. samples are (q, c, elapsed_ms)."""
    if len(samples) < 4:
        return None
    q = np.asarray([s[0] for s in samples], dtype=np.float64)
    c = np.asarray([s[1] for s in samples], dtype=np.float64)
    t = np.asarray([s[2] for s in samples], dtype=np.float64)
    u = q / SCALE
    v = c / SCALE
    columns = np.column_stack([u * u, u * v, u, np.ones_like(u)])
    coef = _nnls_columns(columns, t)
    if coef is None:
        return None
    return CostFit(
        a=float(coef[0]),
        b=float(coef[1]),
        g=float(coef[2]),
        d=float(coef[3]),
        n=len(samples),
    )


def _nnls_columns(columns: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Least squares, dropping any column that comes out negative and refitting."""
    active = [0, 1, 2, 3]
    for _ in range(4):
        mat = columns[:, active]
        coef, *_ = np.linalg.lstsq(mat, target, rcond=None)
        if np.all(coef >= -1e-8):
            full = np.zeros(4, dtype=np.float64)
            full[active] = np.clip(coef, 0.0, None)
            return full
        worst = int(np.argmin(coef))
        del active[worst]
        if not active:
            return None
    return None


def align_tokens(tokens: int, block_size: int) -> int:
    """Round a chunk down to a whole KV block. A short tail stays as-is."""
    if tokens < block_size:
        return max(tokens, 1)
    return max(block_size, tokens // block_size * block_size)


class ChunkPolicy:
    """Choose the next prefill chunk from a time budget.

    The first `warmup_steps` chunks cycle through fixed sizes so the fit sees
    both short and long chunks. After that, and once at least four usable
    samples exist, the chunk is the largest q whose predicted step time is
    within the budget. Fixed mode never fits; it always returns `fixed_chunk`.
    """

    def __init__(
        self,
        *,
        mode: str,
        budget_ms: float,
        fixed_chunk: int,
        min_chunk: int,
        max_chunk: int,
        warmup_steps: int,
        block_size: int,
    ) -> None:
        if mode not in ("budget", "fixed"):
            raise ValueError("mode must be 'budget' or 'fixed'")
        if min_chunk > max_chunk:
            raise ValueError("min_chunk cannot exceed max_chunk")
        self.mode = mode
        self.budget_ms = budget_ms
        self.fixed_chunk = fixed_chunk
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.block_size = block_size
        self._warmup_left = warmup_steps
        self._warmup_i = 0
        self._samples: list[tuple[int, int, float]] = []
        self.fit: CostFit | None = None

    def cap(self, prefix: int, remaining: int) -> tuple[int, str]:
        if remaining <= 0:
            raise ValueError("remaining must be positive")
        if self.mode == "fixed":
            return align_tokens(min(self.fixed_chunk, remaining), self.block_size), "fixed"
        if self.fit is None or self._warmup_left > 0:
            raw = WARMUP_CHUNKS[self._warmup_i % len(WARMUP_CHUNKS)]
            self._warmup_i += 1
            if self._warmup_left > 0:
                self._warmup_left -= 1
            return align_tokens(min(raw, remaining), self.block_size), "warmup"
        limit = min(self.max_chunk, remaining)
        solved = self.fit.solve_q(prefix, self.budget_ms, self.min_chunk, limit)
        return align_tokens(min(solved, remaining), self.block_size), "budget"

    def observe(self, q: int, c: int, elapsed_ms: float) -> bool:
        """Record one finished prefill step.

        Return whether the sample is a usable timing. Fixed mode still reports
        that, but it does not fit a model from it.
        """
        usable = 1.0 <= elapsed_ms <= 2000.0 and q >= 64
        if not usable or self.mode != "budget":
            return usable
        self._samples.append((q, c, elapsed_ms))
        if len(self._samples) > 32:
            self._samples = self._samples[-32:]
        fitted = fit_cost(self._samples)
        if fitted is not None:
            self.fit = fitted
        return True
