# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Profiling-based Dynamic Chunk Size Predictor.

This module implements a dynamic chunk sizing strategy based on profiling prefill
latency and fitting a quadratic model.

The approach:
1. Profile: Run forward passes with different chunk sizes to measure latency
2. Fit: Use quadratic model f(l) = a*l^2 + b*l + c to fit the latency data
3. Predict: Given current num_computed_tokens, solve for chunk size that achieves
   target latency
"""

import math

import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)


class ChunkSizePredictor:
    """Predictor for dynamic chunk size based on quadratic latency model.

    Models latency as: f(l) = a*l^2 + b*l + c

    Given a target latency T and current history length L, predicts next
    chunk size x such that: f(L+x) - f(L) = T

    This expands to the quadratic equation: a*x^2 + (2aL+b)*x - T = 0
    """

    def __init__(self, smooth_factor: float = 0.8, min_chunk: int = 4096):
        self.quadratic_coeff_a: float = 0.0
        self.linear_coeff_b: float = 0.0
        self.constant_coeff_c: float = 0.0

        self.quadratic_chunk_a: float = 0.0
        self.linear_chunk_b: float = 0.0
        self.constant_chunk_c: float = 0.0

        self.target_latency: float | None = None
        self.is_ready: bool = False
        self.with_history_ready: bool = False
        self.smooth_factor = smooth_factor
        self.min_chunk = min_chunk
        self.history_fitted = False

    def clamp_quadratic_and_linear_if_negative(
        self, fitted_a: float, fitted_b: float
    ) -> tuple[float, float]:
        """In theory, for the Transformer structure of LLM, the fitted quadratic
        and linear terms should not be negative. Can perform zero clamping for
        inaccurate fitting.
        """
        if fitted_a < 0:
            logger.warning(
                "Fitted a=%.2e is not positive. Setting a=1e-9.", fitted_a
            )
            fitted_a = 1e-9
        if fitted_b < 0:
            logger.warning(
                "Fitted b=%.2e is not positive. Setting b=0.0.", fitted_b
            )
            fitted_b = 1e-9

        return fitted_a, fitted_b

    def fit(self, seq_lens: list[int], latencies: list[float]) -> bool:
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points.

        Returns:
            True if fitting succeeded, False otherwise
        """
        L = np.array(seq_lens, dtype=np.float64)
        T = np.array(latencies, dtype=np.float64)
        MIN_FIT_POINTS_NO_CHUNK = 8

        if len(L) < MIN_FIT_POINTS_NO_CHUNK:
            logger.warning(
                "Not enough data points for quadratic fitting (%d < 8)",
                len(L),
            )
            return False

        X = np.column_stack([L * L, L, np.ones_like(L)])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, T, rcond=None)
            fitted_a = float(coeffs[0])
            fitted_b = float(coeffs[1])
            fitted_c = float(coeffs[2])
        except Exception as e:
            try:
                poly = np.polyfit(L, T, 2)
                fitted_a = float(poly[0])
                fitted_b = float(poly[1])
                fitted_c = float(poly[2])
                logger.warning(
                    "Least-squares fitting failed (%s), "
                    "fallback to polyfit succeeded.",
                    e,
                )
            except Exception as fallback_error:
                logger.warning(
                    "Failed to fit quadratic model: %s", fallback_error
                )
                return False

        fitted_a, fitted_b = self.clamp_quadratic_and_linear_if_negative(
            fitted_a, fitted_b
        )

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            "[ProfilingChunk] Fitted: a=%.2e, b=%.2e, c=%.2e",
            fitted_a,
            fitted_b,
            fitted_c,
        )
        return True

    def fit_chunk(self, chunked_data: list) -> bool:
        """Fit time with chunks: f(C,H) = a*C(C+H) + b*C + c*H.

        Returns:
            True if fitting succeeded, False otherwise
        """
        num_points = len(chunked_data)
        MIN_FIT_POINTS_CHUNK = 5
        MAX_FIT_POINTS_CHUNK = 30
        if num_points < MIN_FIT_POINTS_CHUNK:
            logger.warning(
                "Not enough data points for chunked data fitting (%d < 5)",
                num_points,
            )
            return False
        if num_points > MAX_FIT_POINTS_CHUNK:
            self.history_fitted = True
            return False

        chunked_data_array = np.array(chunked_data)
        execute_time = chunked_data_array[:, -1]
        input_x = chunked_data_array[:, :-1]

        try:
            params, _, _, _ = np.linalg.lstsq(input_x, execute_time, rcond=None)
            fitted_a = float(params[0])
            fitted_b = float(params[1])
            fitted_c = float(params[2])
        except np.linalg.LinAlgError as e:
            logger.warning("Failed to fit chunked model: %s", e)
            return False

        fitted_a, fitted_b = self.clamp_quadratic_and_linear_if_negative(
            fitted_a, fitted_b
        )

        self.quadratic_chunk_a = fitted_a
        self.linear_chunk_b = fitted_b
        self.constant_chunk_c = fitted_c

        logger.info(
            "[ProfilingChunk With History] Fitted: a=%.2e, b=%.2e, c=%.2e",
            fitted_a,
            fitted_b,
            fitted_c,
        )
        return True

    def set_target_latency(
        self, base_chunk_size: int, elapsed_time: float = 0.0
    ) -> None:
        """Set target latency based on base chunk size."""

        def f(seq_lens: float) -> float:
            return (
                self.quadratic_coeff_a * seq_lens * seq_lens
                + self.linear_coeff_b * seq_lens
                + self.constant_coeff_c
            )

        if elapsed_time > 0:
            self.target_latency = elapsed_time
        else:
            self.target_latency = f(float(base_chunk_size)) - f(0.0)
        if self.target_latency <= 0:
            self.target_latency = 1.0

        logger.info(
            "[ProfilingChunk] Target latency: %.2f ms (base_chunk=%d)",
            self.target_latency,
            base_chunk_size,
        )

    def get_time(
        self,
        query_len: int,
        num_computed_tokens: int,
    ) -> float:
        """Get time T based on current seq_lens,
        f(l) = al^2 + bl + c, f(L+x) - f(L) = T"""

        def f(seq_lens: float) -> float:
            return (
                self.quadratic_coeff_a * seq_lens * seq_lens
                + self.linear_coeff_b * seq_lens
                + self.constant_coeff_c
            )

        return f(query_len + num_computed_tokens) - f(num_computed_tokens)

    def get_time_with_history(
        self,
        query_len: int,
        num_computed_tokens: int,
    ) -> float:
        """Get time T based on current seq_lens,
        f(C,H) = a*C(C+H) + b*(C+H) + c = T"""
        return (
            self.quadratic_chunk_a
            * query_len
            * (query_len + num_computed_tokens)
            + self.linear_chunk_b * (query_len + num_computed_tokens)
            + self.constant_chunk_c
        )

    def predict(
        self,
        num_computed_tokens: int,
        base_chunk_size: int,
        page_size: int,
    ) -> int | None:
        """Predict next chunk size x such that
        f(L+x) - f(L) = target_latency."""
        if not self.is_ready or self.target_latency is None:
            return None

        if self.quadratic_coeff_a <= 0:
            return None

        A = self.quadratic_coeff_a
        B = 2 * self.quadratic_coeff_a * num_computed_tokens + self.linear_coeff_b
        C = -self.target_latency

        discriminant = B * B - 4 * A * C
        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        x = (-B + sqrt_disc) / (2 * A)

        if x <= 0:
            return None

        smoothed = base_chunk_size + self.smooth_factor * (x - base_chunk_size)
        chunk_size = max(int(smoothed), self.min_chunk)

        align = max(page_size, 64)
        chunk_size = ((chunk_size + align - 1) // align) * align
        if chunk_size < align:
            chunk_size = align

        logger.debug("[ProfilingChunk] Predicted chunk_size=%d", chunk_size)
        return chunk_size if chunk_size >= align else None

    def predict_with_history(
        self,
        num_computed_tokens: int,
        base_chunk_size: int,
        page_size: int,
    ) -> int | None:
        """Predict next chunk size x using the history-aware model
        f(C,H) = a*C(C+H) + b*C + c*H."""
        if not self.is_ready or self.target_latency is None:
            return None

        if not self.with_history_ready:
            return None

        if self.quadratic_chunk_a <= 0:
            return None

        # a*C^2 + (a*H + b)*C + b*H + c - T = 0
        A = self.quadratic_chunk_a
        B = self.quadratic_chunk_a * num_computed_tokens + self.linear_chunk_b
        C = (
            self.linear_chunk_b * num_computed_tokens
            + self.constant_chunk_c
            - self.target_latency
        )

        discriminant = B * B - 4 * A * C
        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        x = (-B + sqrt_disc) / (2 * A)

        if x <= 0:
            return None

        logger.debug(
            "[ProfilingChunk] History-aware raw prediction: %.1f", x
        )
        smoothed = base_chunk_size + self.smooth_factor * (x - base_chunk_size)
        chunk_size = max(int(smoothed), self.min_chunk)

        align = max(page_size, 64)
        chunk_size = ((chunk_size + align - 1) // align) * align
        if chunk_size < align:
            chunk_size = align

        return chunk_size if chunk_size >= align else None


class ProfilingChunkManager:
    """Manager for profiling-based dynamic chunk sizing.

    Handles the profiling process and maintains the ChunkSizePredictor.
    """

    def __init__(
        self,
        base_chunk_size: int,
        page_size: int,
        smooth_factor: float = 0.8,
        min_chunk: int = 4096,
    ):
        self.base_chunk_size = base_chunk_size
        self.page_size = page_size
        self.chunked_fit_data: list = []

        self.predictor = ChunkSizePredictor(
            smooth_factor=smooth_factor, min_chunk=min_chunk
        )
        self._profiling_done = False
        self._set_time_done = False

    @property
    def is_ready(self) -> bool:
        return self._profiling_done and self.predictor.is_ready

    @property
    def history_ready(self) -> bool:
        return self.is_ready and self.predictor.with_history_ready

    def predict_chunk_size(
        self, num_computed_tokens: int, target_time: float
    ) -> int | None:
        """Predict optimal chunk size for given history length."""
        if not self.is_ready:
            return None

        self.predictor.target_latency = target_time

        if not self.history_ready:
            predict_func = self.predictor.predict
        else:
            predict_func = self.predictor.predict_with_history
        return predict_func(
            num_computed_tokens=num_computed_tokens,
            base_chunk_size=self.base_chunk_size,
            page_size=self.page_size,
        )

    def predict_time(
        self, num_new_tokens: int, num_computed_tokens: int
    ) -> float:
        """Get the consumed time of scheduled reqs for time_budget."""
        if not self.is_ready:
            return 0.0

        if not self.history_ready:
            predict_func = self.predictor.get_time
        else:
            predict_func = self.predictor.get_time_with_history
        return predict_func(
            query_len=num_new_tokens,
            num_computed_tokens=num_computed_tokens,
        )

    def record_batch_execution_time(
        self, request_chunks: list, elapsed_time: float
    ) -> bool:
        """Record batch execution time for online model refinement.

        Accumulates (x1, x2, x3, time_ms) data points and re-fits the
        history-aware model once enough points are collected.

        Args:
            request_chunks: List of (chunk_size, num_computed_tokens) per request
            elapsed_time: Total elapsed time in seconds
        """
        x1 = x2 = x3 = 0
        for chunk, hist in request_chunks:
            x1 += (chunk + hist) * chunk
            x2 += chunk + hist
            x3 += 1
        self.chunked_fit_data.append([x1, x2, x3, elapsed_time * 1000])
        if not self.predictor.fit_chunk(self.chunked_fit_data):
            return False

        self.predictor.with_history_ready = True
        return True
