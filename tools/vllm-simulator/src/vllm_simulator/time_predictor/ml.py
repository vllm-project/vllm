"""ML-trained per-iteration latency predictor.

The predictor consumes the 18-feature sklearn bundle produced by the
Qwen3.7-Max baseline training pipeline.  It intentionally has no dependency on
the serving backend beyond the simulator's common ``ScheduleBatch`` interface.
"""

import math
import os
import typing
import warnings

import joblib
from sklearn.exceptions import InconsistentVersionWarning

from vllm_simulator.simulation.types import SchedulerConfig
from vllm_simulator.spec.accelerator import AcceleratorInfo
from vllm_simulator.time_predictor.base import InferTimePredictor, ScheduleBatch
from vllm_simulator.utils import get_logger

if typing.TYPE_CHECKING:
    from vllm.config import ModelConfig


logger = get_logger()


class MLTimePredictor(InferTimePredictor):
    """Predict a step latency from batch-composition features."""

    FEATURE_NAMES = [
        "batch_size",
        "sum_extend",
        "max_extend",
        "min_extend",
        "sum_past",
        "max_past",
        "min_past",
        "sum_extend_x_past",
        "sum_extend_squared",
        "sum_past_squared",
        "sum_attn_flops",
        "sum_extend_x_max_past",
        "log1p_sum_past",
        "log1p_sum_attn_flops",
        "batch_size_x_sum_extend",
        "max_past_minus_min_past",
        "is_decode",
        "is_prefill",
    ]

    def __init__(
        self,
        model: "ModelConfig",
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: str,
        latency_scale: float = 1.0,
        logprobs_cost_us_per_token: float = 0.0,
        sample_tokens_base_ms: float = 0.0,
        sample_tokens_lo_us_per_token: float = 0.0,
        sample_tokens_hi_us_per_token: float = 0.0,
        sample_tokens_breakpoint_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(model, hw, config)
        if not database_path or not os.path.exists(database_path):
            raise FileNotFoundError(
                f"MLTimePredictor database_path not found: {database_path}"
            )

        # A sklearn ensemble can emit one version warning per tree.  Preserve
        # the compatibility signal without flooding a 500-tree simulation log.
        with warnings.catch_warnings(record=True) as version_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            bundle = joblib.load(database_path)
        if version_warnings:
            logger.warning(
                "MLTimePredictor loaded a model produced by another sklearn "
                "version (%d compatibility warnings suppressed)",
                len(version_warnings),
            )
        if isinstance(bundle, dict) and "prefill" in bundle and "decode" in bundle:
            self._prefill_model = bundle["prefill"]
            self._decode_model = bundle["decode"]
            saved_features = bundle.get("features", self.FEATURE_NAMES)
        elif isinstance(bundle, dict) and "model" in bundle:
            self._prefill_model = bundle["model"]
            self._decode_model = bundle["model"]
            saved_features = bundle.get("features", self.FEATURE_NAMES)
        else:
            self._prefill_model = bundle
            self._decode_model = bundle
            saved_features = self.FEATURE_NAMES

        if list(saved_features) != self.FEATURE_NAMES:
            raise ValueError(
                "MLTimePredictor feature schema mismatch: "
                f"saved={list(saved_features)!r}, expected={self.FEATURE_NAMES!r}"
            )

        self._latency_scale = float(latency_scale)
        # Extra per-token cost of logprobs requests, for baselines collected
        # without logprobs.  Must be calibrated separately.
        self.logprobs_cost_s_per_token = float(logprobs_cost_us_per_token) / 1e6

        # RPC-2 (sample_tokens) compensation; the trained label covers RPC-1
        # only.  Hinge function of the step's scheduled tokens:
        #
        #   sample_tokens_s = base + a_lo * tt + (a_hi - a_lo) * max(0, tt - s)
        #
        # Kept separate from the label prediction; disabled when all terms
        # are 0.
        self.sample_tokens_base_s = float(sample_tokens_base_ms) / 1e3
        self.sample_tokens_lo_s_per_token = float(sample_tokens_lo_us_per_token) / 1e6
        self.sample_tokens_hi_s_per_token = float(sample_tokens_hi_us_per_token) / 1e6
        self.sample_tokens_breakpoint_tokens = max(
            0, int(sample_tokens_breakpoint_tokens)
        )
        self._sample_tokens_enabled = (
            self.sample_tokens_base_s > 0.0
            or self.sample_tokens_lo_s_per_token > 0.0
            or self.sample_tokens_hi_s_per_token > 0.0
        )
        logger.info(
            "MLTimePredictor loaded from %s (prefill=%s, decode=%s, "
            "n_features=%d, latency_scale=%.4f)",
            database_path,
            type(self._prefill_model).__name__,
            type(self._decode_model).__name__,
            len(saved_features),
            self._latency_scale,
        )
        if self._sample_tokens_enabled:
            logger.info(
                "MLTimePredictor sample_tokens compensation enabled: "
                "base=%.3fms a_lo=%.4fus/token a_hi=%.4fus/token "
                "breakpoint=%d tokens",
                self.sample_tokens_base_s * 1e3,
                self.sample_tokens_lo_s_per_token * 1e6,
                self.sample_tokens_hi_s_per_token * 1e6,
                self.sample_tokens_breakpoint_tokens,
            )

    @staticmethod
    def _extract_features(batch: ScheduleBatch) -> list[float]:
        exts = [req.extend_length for req in batch.reqs]
        pasts = [req.past_kv_length for req in batch.reqs]

        batch_size = len(exts)
        sum_extend = sum(exts)
        sum_past = sum(pasts)
        max_extend = max(exts)
        min_extend = min(exts)
        max_past = max(pasts)
        min_past = min(pasts)
        sum_attn_flops = sum(
            extend * (past + extend / 2) for extend, past in zip(exts, pasts)
        )

        return [
            batch_size,
            sum_extend,
            max_extend,
            min_extend,
            sum_past,
            max_past,
            min_past,
            sum(extend * past for extend, past in zip(exts, pasts)),
            sum(extend * extend for extend in exts),
            sum(past * past for past in pasts),
            sum_attn_flops,
            sum_extend * max_past,
            math.log1p(sum_past),
            math.log1p(sum_attn_flops),
            batch_size * sum_extend,
            max_past - min_past,
            int(all(extend == 1 for extend in exts)),
            int(any(extend > 1 for extend in exts)),
        ]

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0

        features = self._extract_features(batch)
        is_decode = features[-2] == 1
        model = self._decode_model if is_decode else self._prefill_model
        latency = float(model.predict([features])[0]) * self._latency_scale
        return max(latency, 0.0)

    def predict_sample_tokens_time(self, total_tokens: int) -> float:
        """Return the RPC-2 (sample_tokens) duration in seconds.

        ``total_tokens`` is this step's scheduled token count, matching the
        ``total_tokens`` field written by the GPU-side collection hook.
        Returns 0.0 when the compensation is not configured.

        The calibration was fitted on mixed (chunked-prefill) steps only, so
        applying it to a hypothetical decode-only deployment extrapolates the
        token-independent floor beyond its measured domain.
        """
        if not self._sample_tokens_enabled or total_tokens <= 0:
            return 0.0

        tt = float(total_tokens)
        latency = self.sample_tokens_base_s + self.sample_tokens_lo_s_per_token * tt
        if tt > self.sample_tokens_breakpoint_tokens:
            latency += (
                self.sample_tokens_hi_s_per_token - self.sample_tokens_lo_s_per_token
            ) * (tt - self.sample_tokens_breakpoint_tokens)
        return max(latency, 0.0)
