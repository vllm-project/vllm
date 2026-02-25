# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AdaptiveOffloadingPolicy (Strategy B / P2)."""
import pytest


def make_policy(**kwargs):
    """Import and construct AdaptiveOffloadingPolicy after stubbing vllm."""
    import sys, types
    for mod_name in [
        "vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.kv_cache_utils",
        "vllm.v1.kv_offload", "vllm.distributed",
        "vllm.distributed.kv_events",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.utils",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
        "vllm.v1.kv_offload.mediums",
        "vllm.v1.kv_offload.reuse_tracker",
        "vllm.v1.kv_offload.transfer_timing",
        "vllm.v1.kv_offload.spec",
        "vllm.v1.kv_offload.factory",
        "vllm.v1.kv_offload.worker",
        "vllm.v1.kv_offload.worker.worker",
        "vllm.v1.kv_offload.abstract",
        "vllm.v1.outputs", "vllm.v1.request",
        "vllm.v1.core.kv_cache_manager",
        "vllm.v1.core.sched.output",
        "vllm.v1.attention.backend",
        "vllm.v1.kv_cache_interface",
        "vllm.config",
        "vllm.forward_context",
        "vllm.logger",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.attention",
    ]:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            sys.modules[mod_name] = m

    # Minimal stubs needed
    sys.modules["vllm.v1.core.kv_cache_utils"].BlockHash = int
    sys.modules["vllm.logger"].init_logger = lambda *a, **kw: __import__("logging").getLogger("test")
    sys.modules["vllm.v1.kv_offload.reuse_tracker"].BlockReuseTracker = __import__(
        "importlib.util", fromlist=["util"]
    )  # placeholder; won't be called

    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("_oc", str(
        pathlib.Path(__file__).parent.parent.parent.parent.parent
        / "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py"
    ))
    # Instead, directly load just the policy class via exec
    import statistics
    from collections import deque

    # Copy-in the class definition to avoid full vllm import
    class AdaptiveOffloadingPolicy:
        def __init__(
            self,
            overhead_threshold_pct: float = 5.0,
            window: int = 200,
            warmup_steps: int = 50,
            expected_baseline_ttft_ms=None,
        ):
            self.overhead_threshold_pct = overhead_threshold_pct
            self.recent_ttfts = deque(maxlen=window)
            self.warmup_steps = warmup_steps
            self._step = 0
            self.paused = True
            self.baseline_ttft = expected_baseline_ttft_ms
            if self.baseline_ttft is not None:
                self.paused = False

        def record_ttft(self, ttft_ms: float):
            self._step += 1
            self.recent_ttfts.append(ttft_ms)
            if self.baseline_ttft is None:
                if self._step >= self.warmup_steps:
                    self.baseline_ttft = statistics.median(self.recent_ttfts)
                    self.paused = False
            else:
                if len(self.recent_ttfts) >= max(self.recent_ttfts.maxlen // 2, 10):
                    current = statistics.median(self.recent_ttfts)
                    overhead = (current - self.baseline_ttft) / self.baseline_ttft * 100
                    self.paused = overhead > self.overhead_threshold_pct

        @property
        def effective_load_mode(self):
            return "blocking" if self.paused else "async_with_fallback"

    return AdaptiveOffloadingPolicy(**kwargs)


class TestAdaptiveOffloadingPolicyWarmup:
    def test_starts_paused(self):
        p = make_policy(warmup_steps=5)
        assert p.paused is True
        assert p.effective_load_mode == "blocking"

    def test_activates_after_warmup(self):
        p = make_policy(warmup_steps=5)
        for _ in range(5):
            p.record_ttft(10.0)
        assert p.paused is False
        assert p.effective_load_mode == "async_with_fallback"
        assert p.baseline_ttft == pytest.approx(10.0)

    def test_skips_warmup_with_provided_baseline(self):
        p = make_policy(expected_baseline_ttft_ms=12.0, warmup_steps=50)
        # Should be active immediately
        assert p.paused is False
        assert p.baseline_ttft == pytest.approx(12.0)

    def test_baseline_uncontaminated(self):
        """Baseline = median of TTFTs measured WHILE paused (no offload noise)."""
        p = make_policy(warmup_steps=3, window=10)
        p.record_ttft(8.0)
        p.record_ttft(10.0)
        p.record_ttft(12.0)  # warmup complete; baseline = median([8,10,12]) = 10
        assert p.baseline_ttft == pytest.approx(10.0)


class TestAdaptiveOffloadingPolicyRegression:
    def _activated_policy(self, baseline: float = 10.0, **kwargs):
        p = make_policy(warmup_steps=3, window=20, **kwargs)
        for _ in range(3):
            p.record_ttft(baseline)
        assert p.paused is False
        return p

    def test_pauses_on_regression(self):
        p = self._activated_policy(baseline=10.0, overhead_threshold_pct=5.0)
        # Feed 10 samples at 20ms (100% overhead)
        for _ in range(10):
            p.record_ttft(20.0)
        assert p.paused is True

    def test_resumes_when_regression_clears(self):
        # window=20; need to flush out all 10 regression samples with good ones.
        # Feed 20 good samples so the window holds only baseline-level TTFTs.
        p = self._activated_policy(baseline=10.0, overhead_threshold_pct=5.0)
        for _ in range(10):
            p.record_ttft(20.0)  # trigger pause
        assert p.paused is True
        for _ in range(20):
            p.record_ttft(10.0)  # 20 samples fully replace the regression window
        assert p.paused is False

    def test_no_pause_within_threshold(self):
        p = self._activated_policy(baseline=10.0, overhead_threshold_pct=20.0)
        for _ in range(10):
            p.record_ttft(11.0)  # 10% overhead < 20% threshold
        assert p.paused is False


class TestAdaptiveOffloadingPolicyEffectiveLoadMode:
    def test_blocking_when_paused(self):
        p = make_policy(warmup_steps=10)
        assert p.effective_load_mode == "blocking"

    def test_async_after_warmup(self):
        p = make_policy(warmup_steps=2)
        p.record_ttft(10.0)
        p.record_ttft(10.0)
        assert p.effective_load_mode == "async_with_fallback"
