# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for compile_factors() and compute_hash() across config classes.

These tests verify the refactoring from PR #43527:
  - compile_factors() returns a dict[str, object] of cache-affecting fields
  - compute_hash() is a thin wrapper: hash_factors(self.compile_factors())
  - changing a compile-affecting field changes the hash
  - compile_factors() is composable without triggering double-hashing
"""
import pytest

from vllm.config import CompilationConfig, ParallelConfig, SchedulerConfig
from vllm.config.compilation import DynamicShapesConfig, PassConfig
from vllm.config.offload import OffloadConfig
from vllm.config.utils import hash_factors


# ---------------------------------------------------------------------------
# OffloadConfig
# ---------------------------------------------------------------------------


class TestOffloadConfigHash:
    def test_compile_factors_returns_dict(self):
        cfg = OffloadConfig()
        factors = cfg.compile_factors()
        assert isinstance(factors, dict)
        assert len(factors) > 0

    def test_compute_hash_is_thin_wrapper(self):
        cfg = OffloadConfig()
        assert cfg.compute_hash() == hash_factors(cfg.compile_factors())

    def test_compute_hash_returns_str(self):
        cfg = OffloadConfig()
        result = cfg.compute_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_same_config_same_hash(self):
        cfg1 = OffloadConfig()
        cfg2 = OffloadConfig()
        assert cfg1.compute_hash() == cfg2.compute_hash()

    def test_compile_factors_are_stable(self):
        """compile_factors() must be deterministic across repeated calls."""
        cfg = OffloadConfig()
        assert cfg.compile_factors() == cfg.compile_factors()
        assert cfg.compute_hash() == cfg.compute_hash()


# ---------------------------------------------------------------------------
# PassConfig
# ---------------------------------------------------------------------------


class TestPassConfigHash:
    def test_compile_factors_returns_dict(self):
        cfg = PassConfig()
        factors = cfg.compile_factors()
        assert isinstance(factors, dict)
        assert len(factors) > 0

    def test_compute_hash_is_thin_wrapper(self):
        cfg = PassConfig()
        assert cfg.compute_hash() == hash_factors(cfg.compile_factors())

    def test_compute_hash_returns_str(self):
        assert isinstance(PassConfig().compute_hash(), str)

    def test_same_config_same_hash(self):
        assert PassConfig().compute_hash() == PassConfig().compute_hash()

    def test_changed_field_changes_hash(self):
        base = PassConfig()
        modified = PassConfig(enable_noop=not base.enable_noop)
        assert base.compile_factors() != modified.compile_factors()
        assert base.compute_hash() != modified.compute_hash()

    def test_compile_factors_stable(self):
        cfg = PassConfig()
        assert cfg.compile_factors() == cfg.compile_factors()


# ---------------------------------------------------------------------------
# DynamicShapesConfig
# ---------------------------------------------------------------------------


class TestDynamicShapesConfigHash:
    def test_compile_factors_returns_dict(self):
        cfg = DynamicShapesConfig()
        factors = cfg.compile_factors()
        assert isinstance(factors, dict)

    def test_compute_hash_is_thin_wrapper(self):
        cfg = DynamicShapesConfig()
        assert cfg.compute_hash() == hash_factors(cfg.compile_factors())

    def test_compute_hash_returns_str(self):
        assert isinstance(DynamicShapesConfig().compute_hash(), str)

    def test_same_config_same_hash(self):
        assert (
            DynamicShapesConfig().compute_hash()
            == DynamicShapesConfig().compute_hash()
        )


# ---------------------------------------------------------------------------
# CompilationConfig
# ---------------------------------------------------------------------------


class TestCompilationConfigHash:
    def test_compile_factors_returns_dict(self):
        cfg = CompilationConfig()
        factors = cfg.compile_factors()
        assert isinstance(factors, dict)
        assert len(factors) > 0

    def test_compute_hash_is_thin_wrapper(self):
        cfg = CompilationConfig()
        assert cfg.compute_hash() == hash_factors(cfg.compile_factors())

    def test_compute_hash_returns_str(self):
        assert isinstance(CompilationConfig().compute_hash(), str)

    def test_same_config_same_hash(self):
        assert CompilationConfig().compute_hash() == CompilationConfig().compute_hash()

    def test_compile_factors_contains_sub_config_factors(self):
        """
        CompilationConfig.compile_factors() must incorporate its sub-configs
        (PassConfig, DynamicShapesConfig) so changes in them change the hash.
        """
        base = CompilationConfig()
        modified = CompilationConfig(
            pass_config=PassConfig(enable_noop=not PassConfig().enable_noop)
        )
        assert base.compile_factors() != modified.compile_factors()
        assert base.compute_hash() != modified.compute_hash()

    def test_compile_factors_stable(self):
        cfg = CompilationConfig()
        assert cfg.compile_factors() == cfg.compile_factors()

    def test_composable_without_double_hashing(self):
        """
        compile_factors() should return raw factor data, NOT a nested hash.
        Callers must be able to inspect values — not just compare opaque hashes.
        """
        cfg = CompilationConfig()
        factors = cfg.compile_factors()
        # Values should be primitive types or dicts/lists, not str hex hashes
        for key, value in factors.items():
            assert not (
                isinstance(value, str) and len(value) == 64
            ), (
                f"Factor '{key}' looks like a SHA-256 hash string — "
                "compile_factors() should return raw data, not pre-hashed values"
            )


# ---------------------------------------------------------------------------
# SchedulerConfig
# ---------------------------------------------------------------------------


class TestSchedulerConfigHash:
    def _make(self, **kwargs):
        defaults = dict(
            max_num_seqs=128,
            max_num_batched_tokens=2048,
            max_model_len=2048,
        )
        defaults.update(kwargs)
        return SchedulerConfig(**defaults)

    def test_compile_factors_returns_dict(self):
        factors = self._make().compile_factors()
        assert isinstance(factors, dict)
        assert "max_num_batched_tokens" in factors

    def test_compute_hash_is_thin_wrapper(self):
        cfg = self._make()
        assert cfg.compute_hash() == hash_factors(cfg.compile_factors())

    def test_compute_hash_returns_str(self):
        assert isinstance(self._make().compute_hash(), str)

    def test_same_config_same_hash(self):
        assert self._make().compute_hash() == self._make().compute_hash()

    def test_max_num_batched_tokens_affects_hash(self):
        """max_num_batched_tokens directly affects graph shape → must differ."""
        cfg_a = self._make(max_num_batched_tokens=1024)
        cfg_b = self._make(max_num_batched_tokens=2048)
        assert cfg_a.compile_factors() != cfg_b.compile_factors()
        assert cfg_a.compute_hash() != cfg_b.compute_hash()

    def test_max_num_batched_tokens_is_only_factor(self):
        """SchedulerConfig only includes max_num_batched_tokens as a factor."""
        cfg = self._make()
        factors = cfg.compile_factors()
        assert list(factors.keys()) == ["max_num_batched_tokens"]


# ---------------------------------------------------------------------------
# ParallelConfig
# ---------------------------------------------------------------------------


class TestParallelConfigHash:
    def test_compile_factors_returns_dict(self):
        cfg = ParallelConfig()
        factors = cfg.compile_factors()
        assert isinstance(factors, dict)
        assert len(factors) > 0

    def test_compute_hash_is_thin_wrapper(self):
        cfg = ParallelConfig()
        assert cfg.compute_hash() == hash_factors(cfg.compile_factors())

    def test_compute_hash_returns_str(self):
        result = ParallelConfig().compute_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_same_config_same_hash(self):
        assert ParallelConfig().compute_hash() == ParallelConfig().compute_hash()

    def test_tp_size_affects_hash(self):
        """tensor_parallel_size is a graph-structural field."""
        cfg_tp1 = ParallelConfig(tensor_parallel_size=1)
        cfg_tp2 = ParallelConfig(tensor_parallel_size=2)
        assert cfg_tp1.compile_factors() != cfg_tp2.compile_factors()
        assert cfg_tp1.compute_hash() != cfg_tp2.compute_hash()

    def test_compile_factors_excludes_ignored_fields(self):
        """
        Fields in ParallelConfig.ignored_factors (e.g. numa_bind_nodes)
        must NOT appear in compile_factors().
        """
        cfg = ParallelConfig()
        factors = cfg.compile_factors()
        ignored = {"numa_bind_nodes", "numa_bind_cpus"}
        for field in ignored:
            assert field not in factors, (
                f"Ignored field '{field}' should not be in compile_factors()"
            )

    def test_compile_factors_stable(self):
        cfg = ParallelConfig()
        assert cfg.compile_factors() == cfg.compile_factors()


# ---------------------------------------------------------------------------
# Cross-config: SupportsHash protocol compliance
# ---------------------------------------------------------------------------


class TestSupportsHashProtocol:
    """
    Verify all refactored configs satisfy the SupportsHash contract:
      - compile_factors() -> dict[str, object]
      - compute_hash() -> str
    """

    @pytest.mark.parametrize(
        "cfg",
        [
            OffloadConfig(),
            PassConfig(),
            DynamicShapesConfig(),
            CompilationConfig(),
            ParallelConfig(),
        ],
    )
    def test_compile_factors_type(self, cfg):
        factors = cfg.compile_factors()
        assert isinstance(factors, dict), (
            f"{type(cfg).__name__}.compile_factors() must return dict, "
            f"got {type(factors).__name__}"
        )

    @pytest.mark.parametrize(
        "cfg",
        [
            OffloadConfig(),
            PassConfig(),
            DynamicShapesConfig(),
            CompilationConfig(),
            ParallelConfig(),
        ],
    )
    def test_compute_hash_type(self, cfg):
        h = cfg.compute_hash()
        assert isinstance(h, str), (
            f"{type(cfg).__name__}.compute_hash() must return str, "
            f"got {type(h).__name__}"
        )
        assert len(h) > 0

    @pytest.mark.parametrize(
        "cfg",
        [
            OffloadConfig(),
            PassConfig(),
            DynamicShapesConfig(),
            CompilationConfig(),
            ParallelConfig(),
        ],
    )
    def test_hash_equals_hash_of_factors(self, cfg):
        """Core invariant: compute_hash() == hash_factors(compile_factors())."""
        assert cfg.compute_hash() == hash_factors(cfg.compile_factors()), (
            f"{type(cfg).__name__}: compute_hash() is not a thin wrapper "
            "over hash_factors(compile_factors())"
        )
