# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for steering model-runner mixin methods using a mock model.

All hook-point-aware tests use the default hook point
(``post_mlp``) unless testing multi-hook-point behaviour.

Tests cover three-tier steering (base, prefill, decode) and co-located
scale format (bare list vs dict with scale).
"""

import math

import pytest
import torch
import torch.nn as nn

from vllm.exceptions import SteeringVectorError
from vllm.model_executor.layers.steering import DEFAULT_HOOK_POINT
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin
from vllm.v1.worker.worker_base import WorkerBase

# Shorthand for test readability
_HP = DEFAULT_HOOK_POINT.value  # "post_mlp"


class FakeDecoderLayer(nn.Module):
    """Minimal decoder layer with per-hook-point steering buffers."""

    def __init__(self, layer_idx: int, hidden_size: int, max_steering_configs: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        # Default hook point buffers (post_mlp) — table + index only
        self.register_buffer(
            "steering_table_post_mlp",
            torch.zeros(max_steering_configs + 2, hidden_size),
            persistent=False,
        )
        self.register_buffer(
            "steering_index",
            torch.zeros(16, dtype=torch.long),  # small for testing
            persistent=False,
        )


class FakeModel(nn.Module):
    """Model with a few steerable decoder layers."""

    def __init__(
        self,
        num_layers: int = 4,
        hidden_size: int = 8,
        max_steering_configs: int = 0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FakeDecoderLayer(i, hidden_size, max_steering_configs)
                for i in range(num_layers)
            ]
        )
        self.hidden_size = hidden_size
        # Share steering_index across layers (matches real model behaviour)
        if self.layers:
            shared_index = self.layers[0].steering_index
            for layer in self.layers[1:]:
                layer.steering_index = shared_index


class FakeModelRunner(SteeringModelRunnerMixin):
    """Minimal model runner mixing in the steering methods."""

    def __init__(self, model: nn.Module):
        self._model = model
        self._steering_manager: SteeringManager | None = None
        self._steerable_layers_cache = None

    def get_model(self) -> nn.Module:
        return self._model


class FakeWorker(WorkerBase):
    """Concrete WorkerBase for testing (abstract methods stubbed).

    Delegates the four public steering methods to ``self.model_runner``
    the same way ``gpu_worker.py`` does, while still tolerating a
    ``None`` model runner (so the legacy "no model runner" tests keep
    exercising the graceful-degradation path).
    """

    def __init__(self, model: nn.Module):
        # Don't call super().__init__ — just set model_runner directly
        self.model_runner: FakeModelRunner | None = FakeModelRunner(model)  # type: ignore[assignment]

    def init_device(self):
        pass

    def get_model(self):
        assert self.model_runner is not None
        return self.model_runner.get_model()

    def set_steering_vectors(self, **kwargs):
        if self.model_runner is None:
            return (0, 0, [])
        return self.model_runner.set_steering_vectors(**kwargs)

    def clear_steering_vectors(self):
        if self.model_runner is None:
            return
        return self.model_runner.clear_steering_vectors()

    def list_steerable_layers(self):
        if self.model_runner is None:
            return {}
        return self.model_runner.list_steerable_layers()

    def get_steering_status(self):
        if self.model_runner is None:
            return {}
        return self.model_runner.get_steering_status()

    def _steerable_layers(self):
        if self.model_runner is None:
            return {}
        return self.model_runner._steerable_layers()


@pytest.fixture
def model():
    return FakeModel(num_layers=4, hidden_size=8)


@pytest.fixture
def worker(model):
    return FakeWorker(model)


@pytest.fixture
def worker_with_manager(model):
    """Worker whose FakeModelRunner has a live SteeringManager.

    Mirrors the post-eager-init runtime state where
    ``_init_steering_state`` has already constructed the manager.
    """
    w = FakeWorker(model)
    assert w.model_runner is not None
    mgr = SteeringManager(max_steering_configs=4)
    w.model_runner._steering_manager = mgr
    return w


# --- _steerable_layers ---


class TestSteerableLayers:
    def test_finds_all_layers(self, worker, model):
        layers = worker._steerable_layers()
        assert set(layers.keys()) == {0, 1, 2, 3}

    def test_no_model_runner(self):
        w = FakeWorker.__new__(FakeWorker)
        w.model_runner = None
        layers = w._steerable_layers()
        assert layers == {}


# --- set_steering_vectors: base tier ---


class TestSetSteeringVectorsBase:
    """Tests for base-tier vector setting."""

    def test_set_single_layer(self, worker_with_manager):
        vec = [1.0] * 8
        result = worker_with_manager.set_steering_vectors(vectors={_HP: {2: vec}})
        assert result[2] == [2]
        mgr = worker_with_manager.model_runner._steering_manager
        assert _HP in mgr.global_base_vectors
        assert 2 in mgr.global_base_vectors[_HP]
        stored = mgr.global_base_vectors[_HP][2]
        assert torch.allclose(stored, torch.tensor(vec))

    def test_set_multiple_layers(self, worker_with_manager):
        vec_a = [1.0] * 8
        vec_b = [2.0] * 8
        result = worker_with_manager.set_steering_vectors(
            vectors={_HP: {0: vec_a, 3: vec_b}}
        )
        assert result[2] == [0, 3]
        mgr = worker_with_manager.model_runner._steering_manager
        assert mgr.global_base_vectors[_HP][0].sum().item() == 8.0
        assert mgr.global_base_vectors[_HP][3].sum().item() == 16.0

    def test_unspecified_layers_unchanged(self, worker_with_manager):
        mgr = worker_with_manager.model_runner._steering_manager
        # Pre-set layer 1 in manager
        mgr.update_global_vectors(_HP, 1, torch.full((8,), 5.0), phase="base")
        worker_with_manager.set_steering_vectors(vectors={_HP: {0: [1.0] * 8}})
        assert mgr.global_base_vectors[_HP][1].sum().item() == 40.0

    def test_nonexistent_layer_ignored(self, worker_with_manager):
        result = worker_with_manager.set_steering_vectors(
            vectors={_HP: {999: [1.0] * 8}}
        )
        assert result[2] == []

    def test_mixed_valid_and_invalid_layers(self, worker_with_manager):
        vec = [3.0] * 8
        result = worker_with_manager.set_steering_vectors(
            vectors={_HP: {1: vec, 999: [1.0] * 8}}
        )
        assert result[2] == [1]
        mgr = worker_with_manager.model_runner._steering_manager
        assert mgr.global_base_vectors[_HP][1].sum().item() == 24.0

    def test_wrong_vector_size_raises(self, worker_with_manager):
        with pytest.raises(SteeringVectorError, match="expected vector of size 8"):
            worker_with_manager.set_steering_vectors(vectors={_HP: {0: [1.0, 2.0]}})

    def test_nan_raises(self, worker_with_manager):
        vec = [1.0] * 7 + [float("nan")]
        with pytest.raises(SteeringVectorError, match="non-finite"):
            worker_with_manager.set_steering_vectors(vectors={_HP: {0: vec}})

    def test_inf_raises(self, worker_with_manager):
        vec = [float("inf")] + [1.0] * 7
        with pytest.raises(SteeringVectorError, match="non-finite"):
            worker_with_manager.set_steering_vectors(vectors={_HP: {0: vec}})

    def test_negative_inf_raises(self, worker_with_manager):
        vec = [float("-inf")] + [1.0] * 7
        with pytest.raises(SteeringVectorError, match="non-finite"):
            worker_with_manager.set_steering_vectors(vectors={_HP: {0: vec}})

    def test_validate_only_does_not_mutate(self, worker_with_manager):
        vec = [5.0] * 8
        result = worker_with_manager.set_steering_vectors(
            vectors={_HP: {0: vec}}, validate_only=True
        )
        assert result[2] == [0]
        mgr = worker_with_manager.model_runner._steering_manager
        # Manager should have no base vectors after validate_only
        assert (
            _HP not in mgr.global_base_vectors
            or 0 not in mgr.global_base_vectors.get(_HP, {})
        )

    def test_validate_only_still_checks_size(self, worker_with_manager):
        with pytest.raises(SteeringVectorError, match="expected vector of size"):
            worker_with_manager.set_steering_vectors(
                vectors={_HP: {0: [1.0]}}, validate_only=True
            )

    def test_validate_only_still_checks_finite(self, worker_with_manager):
        vec = [float("nan")] * 8
        with pytest.raises(SteeringVectorError, match="non-finite"):
            worker_with_manager.set_steering_vectors(
                vectors={_HP: {0: vec}}, validate_only=True
            )

    def test_validation_error_prevents_mutation(self, worker_with_manager):
        with pytest.raises(SteeringVectorError, match="expected vector of size"):
            worker_with_manager.set_steering_vectors(
                vectors={_HP: {0: [1.0] * 8, 1: [1.0] * 3}},
            )
        mgr = worker_with_manager.model_runner._steering_manager
        # Manager should have no base vectors after validation error
        assert (
            _HP not in mgr.global_base_vectors
            or 0 not in mgr.global_base_vectors.get(_HP, {})
        )

    def test_empty_vectors_is_noop(self, worker_with_manager):
        result = worker_with_manager.set_steering_vectors(vectors={})
        assert result[2] == []

    def test_no_vectors_is_noop(self, worker_with_manager):
        result = worker_with_manager.set_steering_vectors()
        assert result[2] == []

    def test_invalid_hook_point_raises(self, worker_with_manager):
        with pytest.raises(SteeringVectorError, match="Invalid hook point"):
            worker_with_manager.set_steering_vectors(
                vectors={"not_a_hook": {0: [1.0] * 8}}
            )

    def test_inactive_hook_point_raises(self, worker_with_manager):
        with pytest.raises(SteeringVectorError, match="not active"):
            worker_with_manager.set_steering_vectors(
                vectors={"pre_attn": {0: [1.0] * 8}}
            )


# --- set_steering_vectors: three-tier ---


class TestSetSteeringVectorsThreeTier:
    """Tests for three-tier steering (base, prefill, decode)."""

    def test_set_prefill_only(self, worker_with_manager):
        """Prefill-only vectors go to manager, not buffers."""
        vec = [2.0] * 8
        result = worker_with_manager.set_steering_vectors(
            prefill_vectors={_HP: {0: vec}}
        )
        assert result[2] == [0]
        mgr = worker_with_manager.model_runner._steering_manager
        assert _HP in mgr.global_prefill_vectors
        assert 0 in mgr.global_prefill_vectors[_HP]
        # No base vectors
        assert (
            _HP not in mgr.global_base_vectors
            or 0 not in mgr.global_base_vectors.get(_HP, {})
        )

    def test_set_decode_only(self, worker_with_manager):
        """Decode-only vectors go to manager, not buffers."""
        vec = [3.0] * 8
        result = worker_with_manager.set_steering_vectors(
            decode_vectors={_HP: {1: vec}}
        )
        assert result[2] == [1]
        mgr = worker_with_manager.model_runner._steering_manager
        assert _HP in mgr.global_decode_vectors
        assert 1 in mgr.global_decode_vectors[_HP]

    def test_set_all_three_tiers(self, worker_with_manager):
        """Setting all three tiers in one call."""
        result = worker_with_manager.set_steering_vectors(
            vectors={_HP: {0: [1.0] * 8}},
            prefill_vectors={_HP: {1: [2.0] * 8}},
            decode_vectors={_HP: {2: [3.0] * 8}},
        )
        assert result[2] == [0, 1, 2]

    def test_validate_only_checks_all_tiers(self, worker_with_manager):
        """Validate mode checks all tiers without mutation."""
        result = worker_with_manager.set_steering_vectors(
            vectors={_HP: {0: [1.0] * 8}},
            prefill_vectors={_HP: {1: [2.0] * 8}},
            validate_only=True,
        )
        assert result[2] == [0, 1]
        # Nothing mutated
        mgr = worker_with_manager.model_runner._steering_manager
        assert (
            _HP not in mgr.global_base_vectors
            or 0 not in mgr.global_base_vectors.get(_HP, {})
        )
        assert (
            _HP not in mgr.global_prefill_vectors
            or 1 not in mgr.global_prefill_vectors.get(_HP, {})
        )

    def test_validation_error_in_prefill_tier(self, worker_with_manager):
        """Validation error in prefill tier prevents all mutation."""
        with pytest.raises(SteeringVectorError, match="expected vector of size"):
            worker_with_manager.set_steering_vectors(
                vectors={_HP: {0: [1.0] * 8}},
                prefill_vectors={_HP: {1: [1.0] * 3}},  # wrong size
            )

    def test_validation_error_in_decode_tier(self, worker_with_manager):
        """Validation error in decode tier prevents all mutation."""
        with pytest.raises(SteeringVectorError, match="non-finite"):
            worker_with_manager.set_steering_vectors(
                vectors={_HP: {0: [1.0] * 8}},
                decode_vectors={_HP: {1: [float("nan")] * 8}},
            )

    def test_replace_clears_all_before_applying(self, worker_with_manager):
        """replace=True clears manager before setting new vectors."""
        mgr = worker_with_manager.model_runner._steering_manager
        # Set some initial values
        worker_with_manager.set_steering_vectors(
            vectors={_HP: {0: [5.0] * 8, 1: [5.0] * 8}}
        )
        # Replace with only layer 2
        worker_with_manager.set_steering_vectors(
            vectors={_HP: {2: [1.0] * 8}},
            replace=True,
        )
        # Layers 0, 1 should be gone from manager (cleared)
        base = mgr.global_base_vectors.get(_HP, {})
        assert 0 not in base
        assert 1 not in base
        # Layer 2 should have the new values
        assert base[2].sum().item() == 8.0

    def test_replace_with_no_new_vectors_clears_all(self, worker_with_manager):
        """replace=True with no vector arguments clears all existing vectors."""
        mgr = worker_with_manager.model_runner._steering_manager
        # Set some initial values on layers 0 and 1
        worker_with_manager.set_steering_vectors(
            vectors={_HP: {0: [5.0] * 8, 1: [5.0] * 8}}
        )
        # Verify they are set
        assert mgr.global_base_vectors[_HP][0].sum().item() == 40.0
        assert mgr.global_base_vectors[_HP][1].sum().item() == 40.0

        # Call replace=True with NO vector arguments
        result = worker_with_manager.set_steering_vectors(replace=True)

        # Should return empty (no new layers updated)
        assert result[2] == []
        # But all existing vectors should be cleared
        assert not mgr.global_base_vectors

    def test_base_vectors_go_to_manager(self, worker_with_manager):
        """When all three tiers target the same layer, base vectors go
        to the manager as phase='base'."""
        base_vec = [1.0] * 8
        prefill_vec = [10.0] * 8
        decode_vec = [100.0] * 8
        worker_with_manager.set_steering_vectors(
            vectors={_HP: {0: base_vec}},
            prefill_vectors={_HP: {0: prefill_vec}},
            decode_vectors={_HP: {0: decode_vec}},
        )
        mgr = worker_with_manager.model_runner._steering_manager
        # Base should be in manager
        assert mgr.global_base_vectors[_HP][0].sum().item() == pytest.approx(8.0)
        # Prefill and decode should also be in manager
        assert mgr.global_prefill_vectors[_HP][0].sum().item() == pytest.approx(80.0)
        assert mgr.global_decode_vectors[_HP][0].sum().item() == pytest.approx(800.0)


# --- clear_steering_vectors ---


class TestClearSteeringVectors:
    def test_clears_manager(self, worker_with_manager):
        mgr = worker_with_manager.model_runner._steering_manager
        worker_with_manager.set_steering_vectors(
            vectors={_HP: {0: [1.0] * 8, 2: [2.0] * 8}}
        )
        worker_with_manager.clear_steering_vectors()
        assert not mgr.global_base_vectors

    def test_clear_on_already_empty(self, worker_with_manager):
        worker_with_manager.clear_steering_vectors()
        mgr = worker_with_manager.model_runner._steering_manager
        assert not mgr.global_base_vectors

    def test_clear_after_three_tier_set(self, worker_with_manager):
        """Clearing after setting all three tiers clears manager."""
        worker_with_manager.set_steering_vectors(
            vectors={_HP: {0: [1.0] * 8}},
            prefill_vectors={_HP: {1: [2.0] * 8}},
            decode_vectors={_HP: {2: [3.0] * 8}},
        )
        worker_with_manager.clear_steering_vectors()
        mgr = worker_with_manager.model_runner._steering_manager
        assert not mgr.global_base_vectors
        assert not mgr.global_prefill_vectors
        assert not mgr.global_decode_vectors


# --- get_steering_status ---


class TestGetSteeringStatus:
    def test_empty_when_no_steering(self, worker_with_manager):
        status = worker_with_manager.get_steering_status()
        assert status == {}

    def test_reports_active_layers(self, worker_with_manager):
        worker_with_manager.set_steering_vectors(
            vectors={_HP: {1: [1.0] * 8, 3: [2.0] * 8}}
        )
        status = worker_with_manager.get_steering_status()
        assert 1 in status
        assert 3 in status
        assert 0 not in status
        assert 2 not in status
        assert _HP in status[1]
        assert _HP in status[3]

    def test_norm_values(self, worker_with_manager):
        worker_with_manager.set_steering_vectors(vectors={_HP: {0: [3.0] * 8}})
        status = worker_with_manager.get_steering_status()
        expected_norm = round(math.sqrt(8 * 9.0), 6)
        assert status[0][_HP]["norm"] == expected_norm

    def test_cleared_after_set(self, worker_with_manager):
        worker_with_manager.set_steering_vectors(vectors={_HP: {0: [1.0] * 8}})
        worker_with_manager.clear_steering_vectors()
        status = worker_with_manager.get_steering_status()
        assert status == {}

    def test_status_reports_prefill_norm(self, worker_with_manager):
        """Prefill-only vectors appear as ``prefill_norm`` in status."""
        vec = [2.0] * 8
        worker_with_manager.set_steering_vectors(
            prefill_vectors={_HP: {0: vec}},
        )
        status = worker_with_manager.get_steering_status()
        assert 0 in status
        assert _HP in status[0]
        expected_norm = round(torch.tensor(vec).norm().item(), 6)
        assert status[0][_HP]["prefill_norm"] == expected_norm
        # No base norm since we only set prefill
        assert "norm" not in status[0][_HP]

    def test_status_reports_decode_norm(self, worker_with_manager):
        """Decode-only vectors appear as ``decode_norm`` in status."""
        vec = [3.0] * 8
        worker_with_manager.set_steering_vectors(
            decode_vectors={_HP: {1: vec}},
        )
        status = worker_with_manager.get_steering_status()
        assert 1 in status
        assert _HP in status[1]
        expected_norm = round(torch.tensor(vec).norm().item(), 6)
        assert status[1][_HP]["decode_norm"] == expected_norm
        # No base norm since we only set decode
        assert "norm" not in status[1][_HP]

    def test_status_base_only_no_phase_keys(self, worker_with_manager):
        """Base-only vectors produce ``norm`` but no phase-specific keys."""
        vec = [1.0] * 8
        worker_with_manager.set_steering_vectors(
            vectors={_HP: {0: vec}},
        )
        status = worker_with_manager.get_steering_status()
        assert 0 in status
        assert _HP in status[0]
        assert "norm" in status[0][_HP]
        assert "prefill_norm" not in status[0][_HP]
        assert "decode_norm" not in status[0][_HP]

    def test_status_all_tiers(self, worker_with_manager):
        """All three tiers on the same layer produce all three norm keys."""
        base_vec = [1.0] * 8
        prefill_vec = [2.0] * 8
        decode_vec = [3.0] * 8
        worker_with_manager.set_steering_vectors(
            vectors={_HP: {0: base_vec}},
            prefill_vectors={_HP: {0: prefill_vec}},
            decode_vectors={_HP: {0: decode_vec}},
        )
        status = worker_with_manager.get_steering_status()
        assert 0 in status
        assert _HP in status[0]
        layer_status = status[0][_HP]
        # Base norm from manager
        expected_base = round(torch.tensor(base_vec).norm().item(), 6)
        assert layer_status["norm"] == expected_base
        # Prefill norm from manager
        expected_prefill = round(torch.tensor(prefill_vec).norm().item(), 6)
        assert layer_status["prefill_norm"] == expected_prefill
        # Decode norm from manager
        expected_decode = round(torch.tensor(decode_vec).norm().item(), 6)
        assert layer_status["decode_norm"] == expected_decode


# --- no model runner ---


class TestNoModelRunner:
    def test_set_with_no_runner(self):
        w = FakeWorker.__new__(FakeWorker)
        w.model_runner = None
        assert w.set_steering_vectors(vectors={_HP: {0: [1.0]}}) == (0, 0, [])

    def test_clear_with_no_runner(self):
        w = FakeWorker.__new__(FakeWorker)
        w.model_runner = None
        w.clear_steering_vectors()  # should not raise

    def test_status_with_no_runner(self):
        w = FakeWorker.__new__(FakeWorker)
        w.model_runner = None
        assert w.get_steering_status() == {}
