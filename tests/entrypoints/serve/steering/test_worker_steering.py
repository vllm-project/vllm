# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for WorkerBase steering methods using a mock model."""

import math

import pytest
import torch
import torch.nn as nn

from vllm.exceptions import SteeringVectorError
from vllm.v1.worker.worker_base import WorkerBase


class FakeDecoderLayer(nn.Module):
    """Minimal decoder layer with a steering buffer."""

    def __init__(self, layer_idx: int, hidden_size: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.register_buffer(
            "steering_vector",
            torch.zeros(1, hidden_size),
            persistent=False,
        )


class FakeModel(nn.Module):
    """Model with a few steerable decoder layers."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeDecoderLayer(i, hidden_size) for i in range(num_layers)]
        )
        self.hidden_size = hidden_size


class FakeModelRunner:
    """Minimal model runner exposing get_model()."""

    def __init__(self, model: nn.Module):
        self._model = model

    def get_model(self) -> nn.Module:
        return self._model


class FakeWorker(WorkerBase):
    """Concrete WorkerBase for testing (abstract methods stubbed)."""

    def __init__(self, model: nn.Module):
        # Don't call super().__init__ — just set model_runner directly
        self.model_runner = FakeModelRunner(model)

    def init_device(self):
        pass

    def get_model(self):
        return self.model_runner.get_model()


@pytest.fixture
def model():
    return FakeModel(num_layers=4, hidden_size=8)


@pytest.fixture
def worker(model):
    return FakeWorker(model)


# --- _steerable_layers ---


class TestSteerableLayers:
    def test_finds_all_layers(self, worker, model):
        layers = worker._steerable_layers()
        assert set(layers.keys()) == {0, 1, 2, 3}

    def test_no_model_runner(self):
        w = FakeWorker.__new__(FakeWorker)
        w.model_runner = None
        layers = WorkerBase._steerable_layers(w)
        assert layers == {}


# --- set_steering_vectors ---


class TestSetSteeringVectors:
    def test_set_single_layer(self, worker, model):
        vec = [1.0] * 8
        result = worker.set_steering_vectors({2: vec})
        assert result == [2]
        assert torch.allclose(
            model.layers[2].steering_vector,
            torch.tensor([vec]),
        )

    def test_set_multiple_layers(self, worker, model):
        vec_a = [1.0] * 8
        vec_b = [2.0] * 8
        result = worker.set_steering_vectors({0: vec_a, 3: vec_b})
        assert result == [0, 3]
        assert model.layers[0].steering_vector.sum().item() == 8.0
        assert model.layers[3].steering_vector.sum().item() == 16.0

    def test_unspecified_layers_unchanged(self, worker, model):
        """Layers not in vectors_data should keep their current state."""
        # Pre-set layer 1
        model.layers[1].steering_vector.fill_(5.0)
        # Set only layer 0
        worker.set_steering_vectors({0: [1.0] * 8})
        # Layer 1 should be untouched
        assert model.layers[1].steering_vector.sum().item() == 40.0

    def test_nonexistent_layer_ignored(self, worker):
        """Requesting a layer that doesn't exist returns empty list."""
        result = worker.set_steering_vectors({999: [1.0] * 8})
        assert result == []

    def test_mixed_valid_and_invalid_layers(self, worker, model):
        """Valid layers are set; invalid layers are silently skipped."""
        vec = [3.0] * 8
        result = worker.set_steering_vectors({1: vec, 999: [1.0] * 8})
        assert result == [1]
        assert model.layers[1].steering_vector.sum().item() == 24.0

    def test_wrong_vector_size_raises(self, worker):
        with pytest.raises(SteeringVectorError, match="expected vector of size 8"):
            worker.set_steering_vectors({0: [1.0, 2.0]})  # too short

    def test_nan_raises(self, worker):
        vec = [1.0] * 7 + [float("nan")]
        with pytest.raises(SteeringVectorError, match="non-finite"):
            worker.set_steering_vectors({0: vec})

    def test_inf_raises(self, worker):
        vec = [float("inf")] + [1.0] * 7
        with pytest.raises(SteeringVectorError, match="non-finite"):
            worker.set_steering_vectors({0: vec})

    def test_negative_inf_raises(self, worker):
        vec = [float("-inf")] + [1.0] * 7
        with pytest.raises(SteeringVectorError, match="non-finite"):
            worker.set_steering_vectors({0: vec})

    def test_validate_only_does_not_mutate(self, worker, model):
        vec = [5.0] * 8
        result = worker.set_steering_vectors({0: vec}, validate_only=True)
        assert result == [0]
        # Buffer should still be zero
        assert model.layers[0].steering_vector.sum().item() == 0.0

    def test_validate_only_still_checks_size(self, worker):
        with pytest.raises(SteeringVectorError, match="expected vector of size"):
            worker.set_steering_vectors({0: [1.0]}, validate_only=True)

    def test_validate_only_still_checks_finite(self, worker):
        vec = [float("nan")] * 8
        with pytest.raises(SteeringVectorError, match="non-finite"):
            worker.set_steering_vectors({0: vec}, validate_only=True)

    def test_validation_error_prevents_mutation(self, worker, model):
        """If layer 0 is valid but layer 1 has wrong size, no layer is
        mutated — validation runs for all layers before any copy."""
        # Layer 0 valid, layer 1 invalid size
        with pytest.raises(SteeringVectorError, match="expected vector of size"):
            worker.set_steering_vectors(
                {0: [1.0] * 8, 1: [1.0] * 3},
            )
        # Layer 0 should NOT have been updated
        assert model.layers[0].steering_vector.sum().item() == 0.0

    def test_empty_dict_is_noop(self, worker):
        result = worker.set_steering_vectors({})
        assert result == []


# --- clear_steering_vectors ---


class TestClearSteeringVectors:
    def test_clears_all_layers(self, worker, model):
        # Set some vectors first
        worker.set_steering_vectors({0: [1.0] * 8, 2: [2.0] * 8})
        worker.clear_steering_vectors()
        for layer in model.layers:
            assert layer.steering_vector.sum().item() == 0.0

    def test_clear_on_already_zero(self, worker, model):
        """Clearing when already zero is a no-op — no errors."""
        worker.clear_steering_vectors()
        for layer in model.layers:
            assert layer.steering_vector.sum().item() == 0.0


# --- get_steering_status ---


class TestGetSteeringStatus:
    def test_empty_when_no_steering(self, worker):
        status = worker.get_steering_status()
        assert status == {}

    def test_reports_active_layers(self, worker):
        worker.set_steering_vectors({1: [1.0] * 8, 3: [2.0] * 8})
        status = worker.get_steering_status()
        assert 1 in status
        assert 3 in status
        assert 0 not in status
        assert 2 not in status

    def test_norm_values(self, worker):
        worker.set_steering_vectors({0: [3.0] * 8})
        status = worker.get_steering_status()
        expected_norm = round(math.sqrt(8 * 9.0), 6)
        assert status[0]["norm"] == expected_norm

    def test_cleared_after_set(self, worker):
        worker.set_steering_vectors({0: [1.0] * 8})
        worker.clear_steering_vectors()
        status = worker.get_steering_status()
        assert status == {}


# --- no model runner ---


class TestNoModelRunner:
    def test_set_with_no_runner(self):
        w = FakeWorker.__new__(FakeWorker)
        w.model_runner = None
        assert w.set_steering_vectors({0: [1.0]}) == []

    def test_clear_with_no_runner(self):
        w = FakeWorker.__new__(FakeWorker)
        w.model_runner = None
        w.clear_steering_vectors()  # should not raise

    def test_status_with_no_runner(self):
        w = FakeWorker.__new__(FakeWorker)
        w.model_runner = None
        assert w.get_steering_status() == {}
