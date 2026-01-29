# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for weight transfer APIs via LLM class.

These tests use a mock weight transfer engine to verify that the API
calls the correct methods with the right arguments, without requiring
actual NCCL communication.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from vllm import LLM
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferInitRequest,
    WeightTransferUpdateInfo,
    WeightTransferUpdateRequest,
)

from ...utils import create_new_process_for_each_test

# Use a tiny model for fast testing
MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


# --- Mock Weight Transfer Engine ---


@dataclass
class MockInitInfo(WeightTransferInitInfo):
    """Mock initialization info."""

    test_param: str = "test"


@dataclass
class MockUpdateInfo(WeightTransferUpdateInfo):
    """Mock update info."""

    names: list[str] | None = None
    dtype_names: list[str] | None = None
    shapes: list[list[int]] | None = None


class MockWeightTransferEngine(WeightTransferEngine[MockInitInfo, MockUpdateInfo]):
    """Mock weight transfer engine that tracks method calls."""

    init_info_cls = MockInitInfo
    update_info_cls = MockUpdateInfo

    # Class-level tracking for verification across processes
    init_transfer_engine_called: bool = False
    receive_weights_called: bool = False
    shutdown_called: bool = False
    last_init_info: MockInitInfo | None = None
    last_update_info: MockUpdateInfo | None = None

    def __init__(self, config, parallel_config):
        super().__init__(config, parallel_config)
        # Reset tracking on init
        MockWeightTransferEngine.init_transfer_engine_called = False
        MockWeightTransferEngine.receive_weights_called = False
        MockWeightTransferEngine.shutdown_called = False
        MockWeightTransferEngine.last_init_info = None
        MockWeightTransferEngine.last_update_info = None

    def init_transfer_engine(self, init_info: MockInitInfo) -> None:
        MockWeightTransferEngine.init_transfer_engine_called = True
        MockWeightTransferEngine.last_init_info = init_info

    def receive_weights(
        self,
        update_info: MockUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        MockWeightTransferEngine.receive_weights_called = True
        MockWeightTransferEngine.last_update_info = update_info
        # Simulate loading weights by calling load_weights with empty list
        # (In real implementation, this would receive and load actual weights)
        load_weights([])

    def shutdown(self) -> None:
        MockWeightTransferEngine.shutdown_called = True


def mock_create_engine(config, parallel_config):
    """Mock factory function that returns our mock engine."""
    return MockWeightTransferEngine(config, parallel_config)


# --- Tests ---


@create_new_process_for_each_test()
def test_get_world_size_tp1():
    """Test world_size is correctly configured for TP=1."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    llm = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        load_format="dummy",
        tensor_parallel_size=1,
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
    )

    world_size = llm.llm_engine.vllm_config.parallel_config.world_size
    assert world_size == 1


@create_new_process_for_each_test()
def test_init_weight_transfer_engine_calls_engine():
    """Test that init_weight_transfer_engine calls the engine's
    init_transfer_engine method."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    # Enable insecure serialization to allow pickling functions for collective_rpc
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    with patch(
        "vllm.v1.worker.gpu_worker.WeightTransferEngineFactory.create_engine",
        mock_create_engine,
    ):
        llm = LLM(
            model=MODEL_NAME,
            enforce_eager=True,
            load_format="dummy",
            tensor_parallel_size=1,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        )

        # Verify engine was created
        def check_engine_exists(self):
            return self.weight_transfer_engine is not None

        results = llm.collective_rpc(check_engine_exists)
        assert all(results), "Weight transfer engine should be initialized"

        # Call init_weight_transfer_engine
        llm.init_weight_transfer_engine(
            WeightTransferInitRequest(init_info={"test_param": "hello"})
        )

        # Verify init_transfer_engine was called on the engine
        def check_init_called(self):
            engine = self.weight_transfer_engine
            return (
                engine.init_transfer_engine_called,
                engine.last_init_info.test_param if engine.last_init_info else None,
            )

        results = llm.collective_rpc(check_init_called)
        for called, param in results:
            assert called, "init_transfer_engine should have been called"
            assert param == "hello", f"Expected 'hello', got {param}"


@create_new_process_for_each_test()
def test_update_weights_calls_engine():
    """Test that update_weights calls the engine's receive_weights method."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    # Enable insecure serialization to allow pickling functions for collective_rpc
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    with patch(
        "vllm.v1.worker.gpu_worker.WeightTransferEngineFactory.create_engine",
        mock_create_engine,
    ):
        llm = LLM(
            model=MODEL_NAME,
            enforce_eager=True,
            load_format="dummy",
            tensor_parallel_size=1,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        )

        # First init the weight transfer
        llm.init_weight_transfer_engine(
            WeightTransferInitRequest(init_info={"test_param": "init"})
        )

        # Call update_weights
        test_names = ["layer.weight", "layer.bias"]
        test_dtypes = ["float32", "float32"]
        test_shapes = [[10, 10], [10]]

        llm.update_weights(
            WeightTransferUpdateRequest(
                update_info={
                    "names": test_names,
                    "dtype_names": test_dtypes,
                    "shapes": test_shapes,
                }
            )
        )

        # Verify receive_weights was called with correct info
        def check_update_called(self):
            engine = self.weight_transfer_engine
            if not engine.receive_weights_called:
                return False, None, None, None
            info = engine.last_update_info
            return (True, info.names, info.dtype_names, info.shapes)

        results = llm.collective_rpc(check_update_called)
        for called, names, dtypes, shapes in results:
            assert called, "receive_weights should have been called"
            assert names == test_names
            assert dtypes == test_dtypes
            assert shapes == test_shapes


@create_new_process_for_each_test()
def test_finalize_weight_update_runs():
    """Test that finalize_weight_update completes without error."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    with patch(
        "vllm.v1.worker.gpu_worker.WeightTransferEngineFactory.create_engine",
        mock_create_engine,
    ):
        llm = LLM(
            model=MODEL_NAME,
            enforce_eager=True,
            load_format="dummy",
            tensor_parallel_size=1,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        )

        # finalize_weight_update should run without error
        # (it calls process_weights_after_loading internally)
        llm.finalize_weight_update()


@create_new_process_for_each_test()
def test_full_weight_transfer_flow():
    """Test the complete weight transfer flow: init -> update -> finalize."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    # Enable insecure serialization to allow pickling functions for collective_rpc
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    with patch(
        "vllm.v1.worker.gpu_worker.WeightTransferEngineFactory.create_engine",
        mock_create_engine,
    ):
        llm = LLM(
            model=MODEL_NAME,
            enforce_eager=True,
            load_format="dummy",
            tensor_parallel_size=1,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        )

        # Step 1: Initialize
        llm.init_weight_transfer_engine(
            WeightTransferInitRequest(init_info={"test_param": "flow_test"})
        )

        # Step 2: Update weights
        llm.update_weights(
            WeightTransferUpdateRequest(
                update_info={
                    "names": ["test.weight"],
                    "dtype_names": ["bfloat16"],
                    "shapes": [[100, 100]],
                }
            )
        )

        # Step 3: Finalize
        llm.finalize_weight_update()

        # Verify the full flow completed
        def check_flow(self):
            engine = self.weight_transfer_engine
            return {
                "init_called": engine.init_transfer_engine_called,
                "update_called": engine.receive_weights_called,
                "init_param": (
                    engine.last_init_info.test_param if engine.last_init_info else None
                ),
                "update_names": (
                    engine.last_update_info.names if engine.last_update_info else None
                ),
            }

        results = llm.collective_rpc(check_flow)
        for result in results:
            assert result["init_called"], "init_transfer_engine should be called"
            assert result["update_called"], "receive_weights should be called"
            assert result["init_param"] == "flow_test"
            assert result["update_names"] == ["test.weight"]


@create_new_process_for_each_test()
def test_weight_transfer_config_backend():
    """Test that WeightTransferConfig backend is properly configured."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    # Test with nccl backend
    llm = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        load_format="dummy",
        tensor_parallel_size=1,
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
    )

    config = llm.llm_engine.vllm_config.weight_transfer_config
    assert config.backend == "nccl"
