# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for weight transfer APIs via LLM class.

These tests use a mock weight transfer engine to verify that the API
calls the correct methods with the right arguments, without requiring
actual NCCL communication.
"""

import os
import weakref
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferInitRequest,
    WeightTransferUpdateInfo,
    WeightTransferUpdateRequest,
)
from vllm.platforms import current_platform

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
    start_called: bool = False
    receive_weights_called: bool = False
    finish_called: bool = False
    shutdown_called: bool = False
    last_init_info: MockInitInfo | None = None
    last_update_info: MockUpdateInfo | None = None

    def __init__(self, config, vllm_config, device, model):
        super().__init__(config, vllm_config, device, model)
        # Reset tracking on init
        MockWeightTransferEngine.init_transfer_engine_called = False
        MockWeightTransferEngine.start_called = False
        MockWeightTransferEngine.receive_weights_called = False
        MockWeightTransferEngine.finish_called = False
        MockWeightTransferEngine.shutdown_called = False
        MockWeightTransferEngine.last_init_info = None
        MockWeightTransferEngine.last_update_info = None

    def init_transfer_engine(self, init_info: MockInitInfo) -> None:
        MockWeightTransferEngine.init_transfer_engine_called = True
        MockWeightTransferEngine.last_init_info = init_info

    def start_weight_update(self) -> None:
        MockWeightTransferEngine.start_called = True

    def finish_weight_update(self) -> None:
        MockWeightTransferEngine.finish_called = True

    def receive_weights(self, update_info: MockUpdateInfo) -> None:
        MockWeightTransferEngine.receive_weights_called = True
        MockWeightTransferEngine.last_update_info = update_info

    def shutdown(self) -> None:
        MockWeightTransferEngine.shutdown_called = True


def mock_create_engine(config, vllm_config, device, model):
    """Mock factory function that returns our mock engine."""
    return MockWeightTransferEngine(config, vllm_config, device, model)


# --- Tests ---


@create_new_process_for_each_test()
def test_get_world_size_tp1(vllm_runner):
    """Test world_size is correctly configured for TP=1."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    with vllm_runner(
        MODEL_NAME,
        enforce_eager=True,
        load_format="dummy",
        tensor_parallel_size=1,
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
    ) as runner:
        world_size = runner.llm.llm_engine.vllm_config.parallel_config.world_size
        assert world_size == 1


@create_new_process_for_each_test()
def test_init_weight_transfer_engine_calls_engine(vllm_runner):
    """Test that init_weight_transfer_engine calls the engine's
    init_transfer_engine method."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    # Run in-process so mock.patch works (spawn won't inherit the mock)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Enable insecure serialization to allow pickling functions for collective_rpc
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    with (
        patch(
            "vllm.v1.worker.gpu_worker.WeightTransferEngineFactory.create_engine",
            mock_create_engine,
        ),
        vllm_runner(
            MODEL_NAME,
            enforce_eager=True,
            load_format="dummy",
            tensor_parallel_size=1,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        ) as runner,
    ):
        llm = weakref.proxy(runner.llm)

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
def test_update_weights_calls_engine(vllm_runner):
    """Test that update_weights calls the engine's receive_weights method."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    # Run in-process so mock.patch works (spawn won't inherit the mock)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Enable insecure serialization to allow pickling functions for collective_rpc
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    with (
        patch(
            "vllm.v1.worker.gpu_worker.WeightTransferEngineFactory.create_engine",
            mock_create_engine,
        ),
        vllm_runner(
            MODEL_NAME,
            enforce_eager=True,
            load_format="dummy",
            tensor_parallel_size=1,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        ) as runner,
    ):
        llm = weakref.proxy(runner.llm)

        # First init the weight transfer
        llm.init_weight_transfer_engine(
            WeightTransferInitRequest(init_info={"test_param": "init"})
        )
        llm.start_weight_update()

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

        llm.finish_weight_update()
        assert llm.get_weight_version() == "default"


@create_new_process_for_each_test()
def test_weight_update_rejected_while_asleep(vllm_runner):
    """A weight update against unmapped weights is refused, not a crash."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    with (
        patch(
            "vllm.v1.worker.gpu_worker.WeightTransferEngineFactory.create_engine",
            mock_create_engine,
        ),
        vllm_runner(
            MODEL_NAME,
            enforce_eager=True,
            load_format="dummy",
            tensor_parallel_size=1,
            enable_sleep_mode=True,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        ) as runner,
    ):
        llm = weakref.proxy(runner.llm)
        llm.init_weight_transfer_engine(
            WeightTransferInitRequest(init_info={"test_param": "init"})
        )

        def assert_refused_until_weights_wake(level: int) -> None:
            llm.sleep(level=level)
            with pytest.raises(RuntimeError, match="asleep"):
                llm.start_weight_update()
            llm.wake_up(tags=["kv_cache"])  # weights still unmapped
            with pytest.raises(RuntimeError, match="asleep"):
                llm.start_weight_update()
            llm.wake_up()

        assert_refused_until_weights_wake(level=1)

        # A session interrupted by sleep is dropped, not resumed.
        llm.start_weight_update()
        llm.sleep(level=1)
        with pytest.raises(RuntimeError, match="asleep"):
            llm.finish_weight_update()
        llm.wake_up()
        with pytest.raises(RuntimeError, match="without a matching"):
            llm.finish_weight_update()

        # Fully awake again: the normal flow works and the engine is alive.
        llm.start_weight_update()
        llm.update_weights(
            WeightTransferUpdateRequest(
                update_info={
                    "names": ["layer.weight"],
                    "dtype_names": ["float32"],
                    "shapes": [[10, 10]],
                }
            )
        )
        llm.finish_weight_update()
        outputs = llm.generate(["Hello"], use_tqdm=False)
        assert len(outputs) == 1

        # Level 2 discards the weights; with dummy weights there is nothing to
        # reload, so only the refusal itself is checked here.
        assert_refused_until_weights_wake(level=2)


@create_new_process_for_each_test()
def test_full_weight_transfer_flow(vllm_runner):
    """Test the complete weight transfer flow: init -> start -> update -> finish."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    # Run in-process so mock.patch works (spawn won't inherit the mock)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Enable insecure serialization to allow pickling functions for collective_rpc
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    with (
        patch(
            "vllm.v1.worker.gpu_worker.WeightTransferEngineFactory.create_engine",
            mock_create_engine,
        ),
        vllm_runner(
            MODEL_NAME,
            enforce_eager=True,
            load_format="dummy",
            tensor_parallel_size=1,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        ) as runner,
    ):
        llm = weakref.proxy(runner.llm)

        assert llm.get_weight_version() == "default"

        # Step 1: Initialize weight transfer engine
        llm.init_weight_transfer_engine(
            WeightTransferInitRequest(init_info={"test_param": "flow_test"})
        )

        # Step 2: Start weight update
        llm.start_weight_update()

        # Step 3: Update weights
        llm.update_weights(
            WeightTransferUpdateRequest(
                update_info={
                    "names": ["test.weight"],
                    "dtype_names": ["bfloat16"],
                    "shapes": [[100, 100]],
                }
            )
        )

        assert llm.get_weight_version() == "default"

        # Step 4: Finish weight update
        llm.finish_weight_update("step-42")

        assert llm.get_weight_version() == "step-42"

        llm.update_weight_version("manual-version")
        assert llm.get_weight_version() == "manual-version"

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
def test_failed_update_leaves_model_usable(vllm_runner):
    """A chunk that fails inside update_weights must not take the model down.

    The IPC engine parks every weight on the meta device at start_weight_update;
    the worker's abort path has to put them back, or the next forward runs on
    meta tensors and kills the engine.
    """
    # A real IPC engine reaches CUDA IPC calls; the mock-engine tests here do not.
    if not current_platform.is_cuda_alike():
        pytest.skip("Requires a CUDA-like device for the real IPC engine")

    with vllm_runner(
        MODEL_NAME,
        enforce_eager=True,
        load_format="dummy",
        tensor_parallel_size=1,
        weight_transfer_config=WeightTransferConfig(backend="ipc"),
    ) as runner:
        llm = weakref.proxy(runner.llm)
        prompts = ["The capital of France is"]
        before = runner.generate_greedy(prompts, 8)

        llm.init_weight_transfer_engine(WeightTransferInitRequest(init_info={}))
        llm.start_weight_update()
        # The worker error comes back as a plain string flattened into an
        # Exception by the engine-core RPC, so match on the message, not a type.
        with pytest.raises(Exception, match="IPC handle not found"):
            llm.update_weights(
                WeightTransferUpdateRequest(
                    update_info={
                        "names": ["model.embed_tokens.weight"],
                        "dtype_names": ["float32"],
                        "shapes": [[1, 1]],
                        "ipc_handles": [{"not-this-gpu": []}],
                    }
                )
            )

        # The failed session is gone and the weights are back where they were.
        assert runner.generate_greedy(prompts, 8) == before
        with pytest.raises(Exception, match="without a matching"):
            llm.finish_weight_update()

        # A fresh session works on the restored model.
        llm.start_weight_update()
        llm.finish_weight_update()
        assert runner.generate_greedy(prompts, 8) == before


@create_new_process_for_each_test()
def test_weight_transfer_config_backend(vllm_runner):
    """Test that WeightTransferConfig backend is properly configured."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    # Test with nccl backend
    with vllm_runner(
        MODEL_NAME,
        enforce_eager=True,
        load_format="dummy",
        tensor_parallel_size=1,
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
    ) as runner:
        config = runner.llm.llm_engine.vllm_config.weight_transfer_config
        assert config is not None
        assert config.backend == "nccl"
