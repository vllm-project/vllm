# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from tests.utils import create_new_process_for_each_test
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type
HAS_DEVICE_GRAPH = current_platform.is_cuda_alike() or current_platform.is_xpu()

if current_platform.is_xpu():
    from vllm.v1.worker.xpu_model_runner import _torch_cuda_wrapper

    # Route torch.cuda graph/stream APIs to torch.xpu, as XPUModelRunner does.
    with _torch_cuda_wrapper():
        pass


# Helper MLP for testing
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _create_vllm_config(
    compilation_config: CompilationConfig,
    max_num_seqs: int = 8,
) -> MagicMock:
    mock_config = MagicMock(spec=VllmConfig)
    mock_config.compilation_config = compilation_config
    mock_config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs,
    )
    mock_config.parallel_config = ParallelConfig()
    mock_config.speculative_config = None  # No speculative decoding
    mock_config.num_speculative_tokens = 0
    mock_config.use_cumem_cudagraph_pool = False
    mock_config.lora_config = None
    # Mimic the behavior of VllmConfig.__post_init__()
    if compilation_config.mode == CompilationMode.VLLM_COMPILE:
        compilation_config.set_splitting_ops_for_v1(
            all2all_backend=mock_config.parallel_config.all2all_backend,
            data_parallel_size=mock_config.parallel_config.data_parallel_size,
        )

    # mimic VllmConfig.__post_init__
    if compilation_config.cudagraph_capture_sizes:
        compilation_config.max_cudagraph_capture_size = (
            compilation_config.cudagraph_capture_sizes[-1]
        )

        compilation_config.post_init_cudagraph_sizes()

    return mock_config


@pytest.mark.skipif(not HAS_DEVICE_GRAPH, reason="Requires CUDA, ROCm or XPU graph")
class TestCUDAGraphWrapper:
    def setup_method(self):
        self.vllm_config = _create_vllm_config(CompilationConfig())
        self.model = SimpleMLP().to(DEVICE_TYPE)
        self.persistent_input_buffer = torch.zeros(1, 10, device=DEVICE_TYPE)
        self.input_tensor = torch.randn(1, 10, device=DEVICE_TYPE)

    def test_capture_and_replay(self):
        wrapper = CUDAGraphWrapper(
            self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
        )
        batch_descriptor = BatchDescriptor(num_tokens=10)

        # 0. global warmup
        with set_forward_context(
            attn_metadata=None,
            vllm_config=self.vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=None,
        ):
            wrapper(self.input_tensor)

        # 1. Capture
        with (
            set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=batch_descriptor,
            ),
            patch("torch.cuda.graph", wraps=torch.cuda.graph) as mock_cuda_graph,
        ):
            output1 = wrapper(self.input_tensor)
            # capturing phase should generate a zero output
            assert torch.allclose(output1, torch.zeros_like(output1))
            mock_cuda_graph.assert_called_once()

        assert batch_descriptor in wrapper.concrete_cudagraph_entries
        entry = wrapper.concrete_cudagraph_entries[batch_descriptor]
        assert entry.cudagraph is not None

        # 2. Replay
        with (
            set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=batch_descriptor,
            ),
            patch.object(
                entry.cudagraph, "replay", wraps=entry.cudagraph.replay
            ) as mock_replay,
        ):
            output2 = wrapper(self.input_tensor)
            mock_replay.assert_called_once()

        # Compare with eager output
        eager_output = self.model(self.input_tensor)
        torch.testing.assert_close(eager_output, output2)

    def test_bypass_on_mode_mismatch(self):
        wrapper = CUDAGraphWrapper(
            self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
        )
        batch_descriptor = BatchDescriptor(num_tokens=10)

        with (
            set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                batch_descriptor=batch_descriptor,
            ),
            patch("torch.cuda.graph", wraps=torch.cuda.graph) as mock_cuda_graph,
            patch.object(
                self.model, "forward", wraps=self.model.forward
            ) as mock_forward,
        ):
            wrapper(self.input_tensor)
            mock_cuda_graph.assert_not_called()
            mock_forward.assert_called_once()
        assert not wrapper.concrete_cudagraph_entries

    def test_bypass_on_mode_none(self):
        wrapper = CUDAGraphWrapper(
            self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
        )
        batch_descriptor = BatchDescriptor(num_tokens=10)

        with (
            set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=batch_descriptor,
            ),
            patch("torch.cuda.graph", wraps=torch.cuda.graph) as mock_cuda_graph,
        ):
            wrapper(self.input_tensor)
            mock_cuda_graph.assert_not_called()
        assert not wrapper.concrete_cudagraph_entries


def _run_and_monitor_call(
    wrapper, input_tensor, runtime_mode, batch_descriptor, vllm_config
):
    """Helper to run a single call and monitor the action."""
    with (
        patch("torch.cuda.graph", wraps=torch.cuda.graph) as mock_graph_context,
        patch.object(wrapper, "runnable", wraps=wrapper.runnable) as mock_runnable,
    ):
        entry = wrapper.concrete_cudagraph_entries.get(batch_descriptor, None)

        context = set_forward_context(
            attn_metadata=None,
            vllm_config=vllm_config,
            cudagraph_runtime_mode=runtime_mode,
            batch_descriptor=batch_descriptor,
        )
        mock_replay = MagicMock()
        if entry and entry.cudagraph:
            with (
                context,
                patch.object(
                    entry.cudagraph, "replay", new_callable=MagicMock
                ) as mock_replay,
            ):
                wrapper(input_tensor)
        else:
            with context:
                wrapper(input_tensor)

        if mock_graph_context.called:
            # note that this is globally mocked, so it will be detected
            # even whether called by the inner or outer wrapper
            return "capture_global"
        if mock_replay.called:
            # only for outer wrapper
            return "replay"
        if mock_runnable.call_count > 0:
            # only for outer wrapper
            return "bypass"
        return "unknown"


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(not HAS_DEVICE_GRAPH, reason="Requires CUDA, ROCm or XPU graph")
def test_capture_replay_bypass_logic():
    comp_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_capture_sizes=[1, 2],
    )
    vllm_config = _create_vllm_config(comp_config)
    model = SimpleMLP().to(DEVICE_TYPE)
    full_wrapper = CUDAGraphWrapper(model, vllm_config, CUDAGraphMode.FULL)
    max_bs = 16
    persistent_input_buffer = torch.zeros(max_bs, 10, device=DEVICE_TYPE)
    input_1 = persistent_input_buffer[:1]
    input_2 = persistent_input_buffer[:2]
    input_3 = persistent_input_buffer[:3]

    desc_1 = BatchDescriptor(num_tokens=1)
    desc_2 = BatchDescriptor(num_tokens=2)
    desc_3_unseen = BatchDescriptor(num_tokens=3)

    # 0. global warmup
    with set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        full_wrapper(input_1)

    # 1. Capture first shape
    action = _run_and_monitor_call(
        full_wrapper, input_1, CUDAGraphMode.FULL, desc_1, vllm_config
    )
    assert action == "capture_global"

    # 2. Replay first shape
    action = _run_and_monitor_call(
        full_wrapper, input_1, CUDAGraphMode.FULL, desc_1, vllm_config
    )
    assert action == "replay"

    # 3. Capture second shape
    action = _run_and_monitor_call(
        full_wrapper, input_2, CUDAGraphMode.FULL, desc_2, vllm_config
    )
    assert action == "capture_global"

    # 4. Replay second shape
    action = _run_and_monitor_call(
        full_wrapper, input_2, CUDAGraphMode.FULL, desc_2, vllm_config
    )
    assert action == "replay"

    # 5. Bypass when dispatched with no graph
    action = _run_and_monitor_call(
        full_wrapper, input_3, CUDAGraphMode.NONE, desc_3_unseen, vllm_config
    )
    assert action == "bypass"

    # capture unseen shape is not allowed after disable
    set_cudagraph_capturing_enabled(False)
    with pytest.raises(RuntimeError):
        _run_and_monitor_call(
            full_wrapper, input_3, CUDAGraphMode.FULL, desc_3_unseen, vllm_config
        )
    set_cudagraph_capturing_enabled(True)


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(not HAS_DEVICE_GRAPH, reason="Requires CUDA, ROCm or XPU graph")
def test_nested_wrappers():
    """Tests a scenario with a PIECEWISE wrapper inside a FULL one."""
    comp_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_capture_sizes=[1],
    )
    vllm_config = _create_vllm_config(comp_config)
    model = SimpleMLP().to(DEVICE_TYPE)
    full_wrapper = CUDAGraphWrapper(model, vllm_config, CUDAGraphMode.FULL)
    input_1 = torch.randn(1, 10, device=DEVICE_TYPE)

    # Setup: Inner model is wrapped with PIECEWISE, outer with FULL
    inner_model = SimpleMLP().to(DEVICE_TYPE)
    piecewise_wrapper = CUDAGraphWrapper(
        inner_model, vllm_config, CUDAGraphMode.PIECEWISE
    )
    inner_model.forward = MagicMock(wraps=inner_model.forward)
    outer_model = SimpleMLP().to(DEVICE_TYPE)
    # When outer model is called, it calls the piecewise_wrapper
    outer_model.forward = MagicMock(
        wraps=outer_model.forward, side_effect=piecewise_wrapper
    )
    full_wrapper = CUDAGraphWrapper(outer_model, vllm_config, CUDAGraphMode.FULL)

    desc_1 = BatchDescriptor(num_tokens=1)

    # 0. global warmup
    with set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        full_wrapper(input_1)

    # --- Test runtime mode FULL---
    # Run with FULL mode context. Expect outer wrapper to capture.
    # The inner mock should be called once inside the graph capture.
    outer_model.forward.reset_mock()
    inner_model.forward.reset_mock()
    action = _run_and_monitor_call(
        full_wrapper, input_1, CUDAGraphMode.FULL, desc_1, vllm_config
    )
    assert action == "capture_global"
    assert outer_model.forward.call_count == 1
    assert inner_model.forward.call_count == 1

    # Run again. Expect outer wrapper to replay.
    # The outer model should NOT be called because the whole graph
    # is replayed.
    action = _run_and_monitor_call(
        full_wrapper, input_1, CUDAGraphMode.FULL, desc_1, vllm_config
    )
    assert action == "replay"
    assert outer_model.forward.call_count == 1  # No new call
    assert inner_model.forward.call_count == 1

    # --- Test runtime mode PIECEWISE ---
    outer_model.forward.reset_mock()
    inner_model.forward.reset_mock()
    # Run with PIECEWISE mode context.
    # Expect outer wrapper to bypass and call inner wrapper.
    # Inner wrapper should capture.
    action = _run_and_monitor_call(
        full_wrapper, input_1, CUDAGraphMode.PIECEWISE, desc_1, vllm_config
    )
    assert action == "capture_global"
    assert outer_model.forward.call_count == 1
    assert inner_model.forward.call_count == 1

    # Run again with PIECEWISE.
    # Outer bypasses, inner replays.
    action = _run_and_monitor_call(
        full_wrapper, input_1, CUDAGraphMode.PIECEWISE, desc_1, vllm_config
    )
    assert action == "bypass"
    assert outer_model.forward.call_count == 2
    assert inner_model.forward.call_count == 1
