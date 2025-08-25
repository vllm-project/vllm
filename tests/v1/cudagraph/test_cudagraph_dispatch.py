# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from tests.utils import create_new_process_for_each_test
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (CompilationConfig, CompilationLevel, CUDAGraphMode,
                         ParallelConfig, SchedulerConfig, VllmConfig)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.platforms import current_platform
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher


# Helper MLP for testing
class SimpleMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _create_vllm_config(compilation_config: CompilationConfig,
                        max_num_seqs: int = 8) -> MagicMock:
    mock_config = MagicMock(spec=VllmConfig)
    mock_config.compilation_config = compilation_config
    mock_config.scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs)
    mock_config.parallel_config = ParallelConfig()

    # Mimic the behavior of VllmConfig.__post_init__()
    if compilation_config.level == CompilationLevel.PIECEWISE:
        compilation_config.set_splitting_ops_for_v1()

    return mock_config


class TestCudagraphDispatcher:

    @pytest.mark.parametrize(
        "params",
        [
            # Test case 0: Full CG for mixed batches, no separate routine
            {
                "case_id": 0,
                "cudagraph_mode": "FULL",
                "compilation_level": CompilationLevel.NO_COMPILATION,
            },
            # Test case 1: Full CG for uniform batches, piecewise for mixed
            {
                "case_id": 1,
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "compilation_level": CompilationLevel.PIECEWISE,
            },
            # Test case 2: Full CG for uniform batches, no CG for mixed
            {
                "case_id": 2,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "compilation_level": CompilationLevel.NO_COMPILATION,
            },
            # Test case 3: Piecewise for all
            {
                "case_id": 3,
                "cudagraph_mode": "PIECEWISE",
                "compilation_level": CompilationLevel.PIECEWISE,
            },
        ])
    def test_dispatcher(self, params):
        # Setup dispatcher
        comp_config = CompilationConfig(
            cudagraph_mode=params["cudagraph_mode"],
            level=params["compilation_level"],
            cudagraph_capture_sizes=[1, 8])

        config = _create_vllm_config(comp_config, max_num_seqs=8)
        dispatcher = CudagraphDispatcher(config)
        dispatcher.initialize_cudagraph_keys(
            cudagraph_mode=comp_config.cudagraph_mode,
            uniform_decode_query_len=1)

        # Verify the key is initialized correctly
        if params["cudagraph_mode"] in ["FULL_AND_PIECEWISE", "PIECEWISE"]:
            assert len(dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]) == 2
        else:
            assert len(dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]) == 0
        if params["cudagraph_mode"] not in ["NONE", "PIECEWISE"]:
            assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == 2
        else:
            assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == 0

        # Test dispatch logic
        # 1. non-uniform batch, size in cudagraph size list
        desc_full_exact = BatchDescriptor(num_tokens=8, uniform_decode=False)
        rt_mode, key = dispatcher.dispatch(desc_full_exact)
        if params["cudagraph_mode"] == "FULL":
            assert rt_mode == CUDAGraphMode.FULL
            assert key == desc_full_exact
        elif params["cudagraph_mode"] in ["FULL_AND_PIECEWISE", "PIECEWISE"]:
            assert rt_mode == CUDAGraphMode.PIECEWISE
            assert key == desc_full_exact
        else:
            assert rt_mode == CUDAGraphMode.NONE

        # 2. uniform decode batch, size in cudagraph size list
        desc_uniform_exact = BatchDescriptor(num_tokens=8, uniform_decode=True)
        rt_mode, key = dispatcher.dispatch(desc_uniform_exact)
        if params["cudagraph_mode"] == "FULL":
            assert rt_mode == CUDAGraphMode.FULL
            assert key == desc_uniform_exact.non_uniform
        elif params["cudagraph_mode"] in [
                "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"
        ]:
            assert rt_mode == CUDAGraphMode.FULL
            assert key == desc_uniform_exact
        elif params["cudagraph_mode"] == "PIECEWISE":
            assert rt_mode == CUDAGraphMode.PIECEWISE
            assert key == desc_uniform_exact.non_uniform
        else:
            assert rt_mode == CUDAGraphMode.NONE

        # 3. No key match
        desc_no_match = BatchDescriptor(num_tokens=15, uniform_decode=False)
        rt_mode, key = dispatcher.dispatch(desc_no_match)
        assert rt_mode == CUDAGraphMode.NONE
        assert key is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
class TestCUDAGraphWrapper:

    def setup_method(self):
        self.vllm_config = _create_vllm_config(CompilationConfig())
        self.model = SimpleMLP().to("cuda")
        self.persistent_input_buffer = torch.zeros(1, 10, device="cuda")
        self.input_tensor = torch.randn(1, 10, device="cuda")

    @create_new_process_for_each_test("spawn")
    def test_capture_and_replay(self):
        wrapper = CUDAGraphWrapper(self.model,
                                   self.vllm_config,
                                   runtime_mode=CUDAGraphMode.FULL)
        batch_descriptor = BatchDescriptor(num_tokens=10)

        # 0. global warmup
        with set_forward_context(attn_metadata=None,
                                 vllm_config=self.vllm_config,
                                 cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                 batch_descriptor=None):
            wrapper(self.input_tensor)

        # 1. Capture
        with set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=batch_descriptor),\
            patch("torch.cuda.graph",
                       wraps=torch.cuda.graph) as mock_cuda_graph:
            output1 = wrapper(self.input_tensor)
            # capturing phase should generate a zero output
            assert torch.allclose(output1, torch.zeros_like(output1))
            mock_cuda_graph.assert_called_once()

        assert batch_descriptor in wrapper.concrete_cudagraph_entries
        entry = wrapper.concrete_cudagraph_entries[batch_descriptor]
        assert entry.cudagraph is not None

        # 2. Replay
        with set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=batch_descriptor),\
            patch.object(entry.cudagraph, 'replay',
                         wraps=entry.cudagraph.replay) as mock_replay:
            output2 = wrapper(self.input_tensor)
            mock_replay.assert_called_once()

        # Compare with eager output
        eager_output = self.model(self.input_tensor)
        torch.testing.assert_close(eager_output, output2)

    @create_new_process_for_each_test("spawn")
    def test_bypass_on_mode_mismatch(self):
        wrapper = CUDAGraphWrapper(self.model,
                                   self.vllm_config,
                                   runtime_mode=CUDAGraphMode.FULL)
        batch_descriptor = BatchDescriptor(num_tokens=10)

        with set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                batch_descriptor=batch_descriptor), \
            patch('torch.cuda.graph',
                  wraps=torch.cuda.graph) as mock_cuda_graph, \
            patch.object(self.model, 'forward',
                         wraps=self.model.forward) as mock_forward:
            wrapper(self.input_tensor)
            mock_cuda_graph.assert_not_called()
            mock_forward.assert_called_once()
        assert not wrapper.concrete_cudagraph_entries

    @create_new_process_for_each_test("spawn")
    def test_bypass_on_mode_none(self):
        wrapper = CUDAGraphWrapper(self.model,
                                   self.vllm_config,
                                   runtime_mode=CUDAGraphMode.FULL)
        batch_descriptor = BatchDescriptor(num_tokens=10)

        with set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=batch_descriptor), \
            patch('torch.cuda.graph',
                  wraps=torch.cuda.graph) as mock_cuda_graph:
            wrapper(self.input_tensor)
            mock_cuda_graph.assert_not_called()
        assert not wrapper.concrete_cudagraph_entries


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
class TestCudagraphIntegration:

    def setup_method(self):
        # only FULL mode for non-uniform batches
        self.comp_config = CompilationConfig(level=CompilationLevel.PIECEWISE,
                                             cudagraph_mode="FULL",
                                             cudagraph_capture_sizes=[10, 20])
        self.vllm_config = _create_vllm_config(self.comp_config)
        self.dispatcher = CudagraphDispatcher(self.vllm_config)
        self.dispatcher.initialize_cudagraph_keys(
            self.comp_config.cudagraph_mode, uniform_decode_query_len=1)

    def _run_and_monitor_call(self, wrapper, input_tensor, runtime_mode,
                              batch_descriptor):
        """Helper to run a single call and monitor the action."""

        with patch('torch.cuda.graph',
                wraps=torch.cuda.graph) as mock_graph_context, \
            patch.object(wrapper, 'runnable',
                        wraps=wrapper.runnable) as mock_runnable:

            entry = wrapper.concrete_cudagraph_entries.get(
                batch_descriptor, None)

            context = set_forward_context(attn_metadata=None,
                                          vllm_config=self.vllm_config,
                                          cudagraph_runtime_mode=runtime_mode,
                                          batch_descriptor=batch_descriptor)
            mock_replay = MagicMock()
            if entry and entry.cudagraph:
                with context, \
                    patch.object(entry.cudagraph, 'replay',
                                new_callable=MagicMock) as mock_replay:
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
    def test_capture_replay_bypass_logic(self):
        model = SimpleMLP().to("cuda")
        full_wrapper = CUDAGraphWrapper(model, self.vllm_config,
                                        CUDAGraphMode.FULL)
        max_bs = 16
        persistent_input_buffer = torch.zeros(max_bs, 10, device="cuda")
        input_1 = persistent_input_buffer[:1]
        input_2 = persistent_input_buffer[:2]
        input_3 = persistent_input_buffer[:3]

        desc_1 = BatchDescriptor(num_tokens=1)
        desc_2 = BatchDescriptor(num_tokens=2)
        desc_3_unseen = BatchDescriptor(num_tokens=3)

        # 0. global warmup
        with set_forward_context(attn_metadata=None,
                                 vllm_config=self.vllm_config,
                                 cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                 batch_descriptor=None):
            full_wrapper(input_1)

        rt_mode, key = self.dispatcher.dispatch(desc_1)
        # 1. Capture first shape
        action = self._run_and_monitor_call(full_wrapper, input_1, rt_mode,
                                            key)
        assert action == "capture_global"

        # 2. Replay first shape
        action = self._run_and_monitor_call(full_wrapper, input_1, rt_mode,
                                            key)
        assert action == "replay"

        rt_mode, key = self.dispatcher.dispatch(desc_2)
        # 3. Capture second shape
        action = self._run_and_monitor_call(full_wrapper, input_2, rt_mode,
                                            key)
        assert action == "capture_global"

        # 4. Replay second shape
        action = self._run_and_monitor_call(full_wrapper, input_2,
                                            CUDAGraphMode.FULL, desc_2)
        assert action == "replay"

        # 5. Bypass if no key match
        rt_mode, key = self.dispatcher.dispatch(desc_3_unseen)
        assert rt_mode == CUDAGraphMode.NONE
        action = self._run_and_monitor_call(full_wrapper, input_3, rt_mode,
                                            key)
        assert action == "bypass"

        # capture unseen shape is not allowed after disable
        set_cudagraph_capturing_enabled(False)
        with pytest.raises(RuntimeError):
            self._run_and_monitor_call(full_wrapper, input_3,
                                       CUDAGraphMode.FULL, desc_3_unseen)
        set_cudagraph_capturing_enabled(True)

    @create_new_process_for_each_test("spawn")
    def test_nested_wrappers(self):
        """Tests a scenario with a PIECEWISE wrapper inside a FULL one."""
        model = SimpleMLP().to("cuda")
        full_wrapper = CUDAGraphWrapper(model, self.vllm_config,
                                        CUDAGraphMode.FULL)
        input_1 = torch.randn(1, 10, device="cuda")

        # Setup: Inner model is wrapped with PIECEWISE, outer with FULL
        inner_model = SimpleMLP().to("cuda")
        piecewise_wrapper = CUDAGraphWrapper(inner_model, self.vllm_config,
                                             CUDAGraphMode.PIECEWISE)
        inner_model.forward = MagicMock(wraps=inner_model.forward)
        outer_model = SimpleMLP().to("cuda")
        # When outer model is called, it calls the piecewise_wrapper
        outer_model.forward = MagicMock(wraps=outer_model.forward,
                                        side_effect=piecewise_wrapper)
        full_wrapper = CUDAGraphWrapper(outer_model, self.vllm_config,
                                        CUDAGraphMode.FULL)

        desc_1 = BatchDescriptor(num_tokens=1)

        # 0. global warmup
        with set_forward_context(attn_metadata=None,
                                 vllm_config=self.vllm_config,
                                 cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                 batch_descriptor=None):
            full_wrapper(input_1)

        # --- Test runtime mode FULL---
        # Run with FULL mode context. Expect outer wrapper to capture.
        # The inner mock should be called once inside the graph capture.
        outer_model.forward.reset_mock()
        inner_model.forward.reset_mock()
        action = self._run_and_monitor_call(full_wrapper, input_1,
                                            CUDAGraphMode.FULL, desc_1)
        assert action == "capture_global"
        assert outer_model.forward.call_count == 1
        assert inner_model.forward.call_count == 1

        # Run again. Expect outer wrapper to replay.
        # The outer model should NOT be called because the whole graph
        # is replayed.
        action = self._run_and_monitor_call(full_wrapper, input_1,
                                            CUDAGraphMode.FULL, desc_1)
        assert action == "replay"
        assert outer_model.forward.call_count == 1  # No new call
        assert inner_model.forward.call_count == 1

        # --- Test runtime mode PIECEWISE ---
        outer_model.forward.reset_mock()
        inner_model.forward.reset_mock()
        # Run with PIECEWISE mode context.
        # Expect outer wrapper to bypass and call inner wrapper.
        # Inner wrapper should capture.
        action = self._run_and_monitor_call(full_wrapper, input_1,
                                            CUDAGraphMode.PIECEWISE, desc_1)
        assert action == "capture_global"
        assert outer_model.forward.call_count == 1
        assert inner_model.forward.call_count == 1

        # Run again with PIECEWISE.
        # Outer bypasses, inner replays.
        action = self._run_and_monitor_call(full_wrapper, input_1,
                                            CUDAGraphMode.PIECEWISE, desc_1)
        assert action == "bypass"
        assert outer_model.forward.call_count == 2
        assert inner_model.forward.call_count == 1
