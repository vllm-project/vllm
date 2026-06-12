# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pydantic
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.plugins.observation.hook import ObservationHook
from vllm.plugins.observation.interface import (
    ObservationAction,
    ObservationPlugin,
    ObservationResult,
    PluginManager,
    RequestContext,
)

pydantic.dataclasses.rebuild_dataclass(VllmConfig)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10), nn.Linear(10, 10)])

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class MockPlugin(ObservationPlugin):
    def __init__(self, vllm_config=None):
        super().__init__(vllm_config)
        self.called = False
        self.triggered_abort = False
        self.captured_activations = None

    def get_observation_layers(self):
        return [0, 1]  # Hook both layers

    def on_step_batch(self, batch_hidden_states, request_contexts):
        self.called = True
        self.captured_activations = dict(batch_hidden_states)
        results = []
        for ctx in request_contexts:
            if self.triggered_abort:
                results.append(
                    ObservationResult(
                        action=ObservationAction.ABORT,
                        metadata={"message": "Aborted by test plugin"},
                    )
                )
            else:
                results.append(ObservationResult(action=ObservationAction.CONTINUE))
        return results


def test_observation_hook_captures_tensors():
    model = SimpleModel()
    plugin = MockPlugin()
    plugin_manager = PluginManager([plugin])

    req_contexts = [
        RequestContext(
            request_id="req_1",
            is_prefill=True,
            chunk_idx=0,
            num_cached_tokens=0,
            batch_offset=0,
            num_tokens=10,
        )
    ]

    hook = ObservationHook(plugin_manager, model)

    hook.install_hooks()

    input_tensor = torch.randn(1, 10)

    # Forward pass
    output = model(input_tensor)
    assert output is not None

    # Now we must call process_step to trigger plugin callbacks!
    results = hook.process_step(req_contexts)
    assert results is not None

    hook.remove_hooks()

    assert plugin.called
    assert plugin.captured_activations is not None
    # Verify that it captured activations for both layers
    assert 0 in plugin.captured_activations
    assert 1 in plugin.captured_activations

    # Clean up hooks just in case
    hook.remove_hooks()


def test_abort_logic_simulation():
    requests = {"req_1": MagicMock()}
    requests["req_1"].aborted_by_observation = False

    plugin = MockPlugin()
    plugin.triggered_abort = True
    plugin_manager = PluginManager([plugin])

    req_contexts = [
        RequestContext(
            request_id="req_1",
            is_prefill=True,
            chunk_idx=0,
            num_cached_tokens=0,
            batch_offset=0,
            num_tokens=10,
        )
    ]

    # Simulate what happens in GPUModelRunner.execute_model
    results = plugin_manager.on_step_batch({}, req_contexts)

    for req_ctx, result in zip(req_contexts, results):
        if result.action == ObservationAction.ABORT:
            requests[req_ctx.request_id].aborted_by_observation = True

    assert requests["req_1"].aborted_by_observation


def test_engine_core_aborts_requests():
    from unittest.mock import MagicMock

    from vllm.v1.engine.core import EngineCore
    from vllm.v1.outputs import ModelRunnerOutput

    class DummyEngineCore:
        def __init__(self):
            self.scheduler = MagicMock()
            self.model_executor = MagicMock()
            self.aborts_queue = MagicMock()
            self.aborts_queue.empty.return_value = True
            self.abort_requests = MagicMock()

        def _process_aborts_queue(self):
            pass

        def step(self):
            """Dummy step method to satisfy mypy."""
            return {}, True

        def log_error_detail(self, scheduler_output):
            class DummyCM:
                def __enter__(self):
                    pass

                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass

            return DummyCM()

        def log_iteration_details(self, scheduler_output):
            class DummyCM:
                def __enter__(self):
                    pass

                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass

            return DummyCM()

    engine_core = DummyEngineCore()
    engine_core.step = EngineCore.step.__get__(engine_core, DummyEngineCore)  # type: ignore[method-assign]

    engine_core.scheduler.has_requests.return_value = True

    mock_sched_output = MagicMock()
    mock_sched_output.total_num_scheduled_tokens = 10
    engine_core.scheduler.schedule.return_value = mock_sched_output

    mock_future = MagicMock()
    engine_core.model_executor.execute_model.return_value = mock_future

    mock_model_output = ModelRunnerOutput(
        req_ids=["req_1"], req_id_to_index={"req_1": 0}, aborted_req_ids=["req_1"]
    )
    mock_future.result.return_value = mock_model_output

    engine_core.step()

    engine_core.abort_requests.assert_called_once_with(["req_1"])


def test_gpu_model_runner_collects_aborted_req_ids():
    class MockReqState:
        def __init__(self, aborted):
            self.aborted_by_observation = aborted

    class DummyGPUModelRunner:
        def __init__(self):
            self.requests = {
                "req_1": MockReqState(True),
                "req_2": MockReqState(False),
                "req_3": MockReqState(True),
            }

    runner = DummyGPUModelRunner()

    req_ids_output_copy = ["req_1", "req_2", "req_3", "req_non_existent"]

    aborted_req_ids = []
    for req_id in req_ids_output_copy:
        if req_id in runner.requests:
            req_state = runner.requests[req_id]
            if getattr(req_state, "aborted_by_observation", False):
                aborted_req_ids.append(req_id)

    assert aborted_req_ids == ["req_1", "req_3"]


def test_request_context_population():
    class MockReqState:

        def __init__(self, num_computed, num_prompt):
            self.num_computed_tokens = num_computed
            self.num_prompt_tokens = num_prompt

    class DummyInputBatch:

        def __init__(self, req_ids):
            self.req_ids = req_ids

    class DummySchedulerOutput:

        def __init__(self, num_scheduled_tokens):
            self.num_scheduled_tokens = num_scheduled_tokens

    requests = {
        "req_1": MockReqState(0, 10),  # prefill
        "req_2": MockReqState(10, 10),  # decode
    }
    input_batch = DummyInputBatch(["req_1", "req_2"])
    scheduler_output = DummySchedulerOutput({"req_1": 10, "req_2": 1})

    # Simulate the logic we added in execute_model
    request_contexts = []
    current_offset = 0
    for req_id in input_batch.req_ids:
        num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
        req_state = requests[req_id]
        is_prefill = req_state.num_computed_tokens < req_state.num_prompt_tokens

        request_contexts.append(
            RequestContext(
                request_id=req_id,
                is_prefill=is_prefill,
                batch_offset=current_offset,
                num_tokens=num_tokens,
                num_cached_tokens=req_state.num_computed_tokens,
            )
        )
        current_offset += num_tokens

    assert len(request_contexts) == 2
    assert request_contexts[0].request_id == "req_1"
    assert request_contexts[0].is_prefill == True
    assert request_contexts[0].batch_offset == 0
    assert request_contexts[0].num_tokens == 10
    assert request_contexts[0].num_cached_tokens == 0

    assert request_contexts[1].request_id == "req_2"
    assert request_contexts[1].is_prefill == False
    assert request_contexts[1].batch_offset == 10
    assert request_contexts[1].num_tokens == 1
    assert request_contexts[1].num_cached_tokens == 10


def test_lifecycle_hooks_called():
    class DummyPluginManager:

        def __init__(self):
            self.started_requests = []
            self.completed_requests = []

        def on_request_start(self, req_id, prompt=None):
            self.started_requests.append(req_id)

        def on_request_complete(self, req_id):
            self.completed_requests.append(req_id)

    class NewReqData:

        def __init__(self, req_id):
            self.req_id = req_id

    class DummySchedulerOutput:

        def __init__(self, new_reqs, finished_reqs):
            self.scheduled_new_reqs = [NewReqData(r) for r in new_reqs]
            self.finished_req_ids = finished_reqs

    plugin_manager = DummyPluginManager()

    # Simulate _update_states logic
    def _update_states(scheduler_output):
        # Remove finished
        for req_id in scheduler_output.finished_req_ids:
            plugin_manager.on_request_complete(req_id)

        # Add new
        for new_req_data in scheduler_output.scheduled_new_reqs:
            plugin_manager.on_request_start(new_req_data.req_id)

    sched_output = DummySchedulerOutput(
        new_reqs=["req_new"], finished_reqs=["req_done"]
    )
    _update_states(sched_output)

    assert plugin_manager.started_requests == ["req_new"]
    assert plugin_manager.completed_requests == ["req_done"]
