# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.worker.gpu import model_runner as model_runner_module
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner

pytestmark = pytest.mark.cpu_test


def _make_non_last_runner(
    kv_connector_output: KVConnectorOutput | None,
) -> GPUModelRunner:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.execute_model_state = ExecuteModelState(
        input_batch=cast(Any, SimpleNamespace(num_reqs=1)),
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        finished_req_ids=set(),
    )
    runner.kv_connector_output = kv_connector_output
    runner.is_last_pp_rank = False
    runner.num_speculative_steps = 0
    runner.eplb = SimpleNamespace(step=lambda **_: None)
    return runner


def _patch_non_last_pp_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_runner_module,
        "pp_receive",
        lambda num_reqs, max_sample_len: ([], 0, 0),
    )
    monkeypatch.setattr(
        GPUModelRunner,
        "postprocess",
        lambda self, input_batch, sampled, num_sampled, num_rejected: None,
    )


def test_non_last_pp_without_kv_output_returns_empty_model_runner_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_non_last_pp_side_effects(monkeypatch)
    runner = _make_non_last_runner(None)

    output = runner.sample_tokens(None)

    assert output is EMPTY_MODEL_RUNNER_OUTPUT


def test_non_last_pp_with_kv_output_returns_copy_carrying_kv_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_non_last_pp_side_effects(monkeypatch)
    kv_connector_output = KVConnectorOutput(finished_sending={"req-0"})
    runner = _make_non_last_runner(kv_connector_output)

    output = runner.sample_tokens(None)

    assert output is not EMPTY_MODEL_RUNNER_OUTPUT
    assert output is not None
    assert output.kv_connector_output is kv_connector_output
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None


def test_last_pp_rank_still_returns_regular_model_runner_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeAsyncOutput:
        def __init__(self, model_runner_output, *args, **kwargs) -> None:
            self.model_runner_output = model_runner_output

        def get_output(self):
            return self.model_runner_output

    monkeypatch.setattr(model_runner_module, "AsyncOutput", _FakeAsyncOutput)
    monkeypatch.setattr(
        GPUModelRunner,
        "sample",
        lambda self, hidden_states, input_batch, grammar_output: (
            SimpleNamespace(sampled_token_ids=[[1]]),
            1,
            0,
        ),
    )
    monkeypatch.setattr(
        GPUModelRunner,
        "postprocess",
        lambda self, input_batch, sampled, num_sampled, num_rejected: None,
    )

    kv_connector_output = KVConnectorOutput(finished_recving={"req-0"})
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.execute_model_state = ExecuteModelState(
        input_batch=cast(Any, SimpleNamespace(num_reqs=1, req_ids=["req-0"])),
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=cast(Any, object()),
        aux_hidden_states=None,
        finished_req_ids=set(),
    )
    runner.kv_connector = SimpleNamespace(
        post_forward=lambda finished_req_ids: kv_connector_output,
    )
    runner.kv_connector_output = None
    runner.is_last_pp_rank = True
    runner.use_pp = False
    runner.use_async_scheduling = False
    runner.speculator = None
    runner.main_stream = None
    runner.output_copy_stream = None
    runner.eplb = SimpleNamespace(step=lambda **_: None)
    runner.model = SimpleNamespace(compute_logits=lambda *args, **kwargs: None)
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=lambda *args, **kwargs: {}
    )
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=None),
        num_computed_tokens=SimpleNamespace(gpu=None),
        prompt_len=SimpleNamespace(np=None),
        prefill_len=SimpleNamespace(np=None),
        num_computed_prefill_tokens=None,
    )

    output = runner.sample_tokens(None)

    assert output is not None
    assert output.req_ids == ["req-0"]
    assert output.kv_connector_output is kv_connector_output
