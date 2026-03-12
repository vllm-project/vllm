# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from vllm.entrypoints.llm import LLM
from vllm.model_executor.models.moondream3 import (
    DetectPointStateManager,
    Moondream3PerRequestStateAdapter,
)
from vllm.model_executor.models.moondream3_io import (
    MOONDREAM3_RESULT_DATA_KEY,
    MOONDREAM3_RESULT_MODE_KEY,
    build_moondream3_detect_point_prompt,
    encode_moondream3_detect_point_output,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.plugins.io_processors.moondream3 import (
    Moondream3DetectPointIOProcessor,
    Moondream3DetectPointRequest,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


def _make_request_output(
    text: str = "",
    model_extra_output: dict[str, torch.Tensor] | None = None,
) -> RequestOutput:
    return RequestOutput(
        request_id="req-1",
        prompt="prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[],
                cumulative_logprob=None,
                logprobs=None,
                model_extra_output=model_extra_output,
            )
        ],
        finished=True,
    )


def test_moondream3_io_processor_decodes_raw_detect_output():
    processor = Moondream3DetectPointIOProcessor(SimpleNamespace(), MagicMock())
    request = processor.parse_data(
        {
            "task": "detect",
            "object": "sign",
            "image": object(),
            "max_objects": 4,
        }
    )

    assert request == Moondream3DetectPointRequest(
        task="detect",
        target="sign",
        image=request.image,
        max_objects=4,
    )

    params = processor.merge_sampling_params_for_prompt(
        request,
        SamplingParams(max_tokens=8),
    )
    assert params.extra_args == {
        "moondream3_task": "detect",
        "moondream3_max_objects": 4,
    }

    prompt = processor.pre_process(request)
    assert prompt == {
        "prompt": build_moondream3_detect_point_prompt("detect", "sign"),
        "multi_modal_data": {"image": request.image},
    }

    output = _make_request_output(
        model_extra_output=encode_moondream3_detect_point_output(
            "detect",
            [
                {
                    "x_min": 0.1,
                    "y_min": 0.2,
                    "x_max": 0.3,
                    "y_max": 0.4,
                }
            ],
        )
    )
    processed = processor.post_process_generate(output)
    decoded = json.loads(processed.outputs[0].text)
    assert len(decoded["objects"]) == 1
    assert decoded["objects"][0]["x_min"] == pytest.approx(0.1)
    assert decoded["objects"][0]["y_min"] == pytest.approx(0.2)
    assert decoded["objects"][0]["x_max"] == pytest.approx(0.3)
    assert decoded["objects"][0]["y_max"] == pytest.approx(0.4)


def test_get_io_processor_accepts_qualname():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(to_dict=lambda: {}),
        )
    )
    renderer = MagicMock()

    processor = get_io_processor(
        vllm_config,
        renderer,
        "vllm.plugins.io_processors.moondream3.Moondream3DetectPointIOProcessor",
    )

    assert isinstance(processor, Moondream3DetectPointIOProcessor)


def test_llm_generate_uses_io_processor_data_prompt():
    llm = object.__new__(LLM)
    llm.model_config = SimpleNamespace(runner_type="generate")
    llm.io_processor = MagicMock()

    sampling_params = SamplingParams(max_tokens=4)
    validated_prompt = object()
    merged_params = SamplingParams(max_tokens=7)
    rendered_prompt = {"prompt": "engine prompt"}
    expected_output = _make_request_output("decoded")
    captured = {}

    llm.io_processor.parse_data.return_value = validated_prompt
    llm.io_processor.pre_process.return_value = rendered_prompt
    llm.io_processor.merge_sampling_params_for_prompt.return_value = merged_params
    llm.io_processor.post_process_generate.side_effect = lambda output: output

    def fake_run_completion(**kwargs):
        captured.update(kwargs)
        return [expected_output]

    llm._run_completion = fake_run_completion  # type: ignore[method-assign]

    outputs = llm.generate(
        {"data": {"task": "detect", "object": "sign", "image": object()}},
        sampling_params,
        use_tqdm=False,
    )

    assert outputs == [expected_output]
    assert captured["prompts"] == [rendered_prompt]
    assert captured["params"] == [merged_params]
    llm.io_processor.parse_data.assert_called_once()
    llm.io_processor.pre_process.assert_called_once_with(prompt=validated_prompt)
    llm.io_processor.merge_sampling_params_for_prompt.assert_called_once()
    llm.io_processor.post_process_generate.assert_called_once_with(expected_output)


def test_llm_generate_passthrough_for_regular_prompt():
    llm = object.__new__(LLM)
    llm.model_config = SimpleNamespace(runner_type="generate")
    llm.io_processor = MagicMock()

    expected_output = _make_request_output("decoded")
    captured = {}

    llm.io_processor.post_process_generate.side_effect = lambda output: output

    def fake_run_completion(**kwargs):
        captured.update(kwargs)
        return [expected_output]

    llm._run_completion = fake_run_completion  # type: ignore[method-assign]

    outputs = llm.generate(
        "plain prompt",
        SamplingParams(max_tokens=3),
        use_tqdm=False,
    )

    assert outputs == [expected_output]
    assert captured["prompts"] == "plain prompt"
    assert isinstance(captured["params"], SamplingParams)
    assert captured["params"].max_tokens == 3
    llm.io_processor.merge_sampling_params.assert_not_called()
    llm.io_processor.post_process_generate.assert_not_called()


@pytest.mark.asyncio
async def test_async_llm_prepares_data_prompt_with_io_processor():
    async_llm = object.__new__(AsyncLLM)
    async_llm.io_processor = MagicMock()

    original_params = SamplingParams(max_tokens=2)
    prompt_params = SamplingParams(max_tokens=6)
    validated_prompt = object()
    rendered_prompt = {"prompt": "engine prompt"}

    async_llm.io_processor.parse_data.return_value = validated_prompt
    async_llm.io_processor.pre_process_async = AsyncMock(return_value=rendered_prompt)
    async_llm.io_processor.merge_sampling_params_for_prompt.return_value = prompt_params

    prompt, params = await async_llm._prepare_generate_io_processor_input(
        {"data": {"task": "point", "object": "sign", "image": object()}},
        original_params,
        "req-1",
    )

    assert prompt == rendered_prompt
    assert params == prompt_params
    async_llm.io_processor.parse_data.assert_called_once()
    async_llm.io_processor.pre_process_async.assert_awaited_once_with(
        prompt=validated_prompt,
        request_id="req-1",
    )
    async_llm.io_processor.merge_sampling_params_for_prompt.assert_called_once()
    call_args = async_llm.io_processor.merge_sampling_params_for_prompt.call_args
    assert call_args.args[0] is validated_prompt
    assert isinstance(call_args.args[1], SamplingParams)
    assert call_args.args[1] is not original_params
    assert call_args.args[1].max_tokens == original_params.max_tokens


def test_moondream3_adapter_emits_raw_payload_for_io_processor(monkeypatch):
    manager = DetectPointStateManager()
    adapter = Moondream3PerRequestStateAdapter(
        SimpleNamespace(detect_point_manager=manager)
    )

    adapter.on_new_request(
        req_id="req-1",
        sampling_params=SimpleNamespace(
            extra_args={
                "moondream3_task": "point",
            }
        ),
    )
    state = manager.get_state("req-1")
    assert state is not None
    state.objects.append({"x": 0.25, "y": 0.75})

    monkeypatch.setattr(
        "vllm.model_executor.models.moondream3.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    payload = adapter.get_per_request_extra_output(req_ids=["req-1"])
    assert payload is not None
    req_payload = payload["req-1"]
    assert set(req_payload) == {
        MOONDREAM3_RESULT_MODE_KEY,
        MOONDREAM3_RESULT_DATA_KEY,
    }
    assert req_payload[MOONDREAM3_RESULT_MODE_KEY].tolist() == [1]
    assert req_payload[MOONDREAM3_RESULT_DATA_KEY].tolist() == pytest.approx(
        [0.25, 0.75]
    )
