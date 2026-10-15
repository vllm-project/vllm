# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured outputs must remain constrained across pipelined draft handoffs."""

import json

import jsonschema
import pytest

from tests.utils import multi_gpu_only
from vllm import SamplingParams
from vllm.outputs import RequestOutput
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.metrics.reader import Metric

from ..utils import compute_acceptance_len, compute_acceptance_rate
from .test_draft_model import ArgsTest, vllm_runner_kwargs


def case_to_test(pp_size: int, tp_size: int) -> ArgsTest:
    """This test case should surface problems arising from combining
    speculative decoding, structured outputs and pipeline parallelism."""
    return ArgsTest(
        target_model="Qwen/Qwen3-4B",
        draft_model="z-lab/Qwen3-4B-DFlash-b16",
        sampling_config=SamplingParams(temperature=0, max_tokens=256),
        num_speculative_tokens=15,
        method="dflash",
        # reference for AL and AR is TP=1 PP=1
        expected_acceptance_rate=0.0990 * 0.95,  # ref - 5% tolerance
        expected_acceptance_len=2.3215 * 0.95,  # ref - 5% tolerance
        pipeline_parallel_size=pp_size,
        target_tensor_parallel_size=tp_size,
        async_scheduling=True,  # force PP + async scheduling
        max_num_seqs=8,  # force multiple PP batches
        engine_kwargs={
            "structured_outputs_config": {
                "backend": "xgrammar",
                "disable_any_whitespace": True,
            }
        },
    )


@multi_gpu_only(num_gpus=2)
@pytest.mark.parametrize(
    "args",
    [
        # (2026-10-13) The cases below pass, but are commented out to speed up CI.
        # PP1/TP1: AR=0.0990 AL=2.3215
        # pytest.param(case_to_test(pp_size=1, tp_size=1), id="pp1-tp1"),
        # PP1/TP2: AR=0.0999 AL=2.3351
        # pytest.param(case_to_test(pp_size=1, tp_size=2), id="pp1-tp2"),
        pytest.param(case_to_test(pp_size=2, tp_size=1), id="pp2-tp1"),
    ],
)
def test_specdec_pp_structured_output(args: ArgsTest, vllm_runner) -> None:
    messages, sampling_params = _create_requests(
        args.sampling_config, num_reqs=100, num_structured=50
    )
    kwargs = vllm_runner_kwargs(args)
    with vllm_runner(args.target_model, **kwargs) as runner:
        outputs = runner.llm.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        metrics = runner.llm.get_metrics()
    _assert_specdec_metrics(
        metrics,
        expected_acceptance_rate=args.expected_acceptance_rate,
        expected_acceptance_len=args.expected_acceptance_len,
    )
    _assert_outputs_are_structured(outputs, sampling_params)


def _create_requests(sampling: SamplingParams, num_reqs: int, num_structured: int):
    messages = []
    listof_params: list[SamplingParams] = []
    for i in range(num_reqs):
        is_structured = i < num_structured
        messages.append(_make_messages(i, is_structured))
        params = sampling.clone()
        if is_structured:
            schema = {
                "type": "array",
                "minItems": 16,
                "maxItems": 16,
                "items": {"type": "string", "enum": ["red", "blue"]},
            }
            params.structured_outputs = StructuredOutputsParams(json=schema)
        listof_params.append(params)
    return messages, listof_params


def _make_messages(index: int, is_structured: bool):
    task = (
        "list sixteen animals as JSON."
        if is_structured
        else "write a detailed travel story."
    )
    messages = [{"role": "user", "content": f"Case {index}: {task}"}]
    return messages


def _assert_specdec_metrics(
    metrics: list[Metric],
    expected_acceptance_rate: float,
    expected_acceptance_len: float,
) -> None:
    actual_rate = compute_acceptance_rate(metrics)
    actual_len = compute_acceptance_len(metrics)
    print(f"acceptance_rate={actual_rate:.4f}, acceptance_len={actual_len:.4f}")
    assert actual_rate >= expected_acceptance_rate, (
        actual_rate,
        expected_acceptance_rate,
    )
    assert actual_len >= expected_acceptance_len, (actual_len, expected_acceptance_len)


def _assert_outputs_are_structured(
    outputs: list[RequestOutput], sampling_params: list[SamplingParams]
) -> None:
    assert len(outputs) == len(sampling_params)
    for output, params in zip(outputs, sampling_params, strict=True):
        completion = output.outputs[0]
        assert output.finished and completion.text
        schema = params.structured_outputs
        if schema is not None and schema.json is not None:
            assert completion.finish_reason == "stop", completion.text
            jsonschema.validate(json.loads(completion.text), schema.json)
