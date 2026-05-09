# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.inputs import tokens_input
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor


class _DummyRenderer:
    tokenizer = None

    @staticmethod
    def get_eos_token_id() -> int | None:
        return None


class _DummyMultiModalRegistry:
    @staticmethod
    def supports_multimodal_inputs(model_config: object) -> bool:
        return False


def test_input_processor_applies_override_generation_config_eos():
    model_config = SimpleNamespace(
        try_get_generation_config=lambda: {},
        override_generation_config={"eos_token_id": 123},
        max_model_len=16,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=None,
        lora_config=None,
        scheduler_config=None,
        speculative_config=None,
        structured_outputs_config=None,
        observability_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        ),
    )
    input_processor = InputProcessor(
        vllm_config,
        _DummyRenderer(),
        mm_registry=_DummyMultiModalRegistry(),
    )
    input_processor._validate_params = lambda params, supported_tasks: None
    input_processor._validate_model_inputs = (
        lambda encoder_inputs, decoder_inputs: None
    )

    request = input_processor.process_inputs(
        "request-id",
        tokens_input([1]),
        SamplingParams(max_tokens=4),
        supported_tasks=("generate",),
    )

    assert request.sampling_params is not None
    assert request.sampling_params.stop_token_ids == [123]
    assert request.sampling_params.all_stop_token_ids == {123}
