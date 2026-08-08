# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.entrypoints.offline_utils import OfflineInferenceMixin
from vllm.inputs import tokens_input
from vllm.reasoning import ReasoningParserManager
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils.counter import Counter


def test_chat_forwards_reasoning_state_to_sync_engine(monkeypatch):
    captured: dict = {}

    class FakeReasoner:
        def __init__(self, tokenizer, chat_template_kwargs, model_config):
            captured["tokenizer"] = tokenizer
            captured["chat_template_kwargs"] = chat_template_kwargs
            captured["model_config"] = model_config

        def is_reasoning_end_from_prompt(self, prompt_token_ids):
            captured["prompt_token_ids"] = prompt_token_ids
            return False

    monkeypatch.setattr(
        ReasoningParserManager,
        "get_reasoning_parser",
        staticmethod(lambda name: FakeReasoner),
    )

    mixin = OfflineInferenceMixin()
    tokenizer = object()
    mixin.renderer = SimpleNamespace(
        tokenizer=tokenizer,
        get_tokenizer=lambda: tokenizer,
    )
    mixin.model_config = SimpleNamespace(
        enable_prompt_embeds=False,
        is_encoder_decoder=False,
    )
    structured_config = SimpleNamespace(
        reasoning_parser="minimax_m3",
        reasoning_parser_plugin=None,
    )
    mixin.llm_engine = Mock()
    mixin.llm_engine.vllm_config = SimpleNamespace(
        structured_outputs_config=structured_config
    )
    mixin.llm_engine.add_request.return_value = "0"
    mixin.request_counter = Counter()
    prompt = tokens_input([1, 2, 3])
    mixin._preprocess_chat_one = Mock(return_value=prompt)
    params = SamplingParams(
        max_tokens=8,
        structured_outputs=StructuredOutputsParams(json_object=True),
    )

    request_ids = mixin._add_chat_requests(
        messages=[{"role": "user", "content": "hi"}],
        params=params,
        use_tqdm=False,
        chat_template_kwargs={"thinking_mode": "enabled"},
    )

    assert request_ids == ["0"]
    assert captured["tokenizer"] is tokenizer
    assert captured["prompt_token_ids"] == [1, 2, 3]
    assert captured["chat_template_kwargs"]["thinking_mode"] == "enabled"
    mixin.llm_engine.add_request.assert_called_once()
    add_request_kwargs = mixin.llm_engine.add_request.call_args.kwargs
    assert add_request_kwargs["reasoning_ended"] is False
    assert add_request_kwargs["reasoning_parser_kwargs"] == {
        "chat_template_kwargs": captured["chat_template_kwargs"]
    }
