# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm import LLM, SamplingParams


def _make_mock_llm() -> LLM:
    llm = object.__new__(LLM)
    llm.model_config = SimpleNamespace(
        runner_type="generate", enable_prompt_embeds=False
    )
    return llm


def test_generate_forwards_mm_processor_kwargs() -> None:
    llm = _make_mock_llm()
    mm_processor_kwargs = {"num_crops": 4}
    sampling_params = SamplingParams(max_tokens=1)

    llm._run_completion = Mock(return_value=["ok"])

    outputs = llm.generate(
        "prompt",
        sampling_params=sampling_params,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    assert outputs == ["ok"]
    assert llm._run_completion.call_args.kwargs["mm_processor_kwargs"] == (
        mm_processor_kwargs
    )


def test_enqueue_forwards_mm_processor_kwargs() -> None:
    llm = _make_mock_llm()
    mm_processor_kwargs = {"do_resize": False}
    sampling_params = SamplingParams(max_tokens=1)

    llm._add_completion_requests = Mock(return_value=["req-0"])

    request_ids = llm.enqueue(
        "prompt",
        sampling_params=sampling_params,
        use_tqdm=False,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    assert request_ids == ["req-0"]
    assert llm._add_completion_requests.call_args.kwargs["mm_processor_kwargs"] == (
        mm_processor_kwargs
    )


def test_chat_forwards_mm_processor_kwargs() -> None:
    llm = _make_mock_llm()
    mm_processor_kwargs = {"do_pan_and_scan": True}
    sampling_params = SamplingParams(max_tokens=1)
    messages = [{"role": "user", "content": "hello"}]

    llm._run_chat = Mock(return_value=["ok"])

    outputs = llm.chat(
        messages,
        sampling_params=sampling_params,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    assert outputs == ["ok"]
    assert llm._run_chat.call_args.kwargs["mm_processor_kwargs"] == (
        mm_processor_kwargs
    )


def test_enqueue_chat_forwards_mm_processor_kwargs() -> None:
    llm = _make_mock_llm()
    mm_processor_kwargs = {"do_pan_and_scan": True}
    sampling_params = SamplingParams(max_tokens=1)
    messages = [{"role": "user", "content": "hello"}]

    llm._add_chat_requests = Mock(return_value=["req-0"])

    request_ids = llm.enqueue_chat(
        messages,
        sampling_params=sampling_params,
        use_tqdm=False,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    assert request_ids == ["req-0"]
    assert llm._add_chat_requests.call_args.kwargs["mm_processor_kwargs"] == (
        mm_processor_kwargs
    )


def test_run_chat_forwards_mm_processor_kwargs() -> None:
    llm = _make_mock_llm()
    mm_processor_kwargs = {"num_crops": 8}
    sampling_params = SamplingParams(max_tokens=1)
    messages = [{"role": "user", "content": "hello"}]
    sentinel_output = ["done"]

    llm._add_chat_requests = Mock()
    llm._run_engine = Mock(return_value=sentinel_output)

    outputs = llm._run_chat(
        messages=messages,
        params=sampling_params,
        output_type=object,
        use_tqdm=False,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    assert outputs == sentinel_output
    assert llm._add_chat_requests.call_args.kwargs["mm_processor_kwargs"] == (
        mm_processor_kwargs
    )


def test_run_completion_forwards_mm_processor_kwargs() -> None:
    llm = _make_mock_llm()
    mm_processor_kwargs = {"min_pixels": 4 * 28 * 28}
    sampling_params = SamplingParams(max_tokens=1)
    sentinel_output = ["done"]

    llm._add_completion_requests = Mock()
    llm._run_engine = Mock(return_value=sentinel_output)

    outputs = llm._run_completion(
        prompts=["prompt"],
        params=sampling_params,
        output_type=object,
        use_tqdm=False,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    assert outputs == sentinel_output
    assert llm._add_completion_requests.call_args.kwargs["mm_processor_kwargs"] == (
        mm_processor_kwargs
    )


def test_add_completion_requests_forwards_mm_processor_kwargs() -> None:
    llm = _make_mock_llm()
    mm_processor_kwargs = {"max_dynamic_patch": 4}
    sampling_params = SamplingParams(max_tokens=1)

    llm._params_to_seq = Mock(return_value=[sampling_params])
    llm._lora_request_to_seq = Mock(return_value=[None])
    llm._priority_to_seq = Mock(return_value=[0])
    llm._preprocess_cmpl_one = Mock(return_value={"prompt_token_ids": [1]})

    captured_prompts = []

    def fake_render_and_add_requests(*, prompts, **_kwargs):
        captured_prompts.extend(prompts)
        return ["req-0"]

    llm._render_and_add_requests = Mock(side_effect=fake_render_and_add_requests)

    request_ids = llm._add_completion_requests(
        prompts=["prompt"],
        params=sampling_params,
        use_tqdm=False,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    assert request_ids == ["req-0"]
    llm._preprocess_cmpl_one.assert_called_once_with(
        "prompt",
        None,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    assert captured_prompts == [{"prompt_token_ids": [1]}]


def test_preprocess_cmpl_applies_mm_processor_kwargs_to_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _make_mock_llm()
    mm_processor_kwargs = {"num_crops": 8}
    prompt = {"prompt": "<image>", "multi_modal_data": {"image": object()}}

    renderer = Mock()
    renderer.default_cmpl_tok_params = Mock()
    renderer.default_cmpl_tok_params.with_kwargs.return_value = "tok-params"
    renderer.render_cmpl.return_value = ["engine-input"]
    llm.renderer = renderer

    monkeypatch.setattr(
        "vllm.entrypoints.offline_utils.parse_model_prompt",
        lambda _model_config, parsed_prompt: parsed_prompt,
    )

    outputs = llm._preprocess_cmpl(
        [prompt],
        mm_processor_kwargs=mm_processor_kwargs,
    )

    assert outputs == ["engine-input"]
    renderer.render_cmpl.assert_called_once_with(
        [prompt],
        "tok-params",
        prompt_extras={"mm_processor_kwargs": mm_processor_kwargs},
    )


def test_preprocess_cmpl_keeps_prompt_mm_processor_kwargs_when_no_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _make_mock_llm()
    prompt = {
        "prompt": "<image>",
        "multi_modal_data": {"image": object()},
        "mm_processor_kwargs": {"num_crops": 2},
    }

    renderer = Mock()
    renderer.default_cmpl_tok_params = Mock()
    renderer.default_cmpl_tok_params.with_kwargs.return_value = "tok-params"
    renderer.render_cmpl.return_value = ["engine-input"]
    llm.renderer = renderer

    monkeypatch.setattr(
        "vllm.entrypoints.offline_utils.parse_model_prompt",
        lambda _model_config, parsed_prompt: parsed_prompt,
    )

    outputs = llm._preprocess_cmpl([prompt])

    assert outputs == ["engine-input"]
    renderer.render_cmpl.assert_called_once_with(
        [prompt],
        "tok-params",
        prompt_extras=None,
    )


def test_preprocess_chat_applies_mm_processor_kwargs_to_renderer() -> None:
    llm = _make_mock_llm()
    mm_processor_kwargs = {"num_crops": 8}
    messages = [[{"role": "user", "content": "Describe this image."}]]

    renderer = Mock()
    renderer.tokenizer = object()
    renderer.default_chat_tok_params = Mock()
    renderer.default_chat_tok_params.with_kwargs.return_value = "tok-params"
    renderer.render_chat.return_value = (messages, ["engine-input"])
    llm.renderer = renderer

    outputs = llm._preprocess_chat(
        messages,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    assert outputs == ["engine-input"]
    call_args = renderer.render_chat.call_args
    assert call_args.args[0] == messages
    assert call_args.args[1].mm_processor_kwargs == mm_processor_kwargs
    assert call_args.args[2] == "tok-params"
    assert call_args.kwargs["prompt_extras"] == {
        "mm_processor_kwargs": mm_processor_kwargs
    }


def test_preprocess_chat_omits_mm_processor_kwargs_when_no_override() -> None:
    llm = _make_mock_llm()
    messages = [[{"role": "user", "content": "Describe this image."}]]

    renderer = Mock()
    renderer.tokenizer = object()
    renderer.default_chat_tok_params = Mock()
    renderer.default_chat_tok_params.with_kwargs.return_value = "tok-params"
    renderer.render_chat.return_value = (messages, ["engine-input"])
    llm.renderer = renderer

    outputs = llm._preprocess_chat(messages)

    assert outputs == ["engine-input"]
    call_args = renderer.render_chat.call_args
    assert call_args.args[0] == messages
    assert call_args.args[1].mm_processor_kwargs is None
    assert call_args.args[2] == "tok-params"
    assert call_args.kwargs["prompt_extras"] is None


def test_preprocess_chat_defaults_add_special_tokens_to_false() -> None:
    # Matches `ChatCompletionRequest.add_special_tokens` on the server.
    llm = _make_mock_llm()
    messages = [[{"role": "user", "content": "hi"}]]

    renderer = Mock()
    renderer.tokenizer = object()
    renderer.default_chat_tok_params = Mock()
    renderer.default_chat_tok_params.with_kwargs.return_value = "tok-params"
    renderer.render_chat.return_value = (messages, ["engine-input"])
    llm.renderer = renderer

    llm._preprocess_chat(messages)

    renderer.default_chat_tok_params.with_kwargs.assert_called_once_with(
        add_special_tokens=False
    )


def test_preprocess_chat_tokenization_kwargs_override_add_special_tokens() -> None:
    llm = _make_mock_llm()
    messages = [[{"role": "user", "content": "hi"}]]

    renderer = Mock()
    renderer.tokenizer = object()
    renderer.default_chat_tok_params = Mock()
    renderer.default_chat_tok_params.with_kwargs.return_value = "tok-params"
    renderer.render_chat.return_value = (messages, ["engine-input"])
    llm.renderer = renderer

    llm._preprocess_chat(
        messages,
        tokenization_kwargs={"add_special_tokens": True, "truncate_prompt_tokens": 8},
    )

    renderer.default_chat_tok_params.with_kwargs.assert_called_once_with(
        add_special_tokens=True, truncate_prompt_tokens=8
    )


@pytest.fixture(scope="module")
def llava_llm():
    from vllm.config import ModelConfig, VllmConfig
    from vllm.renderers.hf import HfRenderer
    from vllm.tokenizers import cached_tokenizer_from_config

    # A real multimodal model (Llama tokenizer, BOS=1) so that the real
    # processor default (`add_special_tokens=True`) is in play. Only
    # config/tokenizer/processor files are fetched, no weights.
    model_config = ModelConfig(model="llava-hf/llava-1.5-7b-hf", max_model_len=128)
    renderer = HfRenderer(
        VllmConfig(model_config=model_config),
        cached_tokenizer_from_config(model_config),
    )
    assert renderer.default_chat_tok_params.add_special_tokens is True

    llm = _make_mock_llm()
    # `_preprocess_cmpl` parses prompts against the real model config.
    llm.model_config = model_config
    llm.renderer = renderer
    return llm


class TestChatAddSpecialTokensDefault:
    """`LLM.chat()` renders the chat template to text and then tokenizes it.
    The multimodal processor default is `add_special_tokens=True`, so a
    template that emits `bos_token` used to get a second BOS from the
    tokenizer. `_preprocess_chat` now defaults `add_special_tokens=False`
    like the online chat API does (`ChatCompletionRequest`), unless the
    caller overrides it via `tokenization_kwargs`.
    """

    # Mirrors the Gemma 3 chat template and the bundled deepseek_vl2 /
    # deepseek_ocr templates, whose rendered output starts with the BOS token.
    BOS_TEMPLATE = (
        "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"
    )
    MESSAGES = [[{"role": "user", "content": "hi"}]]

    def _chat_token_ids(self, llm, **kwargs):
        (engine_input,) = llm._preprocess_chat(
            self.MESSAGES,
            chat_template=self.BOS_TEMPLATE,
            add_generation_prompt=False,
            **kwargs,
        )
        return engine_input["prompt_token_ids"]

    def test_chat_does_not_duplicate_template_bos(self, llava_llm):
        bos = llava_llm.renderer.tokenizer.bos_token_id
        prompt_token_ids = self._chat_token_ids(llava_llm)

        assert prompt_token_ids[0] == bos
        assert prompt_token_ids.count(bos) == 1

    def test_chat_matches_online_chat_api(self, llava_llm):
        from vllm.renderers.params import ChatParams, TokenizeParams

        # `ChatCompletionRequest.add_special_tokens` defaults to `False`.
        online_tok_params = TokenizeParams(
            max_total_tokens=llava_llm.renderer.model_config.max_model_len,
            add_special_tokens=False,
        )
        chat_params = ChatParams(
            chat_template=self.BOS_TEMPLATE,
            chat_template_kwargs=dict(tokenize=False, add_generation_prompt=False),
        )
        _, (online_input,) = llava_llm.renderer.render_chat(
            self.MESSAGES, chat_params, online_tok_params
        )

        assert self._chat_token_ids(llava_llm) == online_input["prompt_token_ids"]

    def test_explicit_tokenization_kwargs_override_default(self, llava_llm):
        bos = llava_llm.renderer.tokenizer.bos_token_id
        prompt_token_ids = self._chat_token_ids(
            llava_llm, tokenization_kwargs={"add_special_tokens": True}
        )

        assert prompt_token_ids.count(bos) == 2

    def test_generate_keeps_processor_default(self, llava_llm):
        # Raw prompts have no chat template to emit BOS, so `LLM.generate()`
        # must keep letting the tokenizer add it.
        bos = llava_llm.renderer.tokenizer.bos_token_id
        (engine_input,) = llava_llm._preprocess_cmpl(["hi"])

        assert engine_input["prompt_token_ids"][0] == bos
