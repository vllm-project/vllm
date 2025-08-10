# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

import pytest
from mistral_common.multimodal import download_image
from mistral_common.protocol.instruct.messages import ImageURLChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.multimodal import image_from_chunk
from transformers import AutoProcessor

from vllm import RequestOutput, SamplingParams, TextPrompt, TokensPrompt
from vllm.multimodal import MultiModalDataBuiltins
from vllm.multimodal.inputs import PlaceholderRange
from vllm.sequence import Logprob, SampleLogprobs

from ....utils import VLLM_PATH, large_gpu_test
from ...utils import check_logprobs_close, dummy_hf_overrides

if TYPE_CHECKING:
    from _typeshed import StrPath

PIXTRAL_ID = "mistralai/Pixtral-12B-2409"
MISTRAL_SMALL_3_1_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

MODELS = [PIXTRAL_ID, MISTRAL_SMALL_3_1_ID]

IMG_URLS = [
    "https://huggingface.co/datasets/Isotr0py/mistral-test-images/resolve/main/237-400x300.jpg",
    "https://huggingface.co/datasets/Isotr0py/mistral-test-images/resolve/main/231-200x300.jpg",
    "https://huggingface.co/datasets/Isotr0py/mistral-test-images/resolve/main/27-500x500.jpg",
    "https://huggingface.co/datasets/Isotr0py/mistral-test-images/resolve/main/17-150x600.jpg",
]
PROMPT = "Describe each image in one short sentence."


def _create_msg_format(urls: list[str]) -> list[dict[str, Any]]:
    return [{
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": PROMPT,
        }] + [{
            "type": "image_url",
            "image_url": {
                "url": url
            }
        } for url in urls],
    }]


def _create_msg_format_hf(urls: list[str]) -> list[dict[str, Any]]:
    return [{
        "role":
        "user",
        "content": [{
            "type": "text",
            "content": PROMPT,
        }, *({
            "type": "image",
            "image": download_image(url)
        } for url in urls)],
    }]


def _create_engine_inputs(urls: list[str]) -> TokensPrompt:
    msg = _create_msg_format(urls)

    tokenizer = MistralTokenizer.from_model("pixtral")

    request = ChatCompletionRequest(messages=msg)  # type: ignore[type-var]
    tokenized = tokenizer.encode_chat_completion(request)

    engine_inputs = TokensPrompt(prompt_token_ids=tokenized.tokens)

    images = []
    for chunk in request.messages[0].content:
        if isinstance(chunk, ImageURLChunk):
            images.append(image_from_chunk(chunk))

    mm_data = MultiModalDataBuiltins(image=images)
    engine_inputs["multi_modal_data"] = mm_data

    return engine_inputs


def _create_engine_inputs_hf(urls: list[str]) -> TextPrompt:
    msg = _create_msg_format_hf(urls)

    tokenizer = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")
    prompt = tokenizer.apply_chat_template(msg)

    images = []
    for chunk in msg[0]["content"]:
        if chunk["type"] == "image":
            images.append(chunk["image"])

    mm_data = MultiModalDataBuiltins(image=images)
    engine_inputs = TextPrompt(prompt=prompt, multi_modal_data=mm_data)

    return engine_inputs


MSGS = [
    _create_msg_format(IMG_URLS[:1]),
    _create_msg_format(IMG_URLS[:2]),
    _create_msg_format(IMG_URLS),
]

SAMPLING_PARAMS = SamplingParams(max_tokens=512, temperature=0.0, logprobs=5)
LIMIT_MM_PER_PROMPT = dict(image=4)

MAX_MODEL_LEN = [8192, 65536]

FIXTURES_PATH = VLLM_PATH / "tests/models/fixtures"
assert FIXTURES_PATH.exists()

FIXTURE_LOGPROBS_CHAT = {
    PIXTRAL_ID: FIXTURES_PATH / "pixtral_chat.json",
    MISTRAL_SMALL_3_1_ID: FIXTURES_PATH / "mistral_small_3_chat.json",
}

OutputsLogprobs = list[tuple[list[int], str, Optional[SampleLogprobs]]]


# For the test author to store golden output in JSON
def _dump_outputs_w_logprobs(
    outputs: OutputsLogprobs,
    filename: "StrPath",
) -> None:
    json_data = [(tokens, text, [{
        k: asdict(v)
        for k, v in token_logprobs.items()
    } for token_logprobs in (logprobs or [])])
                 for tokens, text, logprobs in outputs]

    with open(filename, "w") as f:
        json.dump(json_data, f)


def load_outputs_w_logprobs(filename: "StrPath") -> OutputsLogprobs:
    with open(filename, "rb") as f:
        json_data = json.load(f)

    return [(tokens, text, [{
        int(k): Logprob(**v)
        for k, v in token_logprobs.items()
    } for token_logprobs in logprobs]) for tokens, text, logprobs in json_data]


@large_gpu_test(min_gb=80)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_model_len", MAX_MODEL_LEN)
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_chat(
    vllm_runner,
    max_model_len: int,
    model: str,
    dtype: str,
) -> None:
    EXPECTED_CHAT_LOGPROBS = load_outputs_w_logprobs(
        FIXTURE_LOGPROBS_CHAT[model])
    with vllm_runner(
            model,
            dtype=dtype,
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
            max_model_len=max_model_len,
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
    ) as vllm_model:
        outputs = []
        for msg in MSGS:
            output = vllm_model.llm.chat(msg, sampling_params=SAMPLING_PARAMS)

            outputs.extend(output)

    logprobs = vllm_runner._final_steps_generate_w_logprobs(outputs)
    # Remove last `None` prompt_logprobs to compare with fixture
    for i in range(len(logprobs)):
        assert logprobs[i][-1] is None
        logprobs[i] = logprobs[i][:-1]
    check_logprobs_close(outputs_0_lst=EXPECTED_CHAT_LOGPROBS,
                         outputs_1_lst=logprobs,
                         name_0="h100_ref",
                         name_1="output")


@pytest.mark.parametrize("prompt,expected_ranges",
                         [(_create_engine_inputs_hf(IMG_URLS[:1]),
                           [PlaceholderRange(offset=11, length=494)]),
                          (_create_engine_inputs_hf(IMG_URLS[1:4]), [
                              PlaceholderRange(offset=11, length=266),
                              PlaceholderRange(offset=277, length=1056),
                              PlaceholderRange(offset=1333, length=418)
                          ])])
def test_multi_modal_placeholders(vllm_runner, prompt: TextPrompt,
                                  expected_ranges: list[PlaceholderRange],
                                  monkeypatch) -> None:

    # This placeholder checking test only works with V0 engine
    # where `multi_modal_placeholders` is returned with `RequestOutput`
    monkeypatch.setenv("VLLM_USE_V1", "0")
    with vllm_runner(
            "mistral-community/pixtral-12b",
            max_model_len=8192,
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
            load_format="dummy",
            hf_overrides=dummy_hf_overrides,
    ) as vllm_model:
        outputs = vllm_model.llm.generate(prompt)

        assert len(outputs) == 1, f"{len(outputs)=}"
        output: RequestOutput = outputs[0]
        assert hasattr(output,
                       "multi_modal_placeholders"), f"{output.__dict__=}"
        assert "image" in output.multi_modal_placeholders, \
            f"{output.multi_modal_placeholders.keys()=}"
        image_placeholder_ranges: list[
            PlaceholderRange] = output.multi_modal_placeholders["image"]
        assert len(image_placeholder_ranges) == len(
            expected_ranges), f"{image_placeholder_ranges=}"
        for real_range, expected_range in zip(image_placeholder_ranges,
                                              expected_ranges):
            assert real_range.offset == expected_range.offset, \
                f"{real_range=} {expected_range=}"
            assert real_range.length == expected_range.length, \
                f"{real_range=} {expected_range=}"
