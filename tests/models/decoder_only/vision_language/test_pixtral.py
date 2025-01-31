"""Compare the outputs of HF and vLLM for Mistral models using greedy sampling.

Run `pytest tests/models/test_mistral.py`.
"""
import json
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pytest
from mistral_common.multimodal import download_image
from mistral_common.protocol.instruct.messages import ImageURLChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.multimodal import image_from_chunk
from transformers import AutoProcessor

from vllm import (EngineArgs, LLMEngine, RequestOutput, SamplingParams,
                  TextPrompt, TokensPrompt)
from vllm.multimodal import MultiModalDataBuiltins
from vllm.multimodal.inputs import PlaceholderRange
from vllm.sequence import Logprob, SampleLogprobs

from ....utils import VLLM_PATH, large_gpu_test
from ...utils import check_logprobs_close

if TYPE_CHECKING:
    from _typeshed import StrPath

MODELS = ["mistralai/Pixtral-12B-2409"]
IMG_URLS = [
    "https://picsum.photos/id/237/400/300",
    "https://picsum.photos/id/231/200/300",
    "https://picsum.photos/id/27/500/500",
    "https://picsum.photos/id/17/150/600",
]
PROMPT = "Describe each image in one short sentence."


def _create_msg_format(urls: List[str]) -> List[Dict[str, Any]]:
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


def _create_msg_format_hf(urls: List[str]) -> List[Dict[str, Any]]:
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


def _create_engine_inputs(urls: List[str]) -> TokensPrompt:
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


def _create_engine_inputs_hf(urls: List[str]) -> TextPrompt:
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
ENGINE_INPUTS = [
    _create_engine_inputs(IMG_URLS[:1]),
    _create_engine_inputs(IMG_URLS[:2]),
    _create_engine_inputs(IMG_URLS),
]

SAMPLING_PARAMS = SamplingParams(max_tokens=512, temperature=0.0, logprobs=5)
LIMIT_MM_PER_PROMPT = dict(image=4)

MAX_MODEL_LEN = [8192, 65536]

FIXTURES_PATH = VLLM_PATH / "tests/models/fixtures"
assert FIXTURES_PATH.exists()

FIXTURE_LOGPROBS_CHAT = FIXTURES_PATH / "pixtral_chat.json"
FIXTURE_LOGPROBS_ENGINE = FIXTURES_PATH / "pixtral_chat_engine.json"

OutputsLogprobs = List[Tuple[List[int], str, Optional[SampleLogprobs]]]


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
    EXPECTED_CHAT_LOGPROBS = load_outputs_w_logprobs(FIXTURE_LOGPROBS_CHAT)
    with vllm_runner(
            model,
            dtype=dtype,
            tokenizer_mode="mistral",
            enable_chunked_prefill=False,
            max_model_len=max_model_len,
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
    ) as vllm_model:
        outputs = []
        for msg in MSGS:
            output = vllm_model.model.chat(msg,
                                           sampling_params=SAMPLING_PARAMS)

            outputs.extend(output)

    logprobs = vllm_runner._final_steps_generate_w_logprobs(outputs)
    check_logprobs_close(outputs_0_lst=EXPECTED_CHAT_LOGPROBS,
                         outputs_1_lst=logprobs,
                         name_0="h100_ref",
                         name_1="output")


@large_gpu_test(min_gb=80)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_model_engine(vllm_runner, model: str, dtype: str) -> None:
    EXPECTED_ENGINE_LOGPROBS = load_outputs_w_logprobs(FIXTURE_LOGPROBS_ENGINE)
    args = EngineArgs(
        model=model,
        tokenizer_mode="mistral",
        enable_chunked_prefill=False,
        limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
        dtype=dtype,
    )
    engine = LLMEngine.from_engine_args(args)

    engine.add_request(uuid.uuid4().hex, ENGINE_INPUTS[0], SAMPLING_PARAMS)
    engine.add_request(uuid.uuid4().hex, ENGINE_INPUTS[1], SAMPLING_PARAMS)

    outputs = []
    count = 0
    while True:
        out = engine.step()
        count += 1
        for request_output in out:
            if request_output.finished:
                outputs.append(request_output)

        if count == 2:
            engine.add_request(uuid.uuid4().hex, ENGINE_INPUTS[2],
                               SAMPLING_PARAMS)
        if not engine.has_unfinished_requests():
            break

    logprobs = vllm_runner._final_steps_generate_w_logprobs(outputs)
    check_logprobs_close(outputs_0_lst=EXPECTED_ENGINE_LOGPROBS,
                         outputs_1_lst=logprobs,
                         name_0="h100_ref",
                         name_1="output")


@large_gpu_test(min_gb=48)
@pytest.mark.parametrize(
    "prompt,expected_ranges",
    [(_create_engine_inputs_hf(IMG_URLS[:1]), [{
        "offset": 10,
        "length": 494
    }]),
     (_create_engine_inputs_hf(IMG_URLS[1:4]), [{
         "offset": 10,
         "length": 266
     }, {
         "offset": 276,
         "length": 1056
     }, {
         "offset": 1332,
         "length": 418
     }])])
def test_multi_modal_placeholders(
        vllm_runner, prompt, expected_ranges: list[PlaceholderRange]) -> None:
    with vllm_runner(
            "mistral-community/pixtral-12b",
            max_model_len=8192,
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
    ) as vllm_model:
        outputs = vllm_model.model.generate(prompt)

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
            assert real_range == expected_range, \
                f"{real_range=} {expected_range=}"
