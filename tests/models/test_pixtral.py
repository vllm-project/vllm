"""Compare the outputs of HF and vLLM for Mistral models using greedy sampling.

Run `pytest tests/models/test_mistral.py`.
"""
import uuid
from typing import Any, Dict, List

import pytest
from mistral_common.protocol.instruct.messages import ImageURLChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.multimodal import image_from_chunk

from vllm import EngineArgs, LLMEngine, SamplingParams, TokensPrompt
from vllm.multimodal import MultiModalDataBuiltins
from vllm.sampling_params import SamplingParams

pytestmark = pytest.mark.vlm

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
        } for url in urls]
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


MSGS = [
    _create_msg_format(IMG_URLS[:1]),
    _create_msg_format(IMG_URLS[:2]),
    _create_msg_format(IMG_URLS)
]
ENGINE_INPUTS = [
    _create_engine_inputs(IMG_URLS[:1]),
    _create_engine_inputs(IMG_URLS[:2]),
    _create_engine_inputs(IMG_URLS)
]

EXPECTED = [
    "The image shows a black dog sitting on a wooden surface.",
    "1. A black dog with floppy ears sits attentively on a wooden surface.\n2. A vast mountain range with rugged peaks stretches under a cloudy sky.",
    "1. A black dog sits attentively on a wooden floor.\n2. A vast mountain range stretches across the horizon under a cloudy sky.\n3. Surfers wait for waves in the ocean at sunset.\n4. A winding gravel path leads through a lush green park."
]

SAMPLING_PARAMS = SamplingParams(max_tokens=512, temperature=0.0)
LIMIT_MM_PER_PROMPT = dict(image=4)


@pytest.mark.skip(
    reason=
    "Model is too big, test passed on A100 locally but will OOM on CI machine."
)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_model_len", [8192, 65536])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_chat(
    vllm_runner,
    max_model_len: int,
    model: str,
    dtype: str,
) -> None:

    with vllm_runner(model,
                     dtype=dtype,
                     tokenizer_mode="mistral",
                     enable_chunked_prefill=False,
                     max_model_len=max_model_len,
                     limit_mm_per_prompt=LIMIT_MM_PER_PROMPT) as vllm_model:
        results = []
        for msg in MSGS:
            outputs = vllm_model.model.chat(msg,
                                            sampling_params=SAMPLING_PARAMS)

            results.append(outputs[0].outputs[0].text)

        assert results == EXPECTED


@pytest.mark.skip(
    reason=
    "Model is too big, test passed on A100 locally but will OOM on CI machine."
)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_model_engine(model: str, dtype: str) -> None:
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

    results = []
    count = 0
    while True:
        out = engine.step()
        count += 1
        for request_output in out:
            if request_output.finished:
                results.append(request_output.outputs[0].text)

        if count == 2:
            engine.add_request(uuid.uuid4().hex, ENGINE_INPUTS[2],
                               SAMPLING_PARAMS)
        if not engine.has_unfinished_requests():
            break

    assert results[0] == EXPECTED[0]
    # the result is a tiny bit different but this is not unexpected given that different kernels are executed
    # and given that flash attention is not deterministic
    assert results[
        1] == "1. A black dog with floppy ears sits attentively on a wooden surface.\n2. A vast mountain range stretches across the horizon under a cloudy sky."
    assert results[2] == EXPECTED[2]
