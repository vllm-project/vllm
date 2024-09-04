import weakref
from typing import List

import pytest

from vllm import LLM, RequestOutput, SamplingParams

from ...conftest import cleanup
from ..openai.test_vision import TEST_IMAGE_URLS

MODEL_NAME = "facebook/opt-125m"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

TOKEN_IDS = [
    [0],
    [0, 1],
    [0, 2, 1],
    [0, 3, 1, 2],
]


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(model=MODEL_NAME,
              max_num_batched_tokens=4096,
              tensor_parallel_size=1,
              gpu_memory_utilization=0.10,
              enforce_eager=True)

    with llm.deprecate_legacy_api():
        yield weakref.proxy(llm)

        del llm

    cleanup()


def assert_outputs_equal(o1: List[RequestOutput], o2: List[RequestOutput]):
    assert [o.outputs for o in o1] == [o.outputs for o in o2]


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize('prompt', PROMPTS)
def test_v1_v2_api_consistency_single_prompt_string(llm: LLM, prompt):
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

    with pytest.warns(DeprecationWarning, match="'prompts'"):
        v1_output = llm.generate(prompts=prompt,
                                 sampling_params=sampling_params)

    v2_output = llm.generate(prompt, sampling_params=sampling_params)
    assert_outputs_equal(v1_output, v2_output)

    v2_output = llm.generate({"prompt": prompt},
                             sampling_params=sampling_params)
    assert_outputs_equal(v1_output, v2_output)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize('prompt_token_ids', TOKEN_IDS)
def test_v1_v2_api_consistency_single_prompt_tokens(llm: LLM,
                                                    prompt_token_ids):
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

    with pytest.warns(DeprecationWarning, match="'prompt_token_ids'"):
        v1_output = llm.generate(prompt_token_ids=prompt_token_ids,
                                 sampling_params=sampling_params)

    v2_output = llm.generate({"prompt_token_ids": prompt_token_ids},
                             sampling_params=sampling_params)
    assert_outputs_equal(v1_output, v2_output)


@pytest.mark.skip_global_cleanup
def test_v1_v2_api_consistency_multi_prompt_string(llm: LLM):
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

    with pytest.warns(DeprecationWarning, match="'prompts'"):
        v1_output = llm.generate(prompts=PROMPTS,
                                 sampling_params=sampling_params)

    v2_output = llm.generate(PROMPTS, sampling_params=sampling_params)
    assert_outputs_equal(v1_output, v2_output)

    v2_output = llm.generate(
        [{
            "prompt": p
        } for p in PROMPTS],
        sampling_params=sampling_params,
    )
    assert_outputs_equal(v1_output, v2_output)


@pytest.mark.skip_global_cleanup
def test_v1_v2_api_consistency_multi_prompt_tokens(llm: LLM):
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

    with pytest.warns(DeprecationWarning, match="'prompt_token_ids'"):
        v1_output = llm.generate(prompt_token_ids=TOKEN_IDS,
                                 sampling_params=sampling_params)

    v2_output = llm.generate(
        [{
            "prompt_token_ids": p
        } for p in TOKEN_IDS],
        sampling_params=sampling_params,
    )
    assert_outputs_equal(v1_output, v2_output)


@pytest.mark.skip_global_cleanup
def test_multiple_sampling_params(llm: LLM):
    sampling_params = [
        SamplingParams(temperature=0.01, top_p=0.95),
        SamplingParams(temperature=0.3, top_p=0.95),
        SamplingParams(temperature=0.7, top_p=0.95),
        SamplingParams(temperature=0.99, top_p=0.95),
    ]

    # Multiple SamplingParams should be matched with each prompt
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(ValueError):
        outputs = llm.generate(PROMPTS, sampling_params=sampling_params[:3])

    # Single SamplingParams should be applied to every prompt
    single_sampling_params = SamplingParams(temperature=0.3, top_p=0.95)
    outputs = llm.generate(PROMPTS, sampling_params=single_sampling_params)
    assert len(PROMPTS) == len(outputs)

    # sampling_params is None, default params should be applied
    outputs = llm.generate(PROMPTS, sampling_params=None)
    assert len(PROMPTS) == len(outputs)


def test_chat():

    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

    prompt1 = "Explain the concept of entropy."
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt1
        },
    ]
    outputs = llm.chat(messages)
    assert len(outputs) == 1


@pytest.mark.parametrize("image_urls",
                         [[TEST_IMAGE_URLS[0], TEST_IMAGE_URLS[1]]])
def test_chat_multi_image(image_urls: List[str]):
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=5,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},
    )

    messages = [{
        "role":
        "user",
        "content": [
            *({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            } for image_url in image_urls),
            {
                "type": "text",
                "text": "What's in this image?"
            },
        ],
    }]
    outputs = llm.chat(messages)
    assert len(outputs) >= 0
