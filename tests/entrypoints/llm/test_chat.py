# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref

import pytest

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.sampling_params import SamplingParams

from ..openai.test_vision import TEST_IMAGE_ASSETS


@pytest.fixture(scope="function")
def text_llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct", enforce_eager=True, seed=0)

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


@pytest.fixture(scope="function")
def llm_for_failure_test():
    """
    Fixture for testing issue #26081.
    Uses a small max_model_len to easily trigger length errors.
    """
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        enforce_eager=True,
        seed=0,
        max_model_len=128,
        disable_log_stats=True,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


def test_chat(text_llm):
    prompt1 = "Explain the concept of entropy."
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt1},
    ]
    outputs = text_llm.chat(messages)
    assert len(outputs) == 1


def test_multi_chat(text_llm):
    prompt1 = "Explain the concept of entropy."
    prompt2 = "Explain what among us is."

    conversation1 = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt1},
    ]

    conversation2 = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt2},
    ]

    messages = [conversation1, conversation2]

    outputs = text_llm.chat(messages)
    assert len(outputs) == 2


@pytest.fixture(scope="function")
def vision_llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        max_model_len=4096,
        max_num_seqs=5,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},
        seed=0,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "image_urls", [[TEST_IMAGE_ASSETS[0], TEST_IMAGE_ASSETS[1]]], indirect=True
)
def test_chat_multi_image(vision_llm, image_urls: list[str]):
    messages = [
        {
            "role": "user",
            "content": [
                *(
                    {"type": "image_url", "image_url": {"url": image_url}}
                    for image_url in image_urls
                ),
                {"type": "text", "text": "What's in this image?"},
            ],
        }
    ]
    outputs = vision_llm.chat(messages)
    assert len(outputs) >= 0


def test_llm_chat_tokenization_no_double_bos(text_llm):
    """
    LLM.chat() should not add special tokens when using chat templates.
    Check we get a single BOS token for llama chat.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
    ]
    outputs = text_llm.chat(messages)
    assert len(outputs) == 1

    prompt_token_ids = outputs[0].prompt_token_ids
    assert prompt_token_ids is not None

    bos_token = text_llm.get_tokenizer().bos_token_id

    # Ensure we have a single BOS
    assert prompt_token_ids[0] == bos_token
    assert prompt_token_ids[1] != bos_token, "Double BOS"


@pytest.fixture(scope="function")
def thinking_llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=4096,
        enforce_eager=True,
        seed=0,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_chat_extra_kwargs(thinking_llm, enable_thinking):
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is 1+1?"},
    ]

    outputs = thinking_llm.chat(
        messages,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    assert len(outputs) == 1

    prompt_token_ids = outputs[0].prompt_token_ids
    assert prompt_token_ids is not None

    think_id = thinking_llm.get_tokenizer().get_vocab()["<think>"]

    if enable_thinking:
        assert think_id not in prompt_token_ids
    else:
        # The chat template includes dummy thinking process
        assert think_id in prompt_token_ids


def test_chat_batch_failure_cleanup(llm_for_failure_test):
    """
    Tests that if a batch call to llm.chat() fails mid-way
    (e.g., due to one invalid prompt), the requests that
    were already enqueued are properly aborted and do not
    pollute the queue for subsequent calls.
    (Fixes Issue #26081)
    """
    llm = llm_for_failure_test
    valid_msg = [{"role": "user", "content": "Hello"}]
    long_text = "This is a very long text to test the error " * 50
    invalid_msg = [{"role": "user", "content": long_text}]
    batch_1 = [
        valid_msg,
        valid_msg,
        invalid_msg,
    ]
    batch_2 = [
        valid_msg,
        valid_msg,
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    with pytest.raises(ValueError, match="context length is only"):
        llm.chat(batch_1, sampling_params=sampling_params)
    outputs_2 = llm.chat(batch_2, sampling_params=sampling_params)
    assert len(outputs_2) == len(batch_2)
    assert llm.llm_engine.get_num_unfinished_requests() == 0
