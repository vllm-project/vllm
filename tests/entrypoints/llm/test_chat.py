# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm import LLM

from ..openai.test_vision import TEST_IMAGE_URLS


def test_chat():
    llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

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


def test_multi_chat():
    llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

    prompt1 = "Explain the concept of entropy."
    prompt2 = "Explain what among us is."

    conversation1 = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt1
        },
    ]

    conversation2 = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt2
        },
    ]

    messages = [conversation1, conversation2]

    outputs = llm.chat(messages)
    assert len(outputs) == 2


@pytest.mark.parametrize("image_urls",
                         [[TEST_IMAGE_URLS[0], TEST_IMAGE_URLS[1]]])
def test_chat_multi_image(image_urls: list[str]):
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
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


def test_llm_chat_tokenization_no_double_bos():
    """
    LLM.chat() should not add special tokens when using chat templates.
    Check we get a single BOS token for llama chat.
    """
    llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct", enforce_eager=True)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello!"
        },
    ]
    outputs = llm.chat(messages)
    assert len(outputs) == 1
    prompt_token_ids = getattr(outputs[0], "prompt_token_ids", None)
    assert prompt_token_ids is not None

    bos_token = llm.get_tokenizer().bos_token_id

    # Ensure we have a single BOS
    assert prompt_token_ids[0] == bos_token
    assert prompt_token_ids[1] != bos_token, "Double BOS"
