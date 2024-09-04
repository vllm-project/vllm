import warnings
from typing import Optional

import pytest
from PIL import Image

from vllm.assets.image import ImageAsset
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (parse_chat_messages,
                                         parse_chat_messages_futures)
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.utils import encode_image_base64
from vllm.transformers_utils.tokenizer_group import TokenizerGroup

PHI3V_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"


@pytest.fixture(scope="module")
def phi3v_model_config():
    return ModelConfig(PHI3V_MODEL_ID,
                       PHI3V_MODEL_ID,
                       tokenizer_mode="auto",
                       trust_remote_code=True,
                       dtype="bfloat16",
                       seed=0,
                       limit_mm_per_prompt={
                           "image": 2,
                       })


@pytest.fixture(scope="module")
def phi3v_tokenizer():
    return TokenizerGroup(
        tokenizer_id=PHI3V_MODEL_ID,
        enable_lora=False,
        max_num_seqs=5,
        max_input_length=None,
    )


@pytest.fixture(scope="module")
def image_url():
    image = ImageAsset('cherry_blossom')
    base64 = encode_image_base64(image.pil_image)
    return f"data:image/jpeg;base64,{base64}"


def _assert_mm_data_is_image_input(
    mm_data: Optional[MultiModalDataDict],
    image_count: int,
) -> None:
    assert mm_data is not None
    assert set(mm_data.keys()) == {"image"}

    image_data = mm_data.get("image")
    assert image_data is not None

    if image_count == 1:
        assert isinstance(image_data, Image.Image)
    else:
        assert isinstance(image_data, list) and len(image_data) == image_count


def test_parse_chat_messages_single_image(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data = parse_chat_messages([{
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }, {
            "type": "text",
            "text": "What's in the image?"
        }]
    }], phi3v_model_config, phi3v_tokenizer)

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in the image?"
    }]
    _assert_mm_data_is_image_input(mm_data, 1)


@pytest.mark.asyncio
async def test_parse_chat_messages_single_image_async(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_future = parse_chat_messages_futures([{
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }, {
            "type": "text",
            "text": "What's in the image?"
        }]
    }], phi3v_model_config, phi3v_tokenizer)

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in the image?"
    }]
    _assert_mm_data_is_image_input(await mm_future, 1)


def test_parse_chat_messages_multiple_images(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data = parse_chat_messages([{
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }, {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }, {
            "type": "text",
            "text": "What's in these images?"
        }]
    }], phi3v_model_config, phi3v_tokenizer)

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_1|>\n<|image_2|>\nWhat's in these images?"
    }]
    _assert_mm_data_is_image_input(mm_data, 2)


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_async(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_future = parse_chat_messages_futures([{
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }, {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }, {
            "type": "text",
            "text": "What's in these images?"
        }]
    }], phi3v_model_config, phi3v_tokenizer)

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_1|>\n<|image_2|>\nWhat's in these images?"
    }]
    _assert_mm_data_is_image_input(await mm_future, 2)


def test_parse_chat_messages_placeholder_already_in_prompt(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data = parse_chat_messages([{
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }, {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }, {
            "type":
            "text",
            "text":
            "What's in <|image_1|> and how does it compare to <|image_2|>?"
        }]
    }], phi3v_model_config, phi3v_tokenizer)

    assert conversation == [{
        "role":
        "user",
        "content":
        "What's in <|image_1|> and how does it compare to <|image_2|>?"
    }]
    _assert_mm_data_is_image_input(mm_data, 2)


def test_parse_chat_messages_placeholder_one_already_in_prompt(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data = parse_chat_messages([{
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }, {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }, {
            "type":
            "text",
            "text":
            "What's in <|image_1|> and how does it compare to the other one?"
        }]
    }], phi3v_model_config, phi3v_tokenizer)

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_2|>\nWhat's in <|image_1|> and how does it compare to the "
        "other one?"
    }]
    _assert_mm_data_is_image_input(mm_data, 2)


def test_parse_chat_messages_multiple_images_across_messages(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data = parse_chat_messages([{
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }, {
            "type": "text",
            "text": "What's in this image?"
        }]
    }, {
        "role": "assistant",
        "content": "Some stuff."
    }, {
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }, {
            "type": "text",
            "text": "What about this one?"
        }]
    }], phi3v_model_config, phi3v_tokenizer)

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\nWhat's in this image?"
        },
        {
            "role": "assistant",
            "content": "Some stuff."
        },
        {
            "role": "user",
            "content": "<|image_2|>\nWhat about this one?"
        },
    ]
    _assert_mm_data_is_image_input(mm_data, 2)


def test_parse_chat_messages_rejects_too_many_images_in_one_message(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="coroutine 'async_get_and_parse_image' was never awaited")
        with pytest.raises(
                ValueError,
                match="At most 2 image\\(s\\) may be provided in one request\\."
        ):
            parse_chat_messages([{
                "role":
                "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }, {
                    "type": "text",
                    "text": "What's in these images?"
                }]
            }], phi3v_model_config, phi3v_tokenizer)


def test_parse_chat_messages_rejects_too_many_images_across_messages(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="coroutine 'async_get_and_parse_image' was never awaited")
        with pytest.raises(
                ValueError,
                match="At most 2 image\\(s\\) may be provided in one request\\."
        ):
            parse_chat_messages([{
                "role":
                "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }, {
                    "type": "text",
                    "text": "What's in this image?"
                }]
            }, {
                "role": "assistant",
                "content": "Some stuff."
            }, {
                "role":
                "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }, {
                    "type": "text",
                    "text": "What about these two?"
                }]
            }], phi3v_model_config, phi3v_tokenizer)
