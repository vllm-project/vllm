# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from collections.abc import Mapping
from typing import Literal, Optional

import pytest
from mistral_common.tokens.tokenizers.base import (SpecialTokenPolicy,
                                                   SpecialTokens)
from mistral_common.tokens.tokenizers.tekken import (SpecialTokenInfo,
                                                     Tekkenizer)

from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (_try_extract_ast, load_chat_template,
                                         parse_chat_messages,
                                         parse_chat_messages_futures,
                                         resolve_chat_template_content_format,
                                         resolve_hf_chat_template)
from vllm.entrypoints.llm import apply_hf_chat_template
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.utils import (encode_audio_base64, encode_image_base64,
                                   encode_video_base64)
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer

from ..models.registry import HF_EXAMPLE_MODELS
from ..utils import VLLM_PATH

EXAMPLES_DIR = VLLM_PATH / "examples"

PHI3V_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
ULTRAVOX_MODEL_ID = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
QWEN2AUDIO_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
QWEN2VL_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
QWEN25VL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN25OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
MLLAMA_MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_GUARD_MODEL_ID = "meta-llama/Llama-Guard-3-1B"
HERMES_MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"
MISTRAL_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


@pytest.fixture(scope="function")
def phi3v_model_config():
    return ModelConfig(PHI3V_MODEL_ID,
                       runner="generate",
                       trust_remote_code=True,
                       limit_mm_per_prompt={
                           "image": 2,
                       })


@pytest.fixture(scope="function")
def phi3v_model_config_mm_interleaved():
    return ModelConfig(PHI3V_MODEL_ID,
                       runner="generate",
                       trust_remote_code=True,
                       interleave_mm_strings=True,
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


@pytest.fixture(scope="function")
def qwen25omni_model_config_mm_interleaved():
    return ModelConfig(QWEN25OMNI_MODEL_ID,
                       runner="generate",
                       interleave_mm_strings=True,
                       limit_mm_per_prompt={
                           "image": 2,
                           "audio": 1,
                           "video": 1,
                       })


@pytest.fixture(scope="module")
def qwen25omni_tokenizer():
    return TokenizerGroup(
        tokenizer_id=QWEN25OMNI_MODEL_ID,
        enable_lora=False,
        max_num_seqs=5,
        max_input_length=None,
    )


@pytest.fixture(scope="module")
def mllama_model_config():
    return ModelConfig(MLLAMA_MODEL_ID,
                       runner="generate",
                       limit_mm_per_prompt={
                           "image": 2,
                       })


@pytest.fixture(scope="module")
def mllama_tokenizer():
    return TokenizerGroup(
        MLLAMA_MODEL_ID,
        enable_lora=False,
        max_num_seqs=5,
        max_input_length=None,
    )


@pytest.fixture(scope="function")
def mistral_model_config():
    return ModelConfig(MISTRAL_MODEL_ID,
                       runner="generate",
                       limit_mm_per_prompt={
                           "image": 2,
                       })


@pytest.fixture(scope="module")
def mistral_tokenizer():
    return TokenizerGroup(
        tokenizer_id=MISTRAL_MODEL_ID,
        enable_lora=False,
        max_num_seqs=5,
        max_input_length=None,
    )


@pytest.fixture(scope="module")
def image_url():
    image = ImageAsset('cherry_blossom')
    base64 = encode_image_base64(image.pil_image)
    return f"data:image/jpeg;base64,{base64}"


@pytest.fixture(scope="module")
def video_url():
    video = VideoAsset('baby_reading', 1)
    base64 = encode_video_base64(video.np_ndarrays)
    return f"data:video/jpeg;base64,{base64}"


@pytest.fixture(scope="module")
def audio_url():
    audio = AudioAsset('mary_had_lamb')
    base64 = encode_audio_base64(*audio.audio_and_sample_rate)
    return f"data:audio/ogg;base64,{base64}"


def _assert_mm_data_is_image_input(
    mm_data: Optional[MultiModalDataDict],
    image_count: int,
) -> None:
    assert mm_data is not None
    assert set(mm_data.keys()) == {"image"}

    image_data = mm_data.get("image")
    assert image_data is not None

    assert isinstance(image_data, list) and len(image_data) == image_count


ModalityType = Literal["image", "video", "audio"]
MultiModalDataCounts = Mapping[ModalityType, int]


def _assert_mm_data_inputs(
    mm_data: Optional[MultiModalDataDict],
    data_count: MultiModalDataCounts,
) -> None:
    assert mm_data is not None
    assert set(data_count.keys()) == (set(mm_data.keys()))

    for modality, n in data_count.items():
        modality_data = mm_data.get(modality)
        assert modality_data is not None
        assert isinstance(modality_data, list) and len(modality_data) == n


def test_parse_chat_messages_single_image(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data = parse_chat_messages(
        [{
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
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in the image?"
    }]
    _assert_mm_data_is_image_input(mm_data, 1)


def test_parse_chat_messages_empty_system(
    mistral_model_config,
    mistral_tokenizer,
):
    # Test string format
    conversation, _ = parse_chat_messages(
        [{
            "role": "system",
            "content": ""
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Who are you?"
            }]
        }],
        mistral_model_config,
        mistral_tokenizer,
        content_format="string",
    )
    assert conversation == [{
        "role": "system",
        "content": ""
    }, {
        "role": "user",
        "content": "Who are you?"
    }]

    # Test openai format
    conversation, _ = parse_chat_messages(
        [{
            "role": "system",
            "content": ""
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Who are you?"
            }]
        }],
        mistral_model_config,
        mistral_tokenizer,
        content_format="openai",
    )
    assert conversation == [{
        "role": "system",
        "content": [{
            "type": "text",
            "text": ""
        }]
    }, {
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": "Who are you?"
        }]
    }]


@pytest.mark.asyncio
async def test_parse_chat_messages_single_image_async(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_future = parse_chat_messages_futures(
        [{
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
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

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
    conversation, mm_data = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }, {
                "type": "image_pil",
                "image_pil": ImageAsset('cherry_blossom').pil_image
            }, {
                "type": "text",
                "text": "What's in these images?"
            }]
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

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
    conversation, mm_future = parse_chat_messages_futures(
        [{
            "role":
            "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }, {
                "type": "image_pil",
                "image_pil": ImageAsset('cherry_blossom').pil_image
            }, {
                "type": "text",
                "text": "What's in these images?"
            }]
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

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
    conversation, mm_data = parse_chat_messages(
        [{
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
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )
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
    conversation, mm_data = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type":
                    "text",
                    "text":
                    "What's in <|image_1|> and how does it compare to the other one?"  # noqa: E501
                }
            ]
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

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
    conversation, mm_data = parse_chat_messages(
        [{
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
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

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


def test_parse_chat_messages_context_text_format(
    phi3v_model_config,
    phi3v_tokenizer,
):
    conversation, mm_data = parse_chat_messages(
        [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "What's in this text?"
            }]
        }, {
            "role": "assistant",
            "content": "Some stuff."
        }, {
            "role": "user",
            "content": "What about this one?"
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="openai",
    )

    assert conversation == [
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "What's in this text?"
            }]
        },
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "Some stuff."
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "What about this one?"
            }]
        },
    ]


def test_parse_chat_messages_rejects_too_many_images_in_one_message(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="coroutine 'async_get_and_parse_image' was never awaited")
        with pytest.raises(ValueError, match="At most"):
            parse_chat_messages(
                [{
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
                }],
                phi3v_model_config,
                phi3v_tokenizer,
                content_format="string",
            )


def test_parse_chat_messages_rejects_too_many_images_across_messages(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="coroutine 'async_get_and_parse_image' was never awaited")
        with pytest.raises(ValueError, match="At most"):
            parse_chat_messages(
                [{
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
                }],
                phi3v_model_config,
                phi3v_tokenizer,
                content_format="string",
            )


def test_parse_chat_messages_multiple_images_uncommon_input(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                "What's in these images?", {
                    "image_url": image_url
                }, {
                    "image_url": image_url
                }
            ]
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_1|>\n<|image_2|>\nWhat's in these images?"
    }]
    _assert_mm_data_is_image_input(mm_data, 2)


def test_parse_chat_messages_multiple_images_interleave(
    phi3v_model_config_mm_interleaved,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [{
                "type": "text",
                "text": "I need you to compare this image"
            }, {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }, {
                "type": "text",
                "text": "and this one"
            }, {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }, {
                "type": "text",
                "text": "Do they have differences?"
            }]
        }],
        phi3v_model_config_mm_interleaved,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "I need you to compare this image\n<|image_1|>\nand this one\n<|image_2|>\n"  # noqa: E501
        "Do they have differences?"
    }]
    _assert_mm_data_is_image_input(mm_data, 2)


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_interleave_async(
    phi3v_model_config_mm_interleaved,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data = parse_chat_messages_futures(
        [{
            "role":
            "user",
            "content": [{
                "type": "text",
                "text": "I need you to compare this image"
            }, {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }, {
                "type": "text",
                "text": "and this one"
            }, {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }, {
                "type": "text",
                "text": "Do they have differences?"
            }]
        }],
        phi3v_model_config_mm_interleaved,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "I need you to compare this image\n<|image_1|>\nand this one\n<|image_2|>\n"  # noqa: E501
        "Do they have differences?"
    }]
    _assert_mm_data_is_image_input(await mm_data, 2)


def test_parse_chat_messages_multiple_images_multiple_messages_interleave(
    phi3v_model_config_mm_interleaved,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's on this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "Be accurate."
                },
            ]
        }, {
            "role": "assistant",
            "content": "Some stuff."
        }, {
            "role":
            "user",
            "content": [{
                "type": "text",
                "text": "What's on this image?"
            }, {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }]
        }],
        phi3v_model_config_mm_interleaved,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "What's on this image?\n<|image_1|>\nBe accurate."
    }, {
        "role": "assistant",
        "content": "Some stuff."
    }, {
        "role": "user",
        "content": "What's on this image?\n<|image_2|>"
    }]
    _assert_mm_data_is_image_input(mm_data, 2)


def test_parse_chat_messages_multiple_modals_multiple_messages_interleave(
        qwen25omni_model_config_mm_interleaved, qwen25omni_tokenizer,
        image_url, video_url, audio_url):
    conversation, mm_data = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's on this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "Now listen to this audio"
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": audio_url
                    }
                },
            ]
        }, {
            "role": "assistant",
            "content": "Some stuff."
        }, {
            "role":
            "user",
            "content": [{
                "type": "text",
                "text": "What's on this image?"
            }, {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }, {
                "type": "text",
                "text": "And what's in the video?"
            }, {
                "type": "video_url",
                "video_url": {
                    "url": video_url
                }
            }]
        }],
        qwen25omni_model_config_mm_interleaved,
        qwen25omni_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>\n"
        "Now listen to this audio\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>"
    }, {
        "role": "assistant",
        "content": "Some stuff."
    }, {
        "role":
        "user",
        "content":
        "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>\n"
        "And what's in the video?\n<|vision_start|><|VIDEO|><|vision_end|>"
    }]

    _assert_mm_data_inputs(mm_data, {"image": 2, "video": 1, "audio": 1})


def test_parse_chat_messages_multiple_images_interleave_with_placeholders(
    phi3v_model_config_mm_interleaved,
    phi3v_tokenizer,
    image_url,
):
    with pytest.raises(
            ValueError,
            match=r"Found more '<|image_1|>' placeholders in input prompt "
            "than actual multimodal data items."):
        parse_chat_messages(
            [{
                "role":
                "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type":
                        "text",
                        "text":
                        "I need you to compare this image\n<|image_1|>\nand this one\n<|image_2|>\n"  # noqa: E501
                        "Do they have differences?"
                    },
                ]
            }],
            phi3v_model_config_mm_interleaved,
            phi3v_tokenizer,
            content_format="string",
        )


### Mllama currently wraps images / texts as interleaved dictionaries
def test_mllama_single_image(
    mllama_model_config,
    mllama_tokenizer,
    image_url,
):
    """Ensures that a single image is parsed correctly mllama."""
    conversation, mm_data = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [{
                'type': 'text',
                'text': 'The content of this image is:'
            }, {
                "image_url": image_url
            }]
        }],
        mllama_model_config,
        mllama_tokenizer,
        content_format="openai",
    )
    _assert_mm_data_is_image_input(mm_data, 1)
    assert conversation == [{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'The content of this image is:'
        }, {
            'type': 'image'
        }]
    }]


def test_mllama_interleaved_images(
    mllama_model_config,
    mllama_tokenizer,
    image_url,
):
    """Ensures that multiple image are parsed as interleaved dicts."""
    conversation, mm_data = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    'type': 'text',
                    'text': 'The content of the first image is:'
                },
                {
                    "image_url": image_url
                },
                {
                    'type': 'text',
                    'text': 'The content of the second image is:'
                },
                {
                    "image_url": image_url
                },
            ]
        }],
        mllama_model_config,
        mllama_tokenizer,
        content_format="openai",
    )
    _assert_mm_data_is_image_input(mm_data, 2)
    assert conversation == [{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'The content of the first image is:'
        }, {
            'type': 'image'
        }, {
            'type': 'text',
            'text': 'The content of the second image is:'
        }, {
            'type': 'image'
        }]
    }]


@pytest.mark.parametrize("model", [MLLAMA_MODEL_ID])
def test_multimodal_image_parsing_matches_hf(model, image_url):
    """Checks end to end hf alignment for multimodal [image] parsing."""

    def get_conversation(is_hf: bool):
        img_part = {"type": "image_url", "image_url": {"url": image_url}}
        if is_hf:
            img_part = {'type': 'image'}
        return [{
            'role':
            'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'The content of the first image is:'
                },
                img_part,
                {
                    'type': 'text',
                    'text': 'The content of the second image is:'
                },
                img_part,
                {
                    'type': 'text',
                    'text': 'What animal is in the first image?'
                },
            ]
        }]

    # Build a config for the model
    model_config = ModelConfig(model,
                               runner="generate",
                               limit_mm_per_prompt={
                                   "image": 2,
                               })

    # Build the tokenizer group and grab the underlying tokenizer
    tokenizer_group = TokenizerGroup(
        model,
        enable_lora=False,
        max_num_seqs=5,
        max_input_length=None,
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer = tokenizer_group.tokenizer

    # Build and parse a conversation with {"type": "image"} using the tokenizer
    hf_conversation = get_conversation(is_hf=True)
    hf_result = tokenizer.apply_chat_template(
        hf_conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Now parse with vLLMs chat utils & apply the template
    vllm_conversation = get_conversation(is_hf=False)
    conversation, _ = parse_chat_messages(
        vllm_conversation,
        model_config,
        tokenizer_group,
        content_format="openai",
    )

    vllm_result = apply_hf_chat_template(
        tokenizer=tokenizer,
        conversation=conversation,
        chat_template=None,
        model_config=model_config,
        tools=None,
        add_generation_prompt=True,
    )

    assert hf_result == vllm_result


@pytest.mark.parametrize(
    "model",
    [
        QWEN2VL_MODEL_ID,  # tokenizer.chat_template is of type str
        HERMES_MODEL_ID,  # tokenizer.chat_template is of type dict
    ])
@pytest.mark.parametrize("use_tools", [True, False])
def test_resolve_hf_chat_template(sample_json_schema, model, use_tools):
    """checks that chat_template is a dict type for HF models."""
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
    )

    # Build the tokenizer group and grab the underlying tokenizer
    tokenizer_group = TokenizerGroup(
        model,
        enable_lora=False,
        max_num_seqs=5,
        max_input_length=None,
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer = tokenizer_group.tokenizer

    tools = [{
        "type": "function",
        "function": {
            "name": "dummy_function_name",
            "description": "This is a dummy function",
            "parameters": sample_json_schema
        }
    }] if use_tools else None

    # Test detecting the tokenizer's chat_template
    chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template=None,
        tools=tools,
        model_config=model_config,
    )
    assert isinstance(chat_template, str)


# NOTE: Qwen2-Audio default chat template is specially defined inside
# processor class instead of using `tokenizer_config.json`
# yapf: disable
@pytest.mark.parametrize(
    ("model", "expected_format"),
    [(PHI3V_MODEL_ID, "string"),
     (QWEN2VL_MODEL_ID, "openai"),
     (QWEN25VL_MODEL_ID, "openai"),
     (ULTRAVOX_MODEL_ID, "string"),
     (QWEN2AUDIO_MODEL_ID, "openai"),
     (MLLAMA_MODEL_ID, "openai"),
     (LLAMA_GUARD_MODEL_ID, "openai")],
)
# yapf: enable
def test_resolve_content_format_hf_defined(model, expected_format):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
    )

    tokenizer_group = TokenizerGroup(
        model,
        enable_lora=False,
        max_num_seqs=5,
        max_input_length=None,
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer = tokenizer_group.tokenizer

    # Test detecting the tokenizer's chat_template
    chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template=None,
        tools=None,
        model_config=model_config,
    )
    assert isinstance(chat_template, str)

    print("[TEXT]")
    print(chat_template)
    print("[AST]")
    print(_try_extract_ast(chat_template))

    resolved_format = resolve_chat_template_content_format(
        None,  # Test detecting the tokenizer's chat_template
        None,
        "auto",
        tokenizer,
        model_config=model_config,
    )

    assert resolved_format == expected_format


# yapf: disable
@pytest.mark.parametrize(
    ("model", "expected_format"),
    [("Salesforce/blip2-opt-2.7b", "string"),
     ("facebook/chameleon-7b", "string"),
     ("deepseek-ai/deepseek-vl2-tiny", "string"),
     ("microsoft/Florence-2-base", "string"),
     ("adept/fuyu-8b", "string"),
     ("google/paligemma-3b-mix-224", "string"),
     ("Qwen/Qwen-VL", "string"),
     ("Qwen/Qwen-VL-Chat", "string")],
)
# yapf: enable
def test_resolve_content_format_fallbacks(model, expected_format):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
    )

    tokenizer_group = TokenizerGroup(
        model_config.tokenizer,
        enable_lora=False,
        max_num_seqs=5,
        max_input_length=None,
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer = tokenizer_group.tokenizer

    # Test detecting the tokenizer's chat_template
    chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template=None,
        tools=None,
        model_config=model_config,
    )
    assert isinstance(chat_template, str)

    print("[TEXT]")
    print(chat_template)
    print("[AST]")
    print(_try_extract_ast(chat_template))

    resolved_format = resolve_chat_template_content_format(
        None,  # Test detecting the tokenizer's chat_template
        None,
        "auto",
        tokenizer,
        model_config=model_config,
    )

    assert resolved_format == expected_format


# yapf: disable
@pytest.mark.parametrize(
    ("template_path", "expected_format"),
    [("template_alpaca.jinja", "string"),
     ("template_baichuan.jinja", "string"),
     ("template_chatglm.jinja", "string"),
     ("template_chatglm2.jinja", "string"),
     ("template_chatml.jinja", "string"),
     ("template_dse_qwen2_vl.jinja", "openai"),
     ("template_falcon_180b.jinja", "string"),
     ("template_falcon.jinja", "string"),
     ("template_inkbot.jinja", "string"),
     ("template_teleflm.jinja", "string"),
     ("template_vlm2vec.jinja", "openai"),
     ("tool_chat_template_granite_20b_fc.jinja", "string"),
     ("tool_chat_template_hermes.jinja", "string"),
     ("tool_chat_template_internlm2_tool.jinja", "string"),
     ("tool_chat_template_llama3.1_json.jinja", "openai"),
     ("tool_chat_template_llama3.2_json.jinja", "openai"),
     ("tool_chat_template_mistral_parallel.jinja", "string"),
     ("tool_chat_template_mistral.jinja", "string")],
)
# yapf: enable
def test_resolve_content_format_examples(template_path, expected_format):
    model_config = ModelConfig(
        PHI3V_MODEL_ID,  # Dummy
        tokenizer=PHI3V_MODEL_ID,  # Dummy
        trust_remote_code=True,
    )

    tokenizer_group = TokenizerGroup(
        PHI3V_MODEL_ID,  # Dummy
        enable_lora=False,
        max_num_seqs=5,
        max_input_length=None,
        trust_remote_code=model_config.trust_remote_code,
    )
    dummy_tokenizer = tokenizer_group.tokenizer
    dummy_tokenizer.chat_template = None

    chat_template = load_chat_template(EXAMPLES_DIR / template_path)
    assert isinstance(chat_template, str)

    print("[TEXT]")
    print(chat_template)
    print("[AST]")
    print(_try_extract_ast(chat_template))

    resolved_format = resolve_chat_template_content_format(
        chat_template,
        None,
        "auto",
        dummy_tokenizer,
        model_config=model_config,
    )

    assert resolved_format == expected_format


def test_parse_chat_messages_include_thinking_chunk(mistral_model_config,
                                                    mistral_tokenizer):
    messages = [{
        "role":
        "system",
        "content": [{
            "type": "text",
            "text": "You are a helpful assistant."
        }, {
            "type":
            "thinking",
            "closed":
            True,
            "thinking":
            "Only return the answer when you are confident."
        }]
    }, {
        "role": "user",
        "content": "What is 2+2?"
    }, {
        "role":
        "assistant",
        "content": [{
            "type": "text",
            "text": "Let me think about it."
        }, {
            "type": "thinking",
            "closed": True,
            "thinking": "2+2 = 4"
        }, {
            "type": "text",
            "text": "The answer is 4.",
        }],
    }]

    conversation_with_thinking, _ = parse_chat_messages(
        messages,
        mistral_model_config,
        mistral_tokenizer,
        content_format="openai",
    )

    expected_conversation = [{
        "role":
        "system",
        "content": [{
            "type": "text",
            "text": "You are a helpful assistant."
        }, {
            "type": "text",
            "text": "Only return the answer when you are confident."
        }],
    }, {
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": "What is 2+2?"
        }],
    }, {
        "role":
        "assistant",
        "content": [
            {
                "type": "text",
                "text": "Let me think about it."
            },
            {
                "type": "text",
                "text": "2+2 = 4"
            },
            {
                "type": "text",
                "text": "The answer is 4."
            },
        ]
    }]

    assert conversation_with_thinking == expected_conversation


def test_apply_mistral_chat_template_thinking_chunk():
    # Moved import here to avoid yapf and isort conflicts
    from vllm.entrypoints.chat_utils import apply_mistral_chat_template
    messages = [{
        "role":
        "system",
        "content": [{
            "type": "text",
            "text": "You are a helpful assistant."
        }, {
            "type":
            "thinking",
            "closed":
            True,
            "thinking":
            "Only return the answer when you are confident."
        }]
    }, {
        "role": "user",
        "content": "What is 2+2?"
    }, {
        "role":
        "assistant",
        "content": [{
            "type": "text",
            "text": "Let me think about it."
        }, {
            "type": "thinking",
            "closed": True,
            "thinking": "2+2 = 4"
        }, {
            "type": "text",
            "text": "The answer is 4.",
        }],
    }, {
        "role": "user",
        "content": "Thanks, what is 3+3?"
    }]

    # TODO(Julien): upon model release change to a tokenizer already configured.
    # =================================================================
    mistral_tokenizer = MistralTokenizer.from_pretrained(
        "mistralai/Devstral-Small-2507")
    assert isinstance(mistral_tokenizer.tokenizer, Tekkenizer)
    # Add think special tokens to the tokenizer
    mistral_tokenizer.tokenizer._all_special_tokens[35] = SpecialTokenInfo(
        rank=35, is_control=True, token_str=SpecialTokens.begin_think.value)
    mistral_tokenizer.tokenizer._all_special_tokens[36] = SpecialTokenInfo(
        rank=36, is_control=True, token_str=SpecialTokens.end_think.value)
    mistral_tokenizer.tokenizer._special_tokens_reverse_vocab = {
        k: v
        for k, v in
        mistral_tokenizer.tokenizer._special_tokens_reverse_vocab.items()
        if v not in {35, 36}
    }
    mistral_tokenizer.tokenizer._special_tokens_reverse_vocab[
        SpecialTokens.begin_think.value] = 35
    mistral_tokenizer.tokenizer._special_tokens_reverse_vocab[
        SpecialTokens.end_think.value] = 36
    mistral_tokenizer.instruct.BEGIN_THINK = 35
    mistral_tokenizer.instruct.END_THINK = 36
    # =================================================================

    tokens_ids = apply_mistral_chat_template(mistral_tokenizer,
                                             messages,
                                             chat_template=None,
                                             tools=None)

    string_tokens = mistral_tokenizer.mistral.decode(
        tokens_ids, special_token_policy=SpecialTokenPolicy.KEEP)

    expected_tokens = (
        r"<s>[SYSTEM_PROMPT]You are a helpful assistant.[THINK]Only return the"
        r" answer when you are confident.[/THINK][/SYSTEM_PROMPT]"
        r"[INST]What is 2+2?[/INST]"
        r"Let me think about it.[THINK]2+2 = 4[/THINK]The answer is 4.</s>"
        r"[INST]Thanks, what is 3+3?[/INST]")

    assert string_tokens == expected_tokens
