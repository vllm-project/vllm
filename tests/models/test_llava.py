import gc
from dataclasses import fields
from enum import Enum
from typing import Dict, List, Tuple

import pytest
import torch
from transformers import AutoTokenizer

from vllm.config import VisionLanguageConfig

model_and_vl_config = [
    ("llava-hf/llava-1.5-7b-hf",
     VisionLanguageConfig(
         image_input_type=VisionLanguageConfig.ImageInputType.PIXEL_VALUES,
         image_feature_size=576,
         image_token_id=32000,
         image_input_shape=(1, 3, 336, 336))),
    ("llava-hf/llava-1.5-7b-hf",
     VisionLanguageConfig(
         image_input_type=VisionLanguageConfig.ImageInputType.IMAGE_FEATURES,
         image_feature_size=576,
         image_token_id=32000,
         image_input_shape=(1, 576, 1024)))
]


def as_dict(vision_language_config: VisionLanguageConfig) -> Dict:
    """Flatten vision language config to pure args.

    Compatible with what llm entrypoint expects.
    """
    result = {}
    for field in fields(vision_language_config):
        value = getattr(vision_language_config, field.name)
        if isinstance(value, Enum):
            result[field.name] = value.name.lower()
        elif isinstance(value, tuple):
            result[field.name] = ",".join([str(item) for item in value])
        else:
            result[field.name] = value
    return result


def sanitize_vllm_output(vllm_output: Tuple[List[int], str],
                         vision_language_config: VisionLanguageConfig,
                         model_id: str):
    """Sanitize vllm output to be comparable with hf output.
    The function reduces `input_ids` from 1, 32000, 32000, ..., 32000,
    x1, x2, x3 ... to 1, 32000, x1, x2, x3 ...
    It also reduces `output_str` from "<image><image>bla" to "bla".
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    image_token_str = tokenizer.decode(vision_language_config.image_token_id)
    image_token_str_len = len(image_token_str)
    input_ids, output_str = vllm_output
    sanitized_input_ids = input_ids[0:2] + input_ids[2 + vision_language_config
                                                     .image_feature_size - 1:]
    sanitzied_output_str = output_str[vision_language_config.
                                      image_feature_size *
                                      image_token_str_len:]
    return sanitized_input_ids, sanitzied_output_str


@pytest.mark.parametrize("worker_use_ray", [False])
@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(hf_runner, vllm_runner, hf_image_prompts, hf_images,
                vllm_image_prompts, vllm_images, model_and_config: tuple,
                dtype: str, max_tokens: int, worker_use_ray: bool) -> None:
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the raw images as input.
    For vllm runner, we provide image tensors and corresponding
    vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    model_id, vision_language_config = model_and_config
    hf_model = hf_runner(model_id, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(hf_image_prompts,
                                          max_tokens,
                                          images=hf_images)
    del hf_model

    vllm_model = vllm_runner(model_id,
                             dtype=dtype,
                             worker_use_ray=worker_use_ray,
                             **as_dict(vision_language_config))
    vllm_outputs = vllm_model.generate_greedy(vllm_image_prompts,
                                              max_tokens,
                                              images=vllm_images)
    del vllm_model

    gc.collect()
    torch.cuda.empty_cache()

    for i in range(len(hf_image_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = sanitize_vllm_output(
            vllm_outputs[i], vision_language_config, model_id)
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
