import math
from typing import List, Tuple

import pytest
import requests
from PIL import Image

from vllm.config import VisionLanguageConfig
from vllm.multimodal.image import ImagePixelData
from vllm.utils import is_cpu

pytestmark = pytest.mark.vlm


def iter_fuyu_configs(model_name: str):
    image_hw_to_feature_size = {
        (1080, 1920): 2304,
    }

    for (h, w), f in image_hw_to_feature_size.items():
        for input_type, input_shape in [
            (VisionLanguageConfig.ImageInputType.PIXEL_VALUES, (1, 3, h, w)),
        ]:
            yield (model_name,
                   VisionLanguageConfig(image_input_type=input_type,
                                        image_feature_size=f,
                                        image_token_id=71011,
                                        image_input_shape=input_shape,
                                        image_processor=model_name,
                                        image_processor_revision=None))


model_and_vl_config = [
    *iter_fuyu_configs("adept/fuyu-8b"),
]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str]):
    """Sanitize vllm output to be comparable with hf output.
    The function reduces `input_ids` from 1, 32000, 32000, ..., 32000,
    x1, x2, x3 ... to 1, 32000, x1, x2, x3 ...
    It also reduces `output_str` from "<image><image>bla" to "bla".
    """
    input_ids, output_str = vllm_output

    hf_input_ids = input_ids[2:]
    hf_output_str = output_str

    return hf_input_ids, hf_output_str


target_dtype = "half"
if is_cpu():
    target_dtype = "bfloat16"


# TODO: Add test for `tensor_parallel_size` [ref: PR #3883]
@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(hf_runner, vllm_runner, model_and_config, dtype: str,
                max_tokens: int) -> None:
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalData objects and corresponding
    vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    model_id, vlm_config = model_and_config

    # the llava image will raise an error for batch inference,
    # because of unsupported dynamic image patch size.
    # use example image from fuyu repo instead
    url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
    image = Image.open(requests.get(url, stream=True).raw)
    hf_images = [image]
    hf_prompts = ["Generate a coco-style caption.\n"]
    vllm_images = [ImagePixelData(img) for img in hf_images]
    vllm_image_prompts = []
    for prompt, img in zip(hf_prompts, hf_images):
        W, H = img.size
        nrow = math.ceil(min(H, 1080) / 30)
        ncol = math.ceil(min(W, 1920) / 30)
        prompt = ("|SPEAKER|" * ncol +
                  "|NEWLINE|") * nrow + "<s> " + prompt + "\x04"
        vllm_image_prompts.append(prompt)

    with hf_runner(model_id, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(
            hf_prompts,
            max_tokens,
            images=hf_images,
            eos_token_id=hf_model.processor.tokenizer.eos_token_id)

    with vllm_runner(model_id,
                     max_model_len=4096,
                     dtype=dtype,
                     enforce_eager=True,
                     **vlm_config.as_cli_args_dict()) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(vllm_image_prompts,
                                                  max_tokens,
                                                  images=vllm_images)

    for i in range(len(hf_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_to_hf_output(vllm_outputs[i])
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
