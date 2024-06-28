import itertools
import re
from typing import List, Optional, Tuple

import pytest
from transformers import AutoTokenizer

from vllm.config import VisionLanguageConfig
from vllm.model_executor.models.blip2 import (BLIP2_IMAGE_TOKEN,
                                              BLIP2_IMAGE_TOKEN_ID)
from vllm.multimodal.image import ImagePixelData
from vllm.multimodal.utils import rescale_image_size

from ..conftest import IMAGE_ASSETS

pytestmark = pytest.mark.vlm

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "Question: What's the content of the image? Answer:",
    "cherry_blossom":
    "Question: What is the season? Answer:",
})


def iter_blip2_configs(model_name: str):
    image_hw_to_feature_size = {
        (224, 224): 32,
    }

    for (h, w), f in image_hw_to_feature_size.items():
        for input_type, input_shape in [
            (VisionLanguageConfig.ImageInputType.PIXEL_VALUES, (1, 3, h, w)),
        ]:
            yield (model_name,
                   VisionLanguageConfig(image_input_type=input_type,
                                        image_feature_size=f,
                                        image_token_id=BLIP2_IMAGE_TOKEN_ID,
                                        image_input_shape=input_shape,
                                        image_processor=model_name,
                                        image_processor_revision=None))


model_and_vl_config = [
    *iter_blip2_configs("Salesforce/blip2-opt-2.7b"),
]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str],
                      vlm_config: VisionLanguageConfig, model_id: str):
    """Sanitize vllm output to be comparable with hf output.
    The function reduces `input_ids` from 1, 32000, 32000, ..., 32000,
    x1, x2, x3 ... to 1, 32000, x1, x2, x3 ...
    It also reduces `output_str` from "<image><image>bla" to "bla".
    """
    output_ids, output_str = vllm_output

    hf_output_str = output_str.replace(BLIP2_IMAGE_TOKEN, "")
    hf_output_str = re.sub(r"Question:.* Answer:", "", hf_output_str)
    hf_output_str += "\n"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_output_ids = tokenizer.encode(hf_output_str)

    return hf_output_ids, hf_output_str


# TODO: Add test for `tensor_parallel_size` [ref: PR #3883]
@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("is_multiscale", [True, False])
def test_models(hf_runner, vllm_runner, image_assets, model_and_config,
                dtype: str, max_tokens: int, is_multiscale: bool) -> None:
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalData objects and corresponding
    vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    model_id, vlm_config = model_and_config
    hf_images = [asset.for_hf() for asset in image_assets]
    vllm_images = [asset.for_vllm(vlm_config) for asset in image_assets]

    size_factors = (0.25, 0.5, 1.0) if is_multiscale else (1, )
    image_inputs = [
        (rescale_image_size(hf_image, factor),
         ImagePixelData(image=rescale_image_size(vllm_image.image, factor)),
         prompt) for hf_image, vllm_image, prompt in zip(
             hf_images, vllm_images, HF_IMAGE_PROMPTS)
        for factor in size_factors
    ]
    prompt_inputs = [prompt for _, _, prompt in image_inputs]
    hf_image_inputs = [hf_image for hf_image, _, _ in image_inputs]
    vllm_image_inputs = [vllm_image for _, vllm_image, _ in image_inputs]

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model_id,
                     dtype=dtype,
                     enforce_eager=True,
                     **vlm_config.as_cli_args_dict()) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(prompt_inputs,
                                                  max_tokens,
                                                  images=vllm_image_inputs)

    with hf_runner(model_id, dtype=dtype, is_vision_model=True) as hf_model:
        hf_outputs = hf_model.generate_greedy(prompt_inputs,
                                              max_tokens,
                                              images=hf_image_inputs)
        hf_dummy_outputs = hf_model.generate_greedy(prompt_inputs,
                                                    max_tokens=1,
                                                    images=hf_image_inputs)

    # There may be numeric differences for multiscale images due to
    # our implementation of CLIPVisionModel
    best_max_tokens_exc_list: List[Tuple[int, Optional[AssertionError]]] = []
    for i in range(len(HF_IMAGE_PROMPTS)):
        try:
            hf_output_ids, hf_output_str = hf_outputs[i]
            vllm_output_ids, vllm_output_str = vllm_to_hf_output(
                vllm_outputs[i], vlm_config, model_id)
            assert hf_output_str == vllm_output_str, (
                f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
            assert hf_output_ids == vllm_output_ids, (
                f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
        except AssertionError as e:
            num_match_tokens = sum(1 for _ in itertools.takewhile(
                lambda pair: pair[0] == pair[1],
                zip(hf_output_ids, vllm_output_ids),
            ))
            num_prefix_tokens = len(hf_dummy_outputs[i][0]) - 1

            best_max_tokens = num_match_tokens - num_prefix_tokens
            best_max_tokens_exc_list.append((best_max_tokens, e))
        else:
            best_max_tokens_exc_list.append((max_tokens, None))

    best_max_tokens = min(pair[0] for pair in best_max_tokens_exc_list)
    if best_max_tokens < max_tokens:
        exc_list = [pair[1] for pair in best_max_tokens_exc_list]

        pytest.xfail(
            f"Test only fully passes when max_tokens={best_max_tokens} "
            f"(instead of {max_tokens}). Errors encountered per item: "
            f"{exc_list}")
