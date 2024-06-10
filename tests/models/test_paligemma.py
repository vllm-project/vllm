from typing import List, Tuple

import pytest
from transformers import AutoTokenizer

from vllm.config import VisionLanguageConfig

from ..conftest import IMAGE_FILES

pytestmark = pytest.mark.vlm

HF_IMAGE_PROMPTS = [
    "caption es",
    "What is in the picture?",
]

assert len(HF_IMAGE_PROMPTS) == len(IMAGE_FILES)


def iter_paligemma_configs(model_name: str):
    image_hw_to_feature_size = {
        (224, 224): 256,
    }

    for (h, w), f in image_hw_to_feature_size.items():
        for input_type, input_shape in [
            (VisionLanguageConfig.ImageInputType.PIXEL_VALUES, (1, 3, h, w)),
        ]:
            yield (model_name,
                   VisionLanguageConfig(image_input_type=input_type,
                                        image_feature_size=f,
                                        image_token_id=257152,
                                        image_input_shape=input_shape,
                                        image_processor=model_name,
                                        image_processor_revision=None))


model_and_vl_config = [
    *iter_paligemma_configs("google/paligemma-3b-pt-224"),
]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str],
                      vlm_config: VisionLanguageConfig, model_id: str):
    """Sanitize vllm output to be comparable with hf output.
    """
    input_ids, output_str = vllm_output
    image_token_id = vlm_config.image_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    image_token_str = tokenizer.decode(image_token_id)

    # remove image token, bos token and the last newline token
    hf_input_ids = [
        input_id for input_id in input_ids
        if input_id != image_token_id and input_id != tokenizer.bos_token_id
    ]
    if hf_input_ids[-1] == 108:
        hf_input_ids = hf_input_ids[:-1]

    # remove image token from the output string
    hf_output_str = output_str \
        .replace(image_token_str * vlm_config.image_feature_size, "")

    return hf_input_ids, hf_output_str


@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(hf_runner, vllm_runner, hf_images, vllm_images,
                model_and_config, dtype: str, max_tokens: int) -> None:
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalData objects and corresponding
    vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    model_id, vlm_config = model_and_config

    with hf_runner(model_id, dtype=dtype, is_vision_model=True) as hf_model:
        hf_outputs = hf_model.generate_greedy(HF_IMAGE_PROMPTS,
                                              max_tokens,
                                              images=hf_images)

    image_token_id = vlm_config.image_token_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    image_token_str = tokenizer.decode(image_token_id)

    vllm_image_prompts = [
        image_token_str * vlm_config.image_feature_size + p + '\n'
        for p in HF_IMAGE_PROMPTS
    ]

    with vllm_runner(model_id,
                     dtype=dtype,
                     enforce_eager=True,
                     **vlm_config.as_cli_args_dict()) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(vllm_image_prompts,
                                                  max_tokens,
                                                  images=vllm_images)

    for i in range(len(HF_IMAGE_PROMPTS)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_to_hf_output(
            vllm_outputs[i], vlm_config, model_id)
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
