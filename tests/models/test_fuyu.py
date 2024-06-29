from typing import List, Tuple

import pytest

from vllm.config import VisionLanguageConfig
from vllm.utils import is_cpu

from ..conftest import IMAGE_ASSETS

pytestmark = pytest.mark.vlm

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "What color is the stop sign?\n",  # noqa: E501
    "cherry_blossom":
    "What is the season?\n",  # noqa: E501
})


def iter_fuyu_configs(model_name: str):
    image_hw_to_feature_size = {
        (420, 660): 308,
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
def test_models(hf_runner, vllm_runner, image_assets, model_and_config,
                dtype: str, max_tokens: int) -> None:
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalData objects and corresponding
    vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    model_id, vlm_config = model_and_config
    _, _, H, W = vlm_config.image_input_shape

    # resize images to the model's input shape
    hf_images = [asset.for_hf().resize((W, H)) for asset in image_assets]
    vllm_images = [asset.for_vllm(vlm_config) for asset in image_assets]
    for i in range(len(image_assets)):
        vllm_images[i].image = vllm_images[i].image.resize((W, H))

    with hf_runner(model_id, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(
            HF_IMAGE_PROMPTS,
            max_tokens,
            images=hf_images,
            eos_token_id=hf_model.processor.tokenizer.eos_token_id)

    ncol, nrow = W // 30, H // 30
    image_prompts = ("|SPEAKER|" * ncol + "|NEWLINE|") * nrow
    vllm_image_prompts = [
        image_prompts + "<s> " + p + "\x04" for p in HF_IMAGE_PROMPTS
    ]

    with vllm_runner(model_id,
                     max_model_len=1024,
                     dtype=dtype,
                     enforce_eager=True,
                     **vlm_config.as_cli_args_dict()) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(vllm_image_prompts,
                                                  max_tokens,
                                                  images=vllm_images)

    for i in range(len(HF_IMAGE_PROMPTS)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_to_hf_output(vllm_outputs[i])
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
