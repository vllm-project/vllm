from typing import List, Tuple

import pytest
from transformers import AutoTokenizer

from vllm.config import VisionLanguageConfig

from ..conftest import IMAGE_ASSETS
from .utils import check_outputs_equal

pytestmark = pytest.mark.vlm

_PREFACE = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's "
    "questions.")

# The image token is placed before "user" on purpose so that the test can pass
HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    f"{_PREFACE} <image>\nUSER: What's the content of the image?\nASSISTANT:",
    "cherry_blossom":
    f"{_PREFACE} <image>\nUSER: What is the season?\nASSISTANT:",
})


def iter_llava_next_configs(model_name: str):
    image_hw_to_feature_size = {
        (336, 336): 1176,
        (672, 672): 2928,
        (1344, 336): 1944,
        (336, 1344): 1890,
    }

    for (h, w), f in image_hw_to_feature_size.items():
        for input_type, input_shape in [
            (VisionLanguageConfig.ImageInputType.PIXEL_VALUES, (1, 3, h, w)),
        ]:
            yield (model_name,
                   VisionLanguageConfig(image_input_type=input_type,
                                        image_feature_size=f,
                                        image_token_id=32000,
                                        image_input_shape=input_shape,
                                        image_processor=model_name,
                                        image_processor_revision=None))


model_and_vl_config = [
    *iter_llava_next_configs("llava-hf/llava-v1.6-vicuna-7b-hf"),
]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str],
                      vlm_config: VisionLanguageConfig, model_id: str):
    """Sanitize vllm output to be comparable with hf output.
    The function reduces `input_ids` from 1, 32000, 32000, ..., 32000,
    x1, x2, x3 ... to 1, 32000, x1, x2, x3 ...
    It also reduces `output_str` from "<image><image>bla" to "bla".
    """
    output_ids, output_str = vllm_output
    image_token_id = vlm_config.image_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    image_token_str = tokenizer.decode(image_token_id)

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id or output_ids[idx - 1] != image_token_id
    ]
    hf_output_str = output_str \
        .replace(image_token_str * vlm_config.image_feature_size, " ")

    return hf_output_ids, hf_output_str


@pytest.mark.xfail(
    reason="Inconsistent image processor being used due to lack "
    "of support for dynamic image token replacement")
@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize("dtype", ["half"])
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
    hf_images = [asset.for_hf() for asset in image_assets]
    vllm_images = [asset.for_vllm(vlm_config) for asset in image_assets]

    with hf_runner(model_id, dtype=dtype, is_vision_model=True) as hf_model:
        hf_outputs = hf_model.generate_greedy(HF_IMAGE_PROMPTS,
                                              max_tokens,
                                              images=hf_images)

    vllm_image_prompts = [
        p.replace("<image>", "<image>" * vlm_config.image_feature_size)
        for p in HF_IMAGE_PROMPTS
    ]

    with vllm_runner(
            model_id,
            dtype=dtype,
            # should be greater than image_feature_size
            max_model_len=4096,
            enforce_eager=True,
            **vlm_config.as_cli_args_dict(),
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(vllm_image_prompts,
                                                  max_tokens,
                                                  images=vllm_images)

    check_outputs_equal(
        hf_outputs,
        [
            vllm_to_hf_output(vllm_output, vlm_config, model_id)
            for vllm_output in vllm_outputs
        ],
        name_0="hf",
        name_1="vllm",
    )
