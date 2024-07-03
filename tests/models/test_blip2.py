from typing import List, Optional, Tuple

import pytest
from transformers import AutoTokenizer

from vllm.config import VisionLanguageConfig
from vllm.model_executor.models.blip2 import BLIP2_IMAGE_TOKEN_ID
from vllm.multimodal.utils import rescale_image_size
from vllm.sequence import SampleLogprobs

from ..conftest import IMAGE_ASSETS
from .utils import check_logprobs_close

pytestmark = pytest.mark.vlm

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "Question: What's the content of the image? Answer:",
    "cherry_blossom":
    "Question: What is the season? Answer:",
    "boardwalk":
    "Question: What's in this image? Answer:",
})


def iter_blip2_configs(model_name: str):
    image_hw_to_feature_size = {
        (224, 224): 32,
    }

    for (h, w), f in image_hw_to_feature_size.items():
        input_shape = (1, 3, h, w)
        yield (model_name,
               VisionLanguageConfig(image_feature_size=f,
                                    image_token_id=BLIP2_IMAGE_TOKEN_ID,
                                    image_input_shape=input_shape))


model_and_vl_config = [
    *iter_blip2_configs("Salesforce/blip2-opt-2.7b"),
]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                         Optional[SampleLogprobs]],
                      vlm_config: VisionLanguageConfig, model_id: str):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "\n"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_output_ids = tokenizer.encode(hf_output_str)

    return hf_output_ids, hf_output_str, out_logprobs


@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize(
    "size_factors",
    [
        # No image
        [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner, vllm_runner, image_assets, model_and_config,
                size_factors, dtype: str, max_tokens: int,
                num_logprobs: int) -> None:
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalData objects and corresponding
    vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    model_id, vlm_config = model_and_config
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model_id,
                     dtype=dtype,
                     enforce_eager=True,
                     **vlm_config.as_cli_args_dict()) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs_per_image
        ]

    with hf_runner(model_id, dtype=dtype, is_vision_model=True) as hf_model:
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images)
            for prompts, images in inputs_per_image
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output, vlm_config, model_id)
                for vllm_output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
        )
