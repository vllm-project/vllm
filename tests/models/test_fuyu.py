from typing import List, Optional, Tuple, Type

import pytest

from vllm.multimodal.utils import rescale_image_size
from vllm.utils import is_cpu

from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from .utils import check_outputs_equal

pytestmark = pytest.mark.vlm

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign": "What color is the stop sign?\n",  # noqa: E501
    "cherry_blossom": "What is the season?\n",
    "boardwalk": "What's in this image?\n",
})

models = ["adept/fuyu-8b"]


def hf_to_vllm_output(hf_output: Tuple[List[int], str]):
    """remove image placeholder from prompts"""
    hf_output_ids, hf_output_str = hf_output
    hf_output_str = hf_output_str.split("<s> ")[1]
    return hf_output_ids, hf_output_str


# FIXME: Fuyu in transformers miss get_output_embeddings method,
# which break the generate_greedy_logprobs for hf_runner.
# Use check_outputs_equal to pass the test temporarily until the issue is fixed.
def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    size_factors: List[float],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalDataDict objects 
    and corresponding vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     max_model_len=4096,
                     max_num_seqs=1,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy(prompts, max_tokens, images=vllm_images)
            for prompts, vllm_images in inputs_per_image
        ]

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs_per_image = [
            hf_model.generate_greedy(prompts, max_tokens, images=hf_images)
            for prompts, hf_images in inputs_per_image
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        check_outputs_equal(
            outputs_0_lst=[
                hf_to_vllm_output(hf_output) for hf_output in hf_outputs
            ],
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


target_dtype = "half"
if is_cpu():
    target_dtype = "bfloat16"


@pytest.mark.parametrize("model", models)
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
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [10])
def test_models(hf_runner, vllm_runner, image_assets, model, size_factors,
                dtype: str, max_tokens: int, num_logprobs: int) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model,
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
