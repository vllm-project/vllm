import pathlib
from typing import List, Optional, Type

import pytest

from vllm.multimodal.utils import rescale_image_size

from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from .utils import check_logprobs_close

pytestmark = pytest.mark.vlm

text_only_models = [
    "Qwen/Qwen-7B-Chat"  # Has no visual component
]

multimodal_models = ["Qwen/Qwen-VL"]

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "Picture 1: <img></img>\nWhat's the content of the image?: ",
    "cherry_blossom":
    "Picture 1: <img></img>\nWhat is the season?: ",
})


### Tests for multimodal Qwen models
def run_test(
    tmp_path: pathlib.PosixPath,
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
    and corresponding MultiModalConfig as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    images = [asset.pil_image for asset in image_assets]

    # Export the images to a tempdir and substitute it into the hf prompt;
    # the contents between <img>/</img> will be ignored by VLLM, but the
    # transformers implementation for the visual transformer parses this to
    # reload it in the forward call; the contents are treated as a URL or a
    # local path.
    for idx, asset in enumerate(image_assets):
        image_tmp_path = tmp_path / f"{asset.name}.jpg"
        asset.pil_image.save(image_tmp_path)
        HF_IMAGE_PROMPTS[idx] = HF_IMAGE_PROMPTS[idx].replace(
            "<img></img>", f"<img>{image_tmp_path}</img>")

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    # Qwen encodes images into a fixed content size of 256
    with vllm_runner(model,
                     max_model_len=300,
                     max_num_seqs=1,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs_per_image
        ]

    with hf_runner(model, dtype=dtype) as hf_model:
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
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.parametrize("model", multimodal_models)
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
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [8])
@pytest.mark.parametrize("num_logprobs", [5])
def test_multimodal_models(tmp_path, hf_runner, vllm_runner, image_assets,
                           model, size_factors, dtype, max_tokens,
                           num_logprobs) -> None:
    run_test(
        tmp_path,
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


# Ensure that a text-only Qwen model can still be loaded and
# used for inference in VLLM without throwing.
@pytest.mark.parametrize("model", text_only_models)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_text_only_qwen_model_can_be_loaded_and_run(
    vllm_runner: Type[VllmRunner],
    example_prompts,
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
):
    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_model.generate_greedy_logprobs(
            example_prompts,
            max_tokens,
            num_logprobs=num_logprobs,
        )
