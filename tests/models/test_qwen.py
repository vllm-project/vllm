import pathlib
from typing import Dict, List, Optional, Type, Union

import pytest
import torch
from PIL.Image import Image

from vllm.config import ModelConfig
from vllm.inputs import InputContext, LLMInputs
from vllm.model_executor.models.qwen import (input_mapper_for_qwen,
                                             input_processor_for_qwen)
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.utils import cached_get_tokenizer, rescale_image_size

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

### Multimodal preprocessing tests
SAMPLE_IMAGE = IMAGE_ASSETS[0].pil_image
# These values are specific to Qwen-VL/Chat; we can get these from the model
# config also, but they are hardcoded here to keep the parameterize/fixtures
# easy to read.
IMG_START_ID = 151857
IMG_END_ID = 151858
IMG_PAD_ID = 151859
TOKS_PER_IMG = 256
VIS_ENC_DIM = 4096
IMG_SIZE = 448


def build_model_context(model_name: str,
                        tokenizer_name: Optional[str] = None,
                        trust_remote_code: bool = False):
    """Creates an InputContext for a given model.
    
    Args:
        model_name: Name of the model being considered.
        tokenizer_name: Name of the tokenizer being considered.
        trust_remote_code: Whether or not to allow loading remote code.

    Returns:
        InputContext for the model being considered.
    """
    if tokenizer_name is None:
        tokenizer_name = model_name
    model_config = ModelConfig(
        model_name,
        tokenizer_name,
        tokenizer_mode="auto",
        trust_remote_code=trust_remote_code,
        dtype="float32",
        seed=0,
    )
    return InputContext(model_config)


@pytest.fixture()
def qwen_vl_context() -> InputContext:
    """Get an InputContext for Qwen-VL."""
    return build_model_context(model_name="Qwen/Qwen-VL",
                               trust_remote_code=True)


# Happy path tests for single/multi-image scenarios for the multimodal
# input processor and mapper, respectively
@pytest.mark.parametrize("num_images", [1, 2])
def test_input_processor_valid_mm_data(qwen_vl_context: InputContext,
                                       num_images: int):
    """Happy cases for image inputs to Qwen's multimodal input processor."""
    prompt = "".join(
        [f"Picture {num}: <img></img>\n" for num in range(1, num_images + 1)])
    inputs = LLMInputs(
        prompt=prompt,
        # When processing multimodal data for a multimodal model, the qwen
        # input processor will overwrite the provided prompt_token_ids with
        # the image prompts
        prompt_token_ids=None,
        multi_modal_data={"image": torch.rand(num_images, TOKS_PER_IMG, 4096)},
    )
    proc_inputs = input_processor_for_qwen(qwen_vl_context, inputs)
    assert isinstance(proc_inputs, dict)

    # Each image should have one start / stop and a fixed context of 256
    proc_tokens = proc_inputs["prompt_token_ids"]
    assert proc_tokens.count(IMG_START_ID) == num_images
    assert proc_tokens.count(IMG_END_ID) == num_images
    assert proc_tokens.count(IMG_PAD_ID) == num_images * TOKS_PER_IMG


@pytest.mark.parametrize(
    "img_data,expected_shape",
    [
        # single / multi-image
        (SAMPLE_IMAGE, (1, 3, IMG_SIZE, IMG_SIZE)),
        (2 * [SAMPLE_IMAGE], (2, 3, IMG_SIZE, IMG_SIZE)),
        # single / multi-image embeddings
        (torch.rand(
            (TOKS_PER_IMG, VIS_ENC_DIM)), (1, TOKS_PER_IMG, VIS_ENC_DIM)),
        (torch.rand(
            (1, TOKS_PER_IMG, VIS_ENC_DIM)), (1, TOKS_PER_IMG, VIS_ENC_DIM)),
        (torch.rand(
            (2, TOKS_PER_IMG, VIS_ENC_DIM)), (2, TOKS_PER_IMG, VIS_ENC_DIM)),
    ])
def test_input_mapper_valid_mm_data(qwen_vl_context: InputContext,
                                    img_data: Union[torch.Tensor, List[Image],
                                                    Image],
                                    expected_shape: List[int]):
    """Happy cases for image inputs to Qwen's multimodal input mapper."""
    mapped_img_data = input_mapper_for_qwen(qwen_vl_context, img_data)
    # Ensure that we get the appropriately shaped pixel_values
    # for images and image embeddings, respectively.
    assert isinstance(mapped_img_data, MultiModalInputs)
    assert "pixel_values" in mapped_img_data
    assert mapped_img_data["pixel_values"].shape == expected_shape


# Sad path tests for the multimodal input processor and mapper, respectively
@pytest.mark.parametrize("mm_data", [
    {
        "image": torch.rand((5))
    },
    {
        "image": torch.rand((5, 5, 5, 5, 5))
    },
])
def test_input_processor_invalid_mm_data(qwen_vl_context: InputContext,
                                         mm_data: Dict[str, torch.Tensor]):
    """Test sad cases validated in Qwen's multimodal input processor."""
    tokenizer = cached_get_tokenizer(qwen_vl_context.model_config.tokenizer,
                                     trust_remote_code=True)
    prompt = "Picture 1: <img></img>\n"
    prompt_token_ids = tokenizer.encode(prompt)
    inputs = LLMInputs(prompt=prompt,
                       prompt_token_ids=prompt_token_ids,
                       multi_modal_data=mm_data)
    # Should fail since we have too many or too few dimensions for embeddings
    with pytest.raises(ValueError):
        input_processor_for_qwen(qwen_vl_context, inputs)


@pytest.mark.parametrize(
    "img_data",
    [
        # Wrong context length
        torch.rand((1, TOKS_PER_IMG + 10, VIS_ENC_DIM)),
        # Wrong visual encoder output size
        torch.rand((1, TOKS_PER_IMG, VIS_ENC_DIM + 10)),
    ])
def test_input_mapper_invalid_mm_data(
    qwen_vl_context: InputContext,
    img_data: Union[torch.Tensor, List[Image], Image],
):
    """Sad cases validated in Qwen VL's multimodal input mapper."""
    with pytest.raises(ValueError):
        input_mapper_for_qwen(qwen_vl_context, img_data)


### End-to-end generation tests
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
