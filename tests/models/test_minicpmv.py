from collections import UserDict
from typing import List, Optional, Tuple, Type

import pytest
import torch
import torch.types
from transformers import BatchFeature

from vllm.multimodal.utils import rescale_image_size
from vllm.sequence import SampleLogprobs

from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from .utils import check_logprobs_close

pytestmark = pytest.mark.vlm


class NestedInputs(UserDict):

    def __init__(self, model_inputs: BatchFeature):
        super().__init__({"model_inputs": model_inputs})

        self.model_inputs = model_inputs

    def to(self, device: torch.types.Device):
        return NestedInputs(self.model_inputs.to(device))


# The image token is placed before "user" on purpose so that the test can pass
HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
        "(<image>./</image>)\nWhat's the content of the image?<|eot_id|>" \
        "<|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
    "cherry_blossom":
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
        "(<image>./</image>)\nWhat is the season?<|eot_id|>" \
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
})

models = ["openbmb/MiniCPM-Llama3-V-2_5"]


def trunc_hf_output(hf_output: Tuple[List[int], str,
                                     Optional[SampleLogprobs]]):
    output_ids, output_str, out_logprobs = hf_output
    if output_str.endswith("<|eot_id|>"):
        output_str = output_str.split("<|eot_id|>")[0]
    return output_ids, output_str, out_logprobs


target_dtype = "half"


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
        tokenizer = vllm_model.model.get_tokenizer()
        stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images,
                                                stop_token_ids=stop_token_ids)
            for prompts, images in inputs_per_image
        ]

    with hf_runner(model, dtype=dtype) as hf_model, torch.no_grad():
        hf_processor = hf_model.processor
        hf_model.processor = lambda **kw: NestedInputs(
            hf_processor(**kw)  # type: ignore
        )
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images,
                                                    tokenizer=tokenizer)
            for prompts, images in inputs_per_image
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        check_logprobs_close(
            outputs_0_lst=[
                trunc_hf_output(hf_output) for hf_output in hf_outputs
            ],
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


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
@pytest.mark.parametrize("num_logprobs", [5])
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


HF_MULTIIMAGE_IMAGE_PROMPT = \
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
    "(<image>./</image>)\n(<image>./</image>)\n" \
    "Describe these images.<|eot_id|>" \
    "<|start_header_id|>assistant<|end_header_id|>\n\n"


def run_multi_image_test(
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

    inputs_per_case = [
        ([HF_MULTIIMAGE_IMAGE_PROMPT for _ in size_factors],
         [[rescale_image_size(image, factor) for image in images]
          for factor in size_factors])
    ]

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
        tokenizer = vllm_model.model.get_tokenizer()
        stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]
        vllm_outputs_per_case = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images,
                                                stop_token_ids=stop_token_ids)
            for prompts, images in inputs_per_case
        ]

    with hf_runner(model, dtype=dtype) as hf_model, torch.no_grad():
        hf_processor = hf_model.processor
        hf_model.processor = lambda **kw: NestedInputs(
            hf_processor(**kw)  # type: ignore
        )
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images,
                                                    tokenizer=tokenizer)
            for prompts, images in inputs_per_case
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case,
                                        vllm_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=[
                trunc_hf_output(hf_output) for hf_output in hf_outputs
            ],
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


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
@pytest.mark.parametrize("num_logprobs", [5])
def test_multi_images_models(hf_runner, vllm_runner, image_assets, model,
                             size_factors, dtype: str, max_tokens: int,
                             num_logprobs: int) -> None:
    run_multi_image_test(
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
