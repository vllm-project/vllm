from typing import List, Optional, Tuple, Type, Union

import pytest
import torch
import torch.types
from PIL import Image
from transformers import BatchEncoding

from vllm.multimodal.utils import rescale_image_size
from vllm.sequence import SampleLogprobs

from ....conftest import IMAGE_ASSETS, HfRunner, VllmRunner
from ...utils import check_logprobs_close

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
HF_MULTIIMAGE_IMAGE_PROMPT = \
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
    "(<image>./</image>)\n(<image>./</image>)\n" \
    "Describe these images.<|eot_id|>" \
    "<|start_header_id|>assistant<|end_header_id|>\n\n"

models = ["openbmb/MiniCPM-Llama3-V-2_5"]


def _wrap_inputs(hf_inputs: BatchEncoding) -> BatchEncoding:
    return BatchEncoding({"model_inputs": hf_inputs})


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
    inputs: List[Tuple[List[str], Union[List[Image.Image],
                                        List[List[Image.Image]]]]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    mm_limit: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test are from IMAGE_ASSETS.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalDataDict objects 
    and corresponding MultiModalConfig as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     max_model_len=4096,
                     max_num_seqs=1,
                     dtype=dtype,
                     limit_mm_per_prompt={"image": mm_limit},
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
            for prompts, images in inputs
        ]

    hf_model = hf_runner(model, dtype=dtype, postprocess_inputs=_wrap_inputs)
    with hf_model, torch.no_grad():
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images,
                                                    tokenizer=tokenizer)
            for prompts, images in inputs
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
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    run_test(
        hf_runner,
        vllm_runner,
        inputs_per_image,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=1,
        tensor_parallel_size=1,
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
    images = [asset.pil_image for asset in image_assets]

    inputs_per_case = [
        ([HF_MULTIIMAGE_IMAGE_PROMPT for _ in size_factors],
         [[rescale_image_size(image, factor) for image in images]
          for factor in size_factors])
    ]

    run_test(
        hf_runner,
        vllm_runner,
        inputs_per_case,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=2,
        tensor_parallel_size=1,
    )
