import os
import re
from typing import Callable, List, Optional, Tuple, Type

import pytest
import torch
from transformers import AutoImageProcessor, AutoTokenizer

from vllm.inputs import InputContext, LLMInputs
from vllm.model_executor.models.phi3v import _IMAGE_TOKEN_ID
from vllm.multimodal import MultiModalRegistry
from vllm.multimodal.utils import rescale_image_size
from vllm.sequence import SampleLogprobs
from vllm.utils import is_cpu, is_hip

from ....conftest import (IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner,
                          _ImageAssets)
from ...utils import build_model_context, check_logprobs_close

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<|user|>\n<|image_1|>\nWhat's the content of the image?<|end|>\n<|assistant|>\n",  # noqa: E501
    "cherry_blossom":
    "<|user|>\n<|image_1|>\nWhat is the season?<|end|>\n<|assistant|>\n",
})
HF_MULTIIMAGE_IMAGE_PROMPT = "<|user|>\n<|image_1|>\n<|image_2|>\nDescribe these images.<|end|>\n<|assistant|>\n"  # noqa: E501

models = ["microsoft/Phi-3.5-vision-instruct"]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                         Optional[SampleLogprobs]],
                      model: str):
    """Sanitize vllm output to be comparable with hf output."""
    _, output_str, out_logprobs = vllm_output

    output_str_without_image = re.sub(r"(<\|image_\d+\|>)+", "", output_str)
    assert output_str_without_image[0] == " "
    output_str_without_image = output_str_without_image[1:]

    hf_output_str = output_str_without_image + "<|end|><|endoftext|>"

    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_output_ids = tokenizer.encode(output_str_without_image)
    assert hf_output_ids[0] == 1
    hf_output_ids = hf_output_ids[1:]

    return hf_output_ids, hf_output_str, out_logprobs


target_dtype = "half"
if is_cpu():
    target_dtype = "bfloat16"

# ROCm Triton FA can run into shared memory issues with these models,
# use other backends in the meantime
# FIXME (mattwong, gshtrasb, hongxiayan)
if is_hip():
    os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    inputs: List[Tuple[List[str], PromptImageInput]],
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
        vllm_outputs_per_case = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs
        ]

    # use eager mode for hf runner, since phi3_v didn't work with flash_attn
    hf_model_kwargs = {"_attn_implementation": "eager"}
    with hf_runner(model, dtype=dtype,
                   model_kwargs=hf_model_kwargs) as hf_model:
        eos_token_id = hf_model.processor.tokenizer.eos_token_id
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images,
                                                    eos_token_id=eos_token_id)
            for prompts, images in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case,
                                        vllm_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output, model)
                for vllm_output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
        )


# Since we use _attn_implementation="eager" for hf_runner, there is more
# significant numerical difference. The basic `logprobs=5` fails to pass.
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
@pytest.mark.parametrize("dtype", [target_dtype])
def test_regression_7840(hf_runner, vllm_runner, image_assets, model,
                         dtype) -> None:
    images = [asset.pil_image for asset in image_assets]

    inputs_regresion_7840 = [
        ([prompt], [image]) for image, prompt in zip(images, HF_IMAGE_PROMPTS)
    ]

    # Regression test for #7840.
    run_test(
        hf_runner,
        vllm_runner,
        inputs_regresion_7840,
        model,
        dtype=dtype,
        max_tokens=128,
        num_logprobs=10,
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
@pytest.mark.parametrize("num_logprobs", [10])
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


### Fast tests for correctness in processor_kwarg override handling


# Wrap lazy imports to avoid initializing CUDA during test collection
@pytest.fixture()
def input_processor_for_phi3v():
    from vllm.model_executor.models.phi3v import input_processor_for_phi3v
    return input_processor_for_phi3v


@pytest.fixture()
def dummy_data_for_phi3v():
    from vllm.model_executor.models.phi3v import dummy_data_for_phi3v
    return dummy_data_for_phi3v


@pytest.fixture()
def get_max_phi3v_image_tokens():
    from vllm.model_executor.models.phi3v import get_max_phi3v_image_tokens
    return get_max_phi3v_image_tokens


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("num_crops", [4, 16, None])
def test_input_mapper_override(model: str, image_assets: _ImageAssets,
                               num_crops: Optional[int]):
    """Ensure that the [default] input mapper handles num_crops properly."""
    # We pass the processor kwargs here since for this model, we fall back to
    # the default mapper; this will fall back to the HF mapper and forward
    # mm_processor_kwargs to it.
    mm_processor_kwargs = {
        "num_crops": num_crops
    } if num_crops is not None else {}
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    hf_processor = AutoImageProcessor.from_pretrained(model,
                                                      trust_remote_code=True,
                                                      **mm_processor_kwargs)

    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(ctx.model_config)

    image = image_assets[0].pil_image
    hf_result = hf_processor.preprocess(
        image,
        return_tensors="pt",
    )

    vllm_result = mm_registry.map_input(
        ctx.model_config,
        {"image": image},
    )

    assert torch.all(hf_result["image_sizes"] == vllm_result["image_sizes"])
    assert torch.all(
        hf_result["num_img_tokens"] == vllm_result["num_img_tokens"])

    # For pixel values, the second axis should be the num_crops + 1
    # for the rescaled original image. The default value in VLLM falls
    # back to the HF config, which is why we compare to the processor num_crops
    assert torch.all(hf_result["pixel_values"] == vllm_result["pixel_values"])
    assert vllm_result["pixel_values"].shape[1] == hf_processor.num_crops + 1


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("num_crops,expected_max_tokens", [
    (4, 781),
    (16, 2653),
])
def test_max_tokens_override(get_max_phi3v_image_tokens: Callable, model: str,
                             num_crops: int, expected_max_tokens: int):
    """Ensure get_max_phi3v_image_tokens handles num_crops properly."""
    # NOTE: mm_processor_kwargs on the context in this test is unused, since
    # this is testing the mapper directly. In practice, the processor kwargs
    # are wrapped in a closure when calling the max tokens func. We explicitly
    # do NOT use the mm_processor_kwargs in the model context here to ensure
    # that the max image tokens implementation is referencing a mix of the
    # kwargs to the function and the original mm_processor_kwargs in case
    # values are somehow updated and end up in a bad state.
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=None,
    )

    actual_max_tokens = get_max_phi3v_image_tokens(
        InputContext(ctx.model_config),
        num_crops=num_crops,
    )

    assert expected_max_tokens == actual_max_tokens


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("num_crops,toks_per_img,num_imgs", [
    (4, 781, 1),
    (4, 781, 2),
    (16, 2653, 1),
    (16, 2653, 2),
])
def test_dummy_data_override(dummy_data_for_phi3v: Callable, model: str,
                             num_crops: int, toks_per_img: int, num_imgs: int):
    """Ensure dummy_data_for_phi3v handles num_crops properly."""
    # Same as the previous test - don't initialize mm_processor_kwargs
    # in this test and assume that the kwargs will be correctly expanded by
    # the partial when calling the dummy data func.
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=None,
    )

    sequence_data, _, = dummy_data_for_phi3v(
        ctx=ctx,
        seq_len=8192,  # Should be bigger than num_imgs * toks_per_img
        mm_counts={"image": num_imgs},
        num_crops=num_crops,
    )
    # Ensure we have the right number of placeholders per num_crops size
    img_tok_count = sequence_data.get_token_ids().count(_IMAGE_TOKEN_ID)
    assert img_tok_count == toks_per_img * num_imgs


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("num_crops,expected_toks_per_img,num_imgs", [
    (4, 757, 1),
    (4, 757, 2),
    (16, 1921, 1),
    (16, 1921, 2),
])
def test_input_processor_override(input_processor_for_phi3v: Callable,
                                  image_assets: _ImageAssets, model: str,
                                  num_crops: int, expected_toks_per_img: int,
                                  num_imgs: int):
    """Ensure input_processor_for_phi3v handles num_crops properly."""
    # Same as the previous test - don't initialize mm_processor_kwargs
    # in this test and assume that the kwargs will be correctly expanded by
    # the partial when calling the custom input processor.
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    # Build the image str / prompt based on the number of images we pass
    img_str = "".join([f"<|image_{idx}|>\n" for idx in range(1, num_imgs + 1)])
    prompt = f"<|user|>\n{img_str}<|end|>\n<|assistant|>\n"
    images = [image_assets[0].pil_image] * num_imgs

    llm_inputs = LLMInputs(prompt_token_ids=tokenizer.encode(prompt),
                           prompt=prompt,
                           multi_modal_data={"image": images})

    proc_llm_inputs = input_processor_for_phi3v(
        ctx=ctx,
        llm_inputs=llm_inputs,
        num_crops=num_crops,
    )

    # Ensure we have the right number of placeholders per num_crops size
    img_tok_count = proc_llm_inputs["prompt_token_ids"].count(_IMAGE_TOKEN_ID)
    assert img_tok_count == expected_toks_per_img * num_imgs
