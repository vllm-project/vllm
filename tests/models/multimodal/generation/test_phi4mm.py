# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Sequence
from typing import Optional

import librosa
import pytest
import regex as re
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest
from vllm.multimodal.image import convert_image_mode, rescale_image_size
from vllm.platforms import current_platform
from vllm.sequence import SampleLogprobs

from ....conftest import (IMAGE_ASSETS, HfRunner, PromptAudioInput,
                          PromptImageInput, VllmRunner)
from ....utils import large_gpu_test
from ...utils import check_logprobs_close

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<|user|>\n<|image_1|>\nWhat's the content of the image?<|end|>\n<|assistant|>\n",  # noqa: E501
    "cherry_blossom":
    "<|user|>\n<|image_1|>\nPlease infer the season with reason in details.<|end|>\n<|assistant|>\n",  # noqa: E501
})
HF_MULTIIMAGE_IMAGE_PROMPT = "<|user|>\n<|image_1|>\n<|image_2|>\nDescribe these images.<|end|>\n<|assistant|>\n"  # noqa: E501

model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")
# Since the vision-lora and speech-lora co-exist with the base model,
# we have to manually specify the path of the lora weights.
vision_lora_path = os.path.join(model_path, "vision-lora")
speech_question = os.path.join(model_path, "examples",
                               "what_is_shown_in_this_image.wav")
models = [model_path]


def vllm_to_hf_output(vllm_output: tuple[list[int], str,
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

# ROCm Triton FA can run into shared memory issues with these models,
# use other backends in the meantime
# FIXME (mattwong, gshtrasb, hongxiayan)
if current_platform.is_rocm():
    os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"


def run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    inputs: Sequence[tuple[list[str], PromptImageInput,
                           Optional[PromptAudioInput]]],
    model: str,
    *,
    max_model_len: int,
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
    with vllm_runner(
            model,
            task="generate",
            max_model_len=max_model_len,
            max_num_seqs=2,
            dtype=dtype,
            limit_mm_per_prompt={"image": mm_limit},
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            enable_lora=True,
            max_lora_rank=320,
            gpu_memory_utilization=0.8,  # set to 0.8 to avoid OOM in CI
            enforce_eager=True,
    ) as vllm_model:
        lora_request = LoRARequest("vision", 1, vision_lora_path)
        vllm_outputs_per_case = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images,
                                                audios=audios,
                                                lora_request=lora_request)
            for prompts, images, audios in inputs
        ]

    # This error occurs inside `get_peft_model`
    # FIXME: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/discussions/75
    pytest.skip("HF impl is not compatible with current transformers")

    hf_model_kwargs = {"_attn_implementation": "sdpa"}
    with hf_runner(model, dtype=dtype,
                   model_kwargs=hf_model_kwargs) as hf_model:

        hf_processor = hf_model.processor
        eos_token_id = hf_processor.tokenizer.eos_token_id

        def patch_hf_processor(*args,
                               text="",
                               images=None,
                               audio=None,
                               sampling_rate=None,
                               **kwargs):
            audios = None
            if audio is not None and sampling_rate is not None:
                audios = [(audio, sampling_rate)]
            return hf_processor(*args,
                                text=text,
                                images=images,
                                audios=audios,
                                **kwargs)

        hf_model.processor = patch_hf_processor

        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images,
                                                    audios=audios,
                                                    eos_token_id=eos_token_id,
                                                    num_logits_to_keep=0)
            for prompts, images, audios in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case,
                                        vllm_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
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
@pytest.mark.parametrize("max_model_len", [12800])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [10])
def test_models(hf_runner, vllm_runner, image_assets, model, size_factors,
                dtype: str, max_model_len: int, max_tokens: int,
                num_logprobs: int) -> None:
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
        None,
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    run_test(
        hf_runner,
        vllm_runner,
        inputs_per_image,
        model,
        dtype=dtype,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=1,
        tensor_parallel_size=1,
    )


@large_gpu_test(min_gb=48)
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # No image
        # [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_model_len", [25600])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [10])
def test_multi_images_models(hf_runner, vllm_runner, image_assets, model,
                             size_factors, dtype: str, max_model_len: int,
                             max_tokens: int, num_logprobs: int) -> None:
    images = [asset.pil_image for asset in image_assets]

    inputs_per_case = [
        (
            [HF_MULTIIMAGE_IMAGE_PROMPT for _ in size_factors],
            [[rescale_image_size(image, factor) for image in images]
             for factor in size_factors],
            None,
        ),
    ]

    run_test(
        hf_runner,
        vllm_runner,
        inputs_per_case,
        model,
        dtype=dtype,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=2,
        tensor_parallel_size=1,
    )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_model_len", [12800])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [10])
def test_vision_speech_models(hf_runner, vllm_runner, model, dtype: str,
                              max_model_len: int, max_tokens: int,
                              num_logprobs: int) -> None:

    # use the example speech question so that the model outputs are reasonable
    audio = librosa.load(speech_question, sr=None)
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    inputs_vision_speech = [
        (
            ["<|user|><|image_1|><|audio_1|><|end|><|assistant|>"],
            [image],
            [audio],
        ),
    ]

    run_test(
        hf_runner,
        vllm_runner,
        inputs_vision_speech,
        model,
        dtype=dtype,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=1,
        tensor_parallel_size=1,
    )
