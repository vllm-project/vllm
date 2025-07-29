# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Sequence
from typing import Optional

import librosa
import pytest
from huggingface_hub import snapshot_download

from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest
from vllm.multimodal.image import rescale_image_size
from vllm.platforms import current_platform

from ....conftest import (IMAGE_ASSETS, HfRunner, PromptAudioInput,
                          PromptImageInput, VllmRunner)
from ....utils import large_gpu_test
from ...utils import check_logprobs_close

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<|user|>\n<|image|>\nWhat's the content of the image?<|end|>\n<|assistant|>\n",  # noqa: E501
    "cherry_blossom":
    "<|user|>\n<|image|>\nPlease infer the season with reason in details.<|end|>\n<|assistant|>\n",  # noqa: E501
})
HF_MULTIIMAGE_IMAGE_PROMPT = "<|user|>\n<|image|>\n<|image|>\nDescribe these images.<|end|>\n<|assistant|>\n"  # noqa: E501

model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct",
                               revision="refs/pr/70")
# Since the vision-lora and speech-lora co-exist with the base model,
# we have to manually specify the path of the lora weights.
vision_lora_path = os.path.join(model_path, "vision-lora")
speech_question = os.path.join(model_path, "examples",
                               "what_is_shown_in_this_image.wav")
models = [model_path]

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
            trust_remote_code=False,
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

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_model.model.load_adapter(
            vision_lora_path,
            adapter_name="vision",
        )
        hf_processor = hf_model.processor
        eos_token_id = hf_processor.tokenizer.eos_token_id
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images,
                                                    audios=audios,
                                                    eos_token_id=eos_token_id)
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
    audio = librosa.load(speech_question, sr=16000)
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")

    inputs_vision_speech = [
        (
            ["<|user|><|image|><|audio|><|end|><|assistant|>"],
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
