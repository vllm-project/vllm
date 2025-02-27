# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Type

import pytest
from PIL import Image

from vllm.inputs.data import ExplicitEncoderDecoderPrompt, TextPrompt
from vllm.multimodal.image import rescale_image_size
from vllm.sequence import SampleLogprobs

from ....conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from ...utils import check_logprobs_close

MODELS = ["microsoft/Florence-2-base"]
# Florence-2 uses BartFastTokenizer which can't be loaded from AutoTokenizer
# Therefore, we borrow the BartTokenizer from the original Bart model
TOKENIZER = "facebook/bart-base"
HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<CAPTION>",  # special task token
    "cherry_blossom":
    "Describe in detail what is shown in the image.",
})


def get_hf_images_prompts(
    prompts_: list[ExplicitEncoderDecoderPrompt[str, TextPrompt]],
) -> tuple[list[ExplicitEncoderDecoderPrompt[str, str]], list[Image.Image]]:
    prompts, images = [], []
    for prompt in prompts_:
        encoder_prompt = prompt["encoder_prompt"]
        prompts.append(
            ExplicitEncoderDecoderPrompt(
                encoder_prompt=encoder_prompt["prompt"],
                decoder_prompt=None,
            ))
        images.append(encoder_prompt["multi_modal_data"]["image"])
    return prompts, images


def hf_to_vllm_output(hf_output: tuple[list[int], str,
                                       Optional[SampleLogprobs]]):
    """Sanitize hf output to be comparable with vllm output."""
    output_ids, output_str, out_logprobs = hf_output

    output_str = output_str.replace("</s>", "").replace("<s>", "")
    output_ids = [ids for ids in output_ids if ids not in [0, 2]]

    return output_ids, output_str, out_logprobs


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    inputs: list[list[ExplicitEncoderDecoderPrompt]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
) -> None:
    with vllm_runner(model,
                     max_num_seqs=8,
                     tokenizer_name=TOKENIZER,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        vllm_outputs_per_case = [
            vllm_model.generate_encoder_decoder_greedy_logprobs(
                prompts, max_tokens, num_logprobs=num_logprobs)
            for prompts in inputs
        ]

    hf_inputs = [get_hf_images_prompts(prompts) for prompts in inputs]

    with hf_runner(model, dtype=dtype, skip_tokenizer_init=True) as hf_model:
        hf_model.model.get_output_embeddings = lambda: \
            hf_model.model.language_model.lm_head
        hf_outputs_per_case = [
            hf_model.generate_encoder_decoder_greedy_logprobs_limit(
                prompts, max_tokens, num_logprobs=num_logprobs, images=images)
            for prompts, images in hf_inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case,
                                        vllm_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=[hf_to_vllm_output(output) for output in hf_outputs],
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.core_model
@pytest.mark.parametrize("model", MODELS)
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
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner: Type[HfRunner], vllm_runner: Type[VllmRunner],
                image_assets: _ImageAssets, model: str,
                size_factors: list[int], dtype: str, max_tokens: int,
                num_logprobs: int) -> None:
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [[
        ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt=prompt,
                multi_modal_data={"image": rescale_image_size(image, factor)}),
            decoder_prompt=None,
        ) for factor in size_factors
    ] for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    run_test(
        hf_runner,
        vllm_runner,
        inputs_per_image,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
