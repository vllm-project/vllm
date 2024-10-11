from typing import List, Optional, Tuple, Type

import pytest

from vllm.multimodal.utils import rescale_image_size
from vllm.transformers_utils.tokenizer import patch_padding_side

from ....conftest import IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner
from ....utils import large_gpu_test
from ...utils import check_logprobs_close

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "What's the content of the image?",
    "cherry_blossom":
    "What is the season?",
})

models = ["THUDM/glm-4v-9b"]
target_dtype = "bfloat16"


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
    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     max_model_len=2048,
                     max_num_seqs=2,
                     dtype=dtype,
                     limit_mm_per_prompt={"image": mm_limit},
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        stop_token_ids = [151329, 151336, 151338]
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images,
                                                stop_token_ids=stop_token_ids)
            for prompts, images in inputs
        ]

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_processor = hf_model.processor
        patch_padding_side(hf_processor)

        def processor(*args, text="", images=None, **kwargs):
            if images is None:
                return hf_processor(*args, **kwargs)

            return hf_processor.apply_chat_template(
                [{
                    "role": "user",
                    "image": images,
                    "content": text
                }],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **kwargs,
            )

        hf_model.processor = processor
        hf_model.model.get_output_embeddings = lambda: \
            hf_model.model.transformer.output_layer
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=images,
            ) for prompts, images in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@large_gpu_test(min_gb=48)
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
