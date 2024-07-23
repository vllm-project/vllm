import re
from typing import List, Optional, Type

import pytest

from vllm.multimodal.utils import rescale_image_size

from ..conftest import IMAGE_ASSETS, VllmRunner, _ImageAssets

pytestmark = pytest.mark.vlm

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "USER: <image>\nWhat's the content of the image?\nASSISTANT:",
    "cherry_blossom":
    "USER: <image>\nWhat is the season?\nASSISTANT:",
})

models = ["facebook/chameleon-7b"]


#TODO (ywang96): Add correctness test when chameleon is
# available on transformers.
def run_test(
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    size_factors: List[float],
    dtype: str,
    max_tokens: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Test if the model can generate text given 
    a batch of images and prompts.

    """
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    with vllm_runner(model,
                     max_model_len=4096,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:

        for prompts, images in inputs_per_image:
            vllm_outputs = vllm_model.generate_greedy(prompts,
                                                      max_tokens,
                                                      images=images)
            for i in range(len(vllm_outputs)):

                # format prompt back to original
                replacements = {
                    "<racm3:break>": "",
                    "<eoss>": "",
                    "<reserved08706>": ""
                }
                pattern = '|'.join(replacements.keys())
                vllm_result = re.sub(
                    pattern,
                    lambda match: replacements[match.group(0)],  #noqa B023
                    vllm_outputs[i][1])
                vllm_result = vllm_result.replace("<image>", "", 1023)
                assert vllm_result[:len(prompts[i])] == prompts[i]

                # assert at least 10 new characters are generated
                # (to take stop token into account)
                assert len(vllm_outputs[i][1]) - len(prompts[i]) > 10


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(vllm_runner, image_assets, model, size_factors, dtype: str,
                max_tokens: int) -> None:
    run_test(
        vllm_runner,
        image_assets,
        model,
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        tensor_parallel_size=1,
    )
