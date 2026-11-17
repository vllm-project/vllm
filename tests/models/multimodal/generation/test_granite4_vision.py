# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from PIL import Image

from ...utils import check_logprobs_close

MODEL = "ibm-granite/granite-vision-4.1-4b"

# (width, height) -> anyres tile count with GV4.1's grid (384px tiles + base view):
# 5, 10 and 7 tiles. Distinct sizes also give distinct hashes, so no image is
# served from the processor or encoder cache.
IMAGE_SIZES = ((640, 480), (1300, 876), (1024, 768))


def _prompt(num_images: int) -> str:
    return f"<|user|>\n{'<image>' * num_images}\nDescribe the image.\n<|assistant|>\n"


@pytest.mark.core_model
@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("num_logprobs", [5])
def test_mixed_tile_counts_across_requests(
    vllm_runner,
    monkeypatch: pytest.MonkeyPatch,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Batch requests whose images have different tile counts.

    The first request's 5-tile image is padded to 10 tiles by the HF processor
    and the second request's image has 7, so when both are encoded together the
    model receives `pixel_values` as a list of differently shaped tensors.
    Outputs must match running each request alone, where every encoder batch
    is a single 5-D tensor.
    """
    # In-process engine: both requests are queued before the first engine step,
    # so their images are encoded in one batch. With a separate engine process
    # the requests can reach the scheduler in different steps and be encoded
    # separately, which never reaches the list path.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    small, large, medium = (
        Image.new("RGB", size, color)
        for size, color in zip(IMAGE_SIZES, ("red", "green", "blue"))
    )
    prompts = [_prompt(2), _prompt(1)]
    images = [[small, large], [medium]]
    runner_kwargs = dict(
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 2},
        # Two engines run one after the other in this process.
        gpu_memory_utilization=0.9,
    )

    with vllm_runner(MODEL, **runner_kwargs) as vllm_model:
        # One call so both requests' images are encoded in the same batch.
        batched = vllm_model.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs=num_logprobs, images=images
        )

    # Fresh engine, so no encoder or prefix cache entry from the batched run.
    with vllm_runner(MODEL, **runner_kwargs) as vllm_model:
        single = [
            vllm_model.generate_greedy_logprobs(
                [prompt], max_tokens, num_logprobs=num_logprobs, images=[image]
            )[0]
            for prompt, image in zip(prompts, images)
        ]

    check_logprobs_close(
        outputs_0_lst=single,
        outputs_1_lst=batched,
        name_0="single",
        name_1="batched",
    )
