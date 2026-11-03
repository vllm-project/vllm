# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from PIL import Image

MODEL = "ibm-granite/granite-vision-4.1-4b"

# (width, height) -> anyres tile count with GV4.1's grid (384px tiles + base view):
# 5, 10 and 7 tiles. Distinct sizes also give distinct hashes, so no image is
# served from the processor or encoder cache.
IMAGE_SIZES = ((640, 480), (1300, 876), (1024, 768))


def _prompt(num_images: int) -> str:
    return f"<|user|>\n{'<image>' * num_images}\nDescribe the image.\n<|assistant|>\n"


@pytest.mark.core_model
@pytest.mark.parametrize("max_tokens", [8])
def test_mixed_tile_counts_across_requests(
    vllm_runner, monkeypatch: pytest.MonkeyPatch, max_tokens: int
) -> None:
    """Batch requests whose images have different tile counts.

    The HF processor pads every image in one call to that call's largest tile
    count, so the first request's 5-tile image arrives padded to 10 tiles,
    while the second request's image has 7. Encoded together, the per-image
    tensors differ in shape and `MultiModalFieldConfig.batched` passes the
    model a list instead of a 5-D tensor. The padding tiles were stripped only
    for 5-D input, so the list path concatenated 27 tiles and failed with

        RuntimeError: split_with_sizes expects split_sizes to sum exactly to
        27 ... but got split_sizes=[5, 10, 7]

    taking EngineCore down for every in-flight request.
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

    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 2},
    ) as vllm_model:
        # One call so both requests' images are encoded in the same batch;
        # issued separately they never reach the list path.
        outputs = vllm_model.generate_greedy(
            [_prompt(2), _prompt(1)],
            max_tokens,
            images=[[small, large], [medium]],
        )

    assert len(outputs) == 2
    for output in outputs:
        assert len(output[1]) > 0, "Expected non-empty output"
