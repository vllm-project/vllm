# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.assets.image import ImageAsset
from vllm.model_executor.models.locate_anything import (
    LocateAnythingSlowGrammarLogitsProcessor,
    LocateAnythingTokenIds,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.locate_anything import LocateAnythingConfig

from ....conftest import VllmRunner

MODEL_ID = "nvidia/LocateAnything-3B"
PROMPT = (
    "<image-1>Locate a single instance that matches the following "
    "description: stop sign."
)

pytestmark = [
    pytest.mark.slow_test,
    pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA"),
]


def _assert_valid_box_block(output_ids: list[int]) -> None:
    token_ids = LocateAnythingTokenIds.from_config(LocateAnythingConfig())

    try:
        box_start = output_ids.index(token_ids.box_start)
    except ValueError:
        pytest.fail("LocateAnything output did not contain a box start token")

    try:
        box_end = output_ids.index(token_ids.box_end, box_start + 1)
    except ValueError:
        pytest.fail("LocateAnything output did not contain a matching box end token")

    box_tokens = output_ids[box_start + 1 : box_end]
    coord_tokens = set(range(token_ids.coord_start, token_ids.coord_end + 1))
    assert box_tokens == [token_ids.none] or (
        len(box_tokens) in (2, 4)
        and all(token_id in coord_tokens for token_id in box_tokens)
    )


def test_locate_anything_slow_generation_e2e(
    vllm_runner: type[VllmRunner],
) -> None:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        skip_special_tokens=False,
    )
    image = ImageAsset("stop_sign").pil_image.convert("RGB")

    with vllm_runner(
        MODEL_ID,
        dtype="half",
        max_model_len=4096,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        enforce_eager=True,
        logits_processors=[LocateAnythingSlowGrammarLogitsProcessor],
    ) as vllm_model:
        [(output_ids, output_text, _)] = vllm_model.generate_w_logprobs(
            [PROMPT],
            sampling_params,
            images=[image],
        )

    assert output_text
    _assert_valid_box_block(output_ids)
