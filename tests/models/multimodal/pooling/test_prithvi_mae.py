# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from ....conftest import VllmRunner


def _run_test(
    vllm_runner: type[VllmRunner],
    model: str,
) -> None:
    prompt = [
        {
            # This model deals with no text input
            "prompt_token_ids": [1],
            "multi_modal_data": {
                "image": {
                    "pixel_values": torch.ones((6, 512, 512), dtype=torch.float16),
                    "location_coords": torch.ones((1, 2), dtype=torch.float16),
                }
            },
        }
        for _ in range(10)
    ]

    with vllm_runner(
        model,
        runner="pooling",
        dtype="half",
        enforce_eager=True,
        skip_tokenizer_init=True,
        enable_mm_embeds=True,
        # Limit the maximum number of sequences to avoid the
        # test going OOM during the warmup run
        max_num_seqs=32,
        default_torch_num_threads=1,
    ) as vllm_model:
        vllm_model.llm.encode(prompt, pooling_task="plugin")


MODELS = ["mgazz/Prithvi-EO-2.0-300M-TL-Sen1Floods11"]


@pytest.mark.core_model
@pytest.mark.parametrize("model", MODELS)
def test_models_image(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
) -> None:
    _run_test(
        vllm_runner,
        model,
    )
