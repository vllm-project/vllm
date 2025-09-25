# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from ....conftest import VllmRunner


def generate_test_mm_data():
    mm_data = {
        "pixel_values": torch.full((6, 512, 512), 1.0, dtype=torch.float16),
        "location_coords": torch.full((1, 2), 1.0, dtype=torch.float16),
    }
    return mm_data


def _run_test(
    vllm_runner: type[VllmRunner],
    model: str,
) -> None:

    prompt = [
        {
            # This model deals with no text input
            "prompt_token_ids": [1],
            "multi_modal_data": generate_test_mm_data(),
        } for _ in range(10)
    ]

    with vllm_runner(
            model,
            runner="pooling",
            dtype="half",
            enforce_eager=True,
            skip_tokenizer_init=True,
            # Limit the maximum number of sequences to avoid the
            # test going OOM during the warmup run
            max_num_seqs=32,
            default_torch_num_threads=1,
    ) as vllm_model:
        vllm_model.encode(prompt)


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
