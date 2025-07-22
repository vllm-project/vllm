# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.utils import set_default_torch_num_threads

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

    mm_data = generate_test_mm_data()
    prompt = {
        # This model deals with no text input
        "prompt_token_ids": [1],
        "multi_modal_data": mm_data
    }
    with (
            set_default_torch_num_threads(1),
            vllm_runner(
                model,
                task="embed",
                dtype=torch.float16,
                enforce_eager=True,
                skip_tokenizer_init=True,
            ) as vllm_model,
    ):
        vllm_model.encode(prompt)


MODELS = ["christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM"]


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
