# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.conftest import VllmRunner


@pytest.mark.parametrize(
    "model",
    [
        "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        "mgazz/Prithvi_v2_eo_300_tl_unet_agb",
    ],
)
def test_inference(
    vllm_runner: type[VllmRunner],
    model: str,
) -> None:
    pixel_values = torch.full((6, 512, 512), 1.0, dtype=torch.float16)
    location_coords = torch.full((1, 2), 1.0, dtype=torch.float16)
    prompt = dict(
        prompt_token_ids=[1],
        multi_modal_data=dict(
            pixel_values=pixel_values, location_coords=location_coords
        ),
    )
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
        vllm_output = vllm_model.llm.encode(prompt)
        assert torch.equal(
            torch.isnan(vllm_output[0].outputs.data).any(), torch.tensor(False)
        )
