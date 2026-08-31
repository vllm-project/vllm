# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.watermarking import WatermarkConfig, WatermarkPRFName
from vllm.v1.watermarking import create_watermarker


@pytest.mark.parametrize("algorithm", ["gumbel"])
@pytest.mark.parametrize("prf_name", ["philox", "hmac_sha256"])
def test_watermarker_contract(algorithm: str, prf_name: WatermarkPRFName):
    watermarker = create_watermarker(
        WatermarkConfig(algorithm=algorithm, key=42, context_width=4, prf=prf_name)
    )
    logits = torch.zeros(2, 128)
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    random_sample = lambda sample_logits: sample_logits.argmax(dim=-1)

    first = watermarker.sample(logits, contexts, random_sample)
    second = watermarker.sample(logits, contexts, random_sample)

    assert first.token_ids.shape == (2,)
    assert first.logits.shape == logits.shape
    assert torch.equal(first.token_ids, second.token_ids)


def test_large_context_width_warns_but_is_allowed():
    config = WatermarkConfig(key=42, context_width=17)

    with pytest.warns(UserWarning, match="reduce robustness to edits"):
        watermarker = create_watermarker(config)

    assert watermarker.context_width == 17
