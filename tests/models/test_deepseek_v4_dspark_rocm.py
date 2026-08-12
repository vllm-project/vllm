# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.amd.dspark import DSparkDeepseekV4ForCausalLM


def _make_uninitialized_model(confidence_head):
    model = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    object.__setattr__(
        model,
        "model",
        SimpleNamespace(confidence_head=confidence_head),
    )
    return model


@pytest.mark.cpu_test
def test_deepseek_v4_rocm_dspark_maps_enabled_confidence_head():
    model = _make_uninitialized_model(object())

    assert (
        model._remap_dspark_name("mtp.2.confidence_head.proj.weight")
        == "model.confidence_head.proj.weight"
    )


@pytest.mark.cpu_test
def test_deepseek_v4_rocm_dspark_skips_disabled_confidence_head():
    model = _make_uninitialized_model(None)

    assert model._remap_dspark_name("mtp.2.confidence_head.proj.weight") is None


@pytest.mark.cpu_test
def test_deepseek_v4_rocm_dspark_confidence_is_probability():
    class ConfidenceHead:
        def __call__(self, head_hidden, markov_embed):
            return (head_hidden[:, 0] + markov_embed[:, 0]).float()

    model = _make_uninitialized_model(ConfidenceHead())
    head_hidden = torch.tensor([[0.0], [1.0]], dtype=torch.bfloat16)
    markov_embed = torch.tensor([[0.0], [-2.0]], dtype=torch.bfloat16)

    confidence = model.compute_confidence(head_hidden, markov_embed)

    torch.testing.assert_close(
        confidence,
        torch.sigmoid(torch.tensor([0.0, -1.0])),
    )
    assert torch.all((confidence >= 0) & (confidence <= 1))
