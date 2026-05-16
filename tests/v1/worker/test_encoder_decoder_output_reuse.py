# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.skip_global_cleanup


def _make_feature(identifier: str) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="audio",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=0, length=4),
    )


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_gpu_model_runner_reuses_encoder_decoder_outputs_in_batch_order():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    first_output = torch.tensor([[1.0]])
    second_output = torch.tensor([[2.0]])

    runner.input_batch = SimpleNamespace(req_ids=["req_b", "req_skip", "req_a"])
    runner.requests = {
        "req_a": SimpleNamespace(
            num_computed_tokens=0,
            mm_features=[_make_feature("audio_a")],
        ),
        "req_b": SimpleNamespace(
            num_computed_tokens=0,
            mm_features=[_make_feature("audio_b")],
        ),
        "req_skip": SimpleNamespace(
            num_computed_tokens=1,
            mm_features=[_make_feature("audio_skip")],
        ),
    }
    runner.encoder_cache = {
        "audio_a": first_output,
        "audio_b": second_output,
        "audio_skip": torch.tensor([[3.0]]),
    }

    encoder_outputs = runner._get_encoder_decoder_outputs()

    assert encoder_outputs == [second_output, first_output]