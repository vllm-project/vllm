# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.vision import resolve_visual_encoder_outputs


@pytest.mark.parametrize(
    ("feature_sample_layers", "num_layers_loaded", "max_possible_layers",
     "expected_features"),
    [
        # All layers loaded
        ([1, 10], 10, 10, [1, 10]),
        ([-10, -1], 10, 10, [1, 10]),
        # Some layers not loaded
        ([1, 10], 10, 20, [1, 10]),
        ([-20, -11], 10, 20, [1, 10]),
    ])
def test_resolve_visual_encoder_outputs(feature_sample_layers,
                                        num_layers_loaded, max_possible_layers,
                                        expected_features):
    """
    Test that offsets are correctly handled for vision feature layers.
    """
    encoder_outputs = [
        torch.tensor([idx]) for idx in range(num_layers_loaded + 1)
    ]
    output_tensor = resolve_visual_encoder_outputs(
        encoder_outputs=encoder_outputs,
        feature_sample_layers=feature_sample_layers,
        post_layer_norm=None,
        max_possible_layers=max_possible_layers)
    assert torch.equal(torch.tensor(expected_features), output_tensor)
