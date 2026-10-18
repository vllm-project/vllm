# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM parent caches must refresh from final child weights after live reload."""

from unittest.mock import Mock

import pytest
import torch

from vllm.config import ModelConfig
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.utils import register_derived_buffer
from vllm.models.glm5next.common.attention import Indexer
from vllm.models.glm5next.common.kda import Glm5NextLinearAttention


@pytest.mark.parametrize("cache", ["projection", "convolution"])
def test_materialized_glm_cache_refreshes_across_repeated_reload(cache):
    # Only the weight/cache fields are needed; avoid distributed/kernel setup.
    cls = Indexer if cache == "projection" else Glm5NextLinearAttention
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    if cache == "projection":
        layer.head_dim = 2
        layer.wk_weights_proj = torch.nn.Linear(3, 4, bias=False)
        buffer_name = "_wp_fp32"
        children = [layer.wk_weights_proj]
    else:
        children = []
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            child = torch.nn.Module()
            child.weight = torch.nn.Parameter(torch.zeros(2, 1, 3))
            setattr(layer, name, child)
            children.append(child)
        buffer_name = "_merged_conv_weight"
    register_derived_buffer(layer, buffer_name)
    first = [torch.full_like(child.weight, i + 1) for i, child in enumerate(children)]
    with torch.no_grad():
        for child, value in zip(children, first):
            child.weight.copy_(value)
    layer._refresh_derived_buffers()
    stable = layer._buffers[buffer_name]
    pointer = stable.data_ptr()
    record_metadata_for_reloading(layer)

    for values in ([value + 5 for value in first], first):
        initialize_layerwise_reload(layer)
        for child, value in zip(children, values):
            child.weight.weight_loader(child.weight, value)
        finalize_layerwise_reload(layer, Mock(spec=ModelConfig))
        expected = (
            values[0][2:].t().contiguous().float()
            if cache == "projection"
            else torch.cat([value.view(2, 3) for value in values])
        )
        assert layer._buffers[buffer_name] is stable
        assert stable.data_ptr() == pointer
        assert torch.equal(stable, expected)
        assert buffer_name not in layer.state_dict()
