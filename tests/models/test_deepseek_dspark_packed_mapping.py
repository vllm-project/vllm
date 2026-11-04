# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    torch.cuda.get_device_properties = lambda *args, **kwargs: SimpleNamespace(
        gcnArchName="gfx942"
    )

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.models.deepseek_v4.amd.dspark import (
    DSparkDeepseekV4ForCausalLM as V4AMD,
)
from vllm.models.deepseek_v4.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM as V4Nvidia,
)
from vllm.models.deepseek_v4.xpu.dspark import (
    DSparkDeepseekV4ForCausalLM as V4XPU,
)
from vllm.models.deepseek_v41.amd.dspark import (
    DSparkDeepseekV4ForCausalLM as V41AMD,
)
from vllm.models.deepseek_v41.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM as V41Nvidia,
)

EXPECTED_MAPPING = {
    "gate_up_proj": ["w1", "w3"],
    "fused_wqa_wkv": ["wq_a", "wkv"],
}


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "model_cls",
    [V4AMD, V4Nvidia, V4XPU, V41AMD, V41Nvidia],
    ids=["v4_amd", "v4_nvidia", "v4_xpu", "v41_amd", "v41_nvidia"],
)
def test_deepseek_dspark_packed_modules_mapping(model_cls):
    """Verify that all DSpark draft model entry points define packed_modules_mapping
    matching their stacked_params_mapping, so that is_layer_skipped() can expand
    fused layer prefixes to their underlying shard names when checking ignored_layers.
    Fixes #65618.
    """
    mapping = getattr(model_cls, "packed_modules_mapping", None)
    assert mapping is not None, f"{model_cls.__name__} missing packed_modules_mapping"
    assert mapping == EXPECTED_MAPPING

    # Verify is_layer_skipped correctly expands fused draft layer names
    ignored_shards = ["wq_a", "wkv", "w1", "w3"]
    assert is_layer_skipped(
        "model.layers.32.attn.fused_wqa_wkv",
        ignored_shards,
        fused_mapping=mapping,
        match_mode="substring",
    )
    assert is_layer_skipped(
        "model.layers.32.ffn.gate_up_proj",
        ignored_shards,
        fused_mapping=mapping,
        match_mode="substring",
    )
