# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""With online quantization of an MXFP8 checkpoint, ROCm attention on CDNA3
builds wo_a on the checkpoint's quant method, since its einsum path reads the
MXFP8 weight itself."""

import pytest

from vllm.config.model import ModelConfig
from vllm.config.quantization import _ONLINE_SHORTHANDS
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.models.deepseek_v4.common.grouped_wo_a import DeepseekV4GroupedWoA
from vllm.models.deepseek_v41.quant_config import (
    DeepseekV4FP8Config,
    _mxfp8_linear_method_cls,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm attention")


def test_cdna3_wo_a_keeps_checkpoint_quant(default_vllm_config, dist_init, monkeypatch):
    import vllm.platforms.rocm as rocm_platform
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention

    monkeypatch.setattr(rocm_platform, "get_cdna_version", lambda: 3)
    default_vllm_config.model_config = ModelConfig(dtype="bfloat16")
    quant_config = DeepseekV4FP8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }
    )
    online_config = OnlineQuantizationConfig(_ONLINE_SHORTHANDS["fp8_per_channel"])
    quant_config.online_quantization_config = online_config
    # Only the wo_a hooks run, so skip the attention layer's __init__.
    attn = object.__new__(DeepseekV41ROCMAiterMLAAttention)

    wo_a = DeepseekV4GroupedWoA(
        512,
        512,
        n_groups=2,
        bias=False,
        quant_config=attn._wo_a_quant_config(quant_config),
        return_bias=False,
        prefix="model.layers.0.attn.wo_a",
    )

    assert type(wo_a.quant_method) is _mxfp8_linear_method_cls()
    # The other layers keep the online quantization.
    assert quant_config.online_quantization_config is online_config
