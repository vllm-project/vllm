# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""With online quantization of an MXFP8 checkpoint, ROCm attention builds wo_a
on the checkpoint's quant method on CDNA3, whose einsum path reads the MXFP8
weight itself, and on the online one on CDNA4."""

import pytest

from vllm.config.model import ModelConfig
from vllm.config.quantization import _ONLINE_SHORTHANDS
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
)
from vllm.models.deepseek_v4.common.grouped_wo_a import DeepseekV4GroupedWoA
from vllm.models.deepseek_v41.quant_config import (
    DeepseekV4FP8Config,
    _mxfp8_linear_method_cls,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm attention")


@pytest.mark.parametrize(
    ("cdna_version", "keeps_checkpoint_quant"), [(3, True), (4, False)]
)
def test_wo_a_online_quant_by_cdna_version(
    default_vllm_config, dist_init, monkeypatch, cdna_version, keeps_checkpoint_quant
):
    import vllm.platforms.rocm as rocm_platform
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention

    monkeypatch.setattr(rocm_platform, "get_cdna_version", lambda: cdna_version)
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

    expected = (
        _mxfp8_linear_method_cls()
        if keeps_checkpoint_quant
        else Fp8PtpcOnlineLinearMethod
    )
    assert type(wo_a.quant_method) is expected
    # The other layers keep the online quantization.
    assert quant_config.online_quantization_config is online_config
