# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash checkpoints name their BF16 layers under `model.layers.N`; the
quant config must still match them under vLLM's module names."""

import pytest
import torch

from vllm.config.model import ModelConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.models.glm5next.common.model import Glm5NextForConditionalGeneration
from vllm.models.glm5next.common.mtp import Glm5NextMTP

KDA = "language_model.model.layers.0.self_attn"
# As in zai-org/GLM-5.3-Flash's config.json.
NOT_CONVERTED = [
    f"model.layers.0.self_attn.{name}"
    for name in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")
    + ("o_proj",)
]


@pytest.mark.parametrize(
    ("prefix", "online", "method_cls"),
    [
        (f"{KDA}.o_proj", False, UnquantizedLinearMethod),
        (f"{KDA}.in_proj_qkvbfg_a", False, UnquantizedLinearMethod),
        (f"{KDA}.o_proj", True, Mxfp8OnlineLinearMethod),
        (
            "language_model.model.layers.3.mlp.shared_experts.down_proj",
            False,
            Fp8LinearMethod,
        ),
    ],
    ids=["o_proj", "fused-in_proj", "o_proj+online", "shared-expert"],
)
def test_fp8_checkpoint_keeps_kda_projections_bf16(
    default_vllm_config, dist_init, prefix, online, method_cls
):
    default_vllm_config.model_config = ModelConfig(dtype="bfloat16")
    quant_config = Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": NOT_CONVERTED,
        }
    )
    if online:
        quant_config.online_quantization_config = OnlineQuantizationConfig(
            QuantizationConfigArgs(targets={f"re:.*{prefix}": "mxfp8"})
        )
    configure_quant_config(quant_config, Glm5NextForConditionalGeneration)

    layer = ColumnParallelLinear(
        input_size=256,
        output_size=256,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix=prefix,
        disable_tp=True,
    )

    assert type(layer.quant_method) is method_cls
    # Online quantization starts from the BF16 weight, not from an FP8 checkpoint.
    assert getattr(layer.quant_method, "requantization_source", None) is None


def test_nvfp4_checkpoint_keeps_mtp_layer_bf16(default_vllm_config, dist_init):
    """The MTP draft shares the target's quant config; the NVFP4 checkpoint
    excludes the MTP layer as `model.layers.N*`, which must still match it."""
    default_vllm_config.model_config = ModelConfig(dtype="bfloat16")
    quant_config = ModelOptNvFp4Config.from_config(
        {
            "quant_algo": "NVFP4",
            "group_size": 16,
            "exclude_modules": ["model.layers.45*"],
        }
    )
    configure_quant_config(quant_config, Glm5NextForConditionalGeneration)
    configure_quant_config(quant_config, Glm5NextMTP)

    layer = ColumnParallelLinear(
        input_size=256,
        output_size=256,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix="model.layers.45.mtp_block.mlp.shared_experts.down_proj",
        disable_tp=True,
    )

    assert type(layer.quant_method) is UnquantizedLinearMethod
