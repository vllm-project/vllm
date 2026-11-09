# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded KDA shards must finish loading without a checkpoint padding shard."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm KDA projection loader"
)


@pytest.mark.parametrize("alignment", [16, 128])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("online", [False, True])
def test_padded_kda_load_completes_and_clears_padding(
    default_vllm_config, dist_init, alignment, reverse, online
):
    from vllm.config import ModelConfig
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
    from vllm.models.kimi_k3.amd.kda import _PaddedKimiGDNMergedColumnParallelLinear

    default_vllm_config.model_config = ModelConfig(
        model="facebook/opt-125m", dtype="bfloat16"
    )
    # The TP8 local Kimi geometry, with a smaller input dimension for this test.
    rows = [1536] * 4 + [128, 12]
    padding = -sum(rows) % alignment
    quant = (
        OnlineQuantizationConfig(QuantizationConfigArgs(linear="fp8_per_channel"))
        if online
        else None
    )
    with torch.device("cuda"), torch.no_grad():
        layer = _PaddedKimiGDNMergedColumnParallelLinear(
            256,
            rows + [padding],
            replicated_shard_id=4,
            tp_size=1,
            padding_size=padding,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant,
            prefix="probe.in_proj_qkvgfab",
        )
        if not online:
            layer.weight.fill_(float("nan"))
        param = layer.weight
        shards = [
            torch.full((n, 256), (i + 1) / 16, dtype=torch.bfloat16)
            for i, n in enumerate(rows)
        ]
        for i in reversed(range(6)) if reverse else range(6):
            param.weight_loader(param, shards[i], i)
        if online:
            assert layer.weight.device.type == "cuda"
            assert not get_layerwise_info(layer).loaded_weights
        else:
            expected = torch.cat(
                shards + [torch.zeros(padding, 256, dtype=torch.bfloat16)]
            )
            torch.testing.assert_close(layer.weight, expected, atol=0, rtol=0)
