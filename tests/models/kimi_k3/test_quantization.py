# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.models.kimi_k3.nvidia.quantization import (
    pad_merged_output_sizes,
    uses_modelopt_fp8_pb_wo,
)


def test_modelopt_fp8_pb_wo_resolution():
    homogeneous = SimpleNamespace(quant_method="FP8_PB_WO")
    mixed = SimpleNamespace(_resolve_quant_algo=lambda _prefix: "FP8_PB_WO")

    assert uses_modelopt_fp8_pb_wo(homogeneous, "model.layers.0.proj")
    assert uses_modelopt_fp8_pb_wo(mixed, "model.layers.0.proj")
    assert not uses_modelopt_fp8_pb_wo(None, "model.layers.0.proj")


def test_fp8_pb_wo_mla_padding():
    output_sizes, padding = pad_merged_output_sizes(
        [2048, 576], 8, disable_tp=True, alignment=128
    )

    assert output_sizes == [2048, 576, 64]
    assert padding == 64


def test_fp8_pb_wo_kda_padding():
    output_sizes, padding = pad_merged_output_sizes(
        [12288] * 4 + [128, 96],
        8,
        disable_tp=False,
        alignment=128,
        replicated_shard_ids=(4,),
    )

    assert output_sizes == [12288] * 4 + [128, 96, 928]
    assert padding == 116
