# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (  # noqa: E501
    _get_moe_scheme_dicts,
)


class _MockRoutedExperts(torch.nn.Module):
    ckpt_gate_proj_name = "w1"
    ckpt_down_proj_name = "w2"
    ckpt_up_proj_name = "w3"


def _make_config(target_scheme_map):
    return CompressedTensorsConfig(
        target_scheme_map=target_scheme_map,
        ignore=[],
        quant_format="mxfp4-pack-quantized",
    )


def test_moe_scheme_lookup_maps_mixtral_container_name():
    scheme = {"weights": object(), "format": "mxfp4-pack-quantized"}
    config = _make_config({r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$": scheme})

    schemes = _get_moe_scheme_dicts(
        config,
        _MockRoutedExperts(),
        "model.layers.4.block_sparse_moe.experts",
    )

    assert schemes == [scheme, scheme, scheme]


def test_moe_scheme_lookup_uses_checkpoint_projection_names():
    scheme = {"weights": object(), "format": "mxfp4-pack-quantized"}
    config = _make_config({r"re:.*\.experts\.\d+\.w[123]$": scheme})

    schemes = _get_moe_scheme_dicts(
        config, _MockRoutedExperts(), "model.layers.4.mlp.experts"
    )

    assert schemes == [scheme, scheme, scheme]


def test_moe_scheme_lookup_preserves_per_layer_mixed_precision():
    layer_3_scheme = {"weights": object(), "format": "mxfp4-pack-quantized"}
    layer_4_scheme = {"weights": object(), "format": "mxfp8-quantized"}
    config = _make_config(
        {
            r"re:.*layers\.3\.mlp\.experts\.\d+\..*_proj$": layer_3_scheme,
            r"re:.*layers\.4\.mlp\.experts\.\d+\..*_proj$": layer_4_scheme,
        }
    )

    schemes = _get_moe_scheme_dicts(
        config,
        _MockRoutedExperts(),
        "model.layers.4.block_sparse_moe.experts",
    )

    assert schemes == [layer_4_scheme, layer_4_scheme, layer_4_scheme]


def test_moe_scheme_lookup_honors_canonical_path_ignore():
    scheme = {"weights": object(), "format": "mxfp4-pack-quantized"}
    config = _make_config({"MockRoutedExperts": scheme})
    config.ignore = [r"re:.*mlp\.experts\.\d+\.up_proj$"]

    schemes = _get_moe_scheme_dicts(
        config,
        _MockRoutedExperts(),
        "model.layers.4.block_sparse_moe.experts",
    )

    assert schemes == [scheme, None, scheme]
