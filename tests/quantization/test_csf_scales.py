# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.layers.quantization.utils.csf_scales import decode_csf_scale
from vllm.model_executor.model_loader.reload.layerwise import (
    record_metadata_for_reloading,
)


def _encode(scales: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reference csf encoder: row base = row min (capped at 240), outliers as
    exceptions."""
    raw = scales.view(torch.uint8).to(torch.int64)
    rows, cols = raw.shape
    bases = raw.min(dim=1, keepdim=True).values.clamp(max=240)
    codes = raw - bases
    outlier = (codes < 0) | (codes > 15)
    codes = torch.where(outlier, 0, codes)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    fixed = torch.cat(
        (
            bases.to(torch.uint8).reshape(rows // 16, 16),
            packed.reshape(rows // 16, -1),
        ),
        dim=1,
    )
    positions = outlier.reshape(-1).nonzero().reshape(-1)
    words = (raw.reshape(-1)[positions] << 24) | positions
    return {
        "nvfp4_csf_fixed": fixed,
        "nvfp4_csf_exceptions": words.to(torch.int32).view(torch.uint32),
    }


def _encode_32x2(raw: np.ndarray) -> dict[str, torch.Tensor]:
    """Reference csf_32x2_v1 encoder, written as the format's loops."""
    rows, cols = raw.shape
    bases = raw.min(axis=1).clip(max=240)
    codes: list[int] = []
    origins: list[int] = []
    offsets: list[int] = []
    words: list[int] = []
    group = 0
    for row in range(0, rows, 128):
        for col in range(0, cols, 4):
            if group % 32 == 0:
                origins.append(2 * len(words))
                offsets.append(0)
                page = len(words)
            for pair in range(col, min(col + 4, cols), 2):
                for block in range(row, min(row + 128, rows), 32):
                    for i in range(32):
                        for j in range(min(2, cols - pair)):
                            value = int(raw[block + i, pair + j])
                            code = value - int(bases[block + i])
                            codes.append(code if 0 <= code <= 15 else 0)
                            if not 0 <= code <= 15:
                                words.append(value << 8 | 2 * i + j)
                    offsets.append(len(words) - page)
            group += 1
    nibbles = np.array(codes, dtype=np.uint8)
    streams = (
        ("bases", bases, np.uint8),
        ("codes", nibbles[0::2] | nibbles[1::2] << 4, np.uint8),
        ("page_bases", origins, np.uint32),
        ("offsets", offsets, np.uint16),
        ("exceptions", words, np.uint16),
    )
    return {
        f"csf32_{name}": torch.from_numpy(np.array(data, dtype=dtype))
        for name, data, dtype in streams
    }


def _raw_scales(shape: tuple[int, int]) -> torch.Tensor:
    torch.manual_seed(0)
    raw = torch.randint(0x30, 0x40, shape, dtype=torch.uint8)
    raw.view(-1)[torch.randperm(raw.numel())[:64]] = 0x7E  # out-of-window bytes
    return raw


@pytest.mark.parametrize("shape", [(640, 160), (2560, 40)])
def test_csf_decode_matches_source_scales(shape):
    raw = _raw_scales(shape)
    streams = _encode(raw.view(torch.float8_e4m3fn))
    assert streams["nvfp4_csf_exceptions"].numel() >= 64

    decoded = decode_csf_scale("csf", streams)

    assert decoded.dtype == torch.float8_e4m3fn
    assert torch.equal(decoded.view(torch.uint8), raw)


def test_csf_32x2_decode_matches_source_scales():
    # 352 rows leave a partial 128-row tile, 43 columns a 1-wide block, and
    # the 33 tiles span two pages.
    raw = _raw_scales((352, 43))
    raw[:32, :2] = 0x7E  # a block made only of exceptions
    streams = _encode_32x2(raw.numpy())

    decoded = decode_csf_scale("csf_32x2_v1", streams)

    assert torch.equal(decoded.view(torch.uint8), raw)


@pytest.mark.parametrize(
    "base, words", [(0xFF, []), (0x38, [5, 3]), (0x38, [3, 3]), (0x38, [64])]
)
def test_invalid_csf_streams_are_rejected(base, words):
    raw = torch.full((16, 4), 0x38, dtype=torch.uint8)
    streams = _encode(raw.view(torch.float8_e4m3fn))
    streams["nvfp4_csf_fixed"][0, 0] = base
    exceptions = torch.tensor(words, dtype=torch.int32).view(torch.uint32)
    streams["nvfp4_csf_exceptions"] = exceptions
    with pytest.raises(ValueError, match="CSF"):
        decode_csf_scale("csf", streams)


@pytest.mark.parametrize(
    "stream, index, value",
    [
        ("csf32_page_bases", 1, 0),  # page origin out of step
        ("csf32_offsets", 1, 65),  # block end past its 64 cells
        ("csf32_exceptions", 0, 0x7E40),  # position past the block
        ("csf32_exceptions", 1, 0x7E00),  # positions out of order
    ],
)
def test_invalid_csf_32x2_streams_are_rejected(stream, index, value):
    raw = torch.full((352, 43), 0x38, dtype=torch.uint8)
    raw[:32, :2] = 0x7E
    raw[128, 4] = 0x7E
    streams = _encode_32x2(raw.numpy())
    streams[stream].numpy()[index] = value
    with pytest.raises(ValueError, match="(?i)csf"):
        decode_csf_scale("csf_32x2_v1", streams)


def _mixed_config(encoding: str) -> ModelOptMixedPrecisionConfig:
    config = ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.layers.0.mlp.experts": {
                        "quant_algo": "NVFP4",
                        "weight_scale_encoding": encoding,
                    }
                },
            }
        }
    )
    assert isinstance(config, ModelOptMixedPrecisionConfig)
    return config


def test_config_maps_csf_streams_to_method_params():
    config = _mixed_config("csf_32x2_v1")

    mapped = config.get_checkpoint_weight_mapper()._map_name(
        "model.layers.0.mlp.experts.3.up_proj.weight_scale.csf32_codes"
    )

    assert mapped == "model.layers.0.mlp.experts.3.up_proj.weight_scale_csf32_codes"
    assert config._resolve_scale_encoding("model.layers.0.mlp.experts") == (
        "csf_32x2_v1"
    )
    with pytest.raises(ValueError, match="weight_scale_encoding"):
        _mixed_config("csf_64x1")


def test_stream_params_load_decoded_scale_once_streams_arrive(monkeypatch):
    for name, value in (("rank", 0), ("world_size", 1)):
        monkeypatch.setattr(
            f"vllm.model_executor.parameter.get_tensor_model_parallel_{name}",
            lambda value=value: value,
        )
    method = object.__new__(ModelOptNvFp4FusedMoE)
    method.scale_encoding = "csf"
    method._csf_parts = {}
    method.quant_config = SimpleNamespace(
        is_checkpoint_nvfp4_serialized=True, group_size=16
    )
    method.moe = SimpleNamespace(is_act_and_mul=True)
    method.use_global_sf = False
    layer = torch.nn.Module()
    layer._map_global_expert_id_to_local_expert_id = lambda e: e - 1
    method.create_weights(layer, 1, 64, 32, torch.bfloat16, weight_loader=None)
    # Reload wraps the scale's loader after create_weights; streams must use it.
    scale_loader = layer.w2_weight_scale.weight_loader = MagicMock(return_value=True)
    raw = _raw_scales((32, 4))
    streams = _encode(raw.view(torch.float8_e4m3fn))

    for expert_id in (0, 1):
        for stream in ("nvfp4_csf_fixed", "nvfp4_csf_exceptions"):
            param = getattr(layer, f"w2_weight_scale_{stream}")
            assert param.weight_loader_target == "w2_weight_scale"
            loaded = param.weight_loader(
                param,
                streams[stream],
                f"e.w2_weight_scale_{stream}",
                "w2",
                expert_id,
                return_success=True,
            )
            assert loaded is (expert_id == 1)

    (param, scale, name, shard_id, expert_id, _), _ = scale_loader.call_args
    assert scale_loader.call_count == 1 and not method._csf_parts
    assert param is layer.w2_weight_scale
    assert (name, shard_id, expert_id) == ("e.w2_weight_scale", "w2", 1)
    assert torch.equal(scale.view(torch.uint8), raw)

    # The quant method reaches the layer through its MoE quant config, so the
    # stream loaders must not keep it alive from the reload metadata.
    layer.quant_method, method.moe_quant_config = method, layer
    record_metadata_for_reloading(layer)
    layer_ref = weakref.ref(layer)
    del layer, method
    gc.collect()
    assert layer_ref() is None
