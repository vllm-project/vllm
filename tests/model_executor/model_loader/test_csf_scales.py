# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    CSF_SCALE_ENCODINGS,
    csf_weights_transform,
    decode_csf_scale,
)
from vllm.model_executor.models.utils import WeightsMapper


def _mapper(encodings, **mappings) -> WeightsMapper:
    return WeightsMapper(
        weights_transforms=[csf_weights_transform(encodings)],
        **mappings,
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


def _raw_scales(shape: tuple[int, int], seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    raw = torch.randint(0x30, 0x40, shape, dtype=torch.uint8)
    raw.view(-1)[torch.randperm(raw.numel())[:64]] = 0x7E  # out-of-window bytes
    return raw


def _encode_scales(encoding: str, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    if encoding == "csf":
        return _encode(raw.view(torch.float8_e4m3fn))
    return _encode_32x2(raw.numpy())


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


@pytest.mark.parametrize("encoding", list(CSF_SCALE_ENCODINGS))
def test_transform_decodes_interleaved_stream_sets(encoding):
    raws = {
        name: _raw_scales((64, 4), seed)
        for name, seed in (("a.weight_scale", 1), ("b.weight_scale", 2))
    }
    streams = {name: _encode_scales(encoding, raw) for name, raw in raws.items()}
    weight = torch.zeros(1)
    # Stream order permuted within each set, sets interleaved, and a
    # non-CSF tensor in the middle.
    arrivals = [("a.weight", weight)]
    for (sa, ta), (sb, tb) in zip(
        reversed(list(streams["a.weight_scale"].items())),
        reversed(list(streams["b.weight_scale"].items())),
    ):
        arrivals.append((f"b.weight_scale.{sb}", tb))
        arrivals.append((f"a.weight_scale.{sa}", ta))

    transform = _mapper((encoding,), orig_to_new_suffix={".weight_scale": ".scale"})
    out = list(transform.apply(iter(arrivals)))

    # b's last stream arrives one slot before a's, so b decodes first, and
    # the decoded canonical names flow through the name mapping.
    assert [name for name, _ in out] == ["a.weight", "b.scale", "a.scale"]
    assert out[0][1] is weight
    expected = (raws["b.weight_scale"], raws["a.weight_scale"])
    for (_, tensor), raw in zip(out[1:], expected):
        assert torch.equal(tensor.view(torch.uint8), raw)


@pytest.mark.parametrize("encoding", list(CSF_SCALE_ENCODINGS))
def test_transform_rejects_unpaired_stream(encoding):
    raw = _raw_scales((64, 4))
    streams = _encode_scales(encoding, raw)
    stream, tensor = next(iter(streams.items()))
    mapper = _mapper((encoding,))
    with pytest.raises(ValueError, match="without a partner"):
        list(mapper.apply(iter([(f"e.weight_scale.{stream}", tensor)])))


def test_transform_rejects_mixed_encodings():
    raw = _raw_scales((64, 4))
    fixed = _encode(raw.view(torch.float8_e4m3fn))["nvfp4_csf_fixed"]
    bases = _encode_32x2(raw.numpy())["csf32_bases"]
    mapper = _mapper(("csf", "csf_32x2_v1"))
    with pytest.raises(ValueError, match="Mixed CSF"):
        list(
            mapper.apply(
                iter(
                    [
                        ("e.weight_scale.nvfp4_csf_fixed", fixed),
                        ("e.weight_scale.csf32_bases", bases),
                    ]
                )
            )
        )


def test_transform_passes_other_tensors_through_untouched():
    mapper = _mapper(("csf", "csf_32x2_v1"))
    tensors = [
        ("e.weight", torch.zeros(2)),
        ("e.weight_scale", torch.zeros(2)),  # canonical scale: not a stream
        ("e.weight_scale_2", torch.zeros(1)),
        ("e.weight_scale_csf32_codes", torch.zeros(1)),  # mapper-era spelling
    ]

    out = list(mapper.apply(iter(tensors)))

    assert [name for name, _ in out] == [name for name, _ in tensors]
    assert all(t1 is t2 for (_, t1), (_, t2) in zip(out, tensors))


def _mixed_config(encoding: str | None) -> ModelOptMixedPrecisionConfig:
    recipe = {"quant_algo": "NVFP4"}
    if encoding is not None:
        recipe["weight_scale_encoding"] = encoding
    config = ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {"model.layers.0.mlp.experts": recipe},
            }
        }
    )
    assert isinstance(config, ModelOptMixedPrecisionConfig)
    return config


def _has_transform(config) -> bool:
    mapper = config.get_checkpoint_weight_mapper()
    return len(mapper.weights_transforms) > 0


def test_mixed_config_gates_the_weights_transform():
    assert not _has_transform(_mixed_config(None))
    assert _has_transform(_mixed_config("csf"))
    assert _has_transform(_mixed_config("csf_32x2_v1"))
    with pytest.raises(ValueError, match="weight_scale_encoding"):
        _mixed_config("csf_64x1")


def _fp4_config(extra: dict) -> ModelOptNvFp4Config:
    quantization = {
        "quant_algo": "NVFP4",
        "group_size": 16,
        "kv_cache_quant_algo": None,
        "exclude_modules": [],
        **extra,
    }
    config = ModelOptNvFp4Config.from_config({"quantization": quantization})
    assert isinstance(config, ModelOptNvFp4Config)
    return config


def test_plain_nvfp4_config_gates_the_weights_transform():
    assert not _has_transform(_fp4_config({}))
    assert _has_transform(_fp4_config({"weight_scale_encoding": "csf"}))
    with pytest.raises(ValueError, match="weight_scale_encoding"):
        _fp4_config({"weight_scale_encoding": "csf_64x1"})


def test_hook_mapper_decodes_checkpoint_stream():
    raw = _raw_scales((64, 4))
    streams = _encode(raw.view(torch.float8_e4m3fn))
    mapper = _mixed_config("csf").get_checkpoint_weight_mapper()

    out = dict(mapper.apply((f"e.weight_scale.{s}", t) for s, t in streams.items()))

    assert list(out) == ["e.weight_scale"]
    assert torch.equal(out["e.weight_scale"].view(torch.uint8), raw)
