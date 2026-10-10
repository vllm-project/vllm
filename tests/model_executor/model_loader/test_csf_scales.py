# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.model_loader.csf_scales import (
    decode_csf_scale_streams,
    decode_nvfp4_csf_scale,
)


def _encode(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference encoder: row base = row min (capped at 240), outliers as
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
    return fixed, words.to(torch.int32).view(torch.uint32)


@pytest.mark.parametrize("shape", [(640, 160), (2560, 40)])
def test_decode_matches_source_scales(shape):
    torch.manual_seed(0)
    raw = torch.randint(0x30, 0x40, shape, dtype=torch.uint8)
    raw.view(-1)[torch.randperm(raw.numel())[:64]] = 0x7E  # out-of-window bytes
    scales = raw.view(torch.float8_e4m3fn)
    fixed, exceptions = _encode(scales)
    assert exceptions.numel() >= 64

    decoded = decode_nvfp4_csf_scale(fixed, exceptions)

    assert decoded.dtype == torch.float8_e4m3fn
    assert torch.equal(decoded.view(torch.uint8), raw)


def test_stream_pairs_are_replaced_by_the_scale():
    raw = torch.full((16, 4), 0x38, dtype=torch.uint8)
    fixed, exceptions = _encode(raw.view(torch.float8_e4m3fn))
    weights = [
        ("e.weight", torch.zeros(1)),
        ("e.weight_scale.nvfp4_csf_exceptions", exceptions),
        ("e.weight_scale.nvfp4_csf_fixed", fixed),
    ]

    out = dict(decode_csf_scale_streams(weights))

    assert list(out) == ["e.weight", "e.weight_scale"]
    assert torch.equal(out["e.weight_scale"].view(torch.uint8), raw)


def test_unpaired_stream_is_rejected():
    fixed, _ = _encode(
        torch.full((16, 4), 0x38, dtype=torch.uint8).view(torch.float8_e4m3fn)
    )
    with pytest.raises(ValueError, match="without a partner"):
        list(decode_csf_scale_streams([("e.weight_scale.nvfp4_csf_fixed", fixed)]))


def test_hub_download_finds_shards_listed_by_the_index(tmp_path, monkeypatch):
    """A Hub snapshot whose shards sit under tensors/ is found once the index
    is fetched; only the shards are downloaded by the allow patterns."""
    import json

    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import default_loader

    shard = tmp_path / "tensors" / "model-00001-of-00001.safetensors"
    shard.parent.mkdir()
    shard.write_bytes(b"")
    index = {"weight_map": {"w": "tensors/model-00001-of-00001.safetensors"}}

    def fake_download_index(model, index_file, cache_dir, subfolder, revision):
        (tmp_path / index_file).write_text(json.dumps(index))

    monkeypatch.setattr(
        default_loader, "download_weights_from_hf", lambda *a, **k: str(tmp_path)
    )
    monkeypatch.setattr(
        default_loader,
        "download_safetensors_index_file_from_hf",
        fake_download_index,
    )
    loader = default_loader.DefaultModelLoader(LoadConfig(load_format="safetensors"))

    _, files, use_safetensors, _ = loader._prepare_weights(
        "org/model", None, None, fall_back_to_pt=False, allow_patterns_overrides=None
    )

    assert use_safetensors
    assert files == [str(shard)]
