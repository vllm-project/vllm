# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint's NVFP4 GGUF loader must repack llama.cpp NVFP4 blocks into the
ModelOpt layout without changing any weight value."""

import numpy as np
import pytest
import torch

from vllm.models.flint import gguf_nvfp4
from vllm.models.flint.gguf_nvfp4 import (
    _repack_nvfp4,
    dequantize_nvfp4,
    find_gguf_file,
    gguf_nvfp4_weights_iterator,
)

gguf = pytest.importorskip("gguf", minversion="0.19.0")

NVFP4 = gguf.GGMLQuantizationType.NVFP4


def _random_blocks(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    # Random bytes cover every E2M1 code and every representable scale byte,
    # including the sign bit and the 0x7F NaN pattern that ggml decodes as 0.
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (rows, cols // 64, 36), dtype=np.uint8)
    blocks[..., :4] = np.where(blocks[..., :4] == 0xFF, 0x7F, blocks[..., :4])
    return blocks.reshape(rows, -1)


def test_repack_matches_gguf_dequantize():
    rows, cols = 32, 512
    blocks = _random_blocks(rows, cols)
    weight, weight_scale = _repack_nvfp4(torch.from_numpy(blocks), cols)
    assert weight.shape == (rows, cols // 2)
    assert weight_scale.shape == (rows, cols // 16)
    ours = dequantize_nvfp4(weight, weight_scale)
    ref = gguf.quants.dequantize(blocks, NVFP4).reshape(rows, cols)
    torch.testing.assert_close(ours, torch.from_numpy(ref), atol=0, rtol=0)


def test_unrepresentable_scale_raises():
    blocks = _random_blocks(1, 64)
    blocks[0, 0] = 0xFF
    with pytest.raises(ValueError):
        _repack_nvfp4(torch.from_numpy(blocks), 64)


def test_iterator_maps_names_and_scales(tmp_path):
    hidden, inter, vocab, experts = 128, 64, 256, 4
    path = tmp_path / "tiny.gguf"
    writer = gguf.GGUFWriter(str(path), "flint")

    def add_nvfp4(name, rows, cols, num, seed):
        blocks = _random_blocks(num * rows, cols, seed).reshape(num, rows, -1)
        data = blocks[0] if num == 1 else blocks
        writer.add_tensor(name + ".weight", data, raw_dtype=NVFP4)
        scale = np.full((num,), 0.5, dtype=np.float32) * np.arange(1, num + 1)
        writer.add_tensor(name + ".scale", scale)
        return data

    q = add_nvfp4("blk.2.attn_q", hidden, hidden, 1, 1)
    add_nvfp4("blk.2.ffn_up_exps", inter, hidden, experts, 2)
    head = add_nvfp4("output", vocab, hidden, 1, 3)
    writer.add_tensor("blk.2.attn_sinks.weight", np.zeros(4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    tensors = dict(
        gguf_nvfp4_weights_iterator(str(path), "cpu", frozenset({"lm_head"}))
    )

    q_w = tensors["model.layers.2.self_attn.q_proj.weight"]
    q_s = tensors["model.layers.2.self_attn.q_proj.weight_scale"]
    assert q_w.dtype == torch.uint8 and q_s.dtype == torch.float8_e4m3fn
    ref = gguf.quants.dequantize(q, NVFP4).reshape(hidden, hidden) * 0.5
    torch.testing.assert_close(
        dequantize_nvfp4(
            q_w, q_s, tensors["model.layers.2.self_attn.q_proj.weight_scale_2"]
        ),
        torch.from_numpy(ref),
        atol=0,
        rtol=0,
    )
    # Stacked experts are split per expert with their own global scale.
    for e in range(experts):
        gs = tensors[f"model.layers.2.mlp.experts.{e}.up_proj.weight_scale_2"]
        assert float(gs) == 0.5 * (e + 1)
    # The LM head is dequantized exactly to bf16 (global scale kept apart).
    head_w = tensors["lm_head.weight"]
    assert head_w.dtype == torch.bfloat16
    head_ref = gguf.quants.dequantize(head, NVFP4).reshape(vocab, hidden)
    torch.testing.assert_close(
        head_w.float(), torch.from_numpy(head_ref), atol=0, rtol=0
    )
    assert float(tensors["lm_head.weight_scale_2"]) == 0.5
    assert "model.layers.2.self_attn.softmax_offset" in tensors


def test_find_gguf_file_resolves_hub_repo_ids(tmp_path, monkeypatch):
    """`vllm serve <org>/<repo>` must detect a GGUF-only repo, not just local paths."""
    (tmp_path / "flint.gguf").touch()
    monkeypatch.setattr(
        gguf_nvfp4,
        "any_pattern_in_repo_files",
        lambda repo, patterns, revision: repo == "org/gguf",
    )
    monkeypatch.setattr(
        gguf_nvfp4, "download_weights_from_hf", lambda *args: str(tmp_path)
    )
    assert find_gguf_file("org/gguf") == str(tmp_path / "flint.gguf")
    assert find_gguf_file("org/safetensors") is None
