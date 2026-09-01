# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1 tests for the mmap-backed PLE table (VLLM_PLE_MMAP).

No GPU, no real checkpoint: synthetic fp8 safetensors fixtures stand in for
the checkpoint's PLE shards, and the custom op is
exercised through its CPU dispatch key.
"""

from __future__ import annotations

import errno
import gc
import inspect
import json
import logging
import os
import warnings
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import safetensors.torch
import torch
from torch import nn

import vllm.envs as envs
import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp.nvidia.model as model_module
import vllm.models.qwen4_exp.nvidia.model_state as model_state_module
import vllm.models.qwen4_exp.nvidia.ple_mmap as ple_mmap
import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
from vllm.config import CompilationConfig, ParallelConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.fp8_utils import is_fp8
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.qwen4_exp.common.ple import (
    PLEVocabParallelEmbedding,
    copy_ple_embedding_shard_,
)
from vllm.models.qwen4_exp.nvidia import ple_layer as ple_layer_module
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)
from vllm.v1.worker.gpu import cudagraph_utils as cudagraph_utils_module
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_ple_mmap_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from a clean, default-off environment."""
    for name in (
        "VLLM_PLE_MMAP",
        "VLLM_PLE_MMAP_WORKERS",
        "VLLM_PLE_MMAP_CHUNK",
        "VLLM_PLE_MMAP_PREWARM",
        "VLLM_PLE_MMAP_READAHEAD",
        "VLLM_PLE_MMAP_PINNED",
        "VLLM_PLE_MMAP_SERIAL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _allow_single_rank_tensor_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stock VocabParallelEmbedding needs a TP group; stand in a rank-0/size-1
    world without paying for real torch.distributed init (mirrors test_ple.py).
    """
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )


def _make_text_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = dict(
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=0,
        vocab_size=200,
        split_ngram_parts=4,
        ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _synthetic_weight(
    vocab: int,
    cols: int,
    layer_idx: int = 0,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Deterministic, layer-dependent values (never all-zero/uniform, and
    distinguishable across layers so per-layer-keying tests are meaningful).
    """
    raw = torch.arange(vocab * cols, dtype=torch.float32).reshape(vocab, cols)
    raw = torch.remainder(raw + layer_idx * 97, 6.0) - 3.0
    return raw.to(dtype)


def _write_ple_layer(
    directory: Path,
    *,
    layer_idx: int,
    vocab: int,
    parts: int,
    cols: int,
    scale: float,
    write_scale: bool = True,
    scale_dtype: torch.dtype = torch.bfloat16,
    table_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Write one PLE layer's shard + weight_scale tensors as synthetic
    safetensors files (no model.safetensors.index.json, matching the real
    checkpoint). Returns the full logical [vocab, cols] table in table_dtype.
    """
    prefix = (
        f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding"
    )
    shard_size = (vocab + parts - 1) // parts
    full = _synthetic_weight(vocab, cols, layer_idx, dtype=table_dtype)
    for shard_index in range(parts):
        start = shard_index * shard_size
        rows = max(0, min(shard_size, vocab - start))
        tensors: dict[str, torch.Tensor] = {}
        if rows > 0:
            tensors[f"{prefix}.shard_{shard_index}.weight"] = full[start : start + rows]
        if write_scale and shard_index == 0:
            tensors[f"{prefix}.weight_scale"] = torch.tensor([scale], dtype=scale_dtype)
        if tensors:
            safetensors.torch.save_file(
                tensors,
                str(directory / f"model-ple-{layer_idx}-{shard_index:05d}.safetensors"),
            )
    return full


def _write_safetensors_index(directory: Path, weight_map: dict[str, str]) -> None:
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )


def _attached_embedding(
    directory: Path, layer_idx: int, vocab: int, parts: int, cols: int, scale: float
) -> ple_mmap.MmapNgramEmbedding:
    """Build a placeholder wired to an on-disk checkpoint via the same path
    build_tables uses, for tests that don't need the full static_forward_context
    walk.
    """
    shard_map = ple_mmap.discover_shards(str(directory))
    embedding = ple_mmap.MmapNgramEmbedding(vocab, cols)
    ple_mmap.set_weight_scale(
        embedding, torch.tensor([scale], dtype=torch.bfloat16), torch.device("cpu")
    )
    ple_mmap._attach_table(
        embedding,
        shard_map[layer_idx],
        split_ngram_parts=parts,
        layer_idx=layer_idx,
        model_path=str(directory),
    )
    return embedding


# --------------------------------------------------------------------------- #
# safetensors header parsing
# --------------------------------------------------------------------------- #


def test_parse_safetensors_header_returns_metadata_and_data_start(
    tmp_path: Path,
) -> None:
    path = tmp_path / "x.safetensors"
    safetensors.torch.save_file({"a": torch.zeros(3, 2)}, str(path))

    header, data_start = ple_mmap.parse_safetensors_header(str(path))

    assert header["a"]["shape"] == [3, 2]
    assert data_start == path.stat().st_size - (3 * 2 * 4)  # F32 = 4 bytes/elem


def test_parse_safetensors_header_rejects_oversized_header(tmp_path: Path) -> None:
    path = tmp_path / "big_header.safetensors"
    with open(path, "wb") as f:
        f.write((ple_mmap._MAX_HEADER_BYTES + 1).to_bytes(8, "little"))
        f.write(b"\x00" * 16)

    with pytest.raises(ValueError, match="exceeding the"):
        ple_mmap.parse_safetensors_header(str(path))


def test_parse_safetensors_header_rejects_offsets_outside_file(tmp_path: Path) -> None:
    import json
    import struct

    header = {"a": {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 1000]}}
    body = json.dumps(header).encode()
    path = tmp_path / "bad_offsets.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        f.write(b"\x00" * 16)  # far short of the declared 1000-byte tensor

    with pytest.raises(ValueError, match="fall outside the file"):
        ple_mmap.parse_safetensors_header(str(path))


def test_parse_safetensors_header_rejects_truncated_length(tmp_path: Path) -> None:
    path = tmp_path / "truncated.safetensors"
    path.write_bytes(b"\x01\x02\x03")

    with pytest.raises(ValueError, match="truncated safetensors header length"):
        ple_mmap.parse_safetensors_header(str(path))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_read_scale_round_trips_across_dtypes(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    value = -1.5  # exactly representable in fp32/fp16/bf16
    path = tmp_path / "scale.safetensors"
    safetensors.torch.save_file(
        {"scale": torch.tensor([value], dtype=dtype)}, str(path)
    )
    header, data_start = ple_mmap.parse_safetensors_header(str(path))
    start, end = header["scale"]["data_offsets"]
    entry = (str(path), data_start + start, end - start, header["scale"]["dtype"])

    got = ple_mmap._read_scale(entry)

    assert got.item() == pytest.approx(value)


# --------------------------------------------------------------------------- #
# Shard discovery
# --------------------------------------------------------------------------- #


def test_discover_shards_finds_layer_shards_and_scale(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=1, vocab=37, parts=5, cols=4, scale=0.5)

    result = ple_mmap.discover_shards(str(tmp_path))

    assert set(result.keys()) == {1}
    layer = result[1]
    assert layer.cols == 4
    assert layer.dtype_str == "F8_E4M3"
    assert layer.scale_entry is not None
    # shard_size = ceil(37/5) = 8; last shard truncated to 5 rows.
    assert set(layer.shards.keys()) == {0, 1, 2, 3, 4}
    assert layer.shards[4][2] == 5


def test_discover_shards_separates_multiple_ple_layers(tmp_path: Path) -> None:
    """(b): two-PLE-layer synthetic case proving per-layer keying — discovery
    must not mix shard tensors across layers even when files share a
    directory."""
    full0 = _write_ple_layer(
        tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25
    )
    full1 = _write_ple_layer(
        tmp_path, layer_idx=1, vocab=20, parts=4, cols=2, scale=0.75
    )

    result = ple_mmap.discover_shards(str(tmp_path))

    assert set(result.keys()) == {0, 1}
    assert {p for p, _o, _r in result[0].shards.values()}.isdisjoint(
        {p for p, _o, _r in result[1].shards.values()}
    )
    assert not full0.equal(full1[:10])  # fixtures are genuinely distinct


def test_discover_shards_honors_the_safetensors_index(tmp_path: Path) -> None:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=4, parts=1, cols=2, scale=0.5)
    indexed_name = "model-ple-0-00000.safetensors"
    excluded = torch.ones((4, 2), dtype=torch.float32).to(torch.float8_e4m3fn)
    safetensors.torch.save_file(
        {
            f"{prefix}.shard_0.weight": excluded,
            f"{prefix}.weight_scale": torch.tensor([9.0], dtype=torch.bfloat16),
        },
        str(tmp_path / "model.safetensors"),
    )
    _write_safetensors_index(
        tmp_path,
        {
            f"{prefix}.shard_0.weight": indexed_name,
            f"{prefix}.weight_scale": indexed_name,
        },
    )

    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    assert {
        Path(path).name for path, _offset, _rows in layer_shards.shards.values()
    } == {indexed_name}

    embedding = ple_mmap.MmapNgramEmbedding(4, 2)
    ple_mmap._attach_table(
        embedding,
        layer_shards,
        split_ngram_parts=1,
        layer_idx=0,
        model_path=str(tmp_path),
    )
    actual = embedding(torch.arange(4, dtype=torch.long))
    assert torch.equal(actual, full)
    assert embedding.table is not None
    embedding.table.close()


def test_discover_shards_refreshes_when_the_index_changes(tmp_path: Path) -> None:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    shard_name = f"{prefix}.shard_0.weight"
    scale_name = f"{prefix}.weight_scale"
    scale = torch.tensor([0.5], dtype=torch.bfloat16)
    rows_a = torch.zeros((4, 2), dtype=torch.float32).to(torch.float8_e4m3fn)
    rows_b = torch.ones((4, 2), dtype=torch.float32).to(torch.float8_e4m3fn)
    for filename, rows in (
        ("a.safetensors", rows_a),
        ("b.safetensors", rows_b),
    ):
        safetensors.torch.save_file(
            {shard_name: rows, scale_name: scale},
            str(tmp_path / filename),
        )

    _write_safetensors_index(
        tmp_path, {shard_name: "a.safetensors", scale_name: "a.safetensors"}
    )
    first = ple_mmap.discover_shards(str(tmp_path))[0]
    assert Path(first.shards[0][0]).name == "a.safetensors"

    _write_safetensors_index(
        tmp_path, {shard_name: "b.safetensors", scale_name: "b.safetensors"}
    )
    second = ple_mmap.discover_shards(str(tmp_path))[0]
    assert Path(second.shards[0][0]).name == "b.safetensors"

    embedding = ple_mmap.MmapNgramEmbedding(4, 2)
    ple_mmap._attach_table(
        embedding,
        second,
        split_ngram_parts=1,
        layer_idx=0,
        model_path=str(tmp_path),
    )
    assert torch.equal(embedding(torch.arange(4, dtype=torch.long)), rows_b)
    assert embedding.table is not None
    embedding.table.close()


def test_discover_shards_rejects_duplicate_logical_shards(tmp_path: Path) -> None:
    name = (
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    )
    tensor = torch.zeros((4, 2), dtype=torch.float32).to(torch.float8_e4m3fn)
    for filename in ("a.safetensors", "b.safetensors"):
        safetensors.torch.save_file({name: tensor}, str(tmp_path / filename))
    _write_safetensors_index(
        tmp_path, {"include.a": "a.safetensors", "include.b": "b.safetensors"}
    )

    with pytest.raises(ValueError, match="duplicate shard 0"):
        ple_mmap.discover_shards(str(tmp_path))


def test_discover_shards_rejects_duplicate_logical_scales(tmp_path: Path) -> None:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    shard = torch.zeros((4, 2), dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], dtype=torch.bfloat16)
    safetensors.torch.save_file(
        {
            f"{prefix}.shard_0.weight": shard,
            f"{prefix}.weight_scale": scale,
        },
        str(tmp_path / "a.safetensors"),
    )
    safetensors.torch.save_file(
        {f"{prefix}.weight_scale": scale},
        str(tmp_path / "b.safetensors"),
    )
    _write_safetensors_index(
        tmp_path, {"include.a": "a.safetensors", "include.b": "b.safetensors"}
    )

    with pytest.raises(ValueError, match="duplicate weight_scale"):
        ple_mmap.discover_shards(str(tmp_path))


def test_discover_shards_rejects_mixed_dtype_within_a_layer(tmp_path: Path) -> None:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    safetensors.torch.save_file(
        {f"{prefix}.shard_0.weight": torch.zeros(2, 2).to(torch.float8_e4m3fn)},
        str(tmp_path / "a.safetensors"),
    )
    safetensors.torch.save_file(
        {f"{prefix}.shard_1.weight": torch.zeros(2, 2).to(torch.float8_e5m2)},
        str(tmp_path / "b.safetensors"),
    )

    with pytest.raises(ValueError, match="mixed shard dtypes"):
        ple_mmap.discover_shards(str(tmp_path))


def test_discover_shards_rejects_mixed_width_within_a_layer(tmp_path: Path) -> None:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    safetensors.torch.save_file(
        {f"{prefix}.shard_0.weight": torch.zeros(2, 4).to(torch.float8_e4m3fn)},
        str(tmp_path / "a.safetensors"),
    )
    safetensors.torch.save_file(
        {f"{prefix}.shard_1.weight": torch.zeros(2, 8).to(torch.float8_e4m3fn)},
        str(tmp_path / "b.safetensors"),
    )

    with pytest.raises(ValueError, match="mixed shard widths"):
        ple_mmap.discover_shards(str(tmp_path))


def test_discover_shards_rejects_a_header_whose_span_disagrees_with_its_shape(
    tmp_path: Path,
) -> None:
    """a header entry whose data_offsets span doesn't match
    rows * cols * itemsize must be refused with a named error, rather than
    silently under-reading a truncated row."""
    import json
    import struct

    name = (
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    )
    # shape [4, 4] of F8_E4M3 (itemsize 1) needs a 16-byte span; the header
    # declares only 12.
    header = {name: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 12]}}
    body = json.dumps(header).encode()
    path = tmp_path / "bad_span.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        f.write(b"\x00" * 12)  # exactly the declared (too-small) span

    with pytest.raises(ValueError, match="does not match"):
        ple_mmap.discover_shards(str(tmp_path))


def test_discover_shards_rejects_a_header_with_an_unrecognized_dtype(
    tmp_path: Path,
) -> None:
    """_itemsize's None-guard: a dtype string absent from safetensors'
    _TYPES table must raise a named ValueError, not a bare KeyError."""
    import json
    import struct

    name = (
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    )
    header = {
        name: {"dtype": "NOT_A_REAL_DTYPE", "shape": [4, 4], "data_offsets": [0, 16]}
    }
    body = json.dumps(header).encode()
    path = tmp_path / "bad_dtype.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        f.write(b"\x00" * 16)

    with pytest.raises(ValueError, match="unrecognized safetensors dtype"):
        ple_mmap.discover_shards(str(tmp_path))


# --------------------------------------------------------------------------- #
# (d) shard-mapping contract: quotes Qwen4ExpNGramEmbedding.load_weights's
# shard-placement math verbatim.
# --------------------------------------------------------------------------- #


def _upstream_expected_rows(
    embedding: SimpleNamespace, split_ngram_parts: int, shard_index: int
) -> int:
    # Verbatim from Qwen4ExpNGramEmbedding.load_weights, including the
    # outer max(0, ...) clamp. A paraphrase dropping max(0, ...) encodes a
    # DIFFERENT function exactly at the boundary indices this test targets.
    shard_size = (embedding.org_vocab_size + split_ngram_parts - 1) // split_ngram_parts
    checkpoint_start = shard_index * shard_size
    expected_rows = max(
        0,
        min(shard_size, embedding.org_vocab_size - checkpoint_start),
    )
    return expected_rows


@pytest.mark.parametrize(
    ("org_vocab_size", "split_ngram_parts"),
    [
        (37, 5),  # last shard partially truncated (nonzero, < shard_size)
        (10, 8),  # trailing shards fully out of range (rows == 0)
    ],
)
def test_shard_mapping_matches_upstream_checkpoint_math_at_boundaries(
    tmp_path: Path, org_vocab_size: int, split_ngram_parts: int
) -> None:
    # org_vocab_size == padded_vocab_size here: VocabParallelEmbedding is
    # constructed positionally with no org_num_embeddings.
    embedding = SimpleNamespace(org_vocab_size=org_vocab_size, embedding_dim=4)
    shard_size = (org_vocab_size + split_ngram_parts - 1) // split_ngram_parts
    cols = 4
    full = _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=org_vocab_size,
        parts=split_ngram_parts,
        cols=cols,
        scale=1.0,
    )
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        shard_size,
        cols,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(tmp_path),
    )

    for shard_index in (0, 1, split_ngram_parts - 2, split_ngram_parts - 1):
        expected_rows = _upstream_expected_rows(
            embedding, split_ngram_parts, shard_index
        )
        if expected_rows == 0:
            continue
        checkpoint_start = shard_index * shard_size
        # Drive the actual boundary rows (first and last of this shard)
        # through the REAL gather path against the logical table, rather
        # than re-implementing the // and - shard/local math by hand.
        boundary_ids = np.array(
            [checkpoint_start, checkpoint_start + expected_rows - 1], dtype=np.int64
        )
        got = torch.from_numpy(table.gather(boundary_ids)).view(torch.float8_e4m3fn)
        assert torch.equal(got, full[boundary_ids])


# --------------------------------------------------------------------------- #
# MmapPleTable gather
# --------------------------------------------------------------------------- #


def test_mmap_table_gather_matches_naive_lookup(tmp_path: Path) -> None:
    full = _write_ple_layer(tmp_path, layer_idx=1, vocab=37, parts=5, cols=4, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[1]
    shard_size = (37 + 5 - 1) // 5
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        shard_size,
        4,
        torch.float8_e4m3fn,
        workers=2,
        chunk=3,
        model_path=str(tmp_path),
    )

    ids = np.array([0, 36, 5, 5, 20, 1, 31], dtype=np.int64)
    got = torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn)

    assert torch.equal(got, full[ids])


def test_mmap_table_gather_dedupes_and_preserves_input_order(tmp_path: Path) -> None:
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=1.0)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        3,
        2,
        torch.float8_e4m3fn,
        workers=1,
        chunk=1,
        model_path=str(tmp_path),
    )

    ids = np.array([4, 4, 0, 8, 4], dtype=np.int64)
    got = torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn)

    assert torch.equal(got, full[ids])
    assert torch.equal(got[0], got[1])  # the duplicate resolves to the same row


def test_mmap_table_gather_rejects_out_of_range_ids(tmp_path: Path) -> None:
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=1.0)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        3,
        2,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(tmp_path),
    )
    assert table.rows_total == 9

    with pytest.raises(IndexError, match=r"row id out of range"):
        table.gather(np.array([9_999], dtype=np.int64))

    # Exact boundary: rows_total itself is one past the last valid row.
    with pytest.raises(IndexError, match=r"row id out of range"):
        table.gather(np.array([table.rows_total], dtype=np.int64))

    # rows_total - 1 is the last valid row and must succeed.
    got = torch.from_numpy(
        table.gather(np.array([table.rows_total - 1], dtype=np.int64))
    ).view(torch.float8_e4m3fn)
    assert torch.equal(got, full[table.rows_total - 1 : table.rows_total])


def test_mmap_table_gather_empty_input_returns_empty(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=1.0)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        3,
        2,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(tmp_path),
    )

    out = table.gather(np.empty(0, dtype=np.int64))

    assert out.shape == (0, 2)


# --------------------------------------------------------------------------- #
# Serial small-gather dispatch (VLLM_PLE_MMAP_SERIAL)
# --------------------------------------------------------------------------- #


def test_serial_gather_matches_pool_gather_for_the_same_ids(tmp_path: Path) -> None:
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    ids = np.array([0, 39, 12, 13, 14, 5, 5, 20, 31], dtype=np.int64)

    gathered = []
    for serial in (0, 64):  # off (pool) vs on (inline, uniq.size <= 64)
        table = ple_mmap.MmapPleTable(
            layer_shards.shards,
            10,
            8,
            torch.float8_e4m3fn,
            workers=4,
            chunk=2,
            model_path=str(tmp_path),
            serial=serial,
        )
        gathered.append(torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn))
        table.close()

    assert torch.equal(gathered[0], full[ids])
    assert torch.equal(gathered[1], gathered[0])


def _count_pool_dispatches(
    monkeypatch: pytest.MonkeyPatch, table: ple_mmap.MmapPleTable
) -> list[int]:
    """Count table.pool.map calls without disturbing what it returns.

    One sentinel appended per call, so len(...) reads as the dispatch count.
    """
    calls: list[int] = []
    real_map = table.pool.map

    def _counting_map(fn: object, tasks: object) -> object:
        calls.append(1)
        return real_map(fn, tasks)

    monkeypatch.setattr(table.pool, "map", _counting_map)
    return calls


def test_serial_threshold_boundary_switches_inline_vs_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """uniq.size == N stays inline (pool.map never called); uniq.size ==
    N + 1 crosses the threshold back onto the pool. Keyed on uniq.size
    (distinct rows), not task count -- the boundary here spans a gather
    whose task count also happens to differ, which is exactly the
    distinction the knob is keyed to catch."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        10,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=1,
        model_path=str(tmp_path),
        serial=2,
    )
    calls = _count_pool_dispatches(monkeypatch, table)

    table.gather(np.array([0, 5], dtype=np.int64))  # uniq.size == 2 == N
    assert len(calls) == 0

    table.gather(np.array([0, 5, 12], dtype=np.int64))  # uniq.size == 3 > N
    assert len(calls) == 1

    table.close()


def test_serial_threshold_keys_on_distinct_rows_not_task_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large chunk coalesces each shard's span into a single task, so a
    16-distinct-row gather spanning exactly 2 shards produces only 2 tasks
    -- fewer than the serial=6 threshold. The previous boundary test used
    chunk=1, where len(tasks) == uniq.size, so it could not tell a
    uniq.size-keyed gate from a len(tasks)-keyed one; this one can: with
    uniq.size=16 > 6 the gate must still route to the pool even though
    len(tasks)=2 <= 6."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=20, parts=2, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        10,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=2048,
        model_path=str(tmp_path),
        serial=6,
    )
    calls = _count_pool_dispatches(monkeypatch, table)

    # 8 rows from shard 0 ([0, 9]) + 8 rows from shard 1 ([10, 19]):
    # uniq.size == 16, but chunk=2048 coalesces each shard's span into one
    # task each, so len(tasks) == 2.
    ids = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int64
    )
    table.gather(ids)

    assert len(calls) == 1  # pooled: uniq.size (16) > serial (6)

    table.close()


def test_serial_branch_raises_the_same_named_indexerror_on_a_closed_table(
    tmp_path: Path,
) -> None:
    """The serial dispatch loop reuses run() verbatim, so a missing shard
    raises the identical named IndexError regardless of branch -- exercised
    here with more than one task, which the existing len(tasks) == 1
    special case never reaches. The contract under test is "a missing shard
    slot" (run()'s `mm is None` check); this test simulates that cheaply by
    closing the table first, which nulls every mm slot, rather than
    constructing a table with a genuinely missing shard. That's also why
    this test only exercises the serial branch: gathering through the pool
    branch on a CLOSED table instead hits a pre-existing, unrelated
    divergence -- pool.map raises RuntimeError("cannot schedule new futures
    after shutdown") straight from the shut-down executor, before run()
    ever gets a chance to raise its own IndexError. That divergence
    predates SERIAL and is not this knob's contract."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        10,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=1,
        model_path=str(tmp_path),
        serial=64,
    )
    table.close()

    with pytest.raises(IndexError, match="shard"):
        table.gather(np.array([0, 5], dtype=np.int64))


def test_serial_composes_with_readahead_and_gathers_correctly(
    tmp_path: Path,
) -> None:
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        10,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=2,
        model_path=str(tmp_path),
        readahead=64,
        serial=64,
    )
    ids = np.array([0, 39, 12, 13, 14, 5, 5, 20, 31], dtype=np.int64)

    got = torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn)

    assert torch.equal(got, full[ids])
    assert table._latencies_ms[-1][3] > 0  # the readahead pre-pass ran too
    # Prove serial itself engaged too, not just readahead: uniq.size == 8
    # (see the ids above) is <= serial=64.
    assert table._serial_engaged_since_log == 1
    assert len(table._latencies_ms) == 1
    table.close()


def test_serial_field_in_the_gather_log_line_reflects_the_engaged_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_record's rate-limited log line gains an appended serial= field
    (append-only -- rows=/p99_ms=/populate_ms=/copy_ms=/runs=/pending=/
    errors= keep their names, order, and meaning) reporting engaged/total
    gathers in the window, not the p99 SAMPLE's own engaged flag: keying on
    the p99 sample -- by construction the window's biggest gather -- would
    report the wrong branch whenever the biggest gather in a window isn't
    representative of how most calls in it were actually dispatched (see
    the mixed-window test below for the failure this avoids)."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        10,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=2,
        model_path=str(tmp_path),
        serial=2,
    )
    logged = _record_info(monkeypatch)

    table._last_log = 0.0  # simulate the interval having elapsed
    table.gather(np.array([0, 5], dtype=np.int64))  # uniq.size == 2 <= serial
    assert len(logged) == 1
    msg, args = logged[0]
    assert "serial=" in msg
    assert args[-2:] == (1, 1)  # this window's one gather engaged serial

    logged.clear()
    table._last_log = 0.0
    table.gather(np.array([0, 5, 12], dtype=np.int64))  # uniq.size == 3 > serial
    assert len(logged) == 1
    assert logged[0][1][-2:] == (0, 1)  # this window's one gather did not

    table.close()


def test_mixed_window_serial_field_reports_engaged_over_total_gathers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape the serial= field's window-keying exists for: a window
    mixing many small (serial-engaged) gathers with one large pooled
    gather. p99 is, by construction, the window's slowest call -- for a
    mixed-size window that's the large pooled gather, not a representative
    sample of the window's actual engagement. Keying serial= on that one
    sample's flag reports serial=0 for the 19-of-20-engaged window driven
    below; this test asserts the field instead reports the window's true
    engaged/total gather counts, independent of which single call happened
    to be slowest."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=600, parts=4, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        150,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=16,
        model_path=str(tmp_path),
        serial=5,
    )
    logged = _record_info(monkeypatch)

    small_ids = np.array([0, 1], dtype=np.int64)  # uniq.size == 2 <= serial
    for _ in range(19):
        table.gather(small_ids)
    large_ids = np.arange(500, dtype=np.int64)  # uniq.size == 500 > serial
    table._last_log = 0.0  # simulate the interval having elapsed on this call
    table.gather(large_ids)  # pooled, the window's slowest call by far

    assert len(logged) == 1
    msg, args = logged[0]
    assert "serial=" in msg
    assert args[-2:] == (19, 20)  # 19 of this window's 20 gathers engaged

    table.close()


# --------------------------------------------------------------------------- #
# Readahead pre-pass (VLLM_PLE_MMAP_READAHEAD)
# --------------------------------------------------------------------------- #


def _readahead_table(directory: Path, readahead: int) -> ple_mmap.MmapPleTable:
    """A 40-row / 4-shard table over the fixture written by
    _write_ple_layer(vocab=40, parts=4, cols=8): shard_size 10, so ids
    0/5/12/13/14/20/31/39 land in four segments and coalesce to six runs.
    """
    layer_shards = ple_mmap.discover_shards(str(directory))[0]
    return ple_mmap.MmapPleTable(
        layer_shards.shards,
        10,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=2,
        model_path=str(directory),
        readahead=readahead,
    )


def _small_readahead_table(
    directory: Path, readahead: int = 64
) -> ple_mmap.MmapPleTable:
    """A 9-row / 3-shard table over the fixture written by
    _write_ple_layer(vocab=9, parts=3, cols=2): small enough to assert on
    exact fd/mm counts.
    """
    layer_shards = ple_mmap.discover_shards(str(directory))[0]
    return ple_mmap.MmapPleTable(
        layer_shards.shards,
        3,
        2,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(directory),
        readahead=readahead,
    )


def _record_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple[object, ...]]]:
    """Capture logger.warning_once calls, keeping the (msg, args) dedup key
    the real logger caches on.
    """
    recorded: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        ple_mmap.logger,
        "warning_once",
        lambda msg, *args: recorded.append((msg, args)),
    )
    return recorded


def _record_plain_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple[object, ...]]]:
    """Capture plain logger.warning calls (not warning_once's dedup path).

    **kwargs, not just *args: some call sites (e.g. _open_readahead_fds'
    own per-file warning_once) route through this same logger.warning with
    a stacklevel= kwarg attached.
    """
    recorded: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        ple_mmap.logger,
        "warning",
        lambda msg, *args, **kwargs: recorded.append((msg, args)),
    )
    return recorded


def _record_info(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple[Any, ...]]]:
    """Capture plain logger.info calls, the rate-limited-log-line pattern
    shared by MmapPleTable._record and
    MmapNgramEmbedding._record_input_prep_timing.

    Args are typed Any, not object like the warning recorders above: callers
    unpack these %-format args and do arithmetic on the numeric ones.
    """
    recorded: list[tuple[str, tuple[Any, ...]]] = []
    monkeypatch.setattr(
        ple_mmap.logger, "info", lambda msg, *args: recorded.append((msg, args))
    )
    return recorded


def _single_file_ple_checkpoint(
    directory: Path, cols: int
) -> tuple[torch.Tensor, Path]:
    """Write one 9-row/3-shard PLE layer with every shard packed into a
    SINGLE safetensors file — unlike _write_ple_layer, which always writes
    one file per shard. Shards commonly share a file in a real checkpoint,
    and this is the only fixture shape whose byte offsets land non-zero
    and non-page-aligned within one file.
    """
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    full = _synthetic_weight(9, cols)
    path = directory / "model-ple-0-00000.safetensors"
    safetensors.torch.save_file(
        {
            f"{prefix}.shard_0.weight": full[0:3],
            f"{prefix}.shard_1.weight": full[3:6],
            f"{prefix}.shard_2.weight": full[6:9],
            f"{prefix}.weight_scale": torch.tensor([1.0], dtype=torch.bfloat16),
        },
        str(path),
    )
    return full, path


def test_coalesce_runs_merges_only_abutting_spans() -> None:
    # Rows 10/11/12 abut and become one run; row 40 stays its own.
    offsets = np.array([1280, 1408, 1536, 5120], dtype=np.int64)

    assert ple_mmap._coalesce_runs(offsets, 128) == [(1280, 384), (5120, 128)]


def test_coalesce_runs_keeps_a_gapped_pair_as_two_runs() -> None:
    """gap == 0 is the only merge rule. On a hash-scattered table, bridging
    even a one-row gap fetches pages no row in the gather needs, which is the
    exact I/O amplification the pre-pass exists to avoid.
    """
    offsets = np.array([0, 256], dtype=np.int64)

    assert ple_mmap._coalesce_runs(offsets, 128) == [(0, 128), (256, 128)]


def test_coalesce_runs_returns_no_runs_for_no_rows() -> None:
    assert ple_mmap._coalesce_runs(np.empty(0, dtype=np.int64), 128) == []


def test_gather_returns_identical_rows_with_readahead_on_and_off(
    tmp_path: Path,
) -> None:
    """The pre-pass only hints the page cache, so it must not perturb a
    single gathered byte — and the run count must reach _record either way,
    since an on/off A/B reads it straight off the log line.
    """
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    ids = np.array([0, 39, 12, 13, 14, 5, 5, 20, 31], dtype=np.int64)

    gathered = []
    for readahead in (0, 64):
        table = _readahead_table(tmp_path, readahead)
        gathered.append(torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn))
        runs = table._latencies_ms[-1][3]
        assert runs > 0 if readahead else runs == 0
        table.close()

    assert torch.equal(gathered[0], full[ids])
    assert torch.equal(gathered[1], gathered[0])


def test_readahead_bound_skips_the_pre_pass_but_still_reports_the_run_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent skip is indistinguishable from the feature being off, which
    would void a tester's on/off comparison: the observed run count still
    reaches _record, and the skip names both it and the knob.
    """
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    table = _readahead_table(tmp_path, readahead=2)
    advised: list[tuple[object, ...]] = []
    monkeypatch.setattr(os, "posix_fadvise", lambda *args: advised.append(args))
    # The bound-exceeded warning is a per-table latch (logger.warning), not
    # warning_once, so _record_plain_warnings (not _record_warnings) is the
    # right capture helper here.
    warnings = _record_plain_warnings(monkeypatch)

    ids = np.array([0, 5, 12, 20, 31, 39], dtype=np.int64)
    got = torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn)

    assert torch.equal(got, full[ids])
    assert advised == []
    assert table._latencies_ms[-1][3] == 6
    assert len(warnings) == 1
    assert warnings[0][1] == (6, 2)
    table.close()


def test_bound_exceeded_warns_exactly_once_per_table_across_varying_run_counts(
    tmp_path: Path, caplog_vllm: pytest.LogCaptureFixture
) -> None:
    """warning_once dedups on (msg, *args), and the observed run count is
    part of args and varies almost every gather — that would defeat the
    process-wide lru cache and re-log on nearly every over-bound call. The
    per-table latch (a plain instance flag) must warn exactly once
    regardless of how many distinct run counts are observed, verified
    against a real logging.Handler rather than a monkeypatched recorder.
    """
    _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    table = _readahead_table(tmp_path, readahead=1)  # any gather here exceeds 1 run

    try:
        with caplog_vllm.at_level(
            logging.WARNING, logger="vllm.models.qwen4_exp.nvidia.ple_mmap"
        ):
            for ids in (
                np.array([0, 39], dtype=np.int64),  # 2 shards -> 2 runs
                np.array([0, 5, 39], dtype=np.int64),  # a different run count
                np.array([0, 5, 12, 39], dtype=np.int64),  # different again
            ):
                table.gather(ids)
    finally:
        table.close()

    bound_records = [
        r for r in caplog_vllm.records if "readahead skipped" in r.getMessage()
    ]
    assert len(bound_records) == 1


def test_readahead_bound_skip_avoids_materializing_the_run_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound-skipped gather must build no run list at all: _coalesce_runs
    is the seam where a segment's already-computed offsets turn into the
    (fd, offset, length) Python tuples the active arm advises from — one
    call per segment (see _readahead's single numpy pass). Counting calls
    to it, rather than timing the two arms, proves the count-only path
    stays numpy-only when the bound is exceeded, independent of
    box-to-box noise.
    """
    _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    ids = np.array([0, 5, 12, 20, 31, 39], dtype=np.int64)  # 4 segments, 6 runs

    calls = 0
    real_coalesce = ple_mmap._coalesce_runs

    def _counting_coalesce(
        offsets: np.ndarray, row_bytes: int
    ) -> list[tuple[int, int]]:
        nonlocal calls
        calls += 1
        return real_coalesce(offsets, row_bytes)

    monkeypatch.setattr(ple_mmap, "_coalesce_runs", _counting_coalesce)

    skipped = _readahead_table(tmp_path, readahead=1)  # 6 runs > 1: skip
    try:
        skipped.gather(ids)
    finally:
        skipped.close()
    assert calls == 0

    active = _readahead_table(tmp_path, readahead=64)  # 6 runs <= 64: materialize
    try:
        active.gather(ids)
    finally:
        active.close()
    assert calls == 4  # one _coalesce_runs call per segment


def test_readahead_survives_an_oserror_from_every_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """posix_fadvise failing costs the readahead and nothing else: every run
    is still attempted, the rows are still correct, and the failures share one
    warning_once key so the real logger emits a single line.
    """
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    table = _readahead_table(tmp_path, readahead=64)

    def _raise(fd: int, offset: int, length: int, advice: int) -> None:
        raise OSError(errno.EBADF, "Bad file descriptor")

    monkeypatch.setattr(os, "posix_fadvise", _raise)
    warnings = _record_warnings(monkeypatch)

    ids = np.array([0, 5, 12, 20, 31, 39], dtype=np.int64)
    got = torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn)

    assert torch.equal(got, full[ids])
    assert len(warnings) == 6
    assert len(set(warnings)) == 1
    table.close()


def test_gather_survives_a_pre_pass_that_raises_valueerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-pass never touches gathered data, so it is never the
    correctness path: a bug anywhere inside it must cost the readahead rather
    than fail the request.
    """
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    table = _readahead_table(tmp_path, readahead=64)

    def _raise(offsets: np.ndarray, row_bytes: int) -> list[tuple[int, int]]:
        raise ValueError("synthetic pre-pass bug")

    monkeypatch.setattr(ple_mmap, "_coalesce_runs", _raise)
    warnings = _record_warnings(monkeypatch)

    ids = np.array([0, 5, 12, 20, 31, 39], dtype=np.int64)
    got = torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn)

    assert torch.equal(got, full[ids])
    assert warnings[0][1] == ("ValueError",)
    table.close()


def test_readahead_advises_file_absolute_offsets_across_unaligned_shard_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-pass computes ``mm.offset + local * row_bytes`` — the FILE
    offset a shard's rows live at, not a row-relative one. Shards sharing
    one safetensors file (as here) start at non-zero, non-page-aligned byte
    offsets within it, so a dropped ``mm.offset +`` would advise the wrong
    bytes: reading each advised (offset, length) range straight off disk
    and comparing to the logical table catches that class of regression
    directly, independent of gather()'s own correctness.
    """
    full, path = _single_file_ple_checkpoint(tmp_path, cols=3)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    # Sanity: shard 1/2 must NOT start at a page boundary, else this test
    # would still pass even with `mm.offset +` dropped from the pre-pass.
    for shard_idx in (1, 2):
        _path_, offset, _rows = layer_shards.shards[shard_idx]
        assert offset % 4096 != 0

    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        3,
        3,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(tmp_path),
        readahead=64,
    )
    advised: list[tuple[int, int, int, int]] = []
    monkeypatch.setattr(
        os,
        "posix_fadvise",
        lambda fd, offset, length, advice: advised.append((fd, offset, length, advice)),
    )

    ids = np.array([0, 4, 8], dtype=np.int64)  # one row per shard
    table.gather(ids)
    table.close()

    assert advised
    with open(path, "rb") as f:
        pieces = []
        for _fd, offset, length, _advice in sorted(advised, key=lambda a: a[1]):
            f.seek(offset)
            pieces.append(f.read(length))
    got = np.frombuffer(b"".join(pieces), dtype=np.uint8).reshape(-1, 3)
    expected = full[np.unique(ids)].view(torch.uint8).numpy().reshape(-1, 3)
    assert np.array_equal(got, expected)


def test_readahead_holds_one_fd_per_distinct_shard_file(tmp_path: Path) -> None:
    """Shards routinely share a safetensors file, so keying descriptors on
    the path — not the shard slot — is what keeps a 128-shard layer from
    holding 128 of them. close() must hand every one back.
    """
    full, _path = _single_file_ple_checkpoint(tmp_path, cols=2)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        3,
        2,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(tmp_path),
        readahead=64,
    )

    assert len(table._fds) == 1
    assert len(set(table._shard_fds)) == 1
    ids = np.array([0, 4, 8], dtype=np.int64)
    got = torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn)
    assert torch.equal(got, full[ids])

    table.close()

    # Not os.fstat(fd) after close: the OS is free to reuse a released fd
    # number for something unrelated before this assertion runs, which
    # would make an "fstat still raises" check flaky rather than wrong.
    assert not table._fds
    assert all(shard_fd is None for shard_fd in table._shard_fds)


def test_readahead_defaults_to_off_and_opens_no_fds(tmp_path: Path) -> None:
    """Default-off is the whole shape of this knob: while decode perf is
    under dispute the pre-pass is an opt-in instrument, not a new cost every
    gather pays.
    """
    _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    table = ple_mmap.MmapPleTable(
        ple_mmap.discover_shards(str(tmp_path))[0].shards,
        10,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=2,
        model_path=str(tmp_path),
    )

    assert envs.VLLM_PLE_MMAP_READAHEAD == 0
    assert table.readahead == 0
    assert table._fds == {}


def test_readahead_table_close_is_idempotent_with_fds_open(tmp_path: Path) -> None:
    """close() must guard its fd release too: with readahead fds actually
    open (unlike the plain close-idempotency test, which has none to
    release), a second call must stay a no-op rather than double
    os.close()-ing an fd number the OS is free to reuse.
    """
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=1.0)
    table = _small_readahead_table(tmp_path)
    assert table._fds

    table.close()
    table.close()  # idempotent: must not raise (e.g. a double os.close())

    assert not table._fds
    assert all(mm is None for mm in table.mm)


def test_inert_readahead_warns_and_reports_itself_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_open_readahead_fds already warns per shard file it fails to open;
    when that leaves zero fds covered, table.readahead must say so too — a
    tester comparing an on/off A/B via the knob's own value would otherwise
    see it stay > 0 while the pre-pass never issues a single fadvise call.
    """
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=1.0)

    def _raise_open(*args: object, **kwargs: object) -> int:
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(os, "open", _raise_open)
    # _record_plain_warnings tolerates **kwargs: os.open failing also routes
    # _open_readahead_fds' own per-file warning_once through this same
    # logger.warning, and _print_warning_once calls it with stacklevel=... .
    warnings = _record_plain_warnings(monkeypatch)

    table = _small_readahead_table(tmp_path)
    try:
        assert table.readahead == 0
        inert_warnings = [w for w in warnings if "pre-pass is inert" in w[0]]
        assert len(inert_warnings) == 1
        assert inert_warnings[0][1] == (64,)
    finally:
        table.close()


def test_del_after_threadpool_failure_still_closes_readahead_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure constructing the gather ThreadPool must not leak the fds
    _open_readahead_fds already opened: the fd dict is populated before the
    pool is, so close() — reached via __del__ on the half-built table —
    still has real descriptors to hand back.
    """
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=1.0)

    def _raise_pool(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic ThreadPoolExecutor failure")

    monkeypatch.setattr(ple_mmap, "ThreadPoolExecutor", _raise_pool)
    closed_fds: list[int] = []
    real_close = os.close

    def _spying_close(fd: int) -> None:
        closed_fds.append(fd)
        real_close(fd)

    monkeypatch.setattr(os, "close", _spying_close)

    with pytest.raises(RuntimeError, match="synthetic ThreadPoolExecutor failure"):
        _small_readahead_table(tmp_path)
    gc.collect()

    assert closed_fds


def test_readahead_gathers_correctly_across_a_320_byte_bf16_row_width(
    tmp_path: Path,
) -> None:
    """The pre-pass's row-offset math (_readahead/_coalesce_runs) is
    parameterized on row_bytes, not hardcoded to fp8's 1-byte-per-column
    width — a BF16 table's 320-byte row (cols=160 x itemsize 2, the real
    checkpoint's shape) must coalesce and gather identically to the
    narrower fp8 fixtures exercised elsewhere in this file.
    """
    full = _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=6,
        parts=2,
        cols=160,
        scale=0.0,
        write_scale=False,
        table_dtype=torch.bfloat16,
    )
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        3,
        320,
        torch.bfloat16,
        workers=2,
        chunk=2,
        model_path=str(tmp_path),
        readahead=64,
    )

    ids = np.array([0, 5, 2, 2], dtype=np.int64)
    got = torch.from_numpy(table.gather(ids)).view(torch.bfloat16)

    assert torch.equal(got, full[ids])
    table.close()


# --------------------------------------------------------------------------- #
# Bounded prewarm
# --------------------------------------------------------------------------- #


def test_compute_prewarm_bound_caps_at_table_bytes() -> None:
    assert ple_mmap.compute_prewarm_bound(100, 200 * (1 << 30)) == 100


def test_compute_prewarm_bound_respects_headroom() -> None:
    table_bytes = 100 * (1 << 30)
    mem_available = 20 * (1 << 30)
    bound = ple_mmap.compute_prewarm_bound(table_bytes, mem_available)
    assert bound == mem_available - ple_mmap._PREWARM_HEADROOM_BYTES


def test_compute_prewarm_bound_clamps_negative_to_zero() -> None:
    """A negative bound would slice-read nearly the whole table exactly
    when memory is scarcest."""
    table_bytes = 100 * (1 << 30)
    mem_available = 4 * (1 << 30)  # below the 8 GiB headroom
    assert ple_mmap.compute_prewarm_bound(table_bytes, mem_available) == 0


def test_mem_available_bytes_parses_meminfo_format(tmp_path: Path) -> None:
    fixture = tmp_path / "meminfo"
    fixture.write_text(
        "MemTotal:       131000000 kB\n"
        "MemFree:         20000000 kB\n"
        "MemAvailable:   109051904 kB\n"
        "Cached:          15000000 kB\n"
    )

    assert ple_mmap._mem_available_bytes(str(fixture)) == 109051904 * 1024


def test_prewarm_reads_up_to_the_bound(tmp_path: Path) -> None:
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=1, cols=4, scale=1.0)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        9,
        4,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(tmp_path),
    )
    table_bytes = full.numel()

    assert table.prewarm(0) == 0
    assert table.prewarm(table_bytes // 2) <= table_bytes // 2
    assert table.prewarm(table_bytes * 10) <= table_bytes


# --------------------------------------------------------------------------- #
# Custom op removal: the mmap gather path is now explicit module-owned
# staging + gather_into, driven from prepare_mmap_rows, never a
# torch.ops.vllm custom op. Nothing under this env registers an op anymore.
# --------------------------------------------------------------------------- #


def test_mmap_gather_op_is_not_registered() -> None:
    assert not hasattr(ple_mmap, "OP_NAME")
    assert not hasattr(ple_mmap, "QUALIFIED_OP_NAME")
    assert not hasattr(torch.ops.vllm, "qwen4_exp_ple_mmap_forward")


def test_mmap_gather_op_is_absent_from_v1_splitting_ops() -> None:
    cc = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    cc.set_splitting_ops_for_v1(all2all_backend="naive", data_parallel_size=1)

    assert "vllm::qwen4_exp_ple_mmap_forward" not in cc.splitting_ops
    # The OTHER PLE custom op (non-mmap ID hashing, untouched by this PR)
    # must still be present -- proving the assertion above is actually
    # discriminating between the two ops, not just observing an empty list.
    assert "vllm::qwen4_exp_compute_ple_ngram_ids" in cc.splitting_ops


# --------------------------------------------------------------------------- #
# (c) Startup guard: mmap is Model Runner V2-only. check_cudagraph_safety
# reads only use_v2_model_runner -- the graph mode is deliberately
# irrelevant, since V2's staged design (prepare_mmap_rows runs before the
# compiled/captured forward) makes every cudagraph mode safe.
# --------------------------------------------------------------------------- #


def test_check_cudagraph_safety_accepts_v2_model_runner() -> None:
    vllm_config = SimpleNamespace(use_v2_model_runner=True)

    ple_mmap.check_cudagraph_safety(vllm_config)  # must not raise


def test_check_cudagraph_safety_refuses_v1() -> None:
    """V1 has no working PLE query/context preparation path after #53896
    removed its host-side hash-and-gather, so mmap has no V1 fallback and
    must be refused before weights load."""
    vllm_config = SimpleNamespace(use_v2_model_runner=False)

    with pytest.raises(RuntimeError, match="Model Runner V2"):
        ple_mmap.check_cudagraph_safety(vllm_config)


def test_check_cudagraph_safety_accepts_a_real_compilation_config_under_v2() -> None:
    """Ordering: a VllmConfig carrying a REAL CompilationConfig that has
    gone through its normal init path (set_splitting_ops_for_v1), not a
    hand-stubbed SimpleNamespace, must still pass under V2 -- proving the
    guard composes with a genuine object, not just a minimal stub."""
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.FULL
    )
    cc.set_splitting_ops_for_v1(all2all_backend="naive", data_parallel_size=1)
    vllm_config = SimpleNamespace(compilation_config=cc, use_v2_model_runner=True)

    ple_mmap.check_cudagraph_safety(vllm_config)  # must not raise


# --------------------------------------------------------------------------- #
# check_cudagraph_safety is unit-tested as a free function
# above, but its CALL from Qwen4ExpNGramEmbedding.__init__ (ple_layer.py)
# was never exercised — deleting that call left the whole suite green.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_ngram_embedding_construction_accepts_every_graph_mode_under_v2(
    monkeypatch: pytest.MonkeyPatch, cudagraph_mode: CUDAGraphMode
) -> None:
    """V2's staged design makes every cudagraph mode safe, including the
    FULL-containing ones -- construction must accept all of them under V2,
    not just PIECEWISE."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=cudagraph_mode
    )
    vllm_config = SimpleNamespace(
        compilation_config=cc,
        model_config=SimpleNamespace(),  # unresolvable path -> deferred to load time
        use_v2_model_runner=True,
    )

    with set_current_vllm_config(vllm_config):
        Qwen4ExpNGramEmbedding(
            config,
            8,
            0,
            16,
            4,
            "model.layers.1.ple.ple_embedding",
            "model.layers.1.ple",
            params_dtype=torch.float32,
        )  # must not raise


@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_ngram_embedding_construction_refuses_v1(
    monkeypatch: pytest.MonkeyPatch, cudagraph_mode: CUDAGraphMode
) -> None:
    """V1 has no working PLE query/context preparation path, and this must
    hold regardless of the requested cudagraph mode -- unlike the V2 guard
    (which legitimately depends on graph mode history), the V1 refusal is
    unconditional, so every mode must be refused, not just PIECEWISE."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=cudagraph_mode
    )
    vllm_config = SimpleNamespace(
        compilation_config=cc,
        model_config=SimpleNamespace(),  # unresolvable path -> deferred to load time
        use_v2_model_runner=False,
    )

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="Model Runner V2"),
    ):
        Qwen4ExpNGramEmbedding(
            config,
            8,
            0,
            16,
            4,
            "model.layers.1.ple.ple_embedding",
            "model.layers.1.ple",
            params_dtype=torch.float32,
        )


# --------------------------------------------------------------------------- #
# torch.compile dispatch: the mmap forward branch's token-count read must
# stay symbolic (no ConstraintViolationError across differing token counts).
# --------------------------------------------------------------------------- #


def test_ngram_embedding_forward_compiles_across_two_different_token_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mmap branch reads `input_ids.reshape(-1).shape[0]` (a plain
    shape read, never `.numel()`-derived or wrapped in `int()`) to slice the
    staging buffer. Calling the torch.compile'd forward with two different
    token counts must recompile at most once and never raise
    ConstraintViolationError -- the failure mode of the old whole-forward
    custom op this replaced, which specialized that dimension."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.PIECEWISE
    )
    vllm_config = SimpleNamespace(
        compilation_config=cc,
        model_config=SimpleNamespace(),
        use_v2_model_runner=True,
    )
    with set_current_vllm_config(vllm_config):
        layer = Qwen4ExpNGramEmbedding(
            config,
            8,
            0,
            16,
            4,
            "model.layers.2.ple.ple_embedding",
            "model.layers.2.ple",
            params_dtype=torch.float32,
        )
    layer.initialize_mmap_staging(16, torch.device("cpu"))
    layer._mmap_staging.copy_(
        torch.arange(16 * layer.ngram_heads * layer.head_dim, dtype=torch.float32)
        .reshape(16, layer.ngram_heads, layer.head_dim)
        .to(layer._mmap_staging.dtype)
    )

    compiled_forward = torch.compile(layer.forward)
    dummy_qsl = torch.zeros(2, dtype=torch.int32)
    dummy_ctx = torch.zeros((1, 2), dtype=torch.long)

    for num_tokens in (3, 5):
        ids = torch.zeros((1, num_tokens), dtype=torch.long)
        out = compiled_forward(ids, dummy_qsl, dummy_ctx)
        expected = layer._mmap_staging[:num_tokens].flatten(-2)
        assert out.shape == (num_tokens, layer.ngram_heads * layer.head_dim)
        torch.testing.assert_close(out, expected)


# --------------------------------------------------------------------------- #
# Mandatory end-to-end proof: torch.compile + FULL CUDA graph capture/replay,
# through the production Model Runner V2 capture primitive
# (`CudaGraphManager.capture()` -- the exact method
# `ModelCudaGraphManager.capture()` delegates to via
# `super().capture(create_forward_fn, ...)`), must read the mmap staging
# buffer's CURRENT contents through a stable address and through a REAL
# downstream CUDA kernel -- not a pure view/reshape the compiled graph can
# capture as empty -- for both the FP8 and BF16 checkpoint dtypes.
# --------------------------------------------------------------------------- #


class _PleDownstreamConsumer(nn.Module):
    """Forces the compiled graph to capture a real CUDA kernel (a GEMM)
    reading the PLE layer's output, rather than the pure view/reshape
    `Qwen4ExpNGramEmbedding.forward` returns on its own in mmap mode -- which
    needs no kernel at all and captures an EMPTY CUDA graph. Writes into a
    separate, stable, pre-allocated output buffer, so that reading the
    layer's own output (an alias of `_mmap_staging`) can never stand in for
    proof that a captured kernel actually ran: `self.out` is written ONLY by
    a kernel that executed, at capture or at replay.
    """

    def __init__(
        self,
        ple_layer: Qwen4ExpNGramEmbedding,
        weight: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        super().__init__()
        self.ple_layer = ple_layer
        self.weight = weight
        self.out = out

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.ple_layer(input_ids, query_start_loc, ngram_context)
        num_tokens = embedded.shape[0]
        self.out[:num_tokens] = embedded.to(self.weight.dtype) @ self.weight
        return self.out[:num_tokens]


def _build_v2_model_state_for_ple(
    layer: Qwen4ExpNGramEmbedding,
    config: SimpleNamespace,
    max_num_reqs: int,
    device: torch.device,
) -> Qwen4ExpModelState:
    """A real `Qwen4ExpModelState` wired to `layer`, bypassing `__init__`'s
    KV-cache/attention setup (`object.__new__` + manual attributes), exactly
    like the model-state tests above. Callers must still patch
    `MambaHybridModelState.prepare_inputs` / `prepare_dummy_inputs` (the
    heavy, KV-cache-dependent `super()` calls) before driving
    `prepare_inputs` / `prepare_dummy_inputs` on the result."""
    ngram_context_len = int(config.ngram_size) - 1
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state._mmap_ple_modules = (layer,)
    model_state.ngram_context_len = ngram_context_len
    model_state.ngram_eos_token_id = int(config.eos_token_id)
    model_state.ngram_context = torch.full(
        (max_num_reqs, ngram_context_len),
        model_state.ngram_eos_token_id,
        dtype=torch.int32,
        device=device,
    )
    model_state.ngram_context_offsets = torch.arange(
        -ngram_context_len, 0, dtype=torch.int64, device=device
    )
    model_state.ple_query_start_loc = torch.zeros(
        max_num_reqs + 1, dtype=torch.int32, device=device
    )
    return model_state


def _capture_ple_consumer_fullgraph(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config: SimpleNamespace,
    model_state: Qwen4ExpModelState,
    compiled_consumer: nn.Module,
    input_ids_buf: torch.Tensor,
    padded_tokens: int,
) -> tuple[CudaGraphManager, Any]:
    """Capture through the real `CudaGraphManager.capture()` -- the exact
    primitive `ModelCudaGraphManager.capture()` delegates to -- with a
    single FULL-mode candidate at `padded_tokens`. Neutralizes only the
    distributed-group plumbing (`get_pp_group`/`graph_capture`) that a
    single-process test has no real process group for (mirrors
    tests/v1/cudagraph/test_cudagraph_manager.py); `torch.cuda.graph` /
    `torch.cuda.CUDAGraph` themselves run for real, on the real device.
    """
    monkeypatch.setattr(
        cudagraph_utils_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    @contextmanager
    def _fake_graph_capture(*args: object, **kwargs: object):
        del args, kwargs
        yield None

    monkeypatch.setattr(cudagraph_utils_module, "graph_capture", _fake_graph_capture)

    manager = CudaGraphManager(
        vllm_config=vllm_config,
        device=input_ids_buf.device,
        cudagraph_mode=CUDAGraphMode.FULL,
        decode_query_len=1,
    )

    def create_forward_fn(desc: Any, warmup: bool):
        del warmup
        num_tokens = desc.num_tokens
        num_reqs = desc.num_reqs
        # Dummy staging happens OUTSIDE the graph, exactly as
        # ModelCudaGraphManager.capture()'s own create_forward_fn does.
        model_inputs = model_state.prepare_dummy_inputs(num_reqs, num_tokens)
        input_ids = input_ids_buf[:num_tokens]

        def forward_fn(cg_mode: CUDAGraphMode) -> None:
            with set_forward_context(
                attn_metadata=None,
                vllm_config=vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=cg_mode,
                batch_descriptor=None,
            ):
                compiled_consumer(
                    input_ids,
                    model_inputs["query_start_loc"],
                    model_inputs["ngram_context"],
                )

        return forward_fn

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manager.capture(create_forward_fn)

    empty_graph_warnings = [
        str(w.message) for w in caught if "CUDA Graph is empty" in str(w.message)
    ]
    assert not empty_graph_warnings, empty_graph_warnings
    assert len(manager.graphs) == 1
    (desc,) = manager.graphs.keys()
    assert desc.num_tokens == padded_tokens
    return manager, desc


def _ple_vllm_config_for_cudagraph(
    tmp_path_unused: Path,
    padded_tokens: int,
    max_num_reqs: int,
) -> SimpleNamespace:
    del tmp_path_unused
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_capture_sizes=[padded_tokens],
    )
    cc.max_cudagraph_capture_size = padded_tokens
    cc.post_init_cudagraph_sizes()
    return SimpleNamespace(
        compilation_config=cc,
        model_config=SimpleNamespace(),
        use_v2_model_runner=True,
        parallel_config=ParallelConfig(),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_reqs),
        speculative_config=None,
        num_speculative_tokens=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize(
    "table_dtype", [torch.float8_e4m3fn, torch.bfloat16], ids=["fp8", "bf16"]
)
def test_v2_staging_survives_torch_compile_and_full_cudagraph_capture_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table_dtype: torch.dtype
) -> None:
    """This is the plan's mandatory MRV2 correctness proof, run through the
    real production FULL capture primitive -- `CudaGraphManager.capture()`,
    the exact method `ModelCudaGraphManager.capture()` delegates to via
    `super().capture(create_forward_fn, ...)` -- with a real downstream CUDA
    kernel (a GEMM) capturing and consuming `Qwen4ExpNGramEmbedding`'s
    output, and rows staged the same way the real runner stages them:
    through `Qwen4ExpModelState.prepare_inputs` / `prepare_dummy_inputs`,
    never a raw `gather_into` call from the test.

    `Qwen4ExpNGramEmbedding.forward` in mmap mode returns a bare view of
    `_mmap_staging` (a slice + flatten, no kernel): captured alone, PyTorch
    warns "The CUDA Graph is empty" and the graph replays nothing, yet
    reading that view after capture still shows the right numbers because
    it aliases the buffer the test itself just wrote -- proving nothing
    about replay. The downstream consumer's GEMM forces a real kernel into
    the graph, writing into a SEPARATE stable buffer (`consumer.out`): a
    post-replay read of THAT buffer can only be right if the captured
    kernel genuinely re-executed and re-read the staging buffer's live
    contents.

    It captures ONE graph at a padded token count, stages an ACTUAL (real,
    ngram-hashed) request smaller than that padding through
    `model_state.prepare_inputs`, replays, and checks both the actual rows
    (real gathered content) and the padding tail (must be zero) came
    through the captured kernel correctly. It then re-stages a DIFFERENT
    actual request into the SAME buffer address with no new capture, and
    replays the SAME graph again -- proving replay reads the buffer's live
    contents rather than a value frozen at capture time, which is the
    entire justification for the immutable-address staging design.
    """
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    device = torch.device("cuda")
    embedding_dim, head_dim = 8, 2
    padded_tokens, max_num_reqs = 4, 4
    write_scale = table_dtype == torch.float8_e4m3fn

    config = _make_text_config()
    vllm_config = _ple_vllm_config_for_cudagraph(tmp_path, padded_tokens, max_num_reqs)
    with set_current_vllm_config(vllm_config):
        layer = Qwen4ExpNGramEmbedding(
            config,
            embedding_dim,
            0,
            padded_tokens,
            max_num_reqs,
            "model.layers.0.ple.ple_embedding",
            "model.layers.0.ple",
            params_dtype=torch.float32,
        )
    # The placeholder's own vocab layout (derived from ngram_vocab_size_base
    # etc, NOT a small test constant) is the only vocab size a real
    # checkpoint for this layer can be written against.
    vocab = layer.ngram_embedding.org_vocab_size
    full = _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=vocab,
        parts=1,
        cols=head_dim,
        scale=0.5,
        write_scale=write_scale,
        table_dtype=table_dtype,
    ).to(device)
    shard_map = ple_mmap.discover_shards(str(tmp_path))
    ple_mmap._attach_table(
        layer.ngram_embedding,
        shard_map[0],
        split_ngram_parts=1,
        layer_idx=0,
        model_path=str(tmp_path),
    )
    layer = layer.to(device)
    layer.initialize_mmap_staging(padded_tokens, device)
    staging_ptr = layer._mmap_staging.data_ptr()

    out_dim = 4
    weight = torch.randn(embedding_dim, out_dim, dtype=torch.float32, device=device)
    out_buf = torch.zeros(padded_tokens, out_dim, dtype=torch.float32, device=device)
    out_ptr = out_buf.data_ptr()
    consumer = _PleDownstreamConsumer(layer, weight, out_buf).to(device)
    compiled_consumer = torch.compile(consumer, fullgraph=True)

    model_state = _build_v2_model_state_for_ple(layer, config, max_num_reqs, device)
    input_ids_buf = torch.zeros(padded_tokens, dtype=torch.long, device=device)

    with (
        patch.object(MambaHybridModelState, "prepare_inputs", return_value={}),
        patch.object(MambaHybridModelState, "prepare_dummy_inputs", return_value={}),
    ):
        manager, desc = _capture_ple_consumer_fullgraph(
            monkeypatch,
            vllm_config,
            model_state,
            compiled_consumer,
            input_ids_buf,
            padded_tokens,
        )
        assert layer._mmap_staging.data_ptr() == staging_ptr
        assert out_buf.data_ptr() == out_ptr

        actual_tokens = 3
        num_reqs = 1

        def _stage_actual_request(
            token_history: list[int], input_ids: list[int]
        ) -> torch.Tensor:
            """Stage one ACTUAL (real, ngram-hashed) request through
            `Qwen4ExpModelState.prepare_inputs`, and return the
            independently-derived expected embedded rows: the same
            `compute_ngram_ids` call `prepare_mmap_rows` itself makes, with
            the same inputs -- never a read of `_mmap_staging` used as its
            own proof."""
            input_batch = SimpleNamespace(
                num_reqs=num_reqs,
                num_reqs_after_padding=num_reqs,
                num_tokens=actual_tokens,
                num_tokens_after_padding=padded_tokens,
                idx_mapping=torch.tensor([0], device=device),
                query_start_loc=torch.tensor(
                    [0, actual_tokens], dtype=torch.int32, device=device
                ),
                input_ids=torch.tensor(input_ids, dtype=torch.int32, device=device),
            )
            req_states = SimpleNamespace(
                num_computed_tokens=SimpleNamespace(
                    gpu=torch.tensor([len(token_history)], device=device)
                ),
                all_token_ids=SimpleNamespace(
                    gpu=torch.tensor([token_history], dtype=torch.int32, device=device)
                ),
            )
            expected_ngram_context = model_state._prepare_ngram_context(
                input_batch, req_states
            )[:num_reqs].clone()
            expected_ids = layer.compute_ngram_ids(
                input_batch.input_ids[:actual_tokens].clone(),
                input_batch.query_start_loc[: num_reqs + 1].clone(),
                expected_ngram_context,
            )
            model_state.prepare_inputs(input_batch, req_states)
            return full[expected_ids].flatten(-2)

        # 1. Stage ACTUAL request A: 3 real tokens, padded to 4.
        expected_a = _stage_actual_request(
            token_history=[10, 11, 12, 13, 14, 15], input_ids=[21, 22, 23, 0]
        )
        # Guard the premise: a real, non-trivial gather (never all-zero).
        assert not torch.equal(expected_a, torch.zeros_like(expected_a))
        torch.testing.assert_close(
            layer._mmap_staging[:actual_tokens].flatten(-2), expected_a
        )
        padding_tail = layer._mmap_staging[actual_tokens:padded_tokens]
        assert torch.equal(padding_tail, torch.zeros_like(padding_tail))
        assert layer._mmap_staging.data_ptr() == staging_ptr

        # 2. Replay #1 through the captured graph.
        manager.run_fullgraph(desc)
        torch.accelerator.synchronize()
        expected_out_a = expected_a.to(weight.dtype) @ weight
        torch.testing.assert_close(out_buf[:actual_tokens], expected_out_a)
        torch.testing.assert_close(
            out_buf[actual_tokens:padded_tokens],
            torch.zeros((padded_tokens - actual_tokens, out_dim), device=device),
        )
        replay_a = out_buf.clone()

        # 3. Re-stage a DIFFERENT actual request into the SAME buffer
        # address, with no new capture -- the immutable-address contract
        # this design depends on.
        expected_b = _stage_actual_request(
            token_history=[30, 31, 32, 33, 34, 35], input_ids=[41, 42, 43, 0]
        )
        assert not torch.equal(expected_a, expected_b)  # guard the premise
        assert layer._mmap_staging.data_ptr() == staging_ptr

        # 4. Replay #2 of the SAME captured graph: must reflect the
        # buffer's NEW contents, not a value frozen at capture/first-replay
        # time.
        manager.run_fullgraph(desc)
        torch.accelerator.synchronize()
        expected_out_b = expected_b.to(weight.dtype) @ weight
        torch.testing.assert_close(out_buf[:actual_tokens], expected_out_b)
        torch.testing.assert_close(
            out_buf[actual_tokens:padded_tokens],
            torch.zeros((padded_tokens - actual_tokens, out_dim), device=device),
        )
        assert not torch.equal(replay_a, out_buf)
        assert len(manager.graphs) == 1  # no second capture


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_v2_staging_cudagraph_replay_red_proof_disconnected_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Red-proof for the test above: with mmap-row staging disconnected
    (`Qwen4ExpNGramEmbedding.prepare_mmap_rows` neutralized to a no-op, so
    `gather_into` never runs), the captured graph's downstream consumer must
    NOT produce the real request's expected rows -- it can only keep
    reading whatever `_mmap_staging` held before (the capture-time dummy
    zeros). If this divergence assertion ever failed (i.e. the disconnected
    path still produced the right answer), the test above would be unable
    to detect a regression that severs staging from what the graph actually
    consumes -- it would pass vacuously regardless of whether staging ran.
    """
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    device = torch.device("cuda")
    embedding_dim, head_dim = 8, 2
    padded_tokens, max_num_reqs = 4, 4
    table_dtype = torch.bfloat16

    config = _make_text_config()
    vllm_config = _ple_vllm_config_for_cudagraph(tmp_path, padded_tokens, max_num_reqs)
    with set_current_vllm_config(vllm_config):
        layer = Qwen4ExpNGramEmbedding(
            config,
            embedding_dim,
            0,
            padded_tokens,
            max_num_reqs,
            "model.layers.0.ple.ple_embedding",
            "model.layers.0.ple",
            params_dtype=torch.float32,
        )
    vocab = layer.ngram_embedding.org_vocab_size
    full = _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=vocab,
        parts=1,
        cols=head_dim,
        scale=0.5,
        write_scale=False,
        table_dtype=table_dtype,
    ).to(device)
    shard_map = ple_mmap.discover_shards(str(tmp_path))
    ple_mmap._attach_table(
        layer.ngram_embedding,
        shard_map[0],
        split_ngram_parts=1,
        layer_idx=0,
        model_path=str(tmp_path),
    )
    layer = layer.to(device)
    layer.initialize_mmap_staging(padded_tokens, device)

    out_dim = 4
    weight = torch.randn(embedding_dim, out_dim, dtype=torch.float32, device=device)
    out_buf = torch.zeros(padded_tokens, out_dim, dtype=torch.float32, device=device)
    consumer = _PleDownstreamConsumer(layer, weight, out_buf).to(device)
    compiled_consumer = torch.compile(consumer, fullgraph=True)

    model_state = _build_v2_model_state_for_ple(layer, config, max_num_reqs, device)
    input_ids_buf = torch.zeros(padded_tokens, dtype=torch.long, device=device)

    with (
        patch.object(MambaHybridModelState, "prepare_inputs", return_value={}),
        patch.object(MambaHybridModelState, "prepare_dummy_inputs", return_value={}),
        # Disconnect staging consumption: prepare_inputs still runs (updates
        # query_start_loc/ngram_context), but the PLE layer's own row
        # gather never happens -- _mmap_staging keeps whatever it held
        # before (capture-time dummy zeros).
        patch.object(Qwen4ExpNGramEmbedding, "prepare_mmap_rows", lambda *a, **k: None),
    ):
        manager, desc = _capture_ple_consumer_fullgraph(
            monkeypatch,
            vllm_config,
            model_state,
            compiled_consumer,
            input_ids_buf,
            padded_tokens,
        )

        actual_tokens = 3
        num_reqs = 1
        input_batch = SimpleNamespace(
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            num_tokens=actual_tokens,
            num_tokens_after_padding=padded_tokens,
            idx_mapping=torch.tensor([0], device=device),
            query_start_loc=torch.tensor(
                [0, actual_tokens], dtype=torch.int32, device=device
            ),
            input_ids=torch.tensor([21, 22, 23, 0], dtype=torch.int32, device=device),
        )
        req_states = SimpleNamespace(
            num_computed_tokens=SimpleNamespace(gpu=torch.tensor([6], device=device)),
            all_token_ids=SimpleNamespace(
                gpu=torch.tensor(
                    [[10, 11, 12, 13, 14, 15]], dtype=torch.int32, device=device
                )
            ),
        )
        expected_ngram_context = model_state._prepare_ngram_context(
            input_batch, req_states
        )[:num_reqs].clone()
        expected_ids = layer.compute_ngram_ids(
            input_batch.input_ids[:actual_tokens].clone(),
            input_batch.query_start_loc[: num_reqs + 1].clone(),
            expected_ngram_context,
        )
        expected_real_rows = full[expected_ids].flatten(-2)

        model_state.prepare_inputs(input_batch, req_states)  # gather neutralized above
        manager.run_fullgraph(desc)
        torch.accelerator.synchronize()

        expected_out_if_connected = expected_real_rows.to(weight.dtype) @ weight
        assert not torch.allclose(out_buf[:actual_tokens], expected_out_if_connected)


# --------------------------------------------------------------------------- #
# prepare_mmap_rows / prepare_dummy_mmap_rows: the layer-level staging call
# V2 model state makes from `prepare_inputs`/`prepare_dummy_inputs`, before
# the compiled/captured forward ever runs.
# --------------------------------------------------------------------------- #


def _build_mmap_ngram_layer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    layer_idx: int = 0,
    embedding_dim: int = 8,
    max_total_tokens: int = 16,
    max_num_reqs: int = 4,
) -> Qwen4ExpNGramEmbedding:
    """A real V2-mode Qwen4ExpNGramEmbedding with mmap enabled, with no
    table attached yet (a dummy load) -- callers attach a table themselves
    when a test needs real gathered content."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.FULL
    )
    vllm_config = SimpleNamespace(
        compilation_config=cc,
        model_config=SimpleNamespace(),
        use_v2_model_runner=True,
    )
    with set_current_vllm_config(vllm_config):
        return Qwen4ExpNGramEmbedding(
            config,
            embedding_dim,
            layer_idx,
            max_total_tokens,
            max_num_reqs,
            f"model.layers.{layer_idx}.ple.ple_embedding",
            f"model.layers.{layer_idx}.ple",
            params_dtype=torch.float32,
        )


def test_prepare_mmap_rows_and_dummy_rows_keep_the_staging_buffer_pointer_stable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The staging buffer's address must never change across repeated real
    and dummy preparations -- captured code depends on this."""
    layer = _build_mmap_ngram_layer(monkeypatch)
    vocab = layer.ngram_embedding.org_vocab_size
    _write_ple_layer(
        tmp_path, layer_idx=0, vocab=vocab, parts=1, cols=layer.head_dim, scale=0.5
    )
    shard_map = ple_mmap.discover_shards(str(tmp_path))
    ple_mmap._attach_table(
        layer.ngram_embedding,
        shard_map[0],
        split_ngram_parts=1,
        layer_idx=0,
        model_path=str(tmp_path),
    )
    layer.initialize_mmap_staging(8, torch.device("cpu"))
    ptr = layer._mmap_staging.data_ptr()

    input_ids = torch.zeros((1, 3), dtype=torch.long)
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    ngram_context = torch.zeros((1, 2), dtype=torch.long)
    layer.prepare_mmap_rows(input_ids, query_start_loc, ngram_context, 3, 8)
    assert layer._mmap_staging.data_ptr() == ptr

    layer.prepare_dummy_mmap_rows(8)
    assert layer._mmap_staging.data_ptr() == ptr

    layer.prepare_mmap_rows(input_ids, query_start_loc, ngram_context, 3, 8)
    assert layer._mmap_staging.data_ptr() == ptr


def test_prepare_mmap_rows_overwrites_actual_rows_and_zeros_stale_padded_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A larger batch fills the whole buffer with real rows; a smaller batch
    reusing the same (larger) graph's buffer must overwrite its own actual
    rows AND zero the now-stale tail that used to hold the larger batch's
    real content -- a captured replay must never read leftover data as if
    it were this step's padding."""
    layer = _build_mmap_ngram_layer(monkeypatch, max_total_tokens=8, max_num_reqs=8)
    layer.initialize_mmap_staging(8, torch.device("cpu"))

    # Stand in deterministic, distinguishable "gathered" content keyed by
    # the ids compute_ngram_ids would have produced -- decoupled from real
    # hashing (covered by test_compute_ngram_ids_matches_golden_ids) and
    # from a real on-disk table (covered by the gather_into value tests),
    # so this test isolates prepare_mmap_rows' own indexing/zeroing.
    def _fake_compute_ngram_ids(input_ids, query_start_loc, ngram_context):
        return input_ids.reshape(-1, 1).expand(-1, layer.ngram_heads).clone()

    def _fake_gather_into(ids, destination):
        expanded = ids.unsqueeze(-1).to(destination.dtype)
        destination.copy_(expanded.expand(-1, -1, destination.shape[-1]))

    layer.compute_ngram_ids = _fake_compute_ngram_ids
    monkeypatch.setattr(layer.ngram_embedding, "gather_into", _fake_gather_into)

    # Round 1: a full 8-token batch -- every row gets real (nonzero) content.
    ids_large = torch.arange(1, 9, dtype=torch.long)
    layer.prepare_mmap_rows(
        ids_large, torch.tensor([0, 8], dtype=torch.int32), torch.zeros((1, 2)), 8, 8
    )
    assert torch.all(layer._mmap_staging != 0)

    # Round 2: a 3-token batch reusing the same 8-slot buffer.
    ids_small = torch.tensor([10, 20, 30], dtype=torch.long)
    layer.prepare_mmap_rows(
        ids_small, torch.tensor([0, 3], dtype=torch.int32), torch.zeros((1, 2)), 3, 8
    )

    expected_actual = (
        ids_small.reshape(-1, 1, 1)
        .expand(-1, layer.ngram_heads, layer.head_dim)
        .to(layer._mmap_staging.dtype)
    )
    torch.testing.assert_close(layer._mmap_staging[:3], expected_actual)
    stale_tail = layer._mmap_staging[3:8]
    assert torch.equal(stale_tail, torch.zeros_like(stale_tail))


def test_prepare_mmap_rows_across_two_layers_keep_distinct_buffers_and_table_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two mmap PLE layers must never share a staging buffer address, and
    each must gather from its OWN attached table/hash constants -- feeding
    the SAME input_ids through both must not produce the same rows."""
    layer0 = _build_mmap_ngram_layer(monkeypatch, layer_idx=0)
    layer1 = _build_mmap_ngram_layer(monkeypatch, layer_idx=1)
    vocab0 = layer0.ngram_embedding.org_vocab_size
    vocab1 = layer1.ngram_embedding.org_vocab_size
    _write_ple_layer(
        tmp_path, layer_idx=0, vocab=vocab0, parts=1, cols=layer0.head_dim, scale=0.5
    )
    _write_ple_layer(
        tmp_path, layer_idx=1, vocab=vocab1, parts=1, cols=layer1.head_dim, scale=0.5
    )
    shard_map = ple_mmap.discover_shards(str(tmp_path))
    ple_mmap._attach_table(
        layer0.ngram_embedding,
        shard_map[0],
        split_ngram_parts=1,
        layer_idx=0,
        model_path=str(tmp_path),
    )
    ple_mmap._attach_table(
        layer1.ngram_embedding,
        shard_map[1],
        split_ngram_parts=1,
        layer_idx=1,
        model_path=str(tmp_path),
    )
    layer0.initialize_mmap_staging(4, torch.device("cpu"))
    layer1.initialize_mmap_staging(4, torch.device("cpu"))

    assert layer0._mmap_staging.data_ptr() != layer1._mmap_staging.data_ptr()
    assert layer0.ngram_embedding.table is not layer1.ngram_embedding.table

    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    ngram_context = torch.zeros((1, 2), dtype=torch.long)
    layer0.prepare_mmap_rows(input_ids, query_start_loc, ngram_context, 3, 4)
    layer1.prepare_mmap_rows(input_ids, query_start_loc, ngram_context, 3, 4)

    # Layer 1's table is keyed with a different layer_idx seed (see
    # _synthetic_weight), so identical inputs must not stage identical rows.
    assert not torch.equal(layer0._mmap_staging[:3], layer1._mmap_staging[:3])


def test_prepare_dummy_mmap_rows_with_no_table_never_calls_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dummy preparation performs no table access at all -- not even the
    zero-destination branch of gather_into. It must only ever zero the
    buffer directly."""
    layer = _build_mmap_ngram_layer(monkeypatch)
    assert layer.ngram_embedding.table is None
    layer.initialize_mmap_staging(8, torch.device("cpu"))
    layer._mmap_staging.fill_(1.0)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dummy preparation must never call gather_into")

    monkeypatch.setattr(layer.ngram_embedding, "gather_into", _raise)

    layer.prepare_dummy_mmap_rows(8)  # must not raise

    assert torch.equal(layer._mmap_staging, torch.zeros_like(layer._mmap_staging))


# --------------------------------------------------------------------------- #
# Dispatch red-proofs: preparation is the ONLY call site that ever reaches
# the table's gather; forward has no independent path to real content.
# --------------------------------------------------------------------------- #


def test_forward_never_calls_gather_only_preparation_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monkeypatch the table's gather to raise, then call forward: it must
    not raise, because forward only ever reads the pre-staged buffer.
    Calling prepare_mmap_rows (the actual, single call site that reaches
    gather) with the same patch in place must raise -- proving the patch
    would have caught a regression that let forward reach the table."""
    layer = _build_mmap_ngram_layer(monkeypatch)
    layer.initialize_mmap_staging(8, torch.device("cpu"))

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("forward must never call gather_into")

    monkeypatch.setattr(layer.ngram_embedding, "gather_into", _raise)

    input_ids = torch.zeros((1, 3), dtype=torch.long)
    out = layer.forward(input_ids, None, None)  # must not raise
    assert out.shape == (3, layer.embedding_dim)

    with pytest.raises(AssertionError, match="forward must never call gather_into"):
        layer.prepare_mmap_rows(
            input_ids,
            torch.tensor([0, 3], dtype=torch.int32),
            torch.zeros((1, 2)),
            3,
            8,
        )


def test_forward_without_prior_preparation_never_reproduces_real_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Red-proof for the staging design: if forward ever regressed to
    independently re-gather (rather than only reading the pre-staged
    buffer), this test would start seeing real, non-zero table content
    here even though prepare_mmap_rows was never called. A freshly
    initialized buffer must stay all-zero through forward alone."""
    layer = _build_mmap_ngram_layer(monkeypatch)
    vocab = layer.ngram_embedding.org_vocab_size
    _write_ple_layer(
        tmp_path, layer_idx=0, vocab=vocab, parts=1, cols=layer.head_dim, scale=0.5
    )
    shard_map = ple_mmap.discover_shards(str(tmp_path))
    ple_mmap._attach_table(
        layer.ngram_embedding,
        shard_map[0],
        split_ngram_parts=1,
        layer_idx=0,
        model_path=str(tmp_path),
    )
    layer.initialize_mmap_staging(8, torch.device("cpu"))

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = layer.forward(input_ids, None, None)  # prepare_mmap_rows never called

    assert torch.equal(out, torch.zeros_like(out))


# --------------------------------------------------------------------------- #
# V2 model state: closed-world discovery, aggregate memory preflight, and
# staged input preparation (actual vs. padded extents, dummy-only zeroing).
# --------------------------------------------------------------------------- #


def test_model_state_discovers_matching_mmap_modules_from_static_and_module_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _build_mmap_ngram_layer(monkeypatch)
    model = nn.Module()
    model.add_module("ple0", layer)
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"a.ple": SimpleNamespace(ple_embedding=layer)}
        )
    )
    model_state = object.__new__(Qwen4ExpModelState)

    result = model_state._discover_mmap_ple_modules(vllm_config, model)

    assert result == (layer,)


def test_model_state_raises_when_static_context_and_model_modules_disagree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mismatch between the compiled graph's own view
    (static_forward_context) and the live module tree (model.modules())
    must raise -- FULL capture can never be authorized on an unproven
    inventory."""
    in_static_context_only = _build_mmap_ngram_layer(monkeypatch, layer_idx=0)
    in_model_modules_only = _build_mmap_ngram_layer(monkeypatch, layer_idx=1)
    model = nn.Module()
    model.add_module("ple1", in_model_modules_only)  # NOT in_static_context_only
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={
                "a.ple": SimpleNamespace(ple_embedding=in_static_context_only)
            }
        )
    )
    model_state = object.__new__(Qwen4ExpModelState)

    with pytest.raises(RuntimeError, match="inventories disagree"):
        model_state._discover_mmap_ple_modules(vllm_config, model)


def test_model_state_initialize_mmap_staging_fails_closed_before_allocating_any_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The aggregate allocation preflight must fail BEFORE any individual
    layer's buffer is allocated when the aggregate would not fit."""
    layer0 = _build_mmap_ngram_layer(monkeypatch, layer_idx=0)
    layer1 = _build_mmap_ngram_layer(monkeypatch, layer_idx=1)
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.max_num_tokens = 1024
    model_state.device = torch.device("cpu")

    per_layer_bytes = layer0.mmap_staging_nbytes(1024)
    too_small_free = 2 * per_layer_bytes - 1

    class _FakeSnapshot:
        def __init__(self, device: torch.device) -> None:
            self.free_memory = too_small_free

    monkeypatch.setattr(model_state_module, "MemorySnapshot", _FakeSnapshot)

    with pytest.raises(RuntimeError, match="GiB"):
        model_state._initialize_mmap_staging((layer0, layer1))

    assert layer0._mmap_staging is None
    assert layer1._mmap_staging is None


def test_model_state_initialize_mmap_staging_allocates_every_layer_when_it_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer0 = _build_mmap_ngram_layer(monkeypatch, layer_idx=0)
    layer1 = _build_mmap_ngram_layer(monkeypatch, layer_idx=1)
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.max_num_tokens = 32
    model_state.device = torch.device("cpu")

    class _FakeSnapshot:
        def __init__(self, device: torch.device) -> None:
            self.free_memory = 1 << 40  # 1 TiB: always fits

    monkeypatch.setattr(model_state_module, "MemorySnapshot", _FakeSnapshot)

    model_state._initialize_mmap_staging((layer0, layer1))

    for layer in (layer0, layer1):
        assert layer._mmap_staging is not None
        assert layer._mmap_staging.shape == (32, layer.ngram_heads, layer.head_dim)
    assert layer0._mmap_staging.data_ptr() != layer1._mmap_staging.data_ptr()


class _RecordingMmapModule:
    """Stands in for a Qwen4ExpNGramEmbedding at the model-state boundary:
    model_state.py calls these two methods by duck-typed contract, never by
    isinstance, so a plain recorder is faithful here."""

    def __init__(self) -> None:
        self.prepare_calls: list[tuple[Any, Any, Any, int, int]] = []
        self.dummy_calls: list[int] = []

    def prepare_mmap_rows(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        actual_tokens: int,
        padded_tokens: int,
    ) -> None:
        self.prepare_calls.append(
            (
                input_ids.clone(),
                query_start_loc.clone(),
                ngram_context.clone(),
                actual_tokens,
                padded_tokens,
            )
        )

    def prepare_dummy_mmap_rows(self, padded_tokens: int) -> None:
        self.dummy_calls.append(padded_tokens)


def test_model_state_prepare_inputs_stages_mmap_rows_using_actual_slices() -> None:
    """prepare_inputs must call prepare_mmap_rows with the ACTUAL (unpadded)
    input_ids/query_start_loc/ngram_context extents, and the actual/padded
    token counts -- never graph padding."""
    module = _RecordingMmapModule()
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state._mmap_ple_modules = (module,)
    model_state.ngram_context_len = 3
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((4, 3), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-3, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.empty(5, dtype=torch.int32)

    # num_reqs=2 (actual) padded to num_reqs_after_padding=3; num_tokens=3
    # (actual) padded to num_tokens_after_padding=5.
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        num_tokens=3,
        num_tokens_after_padding=5,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
        input_ids=torch.tensor([11, 12, 13, 0, 0], dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 1])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3, 4], [20, 21, 22, 23]], dtype=torch.int32)
        ),
    )

    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_state.prepare_inputs(input_batch, req_states)

    assert len(module.prepare_calls) == 1
    got_input_ids, got_qsl, got_ctx, actual_tokens, padded_tokens = (
        module.prepare_calls[0]
    )
    assert actual_tokens == 3
    assert padded_tokens == 5
    torch.testing.assert_close(got_input_ids, input_batch.input_ids[:3])
    assert got_qsl.shape == (3,)  # num_reqs (actual) + 1, not padded's 4
    assert got_ctx.shape[0] == 2  # num_reqs (actual), not num_reqs_after_padding


def test_model_state_prepare_dummy_inputs_only_zeros_mmap_staging() -> None:
    """Capture-time dummy preparation must never hash, gather, or touch a
    table -- only prepare_dummy_mmap_rows (a zero) may be called."""
    module = _RecordingMmapModule()
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state._mmap_ple_modules = (module,)
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((4, 3), dtype=torch.int32)
    model_state.ple_query_start_loc = torch.empty(5, dtype=torch.int32)

    with patch.object(MambaHybridModelState, "prepare_dummy_inputs", return_value={}):
        model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=8)

    assert module.dummy_calls == [8]
    assert module.prepare_calls == []


def test_model_state_prepare_runtime_dummy_inputs_only_zeros_at_padded_extent() -> None:
    """A runtime (profile/DP-empty) dummy run has no real request state to
    hash against: it must call prepare_dummy_mmap_rows at the PADDED token
    extent, and must never call prepare_mmap_rows (which would reach a
    table's hash/gather/pin methods)."""
    module = _RecordingMmapModule()
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state._mmap_ple_modules = (module,)
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((4, 3), dtype=torch.int32)
    model_state.ple_query_start_loc = torch.empty(5, dtype=torch.int32)

    input_batch = SimpleNamespace(num_reqs_after_padding=3, num_tokens_after_padding=8)
    req_states = SimpleNamespace()

    # super().prepare_inputs, NOT super().prepare_dummy_inputs -- see the
    # method's own docstring on why.
    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_state.prepare_runtime_dummy_inputs(input_batch, req_states)

    assert module.dummy_calls == [8]
    assert module.prepare_calls == []


# --------------------------------------------------------------------------- #
# Construction-time shard validation: refuses a bad checkpoint BEFORE
# the ~78 GiB backbone streams, not after.
# --------------------------------------------------------------------------- #


def test_validate_shards_for_raises_when_no_shards_at_a_resolved_path(
    tmp_path: Path,
) -> None:
    model_config = _model_config(tmp_path)  # resolves; directory is empty

    with pytest.raises(RuntimeError, match="no shard tensors for layer 1"):
        ple_mmap.validate_shards_for(model_config, "model.layers.1.ple", head_dim=4)


def test_validate_shards_for_raises_on_shard_width_mismatch(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=1, vocab=10, parts=3, cols=2, scale=0.25)
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="shard width"):
        ple_mmap.validate_shards_for(model_config, "model.layers.1.ple", head_dim=4)


def test_validate_shards_for_raises_when_weight_scale_missing(tmp_path: Path) -> None:
    _write_ple_layer(
        tmp_path, layer_idx=1, vocab=10, parts=3, cols=2, scale=0.25, write_scale=False
    )
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="no ngram_embedding.weight_scale"):
        ple_mmap.validate_shards_for(model_config, "model.layers.1.ple", head_dim=2)


def test_validate_shards_for_passes_on_a_well_formed_checkpoint(
    tmp_path: Path,
) -> None:
    _write_ple_layer(tmp_path, layer_idx=1, vocab=10, parts=3, cols=2, scale=0.25)
    model_config = _model_config(tmp_path)

    ple_mmap.validate_shards_for(
        model_config, "model.layers.1.ple", head_dim=2
    )  # must not raise


def test_validate_shards_for_refuses_a_bf16_table_with_a_stray_weight_scale(
    tmp_path: Path,
) -> None:
    """BF16 (unquantized) tables are registered with requires_scale=False —
    a weight_scale present on disk anyway signals exporter confusion (e.g. a
    half fp8-to-bf16 conversion) and must be refused up front, not silently
    ignored, per the NEW fail-closed case in _validate_layer_shards."""
    _write_ple_layer(
        tmp_path,
        layer_idx=1,
        vocab=10,
        parts=3,
        cols=2,
        scale=0.25,
        write_scale=True,
        table_dtype=torch.bfloat16,
    )
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="BF16"):
        ple_mmap.validate_shards_for(model_config, "model.layers.1.ple", head_dim=2)


def test_validate_shards_for_passes_on_a_well_formed_bf16_checkpoint(
    tmp_path: Path,
) -> None:
    _write_ple_layer(
        tmp_path,
        layer_idx=1,
        vocab=10,
        parts=3,
        cols=2,
        scale=0.25,
        write_scale=False,
        table_dtype=torch.bfloat16,
    )
    model_config = _model_config(tmp_path)

    ple_mmap.validate_shards_for(
        model_config, "model.layers.1.ple", head_dim=2
    )  # must not raise


def test_validate_shards_for_tolerates_an_unresolvable_model_path() -> None:
    """A bare repo id with no local snapshot (e.g. --load-format
    dummy/test construction, offline): validation defers to load time
    rather than raising — build_tables still fail-closes there."""
    model_config = SimpleNamespace(
        model_weights="", model="nonexistent-org/nonexistent-repo-xyz", revision=None
    )

    ple_mmap.validate_shards_for(
        model_config, "model.layers.1.ple", head_dim=4
    )  # must not raise


def test_ngram_embedding_construction_refuses_a_bad_checkpoint_before_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A checkpoint whose PLE shards are missing/wrong makes
    __init__ itself raise — before load_weights, before the backbone
    streams — not just at build_tables time."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.PIECEWISE
    )
    vllm_config = SimpleNamespace(
        compilation_config=cc,
        model_config=_model_config(tmp_path),
        use_v2_model_runner=True,
    )

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="no shard tensors for layer 1"),
    ):
        Qwen4ExpNGramEmbedding(
            config,
            8,
            0,
            16,
            4,
            "model.layers.1.ple.ple_embedding",
            "model.layers.1.ple",
            params_dtype=torch.float32,
        )


# --------------------------------------------------------------------------- #
# (e) compile-factor assertions.
# --------------------------------------------------------------------------- #


def test_ple_mmap_flag_is_a_compile_factor() -> None:
    assert "VLLM_PLE_MMAP" in envs.compile_factors()


def test_ple_mmap_tuning_knobs_are_not_compile_factors() -> None:
    factors = envs.compile_factors()
    assert "VLLM_PLE_MMAP_WORKERS" not in factors
    assert "VLLM_PLE_MMAP_CHUNK" not in factors
    assert "VLLM_PLE_MMAP_PREWARM" not in factors
    # An unlisted var becomes a torch.compile cache key, so toggling the
    # readahead knob would force a recompile and poison its own A/B.
    assert "VLLM_PLE_MMAP_READAHEAD" not in factors
    assert "VLLM_PLE_MMAP_PINNED" not in factors
    assert "VLLM_PLE_MMAP_SERIAL" not in factors


# --------------------------------------------------------------------------- #
# Placeholder embedding
# --------------------------------------------------------------------------- #


def test_placeholder_forward_returns_fp8_zeros_when_table_unset() -> None:
    """--load-format dummy: load_weights never runs at all, so
    weights_streamed stays False and the table stays unset; the placeholder
    must still produce a valid (zero) fp8 tensor against the default unit
    weight_scale — this is the ONLY case zeros are legitimate."""
    embedding = ple_mmap.MmapNgramEmbedding(16, 4)
    ids = torch.zeros((2, 3), dtype=torch.long)

    assert embedding.weights_streamed is False
    out = embedding(ids)

    assert out.shape == (2, 3, 4)
    assert out.dtype == torch.float8_e4m3fn
    assert torch.equal(out, torch.zeros_like(out))
    assert embedding.weight_scale.item() == 1.0


def test_placeholder_forward_gathers_from_attached_table(tmp_path: Path) -> None:
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )

    ids = torch.tensor([[0, 8], [3, 3]], dtype=torch.long)
    out = embedding(ids)

    assert out.shape == (2, 2, 2)
    assert out.dtype == torch.float8_e4m3fn
    assert torch.equal(out.reshape(-1, 2), full[ids.reshape(-1)])


# --------------------------------------------------------------------------- #
# gather_into: direct destination validation (dtype/device/shape/
# contiguity) and value/pointer parity with an attached table.
# --------------------------------------------------------------------------- #


def test_gather_into_rejects_a_destination_with_the_wrong_shape(
    tmp_path: Path,
) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    ids = torch.tensor([0, 8], dtype=torch.long)
    destination = torch.empty((2, 3), dtype=torch.float8_e4m3fn)  # want (2, 2)

    with pytest.raises(ValueError, match="destination shape"):
        embedding.gather_into(ids, destination)


def test_gather_into_rejects_a_non_contiguous_destination(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    ids = torch.tensor([0, 8], dtype=torch.long)
    backing = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
    destination = backing[:, ::2]  # shape (2, 2) but not contiguous
    assert not destination.is_contiguous()

    with pytest.raises(ValueError, match="must be contiguous"):
        embedding.gather_into(ids, destination)


def test_gather_into_rejects_a_dtype_mismatched_destination_against_an_attached_table(
    tmp_path: Path,
) -> None:
    """The table is FP8; a BF16 destination must be rejected rather than
    silently reinterpreted or upcast."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    ids = torch.tensor([0, 8], dtype=torch.long)
    destination = torch.empty((2, 2), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="table dtype"):
        embedding.gather_into(ids, destination)


def test_gather_into_rejects_a_dtype_mismatched_destination_against_placeholder() -> (
    None
):
    """No table attached (dummy load): the placeholder's own fallback dtype
    still gates the destination, even though nothing is gathered."""
    embedding = ple_mmap.MmapNgramEmbedding(16, 4)
    ids = torch.zeros((2,), dtype=torch.long)
    destination = torch.empty((2, 4), dtype=torch.bfloat16)  # placeholder is fp8

    with pytest.raises(ValueError, match="placeholder dtype"):
        embedding.gather_into(ids, destination)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_gather_into_rejects_a_destination_on_a_different_device(
    tmp_path: Path,
) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    ids = torch.tensor([0, 8], dtype=torch.long, device="cuda")
    destination = torch.empty((2, 2), dtype=torch.float8_e4m3fn, device="cpu")

    with pytest.raises(ValueError, match="destination device"):
        embedding.gather_into(ids, destination)


def test_gather_into_writes_fp8_rows_directly_into_the_provided_destination(
    tmp_path: Path,
) -> None:
    """Value parity with the naive `forward()` path, but through a
    caller-owned destination whose identity/data_ptr must survive the
    call unchanged -- this is the V2 staging contract."""
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    ids = torch.tensor([[0, 8], [3, 3]], dtype=torch.long)
    destination = torch.empty((2, 2, 2), dtype=torch.float8_e4m3fn)
    ptr_before = destination.data_ptr()

    embedding.gather_into(ids, destination)

    assert destination.data_ptr() == ptr_before
    assert torch.equal(destination.reshape(-1, 2), full[ids.reshape(-1)])


def test_gather_into_writes_bf16_rows_directly_into_the_provided_destination(
    tmp_path: Path,
) -> None:
    full = _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=9,
        parts=3,
        cols=2,
        scale=0.5,
        write_scale=False,
        table_dtype=torch.bfloat16,
    )
    shard_map = ple_mmap.discover_shards(str(tmp_path))
    embedding = ple_mmap.MmapNgramEmbedding(9, 2)
    ple_mmap._attach_table(
        embedding,
        shard_map[0],
        split_ngram_parts=3,
        layer_idx=0,
        model_path=str(tmp_path),
    )
    ids = torch.tensor([[0, 8], [3, 3]], dtype=torch.long)
    destination = torch.empty((2, 2, 2), dtype=torch.bfloat16)
    ptr_before = destination.data_ptr()

    embedding.gather_into(ids, destination)

    assert destination.data_ptr() == ptr_before
    assert torch.equal(destination.reshape(-1, 2), full[ids.reshape(-1)])


def test_gather_into_zeros_the_destination_in_place_for_a_dummy_load() -> None:
    """No table attached and weights never streamed: gather_into must zero
    the caller's destination in place rather than gather from nothing."""
    embedding = ple_mmap.MmapNgramEmbedding(16, 4)
    ids = torch.zeros((2,), dtype=torch.long)
    destination = torch.full((2, 4), 3.0, dtype=torch.float8_e4m3fn)
    ptr_before = destination.data_ptr()

    embedding.gather_into(ids, destination)

    assert destination.data_ptr() == ptr_before
    assert torch.equal(destination, torch.zeros_like(destination))


def test_gather_into_raises_when_weights_streamed_but_no_table_was_ever_built() -> None:
    """load_weights ran (weights_streamed=True) but build_tables never
    attached a table -- this is a broken construction sequence, not a
    legitimate dummy load, and must raise rather than silently zero."""
    embedding = ple_mmap.MmapNgramEmbedding(16, 4)
    embedding.weights_streamed = True
    ids = torch.zeros((2,), dtype=torch.long)
    destination = torch.empty((2, 4), dtype=torch.float8_e4m3fn)

    with pytest.raises(RuntimeError, match="build_tables did not"):
        embedding.gather_into(ids, destination)


# --------------------------------------------------------------------------- #
# Input-preparation timing instrument (ids_d2h_wait_ms / gather_ms / h2d_ms)
# --------------------------------------------------------------------------- #


def test_input_prep_timing_instrument_logs_the_rate_limited_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The split line fires only once per _LOG_INTERVAL_S — poked directly
    via ``_prep_last_log``, the same way this file already pokes
    ``table._last_log``, rather than sleeping or monkeypatching
    time.monotonic — and reports the window's sample count plus a p50 and
    p99 split of the three CPU-blocking components and the pinned= arm
    field."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    logged = _record_info(monkeypatch)
    ids = torch.tensor([0, 8], dtype=torch.long)

    embedding(ids)  # first call: buffers a sample, does not log yet
    assert logged == []

    embedding._prep_last_log = 0.0
    embedding(ids)

    assert len(logged) == 1
    msg, args = logged[0]
    assert msg.startswith("PLE mmap input prep:")
    assert "pinned" in msg
    # rows=/p99_ms= are the gather-side log's own keys (MmapPleTable._record);
    # prep_rows=/prep_p99_ms= remain namespaced from those keys.
    assert msg.count("rows=") == 1
    assert msg.count("p99_ms=") == 1
    for key in ("ids_d2h_wait_ms=", "gather_ms=", "h2d_call_ms="):
        assert f"p50_{key}" in msg and f"p99_{key}" in msg
        assert msg.count(key) == 2
    (
        prep_rows,
        n,
        p50_ms,
        p50_ids_d2h_wait_ms,
        p50_gather_ms,
        p50_h2d_ms,
        prep_p99_ms,
        p99_ids_d2h_wait_ms,
        p99_gather_ms,
        p99_h2d_ms,
        pinned_engaged,
        pinned_total,
    ) = args
    assert prep_rows == 2 * ids.numel()
    assert n == 2  # both calls landed in this window
    for value in (
        p50_ms,
        p50_ids_d2h_wait_ms,
        p50_gather_ms,
        p50_h2d_ms,
        prep_p99_ms,
        p99_ids_d2h_wait_ms,
        p99_gather_ms,
        p99_h2d_ms,
    ):
        assert value >= 0.0
    assert prep_p99_ms == pytest.approx(
        p99_ids_d2h_wait_ms + p99_gather_ms + p99_h2d_ms
    )
    assert p50_ms == pytest.approx(p50_ids_d2h_wait_ms + p50_gather_ms + p50_h2d_ms)
    # p99 indexes at or above p50 into a sorted window, so it can never come
    # out below it.
    assert prep_p99_ms >= p50_ms
    # pinned= is engaged/total across the window, not the p99 sample's own
    # flag -- neither call here engaged pinned staging, so 0 of both.
    assert (pinned_engaged, pinned_total) == (0, n)


# --------------------------------------------------------------------------- #
# Pinned H2D staging (VLLM_PLE_MMAP_PINNED)
# --------------------------------------------------------------------------- #


def test_attach_table_snapshots_pinned_off_by_default(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)

    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )

    assert embedding.pinned is False


def test_attach_table_snapshots_pinned_on_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """embedding.pinned is env AND torch.cuda.is_available() at attach time
    (folded in so forward only re-pays the cheap device-type check) —
    is_available() is mocked True so this asserts the env-reflection half
    of that AND deterministically, independent of whether this box has a
    real CUDA device."""
    monkeypatch.setenv("VLLM_PLE_MMAP_PINNED", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)

    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )

    assert embedding.pinned is True


def test_attach_table_snapshots_pinned_off_without_cuda_even_if_env_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CPU-path guarantee: on a CUDA-less box, torch.cuda.is_available()
    is False at attach time, so the AND keeps the gate dead regardless of
    the env value."""
    monkeypatch.setenv("VLLM_PLE_MMAP_PINNED", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)

    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )

    assert embedding.pinned is False


def test_pinned_flag_on_cpu_ids_takes_the_unchanged_pageable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is `self.pinned and ids.device.type == "cuda"`: on CPU ids
    (every other test in this suite) it must short-circuit before ever
    touching the pinned-allocation indirection, proving the flag cannot
    corrupt — or even reach — the CPU-only path. torch.cuda.is_available()
    is mocked True so embedding.pinned is deterministically True here
    regardless of this box's real hardware -- the point under test is the
    device-type check, not is_available()."""
    monkeypatch.setenv("VLLM_PLE_MMAP_PINNED", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    assert embedding.pinned is True
    calls: list[torch.Tensor] = []

    def _spy(cpu_tensor: torch.Tensor) -> torch.Tensor:
        calls.append(cpu_tensor)
        return cpu_tensor

    monkeypatch.setattr(ple_mmap, "_pin_host_tensor", _spy)

    ids = torch.tensor([[0, 8], [3, 3]], dtype=torch.long)
    out = embedding(ids)

    assert calls == []
    assert torch.equal(out.reshape(-1, 2), full[ids.reshape(-1)])


def test_stage_pinned_falls_back_when_the_indirection_raises(
    monkeypatch: pytest.MonkeyPatch, caplog_vllm: pytest.LogCaptureFixture
) -> None:
    """A RuntimeError from the allocation indirection must not fail the
    request — the caller gets its own pageable tensor back, unchanged.

    Goes through the REAL warning_once (a real logging.Handler via
    caplog_vllm, not a monkeypatched recorder that bypasses its lru_cache),
    and each of the 10 raised errors carries a DIFFERENT message (real
    torch allocator errors interpolate the requested/available byte counts,
    which drift call to call): this is the only shape of test that can
    catch _stage_pinned passing str(exc) (varies -> dedup defeated, fires
    every call) instead of type(exc).__name__ (constant -> dedups
    correctly -> fires once)."""

    call_count = 0

    def _raise(cpu_tensor: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        raise RuntimeError(f"synthetic cudaHostAlloc exhaustion: {call_count * 4096} B")

    monkeypatch.setattr(ple_mmap, "_pin_host_tensor", _raise)
    cpu_tensor = torch.arange(6, dtype=torch.uint8).reshape(3, 2)

    with caplog_vllm.at_level(
        logging.WARNING, logger="vllm.models.qwen4_exp.nvidia.ple_mmap"
    ):
        for _ in range(10):
            staged, engaged = ple_mmap._stage_pinned(cpu_tensor)
            assert engaged is False
            assert staged is cpu_tensor
            assert torch.equal(staged, cpu_tensor)

    emissions = [
        r for r in caplog_vllm.records if "pinned H2D staging failed" in r.getMessage()
    ]
    assert len(emissions) == 1


def test_stage_pinned_copies_values_through_a_fake_pinned_indirection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The indirection is monkeypatched UNCONDITIONALLY, even on a
    CUDA-capable box: a real pin_memory=True call here would init a CUDA
    context, which this CPU-only suite must never trigger. A plain CPU
    tensor stands in as the "pinned" buffer to prove _stage_pinned's own
    copy semantics independent of the real allocator."""
    monkeypatch.setattr(ple_mmap, "_pin_host_tensor", torch.empty_like)
    cpu_tensor = torch.arange(6, dtype=torch.uint8).reshape(3, 2)

    staged, engaged = ple_mmap._stage_pinned(cpu_tensor)

    assert engaged is True
    assert staged is not cpu_tensor
    assert torch.equal(staged, cpu_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_pinned_h2d_matches_pageable_h2d_on_a_real_cuda_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag on vs off, same rows, on a real CUDA device: proves the pinned
    arm cannot silently corrupt values relative to the pageable arm it
    replaces. Gated on torch.cuda.is_available() alone, so it skips in
    CPU-only CI; it is the one test in this file that allocates real pinned
    memory and issues a real H2D rather than standing the allocator in.

    Also spies on _pin_host_tensor (wrapping the real allocator, not
    replacing it) to prove the knob actually ENGAGED the mechanism rather
    than merely producing byte-identical output while dead — value
    equality alone would still pass with a permanently inert gate."""
    real_pin_host_tensor = ple_mmap._pin_host_tensor
    calls: list[int] = []

    def _spy(cpu_tensor: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return real_pin_host_tensor(cpu_tensor)

    monkeypatch.setattr(ple_mmap, "_pin_host_tensor", _spy)

    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    ids = torch.tensor([0, 8, 3, 3], dtype=torch.long, device="cuda")

    outs = []
    call_counts = []
    for pinned in (False, True):
        monkeypatch.setenv("VLLM_PLE_MMAP_PINNED", "1" if pinned else "0")
        calls.clear()
        embedding = _attached_embedding(
            tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
        )
        outs.append(embedding(ids).cpu())
        call_counts.append(len(calls))
        assert embedding.table is not None
        embedding.table.close()  # each arm builds its own table; don't leak it

    assert call_counts[0] == 0  # pageable arm never touches the indirection
    assert call_counts[1] >= 1  # pinned arm actually engaged it
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(outs[0].reshape(-1, 2), full[ids.cpu()])


class _SyntheticPinnedAllocError(RuntimeError):
    """Disjoint RuntimeError subclass for a synthetic allocator failure,
    raised by both
    test_pinned_allocation_failure_latches_off_for_the_rest_of_the_instance
    and
    test_input_prep_timing_mixed_window_reports_pinned_engagement.

    _stage_pinned's warning_once call dedups its process-wide lru_cache key
    on (msg, type(exc).__name__). A bare RuntimeError here would share that
    key with test_stage_pinned_falls_back_when_the_indirection_raises's own
    bare RuntimeError, so whichever of the two tests ran first would consume
    the shared dedup slot and starve the other regardless of run order
    (reproduced: running this latch test before the dedup test makes the
    dedup test observe zero emissions instead of one). A distinct exception
    type keeps the two tests' cache keys independent -- do not introduce a
    THIRD bare RuntimeError into a test reaching _stage_pinned; raise this
    class instead.
    """


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_pinned_allocation_failure_latches_off_for_the_rest_of_the_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistently exhausted pinned allocator (e.g. cudaHostAlloc) would
    otherwise be retried every forward -- a silent per-step tax on the ITL
    path (raise, catch, rebuild the fallback tensor) that a tester would
    only ever see once, in the single warning_once emission. forward
    latches self.pinned = False the first time _stage_pinned reports
    not-engaged, so the indirection is tried at most once per instance.

    Needs a REAL CUDA device for `ids` -- the pinned gate is
    `self.pinned and ids.device.type == "cuda"`, unreachable from CPU-only
    ids (the same reason _stage_pinned is unit-tested directly rather than
    through forward elsewhere in this file)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = 0

    def _raise(cpu_tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        raise _SyntheticPinnedAllocError("synthetic cudaHostAlloc exhaustion")

    monkeypatch.setattr(ple_mmap, "_pin_host_tensor", _raise)
    monkeypatch.setenv("VLLM_PLE_MMAP_PINNED", "1")
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    assert embedding.pinned is True
    ids = torch.tensor([0, 8, 3, 3], dtype=torch.long, device="cuda")

    out1 = embedding(ids)
    assert embedding.pinned is False  # latched off after the failed call

    out2 = embedding(ids)

    assert calls == 1  # the indirection was never retried on the second call
    for out in (out1, out2):
        assert torch.equal(out.cpu().reshape(-1, 2), full[ids.cpu()])

    assert embedding.table is not None
    embedding.table.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_input_prep_timing_mixed_window_reports_pinned_engagement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The latch is what makes a mixed window reachable at all: the first
    call here engages pinned staging, the second's allocation fails and
    latches self.pinned off for the rest of the run. Keying the
    forward-timing log's pinned= field on the p99 SAMPLE's own flag would
    report whichever of those two calls happened to sort last as the
    window's total_ms max, not the window's actual engagement; this test
    drives exactly that mixed shape and asserts the rendered pair is the
    window's true engaged/total count regardless of which call was
    biggest."""
    monkeypatch.setenv("VLLM_PLE_MMAP_PINNED", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    assert embedding.pinned is True
    real_pin_host_tensor = ple_mmap._pin_host_tensor
    call_count = 0

    def _pin_once_then_fail(cpu_tensor: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return real_pin_host_tensor(cpu_tensor)
        raise _SyntheticPinnedAllocError("synthetic cudaHostAlloc exhaustion")

    monkeypatch.setattr(ple_mmap, "_pin_host_tensor", _pin_once_then_fail)
    logged = _record_info(monkeypatch)
    ids = torch.tensor([0, 8, 3, 3], dtype=torch.long, device="cuda")

    embedding(ids)  # call 1: pinned engages
    assert embedding.pinned is True
    assert logged == []  # buffers a sample, does not log yet

    embedding._prep_last_log = 0.0
    embedding(ids)  # call 2: allocation fails, latches pinned off
    assert embedding.pinned is False

    assert len(logged) == 1
    msg, args = logged[0]
    pinned_engaged, pinned_total = args[-2:]
    assert (pinned_engaged, pinned_total) == (1, 2)

    assert embedding.table is not None
    embedding.table.close()


# --------------------------------------------------------------------------- #
# load_weights interception + loaded-set contract
# --------------------------------------------------------------------------- #


def _mmap_ngram_module_for_load_test(
    vocab: int = 8, cols: int = 2
) -> Qwen4ExpNGramEmbedding:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(module)
    module.layer_name = "model.layers.1.ple"
    module.split_ngram_parts = 2
    module.register_buffer("layer_multipliers", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_offsets", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_vocab_sizes", torch.zeros(1, dtype=torch.long))
    module.ngram_embedding = ple_mmap.MmapNgramEmbedding(vocab, cols)
    return module


def test_ngram_embedding_mmap_load_weights_intercepts_shards_and_scale() -> None:
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    shard_0 = shard_0.to(torch.float8_e4m3fn)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    shard_1 = shard_1.to(torch.float8_e4m3fn)
    weight_scale = torch.tensor([0.25], dtype=torch.bfloat16)

    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("ngram_embedding.weight_scale", weight_scale),
        ]
    )

    assert loaded == {"ngram_embedding.weight", "ngram_embedding.weight_scale"}
    assert torch.equal(module.ngram_embedding.weight_scale, weight_scale)
    assert module.ngram_embedding.weight_scale_loaded is True
    assert module.ngram_embedding.weights_streamed is True


def test_forward_raises_named_error_when_streamed_but_build_tables_never_ran() -> None:
    """A real load_weights pass over PLE shards, with
    build_tables never called, must not silently serve fp8 zeros — that
    would be indistinguishable from a legitimate --load-format dummy probe.
    weights_streamed=True (set once load_weights sees a real shard tensor)
    is exactly the signal that distinguishes the two, and forward must
    raise the named error instead."""
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    shard_1 = shard_1.to(torch.float8_e4m3fn)
    weight_scale = torch.tensor([0.25], dtype=torch.bfloat16)

    module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("ngram_embedding.weight_scale", weight_scale),
        ]
    )

    assert module.ngram_embedding.table is None  # build_tables never ran
    with pytest.raises(
        RuntimeError,
        match="PLE mmap table not initialized",
    ):
        module.ngram_embedding(torch.zeros((2,), dtype=torch.long))


def test_ngram_embedding_mmap_load_weights_never_retains_shard_tensors() -> None:
    """Invariant 3: nothing may retain the full table, including transiently
    on the placeholder — it has no .weight attribute to retain into."""
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    shard_0 = shard_0.to(torch.float8_e4m3fn)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    shard_1 = shard_1.to(torch.float8_e4m3fn)

    module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
        ]
    )

    assert not hasattr(module.ngram_embedding, "weight")
    assert module.ngram_embedding.table is None  # only build_tables ever sets it


def test_ngram_embedding_mmap_load_weights_rejects_mismatched_shard_shape() -> None:
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)

    with pytest.raises(ValueError, match=r"Shape mismatch for PLE embedding shard 0"):
        module.load_weights([("ngram_embedding.shard_0.weight", torch.zeros(3, 2))])


def test_ngram_embedding_mmap_load_weights_rejects_same_instance_reload(
    tmp_path: Path,
) -> None:
    """A same-path iterator reload must not mix
    checkpoint A's attached table with checkpoint B's scale. Once a table
    is attached, load_weights must fail closed on any later invocation on
    the SAME module — before consuming or mutating checkpoint B's iterator
    — leaving A's table identity, scale, weights_streamed, and a
    representative gathered row untouched. build_tables' own model_path
    check never catches this: the path never changes, so it silently
    reuses the attached table while an unguarded load_weights would have
    already stomped weight_scale/weights_streamed."""
    full_a = _write_ple_layer(
        tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25
    )
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)
    module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", full_a[0:4]),
            ("ngram_embedding.shard_1.weight", full_a[4:8]),
            (
                "ngram_embedding.weight_scale",
                torch.tensor([0.25], dtype=torch.bfloat16),
            ),
        ]
    )
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, module.ngram_embedding, 2)}
    )
    ple_mmap.build_tables(_model_config(tmp_path), cc)
    embedding = module.ngram_embedding
    table_a = embedding.table
    assert table_a is not None
    scale_a = embedding.weight_scale.clone()
    ids = torch.tensor([0, 7], dtype=torch.long)
    row_a_before = embedding(ids).clone()
    assert torch.equal(row_a_before, full_a[ids])  # sanity: A's gather is correct

    # Checkpoint B: distinct shard values and scale, never written to
    # `tmp_path` — the bug is keyed on the SAME module already holding a
    # table, not on any particular directory, so build_tables must never
    # even be reached to reproduce it.
    full_b = _synthetic_weight(8, 2, layer_idx=1)
    consumed: list[str] = []

    def checkpoint_b_iter():
        for name, tensor in (
            ("ngram_embedding.shard_0.weight", full_b[0:4]),
            ("ngram_embedding.shard_1.weight", full_b[4:8]),
            (
                "ngram_embedding.weight_scale",
                torch.tensor([0.75], dtype=torch.bfloat16),
            ),
        ):
            consumed.append(name)
            yield name, tensor

    with pytest.raises(RuntimeError, match="already has a table attached"):
        module.load_weights(checkpoint_b_iter())

    assert consumed == []  # raised before the iterator was ever advanced
    assert embedding.table is table_a  # A's table identity is unchanged
    assert torch.equal(embedding.weight_scale, scale_a)  # A's scale is unchanged
    assert embedding.weights_streamed is True  # unchanged from A's load
    assert torch.equal(embedding(ids), row_a_before)  # A's gathered row is unchanged

    # Repeated idempotent build_tables calls with no further weight load
    # must keep working, unaffected by the rejected reload above.
    ple_mmap.build_tables(_model_config(tmp_path), cc)
    assert embedding.table is table_a


# --------------------------------------------------------------------------- #
# build_tables: construction hook, fail-closed validation, per-layer keying.
# --------------------------------------------------------------------------- #


def _fake_ple_layer(
    layer_idx: int,
    embedding: ple_mmap.MmapNgramEmbedding,
    split_ngram_parts: int,
    *,
    mmap_staging: torch.Tensor | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        layer_idx=layer_idx,
        ple_embedding=SimpleNamespace(
            ngram_embedding=embedding,
            split_ngram_parts=split_ngram_parts,
            _mmap_staging=mmap_staging,
        ),
    )


def _loaded_placeholder(
    vocab: int, cols: int, scale: float
) -> ple_mmap.MmapNgramEmbedding:
    embedding = ple_mmap.MmapNgramEmbedding(vocab, cols)
    ple_mmap.set_weight_scale(
        embedding, torch.tensor([scale], dtype=torch.bfloat16), torch.device("cpu")
    )
    return embedding


def _model_config(directory: Path) -> SimpleNamespace:
    return SimpleNamespace(model_weights=str(directory), model="ignored", revision=None)


def test_build_tables_wires_the_tuning_knobs_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VLLM_PLE_MMAP_WORKERS/_CHUNK/_READAHEAD/_SERIAL must reach the
    attached MmapPleTable's workers/chunk/readahead/serial in the right
    order — a swapped-args regression would still construct a table, just
    with the wrong concurrency knobs, and nothing else would notice."""
    monkeypatch.setenv("VLLM_PLE_MMAP_WORKERS", "3")
    monkeypatch.setenv("VLLM_PLE_MMAP_CHUNK", "7")
    monkeypatch.setenv("VLLM_PLE_MMAP_READAHEAD", "11")
    monkeypatch.setenv("VLLM_PLE_MMAP_SERIAL", "13")
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})

    ple_mmap.build_tables(_model_config(tmp_path), cc)

    assert emb.table is not None
    assert emb.table.workers == 3
    assert emb.table.chunk == 7
    assert emb.table.readahead == 11
    assert emb.table.serial == 13


def test_build_tables_attaches_a_table_per_ple_layer_without_cross_contamination(
    tmp_path: Path,
) -> None:
    """(b): build_tables must key tables per layer prefix, never a module
    global — attaching layer 0's table must not affect layer 1's."""
    full0 = _write_ple_layer(
        tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25
    )
    full1 = _write_ple_layer(
        tmp_path, layer_idx=1, vocab=20, parts=4, cols=2, scale=0.75
    )

    emb0 = _loaded_placeholder(10, 2, 0.25)
    emb1 = _loaded_placeholder(20, 2, 0.75)
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": _fake_ple_layer(0, emb0, 3),
            "b.ple": _fake_ple_layer(1, emb1, 4),
        }
    )
    model_config = _model_config(tmp_path)

    ple_mmap.build_tables(model_config, cc)

    assert emb0.table is not None and emb1.table is not None
    assert emb0.table is not emb1.table
    out0 = emb0(torch.tensor([0, 9], dtype=torch.long))
    out1 = emb1(torch.tensor([0, 19], dtype=torch.long))
    assert torch.equal(out0, full0[[0, 9]])
    assert torch.equal(out1, full1[[0, 19]])
    # No cross-wiring: layer 0's output must not equal layer 1's data.
    assert not torch.equal(out0.reshape(-1), full1[[0, 9]].reshape(-1))


def test_build_tables_ignores_layers_without_an_mmap_placeholder(
    tmp_path: Path,
) -> None:
    """A layer whose ngram_embedding is not our placeholder (env-off) must be
    skipped, never mistaken for a PLE layer needing a table."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    cc = SimpleNamespace(
        static_forward_context={
            "not_a_ple_layer": SimpleNamespace(),
            "no_embedding_attr": SimpleNamespace(ple_embedding=None),
        }
    )
    model_config = _model_config(tmp_path)

    ple_mmap.build_tables(model_config, cc)  # must not raise


def test_build_tables_raises_when_a_ple_layer_has_no_shards_on_disk(
    tmp_path: Path,
) -> None:
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="no shard tensors for layer 0"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_raises_on_shard_width_mismatch(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 4, 0.25)  # embedding_dim disagrees with on-disk cols
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="shard width"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_refuses_a_uniformly_e5m2_ple_table(tmp_path: Path) -> None:
    """F8_E5M2 was never added to _PLE_DTYPES because is_fp8() does not
    recognize it (dequant would silently never fire) — a UNIFORMLY-e5m2
    checkpoint (not a mixed-dtype one, already covered by discover_shards'
    own check) must still be refused, from BOTH validate_shards_for
    (construction-time) and build_tables (load-time), with the same
    dtype diagnosis.

    Also covered: after a SEPARATE successful e4m3 attach, the
    placeholder's own torch_dtype tracks the attached table's dtype
    exactly.
    """
    assert not is_fp8(torch.float8_e5m2)

    vocab, parts, cols = 10, 3, 2
    shard_size = (vocab + parts - 1) // parts
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    for shard_index in range(parts):
        start = shard_index * shard_size
        rows = max(0, min(shard_size, vocab - start))
        tensors = {
            f"{prefix}.shard_{shard_index}.weight": torch.zeros(rows, cols).to(
                torch.float8_e5m2
            )
        }
        if shard_index == 0:
            tensors[f"{prefix}.weight_scale"] = torch.tensor(
                [0.5], dtype=torch.bfloat16
            )
        safetensors.torch.save_file(
            tensors, str(tmp_path / f"e5m2-{shard_index:05d}.safetensors")
        )

    model_config = _model_config(tmp_path)
    with pytest.raises(RuntimeError, match=r"F8_E5M2"):
        ple_mmap.validate_shards_for(
            model_config, "model.language_model.layers.0.ple", head_dim=cols
        )

    emb = _loaded_placeholder(vocab, cols, 0.5)
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, emb, parts)}
    )
    with pytest.raises(RuntimeError, match=r"F8_E5M2"):
        ple_mmap.build_tables(model_config, cc)

    # a fully separate, successful e4m3 attach.
    e4m3_dir = tmp_path / "e4m3"
    e4m3_dir.mkdir()
    _write_ple_layer(
        e4m3_dir, layer_idx=0, vocab=vocab, parts=parts, cols=cols, scale=0.5
    )
    emb2 = _loaded_placeholder(vocab, cols, 0.5)
    cc2 = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, emb2, parts)}
    )
    ple_mmap.build_tables(_model_config(e4m3_dir), cc2)

    assert emb2.table is not None
    assert emb2.table.torch_dtype is emb2.torch_dtype


def test_build_tables_attaches_a_bf16_table_and_forward_gathers_it_value_exact(
    tmp_path: Path,
) -> None:
    """Intel AutoRound W4A16 exports pass the PLE table through as
    unquantized BF16 with no weight_scale on disk: a real streamed load sets
    weights_streamed True but weight_scale_loaded stays False (there is no
    scale tensor to stream) — the True/False quadrant that the
    streamed-loader's fail-closed error would otherwise refuse. For a
    requires_scale=False dtype this must still attach and serve real values
    through the full load_weights -> build_tables -> forward chain, not
    raise."""
    full = _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=8,
        parts=2,
        cols=2,
        scale=0.0,
        write_scale=False,
        table_dtype=torch.bfloat16,
    )
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)
    module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", full[0:4]),
            ("ngram_embedding.shard_1.weight", full[4:8]),
        ]
    )
    assert module.ngram_embedding.weights_streamed is True
    assert module.ngram_embedding.weight_scale_loaded is False

    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, module.ngram_embedding, 2)}
    )
    ple_mmap.build_tables(_model_config(tmp_path), cc)

    ids = torch.tensor([[0, 7], [3, 3]], dtype=torch.long)
    out = module.ngram_embedding(ids)

    assert out.dtype == torch.bfloat16
    assert torch.equal(out.reshape(-1, 2), full[ids.reshape(-1)])


def test_build_tables_raises_on_missing_shard_file(tmp_path: Path) -> None:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    full = _synthetic_weight(10, 2, layer_idx=0)
    # shard_size = ceil(10/3) = 4; write shards 0 and 2, skip shard 1 entirely.
    safetensors.torch.save_file(
        {
            f"{prefix}.shard_0.weight": full[0:4],
            f"{prefix}.weight_scale": torch.tensor([0.25], dtype=torch.bfloat16),
        },
        str(tmp_path / "shard0.safetensors"),
    )
    safetensors.torch.save_file(
        {f"{prefix}.shard_2.weight": full[8:10]},
        str(tmp_path / "shard2.safetensors"),
    )
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match=r"missing shard\(s\) \[1\]"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_raises_when_weight_scale_was_never_loaded(tmp_path: Path) -> None:
    """Rows streamed but weight_scale absent from the same iterable is a
    broken or truncated weight iterator, not an unstreamed family — this
    must stay in the fail-closed True/False quadrant (weights_streamed
    True, weight_scale_loaded False), never fall back to a header read."""
    full = _write_ple_layer(
        tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25, write_scale=True
    )
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)

    module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", full[0:4]),
            ("ngram_embedding.shard_1.weight", full[4:8]),
        ]
    )

    assert module.ngram_embedding.weights_streamed is True
    assert module.ngram_embedding.weight_scale_loaded is False

    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, module.ngram_embedding, 2)}
    )
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="weight_scale was never loaded"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_falls_back_to_header_scale_when_family_never_streamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A layer whose ngram_embedding family was never routed to this
    worker never streams anything, so weights_streamed stays False along
    with weight_scale_loaded (the False/False quadrant) — that must attach
    off a direct header read and warn, not raise the streamed-loader's
    fail-closed error. The on-disk scale is F32 (0.1) to exercise the
    no-cast rule: casting to the placeholder's default bf16 would silently
    rewrite 0.1 to a different float and trip on nothing, masking the bug.
    """
    _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=8,
        parts=2,
        cols=2,
        scale=0.1,
        write_scale=True,
        scale_dtype=torch.float32,
    )
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    assert embedding.weights_streamed is False
    assert embedding.weight_scale_loaded is False
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, embedding, 2)}
    )
    model_config = _model_config(tmp_path)
    warnings = _record_plain_warnings(monkeypatch)

    ple_mmap.build_tables(model_config, cc)

    assert embedding.weight_scale_loaded is True
    assert embedding.weight_scale.dtype is torch.float32
    assert torch.equal(
        embedding.weight_scale, torch.tensor([0.1], dtype=torch.float32).squeeze()
    )
    assert len(warnings) == 1
    assert warnings[0][1] == (0,)  # layer_idx


def test_build_tables_raises_on_a_non_scalar_weight_scale(tmp_path: Path) -> None:
    """A per-channel (multi-element) weight_scale would silently truncate to
    its first element in _read_scale: _validate_layer_shards must refuse it
    up front, before the False/False fallback (or any other quadrant) ever
    gets a chance to attach off a truncated value.
    """
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    vocab, parts, cols = 8, 2, 2
    shard_size = (vocab + parts - 1) // parts
    full = _synthetic_weight(vocab, cols)
    for shard_index in range(parts):
        start = shard_index * shard_size
        rows = max(0, min(shard_size, vocab - start))
        tensors: dict[str, torch.Tensor] = {
            f"{prefix}.shard_{shard_index}.weight": full[start : start + rows]
        }
        if shard_index == 0:
            tensors[f"{prefix}.weight_scale"] = torch.tensor(
                [0.1, 0.2, 0.3, 0.4], dtype=torch.float32
            )
        safetensors.torch.save_file(
            tensors, str(tmp_path / f"model-ple-0-{shard_index:05d}.safetensors")
        )
    embedding = ple_mmap.MmapNgramEmbedding(vocab, cols)
    assert embedding.weights_streamed is False
    assert embedding.weight_scale_loaded is False
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, embedding, parts)}
    )
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match=r"per-channel"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_raises_when_no_weight_scale_tensor_exists_on_disk(
    tmp_path: Path,
) -> None:
    _write_ple_layer(
        tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25, write_scale=False
    )
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="no ngram_embedding.weight_scale"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_raises_on_scale_mismatch_between_streamed_and_header(
    tmp_path: Path,
) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 2, scale=0.5)  # disagrees with the on-disk 0.25
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="weight_scale mismatch"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_prewarm_is_bounded_by_available_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PREWARM=1 must call MmapPleTable.prewarm exactly once, with
    the clamped bound; PREWARM=0 must never call it at all —
    the env gate, not just the bound math, is under test."""
    prewarm_calls: list[int] = []
    real_prewarm = ple_mmap.MmapPleTable.prewarm

    def spy_prewarm(self: ple_mmap.MmapPleTable, max_bytes: int) -> int:
        prewarm_calls.append(max_bytes)
        return real_prewarm(self, max_bytes)

    monkeypatch.setattr(ple_mmap.MmapPleTable, "prewarm", spy_prewarm)
    # Pretend memory is scarce: bound must clamp to 0, not raise.
    monkeypatch.setattr(ple_mmap, "_mem_available_bytes", lambda: 1 << 20)

    monkeypatch.setenv("VLLM_PLE_MMAP_PREWARM", "1")
    on_dir = tmp_path / "on"
    on_dir.mkdir()
    _write_ple_layer(on_dir, layer_idx=0, vocab=10, parts=1, cols=2, scale=0.25)
    emb_on = _loaded_placeholder(10, 2, 0.25)
    cc_on = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, emb_on, 1)}
    )

    ple_mmap.build_tables(_model_config(on_dir), cc_on)

    assert emb_on.table is not None
    assert prewarm_calls == [0]  # 1 MiB available - 8 GiB headroom, clamped to 0

    monkeypatch.setenv("VLLM_PLE_MMAP_PREWARM", "0")
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    _write_ple_layer(off_dir, layer_idx=0, vocab=10, parts=1, cols=2, scale=0.25)
    emb_off = _loaded_placeholder(10, 2, 0.25)
    cc_off = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, emb_off, 1)}
    )

    ple_mmap.build_tables(_model_config(off_dir), cc_off)

    assert emb_off.table is not None
    assert prewarm_calls == [0]  # unchanged: prewarm was not called again


def test_build_tables_is_idempotent_and_reuses_an_already_attached_table(
    tmp_path: Path,
) -> None:
    """Qwen4ExpForCausalLM.load_weights and
    Qwen4ExpForConditionalGeneration.load_weights both call build_tables on
    a real ConditionalGeneration load (the wrapper composes CausalLM
    internally), so a second call for the same layer must not re-attach —
    or leak the first table's ThreadPool."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    ple_mmap.build_tables(model_config, cc)
    first_table = emb.table
    assert first_table is not None

    ple_mmap.build_tables(model_config, cc)  # second call: must not raise

    assert emb.table is first_table  # skipped, not rebuilt


def test_build_tables_second_call_skips_the_discover_shards_header_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """with everything already attached, the redundant second call
    (CausalLM's build_tables call, then ConditionalGeneration's) must not
    re-scan every checkpoint file's header — only resolve_model_path (cheap:
    a directory check) runs on the empty-pending path."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    calls = 0
    real_discover_shards = ple_mmap.discover_shards

    def counting_discover_shards(path: str):
        nonlocal calls
        calls += 1
        return real_discover_shards(path)

    monkeypatch.setattr(ple_mmap, "discover_shards", counting_discover_shards)

    ple_mmap.build_tables(model_config, cc)
    ple_mmap.build_tables(model_config, cc)

    assert calls == 1


def test_build_tables_raises_when_reloaded_against_a_different_checkpoint(
    tmp_path: Path,
) -> None:
    """gpu_model_runner reload_weights can repoint model_config.model
    at a new checkpoint and re-call load_weights on the SAME live model.
    Silently keeping the already-attached table would serve checkpoint A's
    mmap rows against checkpoint B's scale — fail closed instead."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_ple_layer(dir_a, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    _write_ple_layer(dir_b, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.75)

    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})

    ple_mmap.build_tables(_model_config(dir_a), cc)
    assert emb.table is not None
    assert emb.table.model_path == str(dir_a)

    with pytest.raises(RuntimeError, match="different checkpoint|reloading"):
        ple_mmap.build_tables(_model_config(dir_b), cc)


def test_mmap_ple_table_close_drops_memmaps_and_is_idempotent(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=1.0)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        3,
        2,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(tmp_path),
    )

    table.close()
    table.close()  # idempotent: must not raise

    assert all(mm is None for mm in table.mm)
    with pytest.raises(IndexError, match="shard"):
        table.gather(np.array([0], dtype=np.int64))


def test_del_on_a_half_constructed_table_raises_no_attributeerror(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """__init__ validates ``shards`` before setting any of the real table
    state, and __del__ unconditionally calls close() on whatever exists —
    so a table that never got past that check must still destruct cleanly
    instead of printing a suppressed "Exception ignored" AttributeError to
    stderr (the close()/fd/mm containers are set before the raise).
    """
    with pytest.raises(ValueError, match="no shards to build a table from"):
        ple_mmap.MmapPleTable(
            {},
            10,
            8,
            torch.float8_e4m3fn,
            workers=1,
            chunk=8,
            model_path="/nonexistent",
        )
    gc.collect()

    captured = capsys.readouterr()
    assert "Exception ignored" not in captured.err
    assert "AttributeError" not in captured.err


def test_attach_table_closes_a_stale_table_before_building_the_new_one(
    tmp_path: Path,
) -> None:
    """a direct _attach_table re-entry on an already-populated
    placeholder must close the old table (ThreadPool + memmaps) rather
    than leaking it, even though build_tables' own idempotency skip makes
    this unreachable through the normal path."""
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    stale_table = embedding.table
    assert stale_table is not None
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]

    ple_mmap._attach_table(
        embedding,
        layer_shards,
        split_ngram_parts=3,
        layer_idx=0,
        model_path=str(tmp_path),
    )

    assert embedding.table is not stale_table
    assert all(mm is None for mm in stale_table.mm)  # the old table was closed
    assert stale_table.pool._shutdown  # the old ThreadPool was shut down
    ids = torch.tensor([0, 8], dtype=torch.long)
    assert torch.equal(embedding(ids), full[ids])  # the new table still works


def test_attach_table_closes_the_table_when_the_attach_window_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure anywhere between constructing the table and handing it to
    the placeholder (here: prewarm) must not leak the table's ThreadPool,
    memmaps, or readahead fds — _attach_table must close what construction
    already opened before the exception propagates.
    """
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)
    monkeypatch.setenv("VLLM_PLE_MMAP_PREWARM", "1")
    monkeypatch.setenv("VLLM_PLE_MMAP_READAHEAD", "8")

    built: list[ple_mmap.MmapPleTable] = []
    real_init = ple_mmap.MmapPleTable.__init__

    def spying_init(
        self: ple_mmap.MmapPleTable, *args: object, **kwargs: object
    ) -> None:
        real_init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(ple_mmap.MmapPleTable, "__init__", spying_init)

    def _raise_prewarm(self: ple_mmap.MmapPleTable, max_bytes: int) -> int:
        raise RuntimeError("synthetic prewarm failure")

    monkeypatch.setattr(ple_mmap.MmapPleTable, "prewarm", _raise_prewarm)

    with pytest.raises(RuntimeError, match="synthetic prewarm failure"):
        ple_mmap.build_tables(model_config, cc)

    assert emb.table is None
    assert len(built) == 1
    leaked = built[0]
    assert all(mm is None for mm in leaked.mm)
    assert leaked.pool._shutdown
    assert not leaked._fds


# --------------------------------------------------------------------------- #
# Directory resolution
# --------------------------------------------------------------------------- #


def test_resolve_model_path_uses_existing_directory_verbatim(tmp_path: Path) -> None:
    model_config = SimpleNamespace(model_weights=str(tmp_path), model="ignored")

    assert ple_mmap.resolve_model_path(model_config) == str(tmp_path)


def test_resolve_model_path_falls_back_to_offline_snapshot_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeHfApi:
        def snapshot_download(
            self, repo_id, revision, allow_patterns, local_files_only
        ):
            calls["repo_id"] = repo_id
            calls["revision"] = revision
            calls["allow_patterns"] = allow_patterns
            calls["local_files_only"] = local_files_only
            return "/resolved/snapshot/path"

    import vllm.transformers_utils.repo_utils as repo_utils

    monkeypatch.setattr(repo_utils, "hf_api", lambda: FakeHfApi())
    model_config = SimpleNamespace(
        model_weights="", model="some-org/some-ple-model", revision="deadbeef"
    )

    path = ple_mmap.resolve_model_path(model_config)

    assert path == "/resolved/snapshot/path"
    assert calls == {
        "repo_id": "some-org/some-ple-model",
        "revision": "deadbeef",
        "allow_patterns": ["*.safetensors"],
        "local_files_only": True,
    }


# --------------------------------------------------------------------------- #
# (a) env-on vs env-off FORWARD equivalence.
# env-on now stages explicitly: prepare_mmap_rows computes IDs through the
# SAME Qwen4ExpNGramEmbedding.compute_ngram_ids as the stock arm and
# gathers directly into the module's stable staging buffer; forward() then
# only reads that buffer. This proves the env-on path loads the RIGHT
# weights and gathers and dequantizes them the same way the stock
# PLEVocabParallelEmbedding path does — it does NOT independently verify
# the hashing math itself: a bug in compute_ngram_ids would move both arms
# identically and cancel out here. Hashing correctness is pinned separately
# by test_compute_ngram_ids_matches_golden_ids below.
# --------------------------------------------------------------------------- #


def test_env_on_off_forward_equivalence_fp8_and_dequantized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """env-off: a real stock PLEVocabParallelEmbedding under
    Qwen4ExpPLEFp8EmbeddingMethod (real FP8 weight + weight_scale
    Parameters, mirrors test_ple.py's _make_fp8_embedding_layer). env-on:
    an MmapNgramEmbedding placeholder attached to shard files holding the
    IDENTICAL weight values, staged explicitly via
    initialize_mmap_staging/prepare_mmap_rows, then read back through
    forward(). Same input_ids/query_start_loc/ngram_context on both sides;
    compared byte-equal at fp8 AND through _dequantize_embeddings to bf16.
    Proves weight-loading/gather/dequant equivalence between the two paths,
    not hashing correctness (both arms share the same compute_ngram_ids
    call, see module comment above).
    """
    config = _make_text_config()  # ngram_size=3, heads_per_ngram=2 -> 4 heads
    embedding_dim = 8  # head_dim = 2
    scale = 0.5

    # --- env-off: real stock FP8 VocabParallelEmbedding. ---
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        ignored_layers=[],
        weight_block_size=[128, 128],
    )
    stock = Qwen4ExpNGramEmbedding(
        config,
        embedding_dim,
        0,
        16,
        4,
        "model.layers.1.ple.ple_embedding",
        "model.layers.1.ple",
        quant_config=quant_config,
        params_dtype=torch.bfloat16,
    )
    assert isinstance(stock.ngram_embedding, PLEVocabParallelEmbedding)
    vocab = stock.ngram_embedding.org_vocab_size
    head_dim = stock.head_dim
    parts = stock.split_ngram_parts
    weight = _synthetic_weight(vocab, head_dim, layer_idx=1)
    stock.ngram_embedding.weight.data.copy_(weight)
    stock.ngram_embedding.weight_scale.data.copy_(
        torch.tensor([scale], dtype=torch.bfloat16)
    )

    input_ids = torch.tensor([1, 2], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2], dtype=torch.long)
    ngram_context = torch.zeros((1, 4), dtype=torch.long)

    # The stock branch of forward() routes ID generation through the
    # REGISTERED qwen4_exp_compute_ple_ngram_ids op (upstream architecture,
    # untouched by this PR). That op only has a CUDA dispatch-key impl
    # (matches upstream test_ple.py, which drives it as a plain function for
    # the same reason), so on a CUDA-platform host it cannot run against CPU
    # tensors through torch.ops. Shadow the OpOverloadPacket with the real
    # underlying function, and stand in a no_compile_layers context
    # resolving straight to `stock` -- mirrors test_ple.py's
    # test_ple_ngram_ids_custom_op_uses_current_request_layout. This must
    # compute the SAME real IDs the mmap arm below gets via a direct
    # compute_ngram_ids call, or the equivalence assertion would compare
    # unrelated values.
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            no_compile_layers={stock.layer_name: SimpleNamespace(ple_embedding=stock)}
        ),
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen4_exp_compute_ple_ngram_ids",
        ple_layer_module.qwen4_exp_compute_ple_ngram_ids,
        raising=False,
    )

    reference = stock.forward(input_ids, query_start_loc, ngram_context)
    assert reference.dtype == torch.float8_e4m3fn

    # --- env-on: mmap placeholder backed by shards holding the SAME
    # weight values, staged explicitly and read back through forward(). ---
    _write_ple_layer(
        tmp_path, layer_idx=1, vocab=vocab, parts=parts, cols=head_dim, scale=scale
    )
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.PIECEWISE
    )
    vllm_config = SimpleNamespace(
        compilation_config=cc,
        model_config=_model_config(tmp_path),
        use_v2_model_runner=True,
    )
    with set_current_vllm_config(vllm_config):
        mmap_module = Qwen4ExpNGramEmbedding(
            config,
            embedding_dim,
            0,
            16,
            4,
            "model.layers.1.ple.ple_embedding",
            "model.layers.1.ple",
            params_dtype=torch.bfloat16,
        )
    assert isinstance(mmap_module.ngram_embedding, ple_mmap.MmapNgramEmbedding)
    ple_mmap.set_weight_scale(
        mmap_module.ngram_embedding,
        torch.tensor([scale], dtype=torch.bfloat16),
        torch.device("cpu"),
    )
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[1]
    ple_mmap._attach_table(
        mmap_module.ngram_embedding,
        layer_shards,
        split_ngram_parts=parts,
        layer_idx=1,
        model_path=str(tmp_path),
    )

    num_tokens = input_ids.shape[0]
    mmap_module.initialize_mmap_staging(num_tokens, torch.device("cpu"))
    mmap_module.prepare_mmap_rows(
        input_ids,
        query_start_loc,
        ngram_context,
        actual_tokens=num_tokens,
        padded_tokens=num_tokens,
    )
    got = mmap_module.forward(input_ids, query_start_loc, ngram_context)

    assert torch.equal(got, reference)

    # Real nn.Module chain (mirrors test_ple.py's
    # test_ple_fp8_embedding_dequantizes_in_ple_layer): __new__ + a manual
    # nn.Module.__init__ skips the heavy real __init__, but
    # _get_embedding_weight_scale/_dequantize_embeddings stay the REAL
    # bound methods, exercising the actual getattr chain — no lambda stub.
    stock_ple_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(stock_ple_layer)
    stock_ple_layer.ple_embedding = stock
    dequant_off = stock_ple_layer._dequantize_embeddings(reference, torch.bfloat16)

    mmap_ple_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(mmap_ple_layer)
    mmap_ple_layer.ple_embedding = mmap_module
    dequant_on = mmap_ple_layer._dequantize_embeddings(got, torch.bfloat16)

    assert torch.equal(dequant_on, dequant_off)


def test_dequantize_embeddings_casts_a_bf16_table_to_the_output_dtype() -> None:
    """A BF16 (unquantized) PLE table carries no scale to apply: the
    non-fp8 branch of _dequantize_embeddings must still cast to
    output_dtype, mirroring the fp8 branch's final cast — without it, a
    bf16 table served under e.g. ``--dtype float16`` reaches a downstream
    matmul with a stale bf16 dtype and fails there, unattributably, instead
    of here."""
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    embeddings = torch.tensor([[1.5, -2.25], [0.5, 3.0]], dtype=torch.bfloat16)
    assert not is_fp8(embeddings)

    out = layer._dequantize_embeddings(embeddings, torch.float16)

    assert out.dtype == torch.float16
    assert torch.equal(out, embeddings.to(torch.float16))


# --------------------------------------------------------------------------- #
# compute_ngram_ids golden pin. The equivalence test above drives BOTH
# arms through the same compute_ngram_ids call, so it cannot catch a bug in
# the hashing math itself (xor chain / remainder / offset) — a mutation
# there moves both arms identically and cancels out. This test freezes the
# exact output of a fixed, small, real Qwen4ExpNGramEmbedding on fixed
# inputs, so a hashing regression has to change these hardcoded numbers.
# --------------------------------------------------------------------------- #


def test_compute_ngram_ids_matches_golden_ids() -> None:
    """Golden values captured by running this exact scenario once and
    hardcoding the result — they pin the xor-chain/remainder/offset math
    in compute_ngram_ids (ngram_size=3, heads_per_ngram=2, seed=1234,
    ple_dense_layer_id=0), not merely its shape.
    """
    config = _make_text_config()  # ngram_size=3, heads_per_ngram=2 -> 4 heads
    module = Qwen4ExpNGramEmbedding(
        config,
        8,
        0,
        8,
        2,
        "model.layers.1.ple.ple_embedding",
        "model.layers.1.ple",
        params_dtype=torch.float32,
    )

    input_ids = torch.tensor([11, 22, 33], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.long)
    ngram_context = torch.tensor([[44, 55], [66, 77]], dtype=torch.long)

    ngram_ids = module.compute_ngram_ids(input_ids, query_start_loc, ngram_context)

    assert ngram_ids.shape == (3, 4)
    golden = torch.tensor(
        [
            [647, 1359, 2559, 3257],
            [128, 1518, 2612, 3993],
            [891, 1118, 2768, 3902],
        ],
        dtype=torch.long,
    )
    assert torch.equal(ngram_ids, golden)


# --------------------------------------------------------------------------- #
# Qwen4ExpNGramEmbedding's mmap staging buffer allocates
# at the TABLE's dtype, not params_dtype — zero prior coverage exercised
# this exact allocation through initialize_mmap_staging + the real
# forward().
# --------------------------------------------------------------------------- #


def test_mmap_forward_allocates_an_fp8_output_buffer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A regression here (e.g. back to params_dtype bf16) would leave the
    model serving unscaled embeddings — is_fp8() would stop firing and
    Qwen4ExpPLELayer._dequantize_embeddings would silently skip
    dequantization — while every test that exercises the staging buffer or
    the placeholder in isolation stays green."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()  # ngram_size=3, heads_per_ngram=2 -> 4 heads
    layer_name = "model.language_model.layers.1.ple"
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.PIECEWISE
    )
    # Nonexistent repo: __init__'s validate_shards_for tolerates an
    # unresolvable path and defers — org_vocab_size (needed to write a
    # matching shard fixture) is only known once the module is built.
    unresolvable_config = SimpleNamespace(
        dtype=torch.bfloat16,
        model_weights="",
        model="nonexistent-org/nonexistent-repo-xyz",
        revision=None,
    )
    vllm_config = SimpleNamespace(
        compilation_config=cc,
        model_config=unresolvable_config,
        use_v2_model_runner=True,
    )
    with set_current_vllm_config(vllm_config):
        module = Qwen4ExpNGramEmbedding(
            config,
            8,
            0,
            16,
            4,
            f"{layer_name}.ple_embedding",
            layer_name,
            params_dtype=torch.bfloat16,
        )

    embedding = module.ngram_embedding
    assert isinstance(embedding, ple_mmap.MmapNgramEmbedding)
    vocab = embedding.org_vocab_size
    head_dim = module.head_dim
    parts = module.split_ngram_parts
    _write_ple_layer(
        tmp_path, layer_idx=1, vocab=vocab, parts=parts, cols=head_dim, scale=0.5
    )
    ple_mmap.set_weight_scale(
        embedding, torch.tensor([0.5], dtype=torch.bfloat16), torch.device("cpu")
    )
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[1]
    ple_mmap._attach_table(
        embedding,
        layer_shards,
        split_ngram_parts=parts,
        layer_idx=1,
        model_path=str(tmp_path),
    )

    input_ids = torch.tensor([1, 2], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2], dtype=torch.long)
    ngram_context = torch.zeros((1, 4), dtype=torch.long)

    num_tokens = input_ids.shape[0]
    module.initialize_mmap_staging(num_tokens, torch.device("cpu"))
    module.prepare_mmap_rows(
        input_ids,
        query_start_loc,
        ngram_context,
        actual_tokens=num_tokens,
        padded_tokens=num_tokens,
    )
    out = module.forward(input_ids, query_start_loc, ngram_context)

    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == (2, 8)
    assert is_fp8(out)


# --------------------------------------------------------------------------- #
# layer_name plumbing
# --------------------------------------------------------------------------- #


def test_ngram_embedding_stores_the_layer_name_it_is_constructed_with() -> None:
    config = _make_text_config()
    module = Qwen4ExpNGramEmbedding(
        config,
        8,
        0,
        16,
        4,
        "model.layers.1.ple.ple_embedding",
        "model.layers.1.ple",
        params_dtype=torch.float32,
    )

    assert module.layer_name == "model.layers.1.ple"


def test_ple_layer_registers_its_own_prefix_as_the_static_forward_context_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen4ExpPLELayer.__init__ passes its OWN prefix as
    layer_name — the exact key it registers into
    compilation_config.static_forward_context — never
    f"{prefix}.ple_embedding". Constructs a REAL Qwen4ExpPLELayer (not just
    the embedding in isolation) using the suite's TP rank/size
    monkeypatches, extended to vllm.model_executor.layers.linear for
    ReplicatedLinear's construction.
    """
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    monkeypatch.setattr(linear_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        linear_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    config = _make_text_config(
        hidden_size=8,
        hc_count=2,
        ple_conv_kernel_size=3,
        ple_embed_dim=8,
        rms_norm_eps=1e-5,
    )
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.PIECEWISE
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float32, model_weights="", model="ignored/repo", revision=None
        ),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=4),
        num_speculative_tokens=0,
        compilation_config=cc,
        kernel_config=SimpleNamespace(linear_backend="auto"),
        use_v2_model_runner=True,
    )

    with set_current_vllm_config(vllm_config):
        layer = Qwen4ExpPLELayer(
            config,
            vllm_config=vllm_config,
            layer_idx=1,
            ple_dense_layer_id=0,
            prefix="model.layers.1.ple",
        )

    assert layer.ple_embedding.layer_name == layer.prefix
    assert layer.ple_embedding.layer_name in cc.static_forward_context
    assert cc.static_forward_context[layer.ple_embedding.layer_name] is layer
    # (Seam gap) build_tables' _extract_layer_idx(layer_name) must recover
    # the SAME layer_idx the real layer was constructed with — the two
    # halves (registry key -> layer_idx string parsing, and the layer's own
    # int attribute) must agree on a real, not synthetic, prefix.
    assert ple_mmap._extract_layer_idx(layer.prefix) == layer.layer_idx


# --------------------------------------------------------------------------- #
# model.py load_weights hook: both Qwen4ExpForCausalLM and
# Qwen4ExpForConditionalGeneration must call build_tables when enabled —
# ForCausalLM's call was previously uncovered, since a
# text-only checkpoint served through that class alone previously left its
# PLE layer silently serving fp8 zeros forever.
# --------------------------------------------------------------------------- #


class _FakeAutoWeightsLoader:
    """Stands in for AutoWeightsLoader so the stub `self` below never needs
    to be a real nn.Module (AutoWeightsLoader introspects named_parameters/
    named_buffers/named_modules on construction)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def load_weights(self, weights: object, mapper: object = None) -> set[str]:
        del weights, mapper
        return {"dummy"}


@pytest.mark.parametrize(
    "cls_name", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
)
def test_model_load_weights_calls_build_tables_exactly_once_when_enabled(
    monkeypatch: pytest.MonkeyPatch, cls_name: str, tmp_path: Path
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    monkeypatch.setattr(model_module, "AutoWeightsLoader", _FakeAutoWeightsLoader)
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(ple_mmap, "build_tables", lambda mc, cc: calls.append((mc, cc)))

    cls = getattr(model_module, cls_name)
    model_config = _model_config(tmp_path)
    stub_self = SimpleNamespace(
        hf_to_vllm_mapper=cls.hf_to_vllm_mapper,
        model_config=model_config,
        language_model_only=False,
        language_model=SimpleNamespace(),  # only touched by ConditionalGeneration arm
    )
    cc = SimpleNamespace(static_forward_context={})
    vllm_config = SimpleNamespace(compilation_config=cc)

    with set_current_vllm_config(vllm_config):
        result = cls.load_weights(stub_self, iter([]))

    assert result == {"dummy"}
    assert calls == [(model_config, cc)]


def test_model_load_weights_never_calls_build_tables_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(model_module, "AutoWeightsLoader", _FakeAutoWeightsLoader)
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(ple_mmap, "build_tables", lambda mc, cc: calls.append((mc, cc)))

    stub_self = SimpleNamespace(
        hf_to_vllm_mapper=model_module.Qwen4ExpForCausalLM.hf_to_vllm_mapper,
        model_config=_model_config(tmp_path),
    )
    cc = SimpleNamespace(static_forward_context={})
    vllm_config = SimpleNamespace(compilation_config=cc)

    with set_current_vllm_config(vllm_config):
        result = model_module.Qwen4ExpForCausalLM.load_weights(stub_self, iter([]))

    assert result == {"dummy"}
    assert calls == []


# --------------------------------------------------------------------------- #
# The child-level guard in Qwen4ExpNGramEmbedding.load_weights
# only fires once AutoWeightsLoader's recursive walk reaches that specific PLE
# layer -- by then AutoWeightsLoader may already have mutated earlier,
# unrelated parameters from the same reload. Both top-level load_weights must
# preflight -- via ple_mmap.preflight_reload_check -- BEFORE AutoWeightsLoader
# is even constructed and before `weights` is advanced at all.
# --------------------------------------------------------------------------- #


class _ConstructionRecordingAutoWeightsLoader:
    """Stands in for AutoWeightsLoader, recording every construction attempt
    so a test can assert it was NEVER constructed (the preflight raised
    strictly before this point). If it ever were constructed and reached,
    `load_weights` fully drains the iterator, so a bug that swallows the
    preflight failure would still show up as a consumed iterator."""

    def __init__(self, constructed: list[object], *args: object, **kwargs: object):
        del args, kwargs
        constructed.append(self)

    def load_weights(
        self, weights: Iterable[object], mapper: object = None
    ) -> set[str]:
        del mapper
        list(weights)
        return {"dummy"}


@pytest.mark.parametrize(
    "cls_name", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
)
def test_model_load_weights_preflights_before_autoweightsloader_touches_anything(
    monkeypatch: pytest.MonkeyPatch, cls_name: str, tmp_path: Path
) -> None:
    """A same-instance reload onto a PLE layer that already has a table
    attached must be rejected before AutoWeightsLoader is constructed, before
    the checkpoint iterator is advanced at all, before build_tables runs, and
    without mutating any earlier parameter this reload's stream would
    otherwise reach first."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25
    )
    table_before = embedding.table
    assert table_before is not None
    scale_before = embedding.weight_scale.clone()

    constructed: list[object] = []
    monkeypatch.setattr(
        model_module,
        "AutoWeightsLoader",
        partial(_ConstructionRecordingAutoWeightsLoader, constructed),
    )
    build_table_calls: list[object] = []
    monkeypatch.setattr(
        ple_mmap, "build_tables", lambda mc, cc: build_table_calls.append((mc, cc))
    )

    # An "earlier" parameter this reload's stream would reach well before the
    # PLE layer buried under model.layers.N.ple.ple_embedding.ngram_embedding
    # -- a real Parameter, not a plain attribute, so an in-place weight_loader
    # mutation would actually show up in the before/after comparison below.
    sentinel_param = torch.nn.Parameter(torch.tensor([9.0, 9.0, 9.0]))
    sentinel_before = sentinel_param.detach().clone()

    consumed: list[str] = []

    def reload_checkpoint_iter():
        for name, tensor in (
            ("lm_head.weight", torch.tensor([1.0, 2.0, 3.0])),
            ("model.embed_tokens.weight", torch.tensor([4.0, 5.0, 6.0])),
        ):
            consumed.append(name)
            yield name, tensor

    cls = getattr(model_module, cls_name)
    model_config = _model_config(tmp_path)
    stub_self = SimpleNamespace(
        hf_to_vllm_mapper=cls.hf_to_vllm_mapper,
        model_config=model_config,
        language_model_only=False,
        sentinel_param=sentinel_param,
    )
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, embedding, 2)}
    )
    vllm_config = SimpleNamespace(compilation_config=cc)

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="already has a table attached"),
    ):
        cls.load_weights(stub_self, reload_checkpoint_iter())

    assert consumed == []  # the checkpoint iterator was never advanced
    assert constructed == []  # AutoWeightsLoader was never constructed
    assert build_table_calls == []  # build_tables was never reached
    assert torch.equal(stub_self.sentinel_param, sentinel_before)  # unmutated
    assert embedding.table is table_before  # unchanged identity
    assert torch.equal(embedding.weight_scale, scale_before)  # unchanged


@pytest.mark.parametrize(
    "cls_name", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
)
def test_model_load_weights_preflight_and_build_tables_share_one_captured_config(
    monkeypatch: pytest.MonkeyPatch, cls_name: str, tmp_path: Path
) -> None:
    """Both the preflight call and the build_tables call must use the SAME
    captured compilation_config object -- re-reading
    get_current_vllm_config() a second time after AutoWeightsLoader runs
    would be a needless second lookup and could in principle observe a
    different config."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    monkeypatch.setattr(model_module, "AutoWeightsLoader", _FakeAutoWeightsLoader)

    preflight_calls: list[object] = []
    monkeypatch.setattr(
        ple_mmap,
        "preflight_reload_check",
        lambda cc: preflight_calls.append(cc),
    )
    build_table_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        ple_mmap, "build_tables", lambda mc, cc: build_table_calls.append((mc, cc))
    )
    lookups: list[object] = []
    real_get_current_vllm_config = model_module.get_current_vllm_config

    def _counting_get_current_vllm_config():
        result = real_get_current_vllm_config()
        lookups.append(result)
        return result

    monkeypatch.setattr(
        model_module, "get_current_vllm_config", _counting_get_current_vllm_config
    )

    cls = getattr(model_module, cls_name)
    model_config = _model_config(tmp_path)
    stub_self = SimpleNamespace(
        hf_to_vllm_mapper=cls.hf_to_vllm_mapper,
        model_config=model_config,
        language_model_only=False,
        language_model=SimpleNamespace(),  # only touched by ConditionalGeneration arm
    )
    cc = SimpleNamespace(static_forward_context={})
    vllm_config = SimpleNamespace(compilation_config=cc)

    with set_current_vllm_config(vllm_config):
        result = cls.load_weights(stub_self, iter([]))

    assert result == {"dummy"}
    assert len(lookups) == 1  # captured exactly once, not re-read post-load
    assert preflight_calls == [cc]
    assert build_table_calls == [(model_config, cc)]
    assert preflight_calls[0] is build_table_calls[0][1]  # same captured object


# --------------------------------------------------------------------------- #
# Top-level wrapper reload preflight capability.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cls_name", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
)
def test_preflight_reload_weights_delegates_to_ple_mmap_preflight_reload_check(
    monkeypatch: pytest.MonkeyPatch, cls_name: str
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    preflight_calls: list[tuple[object, str | None, bool, bool]] = []

    def _fake_preflight_reload_check(
        cc: object,
        *,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
        has_weights_iterator: bool = False,
    ) -> None:
        preflight_calls.append(
            (cc, weights_path, is_checkpoint_format, has_weights_iterator)
        )

    monkeypatch.setattr(
        ple_mmap, "preflight_reload_check", _fake_preflight_reload_check
    )

    cls = getattr(model_module, cls_name)
    assert callable(getattr(cls, "preflight_reload_weights", None))

    cc = SimpleNamespace(static_forward_context={})
    vllm_config = SimpleNamespace(compilation_config=cc)
    stub_self = SimpleNamespace()

    with set_current_vllm_config(vllm_config):
        cls.preflight_reload_weights(
            stub_self,
            weights_path="/some/path",
            is_checkpoint_format=False,
            has_weights_iterator=True,
        )

    assert preflight_calls == [(cc, "/some/path", False, True)]


@pytest.mark.parametrize(
    "cls_name", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
)
def test_preflight_reload_weights_is_a_noop_when_mmap_disabled(cls_name: str) -> None:
    """When mmap is off (the default), the hook must not touch
    `get_current_vllm_config()` or `ple_mmap.preflight_reload_check` at all --
    it must be safe for a generic caller to invoke unconditionally without
    knowing whether mmap is enabled or a compilation_config is even set up."""
    assert ple_mmap.enabled() is False
    cls = getattr(model_module, cls_name)
    stub_self = SimpleNamespace()

    cls.preflight_reload_weights(stub_self)  # must not raise


@pytest.mark.parametrize(
    "cls_name", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
)
def test_preflight_reload_weights_rejects_an_already_attached_table(
    monkeypatch: pytest.MonkeyPatch, cls_name: str, tmp_path: Path
) -> None:
    """End-to-end, against the real (unmocked) `preflight_reload_check`: a
    model whose PLE layer already has a table attached must reject through
    this new capability hook exactly like the `load_weights` guard does."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25
    )

    cls = getattr(model_module, cls_name)
    stub_self = SimpleNamespace()
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, embedding, 2)}
    )
    vllm_config = SimpleNamespace(compilation_config=cc)

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="already has a table attached"),
    ):
        cls.preflight_reload_weights(stub_self)


def test_preflight_reload_check_ignores_layers_without_a_ple_embedding() -> None:
    cc = SimpleNamespace(static_forward_context={"a": SimpleNamespace(layer_idx=0)})
    ple_mmap.preflight_reload_check(cc)  # must not raise


def test_preflight_reload_check_ignores_non_mmap_embeddings() -> None:
    """A stock (non-mmap) PLEVocabParallelEmbedding has no `.table` at all --
    the isinstance(embedding, MmapNgramEmbedding) check must gate before any
    attribute access that would otherwise raise."""
    stock_embedding = SimpleNamespace()
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, stock_embedding, 2)}
    )
    ple_mmap.preflight_reload_check(cc)  # must not raise


def test_preflight_reload_check_ignores_an_mmap_embedding_with_no_table_yet() -> None:
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    assert embedding.table is None
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, embedding, 2)}
    )
    ple_mmap.preflight_reload_check(cc)  # must not raise


def test_preflight_reload_check_raises_when_a_layer_already_has_a_table(
    tmp_path: Path,
) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25
    )
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, embedding, 2)}
    )
    with pytest.raises(RuntimeError, match="already has a table attached"):
        ple_mmap.preflight_reload_check(cc)


def test_preflight_reload_check_scans_past_a_clean_layer_to_find_the_attached_one(
    tmp_path: Path,
) -> None:
    """An earlier, never-loaded PLE layer in iteration order must not mask
    an attached table discovered on a later one."""
    clean_embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    _write_ple_layer(tmp_path, layer_idx=1, vocab=8, parts=2, cols=2, scale=0.25)
    attached_embedding = _attached_embedding(
        tmp_path, layer_idx=1, vocab=8, parts=2, cols=2, scale=0.25
    )
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": _fake_ple_layer(0, clean_embedding, 2),
            "b.ple": _fake_ple_layer(1, attached_embedding, 2),
        }
    )
    with pytest.raises(RuntimeError, match="already has a table attached"):
        ple_mmap.preflight_reload_check(cc)


def test_preflight_reload_check_permits_a_matching_dummy_to_real_reload(
    tmp_path: Path,
) -> None:
    """Mmap staging initialized (a dummy run) with no table attached yet
    must be allowed to reload from a resolvable, checkpoint-format path
    whose on-disk shard dtype/width match what is already staged."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    assert embedding.table is None
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
        }
    )

    ple_mmap.preflight_reload_check(
        cc, weights_path=str(tmp_path), is_checkpoint_format=True
    )  # must not raise


def test_preflight_reload_check_rejects_a_dtype_mismatched_reload(
    tmp_path: Path,
) -> None:
    """Staging was initialized for an FP8 checkpoint; the incoming
    checkpoint's shards are BF16 -- reject before anything is mutated."""
    _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=8,
        parts=2,
        cols=2,
        scale=0.25,
        table_dtype=torch.bfloat16,
        write_scale=False,
    )
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
        }
    )

    with pytest.raises(RuntimeError, match="reload rejected"):
        ple_mmap.preflight_reload_check(
            cc, weights_path=str(tmp_path), is_checkpoint_format=True
        )


def test_preflight_reload_check_rejects_a_width_mismatched_reload(
    tmp_path: Path,
) -> None:
    """Staging was initialized for a 5-wide row; the incoming checkpoint's
    shards are 2-wide -- reject before anything is mutated."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 5), dtype=torch.float8_e4m3fn)
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
        }
    )

    with pytest.raises(RuntimeError, match="reload rejected"):
        ple_mmap.preflight_reload_check(
            cc, weights_path=str(tmp_path), is_checkpoint_format=True
        )


def test_preflight_reload_check_rejects_a_same_shape_fp8_reload_missing_weight_scale(
    tmp_path: Path,
) -> None:
    """Dtype and row width both match what is already
    staged exactly, but the incoming FP8 checkpoint carries no
    `weight_scale` at all -- a bare dtype/width comparison would happily
    approve this. The full `_validate_layer_shards` header contract must
    still reject it before an approval can be granted."""
    _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=8,
        parts=2,
        cols=2,
        scale=0.25,
        write_scale=False,
    )
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
        }
    )

    with pytest.raises(RuntimeError, match="weight_scale"):
        ple_mmap.preflight_reload_check(
            cc, weights_path=str(tmp_path), is_checkpoint_format=True
        )


def test_preflight_reload_check_rejects_unsupported_scale_dtype(
    tmp_path: Path,
) -> None:
    _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=8,
        parts=2,
        cols=2,
        scale=1.0,
        scale_dtype=torch.int32,
    )
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
        }
    )

    with pytest.raises(RuntimeError, match="weight_scale has unsupported dtype"):
        ple_mmap.preflight_reload_check(
            cc, weights_path=str(tmp_path), is_checkpoint_format=True
        )


def test_preflight_reload_check_refreshes_replaced_scale_metadata(
    tmp_path: Path,
) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.5)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
        }
    )
    ple_mmap.preflight_reload_check(
        cc, weights_path=str(tmp_path), is_checkpoint_format=True
    )

    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    full = _synthetic_weight(8, 2)
    safetensors.torch.save_file(
        {
            f"{prefix}.shard_0.weight": full[:4],
            f"{prefix}.weight_scale": torch.tensor([1], dtype=torch.int32),
        },
        str(tmp_path / "model-ple-0-00000.safetensors"),
    )

    with pytest.raises(RuntimeError, match="weight_scale has unsupported dtype"):
        ple_mmap.preflight_reload_check(
            cc, weights_path=str(tmp_path), is_checkpoint_format=True
        )


def test_preflight_reload_check_rejects_staged_reload_with_no_weights_path() -> None:
    """No path at all means nothing to prove compatibility against --
    reject rather than optimistically attach."""
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
        }
    )

    with pytest.raises(RuntimeError, match="row-layout compatibility"):
        ple_mmap.preflight_reload_check(
            cc, weights_path=None, is_checkpoint_format=True
        )


def test_preflight_reload_check_rejects_staged_reload_in_kernel_format(
    tmp_path: Path,
) -> None:
    """``is_checkpoint_format=False`` means the incoming weights are already
    repacked kernel-format tensors -- ``discover_shards``' safetensors
    header parsing cannot speak to that layout, so reject even though the
    path itself resolves."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
        }
    )

    with pytest.raises(RuntimeError, match="row-layout compatibility"):
        ple_mmap.preflight_reload_check(
            cc, weights_path=str(tmp_path), is_checkpoint_format=False
        )


def test_preflight_reload_check_rejects_staged_reload_at_an_unresolvable_path() -> None:
    """A ``weights_path`` that is neither a local directory nor an
    offline-cached snapshot must fail closed rather than attach blind."""
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
        }
    )

    with pytest.raises(RuntimeError, match="does not resolve to a local checkpoint"):
        ple_mmap.preflight_reload_check(
            cc,
            weights_path="nonexistent/unresolvable-repo-id",
            is_checkpoint_format=True,
        )


# --------------------------------------------------------------------------- #
# A path-aware preflight (the runner's
# `preflight_reload_weights`, which knows `weights_path`) can prove a
# staged-but-tableless dummy-to-real reload safe. The top-level model's own
# `load_weights` re-runs the same guard pathlessly (`AutoWeightsLoader` has
# no `weights_path` argument) and would otherwise re-reject a reload the
# path-aware call already proved safe. `ple_mmap.approve_reload` /
# `validate_reload_approval` / `clear_reload_approval` reconcile the two
# calls via a transaction-scoped approval stashed on the model instance.
# These tests drive
# the real handshake through the real `GPUModelRunner.reload_weights`, with
# only the heavyweight layerwise-reload/model-loader machinery neutralized
# (mirrors test_gpu_model_runner.py's own preflight-focused stubs).
# --------------------------------------------------------------------------- #


class _ReloadApprovalModel:
    """Minimal top-level-model stand-in wired to the real
    `ple_mmap.approve_reload` / `validate_reload_approval` /
    `clear_reload_approval`, exactly like `Qwen4ExpForCausalLM` /
    `Qwen4ExpForConditionalGeneration` -- without building a real nn.Module
    tree or running `AutoWeightsLoader`, which is irrelevant to the approval
    handshake under test here."""

    def __init__(self, compilation_config: object) -> None:
        self.compilation_config = compilation_config
        self.load_weights_calls = 0

    def preflight_reload_weights(
        self,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
        has_weights_iterator: bool = False,
    ) -> None:
        ple_mmap.approve_reload(
            self,
            self.compilation_config,
            weights_path=weights_path,
            is_checkpoint_format=is_checkpoint_format,
            has_weights_iterator=has_weights_iterator,
        )

    def load_weights(self, weights: Iterable[object]) -> set[str]:
        list(weights)  # drain, like the real AutoWeightsLoader would
        self.load_weights_calls += 1
        ple_mmap.validate_reload_approval(self, self.compilation_config)
        return set()

    def named_parameters(self) -> Iterable[tuple[str, torch.Tensor]]:
        return iter([])

    def clear_reload_approval(self) -> None:
        ple_mmap.clear_reload_approval(self)


def _reload_runner(monkeypatch: pytest.MonkeyPatch, model: object) -> GPUModelRunner:
    """A `GPUModelRunner` exercising the real `reload_weights` body against
    a fake model, with layerwise-reload neutralized (it requires a real
    `nn.Module` tree, orthogonal to the approval handshake under test)."""
    monkeypatch.setattr(
        gpu_model_runner_module, "initialize_layerwise_reload", lambda m: None
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "finalize_layerwise_reload",
        lambda m, model_config: None,
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.lora_config = None
    runner.model_config = SimpleNamespace(
        model="orig", revision=None, quantization=None
    )
    runner.load_config = SimpleNamespace(load_format="auto")
    runner.get_model = lambda: model
    runner.reset_lora_state = lambda: None
    runner.reset_encoder_cache = lambda: None
    runner.reset_mm_cache = lambda: None
    return runner


def _stub_model_loader(
    monkeypatch: pytest.MonkeyPatch,
    weights_factory: Callable[[], Iterable[tuple[str, torch.Tensor]]],
) -> None:
    """Stand in for `get_model_loader(load_config).get_all_weights(...)` so
    a test can drive a real "reload from a path" call
    (`weights_iterator=None`, `weights_path=...`) without a real model
    loader. Supplying BOTH `weights_iterator` and `weights_path` ourselves,
    the way these tests used to, would trip the ambiguous-combination
    guard `preflight_reload_check` now enforces for a model with a
    staged-but-tableless PLE layer -- exactly the state
    every one of these tests sets up. `weights_factory` is called lazily,
    each time `get_all_weights` would be, so a test proving its iterator is
    never ADVANCED (not just never constructed) still holds."""
    loader = SimpleNamespace(
        get_all_weights=lambda model_config, model: weights_factory()
    )
    monkeypatch.setattr(
        gpu_model_runner_module, "get_model_loader", lambda load_config: loader
    )


def test_reload_weights_approves_a_matching_dummy_to_real_reload_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runner's path-aware preflight approves a staged-but-tableless
    layer whose on-disk shard layout matches; the model's own `load_weights`
    must validate against that approval (not re-derive, and not re-reject,
    the same proof pathlessly) and the approval must not survive past the
    runner's `finally`-bounded transaction (`maybe_clear_reload_approval`)."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)
    runner = _reload_runner(monkeypatch, model)
    _stub_model_loader(monkeypatch, lambda: iter([]))

    runner.reload_weights(
        weights_iterator=None,
        weights_path=str(tmp_path),
        is_checkpoint_format=True,
    )

    assert model.load_weights_calls == 1
    assert not hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)


def test_reload_weights_rejects_an_unverifiable_reload_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload with no resolvable path cannot prove row-layout
    compatibility with the already-staged layer; the runner's preflight
    must reject it before `model.load_weights` is ever called, and must not
    leave a stale approval behind."""
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)
    runner = _reload_runner(monkeypatch, model)

    with pytest.raises(RuntimeError, match="row-layout compatibility"):
        runner.reload_weights(
            weights_iterator=iter([]),
            weights_path=None,
            is_checkpoint_format=True,
        )

    assert model.load_weights_calls == 0
    assert not hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)


def test_reload_weights_rejects_a_mismatched_dummy_to_real_reload_before_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On-disk shard width disagrees with what is already staged; the
    runner's path-aware preflight must reject before `model.load_weights` is
    ever called, and must not leave a stale approval behind."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    # 5-wide staging buffer != 2-wide on-disk shard.
    staging = torch.zeros((4, 3, 5), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)
    runner = _reload_runner(monkeypatch, model)
    _stub_model_loader(monkeypatch, lambda: iter([]))

    with pytest.raises(RuntimeError, match="reload rejected"):
        runner.reload_weights(
            weights_iterator=None,
            weights_path=str(tmp_path),
            is_checkpoint_format=True,
        )

    assert model.load_weights_calls == 0
    assert not hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)


def test_reload_weights_rejects_a_same_shape_fp8_reload_missing_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end through the real `GPUModelRunner.
    reload_weights`: on-disk shard dtype/width match the already-staged
    layout exactly, but the FP8 checkpoint carries no `weight_scale` at
    all. The runner's path-aware preflight must still reject it -- via the
    full `_validate_layer_shards` contract, not a bare dtype/width
    comparison -- before the checkpoint iterator is ever advanced, before
    `model.load_weights` is ever called, before any table attaches to the
    backbone, and without leaving a stale approval behind."""
    _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=8,
        parts=2,
        cols=2,
        scale=0.25,
        write_scale=False,
    )
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)
    runner = _reload_runner(monkeypatch, model)

    consumed: list[str] = []

    def weights_iterator() -> Iterable[tuple[str, torch.Tensor]]:
        consumed.append("advanced")
        yield "sentinel.weight", torch.tensor([1.0])

    # weights_iterator=None + a stubbed model loader (not a directly-passed
    # iterator alongside weights_path): supplying both ourselves would trip
    # the ambiguous-combination guard `preflight_reload_check` now enforces
    # for exactly this state (a staged-but-tableless PLE layer). The loader
    # stub still proves the point -- `get_all_weights` is only ever called
    # AFTER preflight passes, so a rejection here means it (and this
    # generator) are never reached either.
    _stub_model_loader(monkeypatch, weights_iterator)

    with pytest.raises(RuntimeError, match="weight_scale"):
        runner.reload_weights(
            weights_iterator=None,
            weights_path=str(tmp_path),
            is_checkpoint_format=True,
        )

    assert consumed == []  # the checkpoint iterator was never advanced
    assert model.load_weights_calls == 0  # model.load_weights was never called
    assert embedding.table is None  # no table ever attached to the backbone
    assert not hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)  # no approval left


def test_reload_weights_rejects_unsupported_scale_dtype_before_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=8,
        parts=2,
        cols=2,
        scale=1.0,
        scale_dtype=torch.int32,
    )
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)
    runner = _reload_runner(monkeypatch, model)
    consumed: list[str] = []

    def weights_iterator() -> Iterable[tuple[str, torch.Tensor]]:
        consumed.append("advanced")
        yield "sentinel.weight", torch.tensor([1.0])

    _stub_model_loader(monkeypatch, weights_iterator)

    with pytest.raises(RuntimeError, match="weight_scale has unsupported dtype"):
        runner.reload_weights(
            weights_iterator=None,
            weights_path=str(tmp_path),
            is_checkpoint_format=True,
        )

    assert consumed == []
    assert model.load_weights_calls == 0
    assert embedding.table is None
    assert not hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)


# --------------------------------------------------------------------------- #
# Shard-placement validation: dtype/width/header checks prove nothing about
# shard CARDINALITY or per-shard ROW placement -- a reload can match dtype
# and width exactly while still being missing a shard, carrying an extra
# one, or holding the wrong row count for its index (e.g. exported with a
# different split_ngram_parts than what is already staged).
# `_validate_shard_placement` closes that gap; these runner-integration
# red-proofs drive the real handshake through `GPUModelRunner.reload_weights`
# end to end, proving each malformed shard layout is rejected before the
# checkpoint iterator is ever advanced, before `model.load_weights` is ever
# called, before any table attaches, and without leaving a stale approval.
# --------------------------------------------------------------------------- #


def test_reload_weights_rejects_a_reload_missing_a_shard_before_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dtype and width both match what is already staged
    exactly, but the on-disk checkpoint is missing shard 1 of 2 entirely --
    a bare dtype/width comparison would happily approve this."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    (tmp_path / "model-ple-0-00001.safetensors").unlink()  # drop shard 1 of 2
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)
    runner = _reload_runner(monkeypatch, model)

    consumed: list[str] = []

    def weights_iterator() -> Iterable[tuple[str, torch.Tensor]]:
        consumed.append("advanced")
        yield "sentinel.weight", torch.tensor([1.0])

    _stub_model_loader(monkeypatch, weights_iterator)

    with pytest.raises(RuntimeError, match=r"missing shard\(s\) \[1\]"):
        runner.reload_weights(
            weights_iterator=None,
            weights_path=str(tmp_path),
            is_checkpoint_format=True,
        )

    assert consumed == []  # the checkpoint iterator was never advanced
    assert model.load_weights_calls == 0  # model.load_weights was never called
    assert embedding.table is None  # no table ever attached to the backbone
    assert not hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)  # no approval left


def test_reload_weights_rejects_a_reload_with_an_extra_shard_before_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dtype and width both match, shards 0 and 1 are
    individually well-formed and correctly sized for the staged
    `split_ngram_parts=2`, but the checkpoint also carries a shard index 2
    that exceeds it -- a bare dtype/width comparison would happily approve
    this."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    safetensors.torch.save_file(
        {f"{prefix}.shard_2.weight": torch.zeros(2, 2, dtype=torch.float8_e4m3fn)},
        str(tmp_path / "model-ple-0-00002.safetensors"),
    )
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)
    runner = _reload_runner(monkeypatch, model)

    consumed: list[str] = []

    def weights_iterator() -> Iterable[tuple[str, torch.Tensor]]:
        consumed.append("advanced")
        yield "sentinel.weight", torch.tensor([1.0])

    _stub_model_loader(monkeypatch, weights_iterator)

    with pytest.raises(RuntimeError, match=r"shard 2 exceeds split_ngram_parts=2"):
        runner.reload_weights(
            weights_iterator=None,
            weights_path=str(tmp_path),
            is_checkpoint_format=True,
        )

    assert consumed == []  # the checkpoint iterator was never advanced
    assert model.load_weights_calls == 0  # model.load_weights was never called
    assert embedding.table is None  # no table ever attached to the backbone
    assert not hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)  # no approval left


def test_reload_weights_rejects_a_reload_with_a_wrong_shard_row_count_before_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dtype and width match exactly -- shard 0's tensor is
    still FP8 and still 2-wide, the same shape family checked above -- but
    it carries 3 rows on disk where its index under the staged
    `split_ngram_parts=2` implies 4. Same dtype/width, wrong placement."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    safetensors.torch.save_file(
        {
            f"{prefix}.shard_0.weight": torch.zeros(3, 2, dtype=torch.float8_e4m3fn),
            f"{prefix}.weight_scale": torch.tensor([0.25], dtype=torch.bfloat16),
        },
        str(tmp_path / "model-ple-0-00000.safetensors"),
    )
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)
    runner = _reload_runner(monkeypatch, model)

    consumed: list[str] = []

    def weights_iterator() -> Iterable[tuple[str, torch.Tensor]]:
        consumed.append("advanced")
        yield "sentinel.weight", torch.tensor([1.0])

    _stub_model_loader(monkeypatch, weights_iterator)

    with pytest.raises(RuntimeError, match=r"shard 0 has 3 rows, expected 4"):
        runner.reload_weights(
            weights_iterator=None,
            weights_path=str(tmp_path),
            is_checkpoint_format=True,
        )

    assert consumed == []  # the checkpoint iterator was never advanced
    assert model.load_weights_calls == 0  # model.load_weights was never called
    assert embedding.table is None  # no table ever attached to the backbone
    assert not hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)  # no approval left


def test_direct_load_weights_call_without_preflight_is_rejected() -> None:
    """A `load_weights` call that skipped `preflight_reload_weights`
    entirely (no runner, no approval) must fall back to the ordinary
    pathless guard and reject a staged-but-tableless layer -- an approval
    is earned, never assumed."""
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)

    with pytest.raises(RuntimeError, match="row-layout compatibility"):
        model.load_weights(iter([]))


def test_a_validated_reload_approval_remains_usable_within_the_same_transaction(
    tmp_path: Path,
) -> None:
    """The approval is transaction-scoped, not single-use: a second
    `load_weights` call on the same model, with no intervening
    `clear_reload_approval`, must still validate against the SAME token --
    the real `AutoWeightsLoader` can call a mapped module's `load_weights`
    more than once inside one reload when its weight stream interleaves
    that module's groups with an unrelated one (see
    `ple_mmap.validate_reload_approval`). Only the runner-equivalent
    `clear_reload_approval` -- standing in for the runner's `finally`
    boundary -- ends the transaction and makes a further call fall back to
    the pathless guard."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)

    model.preflight_reload_weights(
        weights_path=str(tmp_path), is_checkpoint_format=True
    )
    model.load_weights(iter([]))  # first call validates, leaves the token in place
    assert hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)

    model.load_weights(iter([]))  # second call, same transaction, also validates
    assert model.load_weights_calls == 2
    assert hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)

    model.clear_reload_approval()  # the runner-equivalent transaction boundary
    assert not hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)

    with pytest.raises(RuntimeError, match="row-layout compatibility"):
        model.load_weights(iter([]))  # no approval left after the transaction ends


def test_a_rejected_preflight_grants_no_reload_approval(tmp_path: Path) -> None:
    """A preflight that raises must never grant an approval a later,
    unrelated `load_weights` call could spend."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 5), dtype=torch.float8_e4m3fn)  # mismatched width
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)

    with pytest.raises(RuntimeError, match="reload rejected"):
        model.preflight_reload_weights(
            weights_path=str(tmp_path), is_checkpoint_format=True
        )

    assert not hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)


def test_reload_weights_clears_approval_when_named_parameters_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runner's path-aware preflight grants an
    approval, and the very next step -- `model.named_parameters()` -- then
    raises. The approval-clearing `try/finally` must already be wrapping
    the preflight call itself (not started only around the load/copy body
    afterward), so this must still clear the approval. A later, unrelated
    pathless `model.load_weights` call must find no approval left to
    spend."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)

    class _NamedParametersBoom(Exception):
        pass

    approval_seen_at_named_parameters: list[bool] = []

    def _boom() -> Iterable[tuple[str, torch.Tensor]]:
        approval_seen_at_named_parameters.append(
            hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)
        )
        raise _NamedParametersBoom("named_parameters exploded")

    monkeypatch.setattr(model, "named_parameters", _boom)
    runner = _reload_runner(monkeypatch, model)

    with pytest.raises(_NamedParametersBoom):
        runner.reload_weights(
            # weights_iterator=None: `named_parameters` raises before
            # `get_model_loader` is ever reached either way, but passing an
            # iterator here too, alongside weights_path, would trip the
            # ambiguous-combination guard for this staged-but-tableless
            # layer before even reaching the preflight-then-named_parameters
            # sequence this test is red-proofing.
            weights_iterator=None,
            weights_path=str(tmp_path),
            is_checkpoint_format=True,
        )

    # The path-aware preflight really did grant an approval before
    # named_parameters blew up -- otherwise this test would not be
    # red-proofing the try/finally boundary at all.
    assert approval_seen_at_named_parameters == [True]
    assert model.load_weights_calls == 0
    assert not hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)

    # No stale approval survives for a later, unrelated pathless call.
    with pytest.raises(RuntimeError, match="row-layout compatibility"):
        model.load_weights(iter([]))


@pytest.mark.parametrize("is_checkpoint_format", [True, False])
def test_reload_weights_rejects_ambiguous_iterator_and_path_before_any_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, is_checkpoint_format: bool
) -> None:
    """Supplying both a weights_iterator and a
    weights_path for a staged-but-tableless PLE layer is ambiguous --
    `preflight_reload_check` can only validate the checkpoint sitting at
    `weights_path`, but the actual load streams from whichever the caller
    passes to `model.load_weights` (here, the directly-supplied iterator),
    which is not provably the same checkpoint. Must reject before
    `named_parameters`, before the checkpoint iterator advances, before
    `model_config.model` is repointed, and without leaving a residual
    approval -- in BOTH checkpoint format (`is_checkpoint_format=True`) and
    kernel format (`is_checkpoint_format=False`). The on-disk checkpoint
    here is a PERFECTLY VALID, matching layout (see `_write_ple_layer`
    below) -- proving this guard fires on the ambiguity itself, not merely
    as a side effect of some other validation failure."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    model = _ReloadApprovalModel(cc)
    runner = _reload_runner(monkeypatch, model)

    named_parameters_calls: list[bool] = []
    real_named_parameters = model.named_parameters

    def _spy_named_parameters() -> Iterable[tuple[str, torch.Tensor]]:
        named_parameters_calls.append(True)
        return real_named_parameters()

    monkeypatch.setattr(model, "named_parameters", _spy_named_parameters)

    consumed: list[str] = []

    def weights_iterator() -> Iterable[tuple[str, torch.Tensor]]:
        consumed.append("advanced")
        yield "sentinel.weight", torch.tensor([1.0])

    original_model_path = runner.model_config.model

    with pytest.raises(RuntimeError, match="supplied BOTH a weights_path"):
        runner.reload_weights(
            weights_iterator=weights_iterator(),
            weights_path=str(tmp_path),
            is_checkpoint_format=is_checkpoint_format,
        )

    assert consumed == []  # the checkpoint iterator was never advanced
    assert named_parameters_calls == []  # named_parameters was never reached
    assert model.load_weights_calls == 0
    assert runner.model_config.model == original_model_path  # never repointed
    assert not hasattr(model, ple_mmap._RELOAD_APPROVAL_ATTR)  # no residual approval


# --------------------------------------------------------------------------- #
# Nested `Qwen4ExpForConditionalGeneration.load_weights` transactions:
# validates against its own approval, then `AutoWeightsLoader`'s recursive
# walk of its module tree invokes the nested `Qwen4ExpForCausalLM`'s OWN
# `load_weights` directly and pathlessly -- after AutoWeightsLoader may
# already have mutated earlier parameters (e.g. the vision tower) that the
# checkpoint's weight stream reaches first. A transaction-scoped approval,
# granted to BOTH wrappers by the SAME `approve_reload` call and validated
# independently (without either being spent) by each wrapper's own
# `load_weights`, closes that gap. `_NestedReloadApprovalOuter
# Model`/`_NestedReloadApprovalInnerModel` are a faithful `AutoWeightsLoader`
# stub -- like `_ReloadApprovalModel` above, they skip building a real
# nn.Module tree (irrelevant to the approval handshake under test), but
# `_NestedReloadApprovalOuterModel.load_weights` preserves the ONE ordering
# property this fix depends on: an earlier, unrelated parameter mutates in
# stream order BEFORE the walk reaches `language_model`'s own
# `load_weights`, called directly and pathlessly, exactly like the real
# `AutoWeightsLoader._load_module` does for a child module that defines its
# own `load_weights` (vllm/model_executor/models/utils.py). A separate
# red-proof below drives the REAL `AutoWeightsLoader` over a REAL nested
# `nn.Module` tree to prove the transaction survives that same module's
# `load_weights` being called MORE THAN ONCE in one reload.
# --------------------------------------------------------------------------- #


class _NestedReloadApprovalInnerModel:
    """Faithful stand-in for the nested `Qwen4ExpForCausalLM`, wired to the
    real `ple_mmap.validate_reload_approval` exactly like the real class."""

    def __init__(self, compilation_config: object) -> None:
        self.compilation_config = compilation_config
        self.load_weights_calls = 0
        self.sentinel_param = torch.zeros(1)
        self.raise_after_consume: Exception | None = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.load_weights_calls += 1
        ple_mmap.validate_reload_approval(self, self.compilation_config)
        if self.raise_after_consume is not None:
            raise self.raise_after_consume
        loaded = set()
        for name, tensor in weights:
            self.sentinel_param = tensor
            loaded.add(name)
        return loaded


class _NestedReloadApprovalOuterModel:
    """Faithful stand-in for `Qwen4ExpForConditionalGeneration` composing a
    nested `language_model` (see `_NestedReloadApprovalInnerModel` above).
    `load_weights` mimics the ONE property of the real `AutoWeightsLoader`'s
    recursive walk this fix depends on -- see the section comment above."""

    def __init__(self, compilation_config: object) -> None:
        self.compilation_config = compilation_config
        self.language_model = _NestedReloadApprovalInnerModel(compilation_config)
        self.load_weights_calls = 0
        self.sentinel_param = torch.zeros(1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.load_weights_calls += 1
        ple_mmap.validate_reload_approval(self, self.compilation_config)
        loaded: set[str] = set()
        inner_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if name.startswith("language_model."):
                inner_weights.append((name[len("language_model.") :], tensor))
                continue
            # Stands in for AutoWeightsLoader mutating an earlier, unrelated
            # parameter (e.g. the vision tower) in stream order, before its
            # walk ever reaches `language_model`'s prefix.
            self.sentinel_param = tensor
            loaded.add(name)
        if inner_weights:
            loaded |= {
                f"language_model.{n}"
                for n in self.language_model.load_weights(iter(inner_weights))
            }
        return loaded


def _nested_reload_setup(
    tmp_path: Path,
) -> tuple[_NestedReloadApprovalOuterModel, SimpleNamespace]:
    """A staged-but-tableless PLE layer with an on-disk checkpoint whose
    layout matches exactly -- the same fixture every single-wrapper reload
    test above uses, applied to a nested outer+inner model pair."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    vllm_config = SimpleNamespace(compilation_config=cc)
    return _NestedReloadApprovalOuterModel(cc), vllm_config


def test_causal_lm_preflight_grants_no_inner_model_attr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Qwen4ExpForCausalLM` (standalone, no multimodal wrapper) must pass
    `inner_model_attr=None` through to `ple_mmap.approve_reload` -- there is
    no nested inner model to share a second approval with, so a
    CausalLM-only load retains exactly ONE approval."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    captured: dict[str, object] = {}

    def _fake_approve_reload(
        model: object, compilation_config: object, **kwargs: object
    ) -> None:
        del model, compilation_config
        captured.update(kwargs)

    monkeypatch.setattr(ple_mmap, "approve_reload", _fake_approve_reload)
    cc = SimpleNamespace(static_forward_context={})
    vllm_config = SimpleNamespace(compilation_config=cc)
    stub_self = SimpleNamespace()

    with set_current_vllm_config(vllm_config):
        model_module.Qwen4ExpForCausalLM.preflight_reload_weights(
            stub_self, weights_path="/p", is_checkpoint_format=True
        )

    assert captured["inner_model_attr"] is None


def test_conditional_generation_preflight_grants_inner_model_attr_language_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Qwen4ExpForConditionalGeneration` must pass
    `inner_model_attr="language_model"` through to `ple_mmap.approve_reload`
    -- the wiring that lets the nested causal LM's own `load_weights` spend
    a matching copy of the same transaction's token."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    captured: dict[str, object] = {}

    def _fake_approve_reload(
        model: object, compilation_config: object, **kwargs: object
    ) -> None:
        del model, compilation_config
        captured.update(kwargs)

    monkeypatch.setattr(ple_mmap, "approve_reload", _fake_approve_reload)
    cc = SimpleNamespace(static_forward_context={})
    vllm_config = SimpleNamespace(compilation_config=cc)
    stub_self = SimpleNamespace()

    with set_current_vllm_config(vllm_config):
        model_module.Qwen4ExpForConditionalGeneration.preflight_reload_weights(
            stub_self, weights_path="/p", is_checkpoint_format=True
        )

    assert captured["inner_model_attr"] == "language_model"


def test_nested_load_weights_outer_and_inner_each_validate_their_own_approval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The outer wrapper's `load_weights` validates against ITS approval,
    mutates an earlier parameter (standing in for the vision tower), and
    THEN its AutoWeightsLoader-like walk invokes the nested causal LM's OWN
    `load_weights` -- which must find and validate against a MATCHING
    approval of its own, rather than falling back to the pathless guard and
    wrongly rejecting a staged-but-tableless PLE layer this same preflight
    already proved safe. Both approvals are granted by ONE
    `preflight_reload_weights` call (through the real
    `Qwen4ExpForConditionalGeneration` classmethod); neither `load_weights`
    call removes its token (transaction-scoped, not single-use), and both
    are gone only once the runner-equivalent transaction boundary
    (`clear_reload_approval`) closes the reload."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    outer, vllm_config = _nested_reload_setup(tmp_path)

    with set_current_vllm_config(vllm_config):
        model_module.Qwen4ExpForConditionalGeneration.preflight_reload_weights(
            outer, weights_path=str(tmp_path), is_checkpoint_format=True
        )

        assert hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
        assert hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)

        weights = [
            ("vision_tower.weight", torch.tensor([9.0])),
            ("language_model.weight", torch.tensor([7.0])),
        ]
        loaded = outer.load_weights(iter(weights))

        # Validating does not remove either token -- both survive past a
        # successful `load_weights` call, still inside the transaction.
        assert hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
        assert hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)

        model_module.Qwen4ExpForConditionalGeneration.clear_reload_approval(outer)

    assert loaded == {"vision_tower.weight", "language_model.weight"}
    assert outer.sentinel_param.item() == 9.0  # vision-like param mutated
    assert outer.language_model.sentinel_param.item() == 7.0  # inner mutated
    assert outer.load_weights_calls == 1
    assert outer.language_model.load_weights_calls == 1
    assert not hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
    assert not hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)


def test_nested_load_weights_without_active_transaction_rejects_before_vision_mutates(
    tmp_path: Path,
) -> None:
    """A pathless direct call to the outer wrapper's `load_weights` --
    skipping `preflight_reload_weights` entirely, so there is no active
    transaction -- must reject via the ordinary pathless guard as the VERY
    FIRST thing `load_weights` does: before any earlier parameter (e.g. the
    vision tower) mutates, and before the nested causal LM's own
    `load_weights` is ever reached."""
    outer, _ = _nested_reload_setup(tmp_path)

    weights = [
        ("vision_tower.weight", torch.tensor([9.0])),
        ("language_model.weight", torch.tensor([7.0])),
    ]

    with pytest.raises(RuntimeError, match="row-layout compatibility"):
        outer.load_weights(iter(weights))

    assert outer.sentinel_param.item() == 0.0  # never mutated
    assert outer.language_model.sentinel_param.item() == 0.0  # inner never reached
    assert outer.language_model.load_weights_calls == 0
    assert not hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
    assert not hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)


def test_nested_reload_clears_inner_approval_if_outer_raises_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The outer wrapper's own mutation step (standing in for the vision
    tower) raises BEFORE its AutoWeightsLoader-like walk ever reaches the
    nested causal LM's `load_weights` -- so the inner model's copy of the
    shared token is never validated against. The runner-equivalent
    `finally` -> `clear_reload_approval()` (the real
    `Qwen4ExpForConditionalGeneration` classmethod) must still remove that
    stale inner token; otherwise it would survive for a later, unrelated
    reload to validate against."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    outer, vllm_config = _nested_reload_setup(tmp_path)

    class _VisionBoom(Exception):
        pass

    def _boom_load_weights(weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        outer.load_weights_calls += 1
        ple_mmap.validate_reload_approval(outer, vllm_config.compilation_config)
        raise _VisionBoom("vision tower blew up")

    outer.load_weights = _boom_load_weights  # type: ignore[method-assign]

    with set_current_vllm_config(vllm_config):
        model_module.Qwen4ExpForConditionalGeneration.preflight_reload_weights(
            outer, weights_path=str(tmp_path), is_checkpoint_format=True
        )
        assert hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
        assert hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)

        try:
            with pytest.raises(_VisionBoom):
                outer.load_weights(iter([]))
        finally:
            model_module.Qwen4ExpForConditionalGeneration.clear_reload_approval(outer)

    assert not hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
    assert not hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)
    assert outer.language_model.load_weights_calls == 0  # inner never reached


def test_nested_reload_inner_failure_after_its_own_validate_leaves_no_residual_approval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The nested causal LM's OWN `load_weights` validates against its copy
    of the shared token FIRST -- which does NOT remove it, transaction-
    scoped approval being validated, not spent -- then raises for an
    unrelated reason (a real load error): an inner failure. Both wrappers'
    approvals are therefore STILL PRESENT at that point; the
    runner-equivalent `finally` -> `clear_reload_approval()` must be the
    one to actually remove them here, and no approval must survive on
    either wrapper for a later, unrelated reload to validate against."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    outer, vllm_config = _nested_reload_setup(tmp_path)

    class _InnerBoom(Exception):
        pass

    outer.language_model.raise_after_consume = _InnerBoom("inner load blew up")

    with set_current_vllm_config(vllm_config):
        model_module.Qwen4ExpForConditionalGeneration.preflight_reload_weights(
            outer, weights_path=str(tmp_path), is_checkpoint_format=True
        )

        weights = [
            ("vision_tower.weight", torch.tensor([9.0])),
            ("language_model.weight", torch.tensor([7.0])),
        ]
        try:
            with pytest.raises(_InnerBoom):
                outer.load_weights(iter(weights))
            # The inner call validated against its token without removing
            # it -- both approvals are still live right up until the
            # runner-equivalent `finally` below closes the transaction.
            assert hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
            assert hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)
        finally:
            model_module.Qwen4ExpForConditionalGeneration.clear_reload_approval(outer)

    assert outer.sentinel_param.item() == 9.0  # vision mutated before inner raised
    assert outer.language_model.load_weights_calls == 1  # inner was reached
    assert not hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
    assert not hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)


# --------------------------------------------------------------------------- #
# Red-proof: the REAL `AutoWeightsLoader` (not the hand-rolled stand-in
# above) groups an incoming weight stream by CONSECUTIVE runs of a shared
# top-level prefix (`AutoWeightsLoader._groupby_prefix`, via
# `itertools.groupby`) -- not a full partition of the stream. A checkpoint
# that interleaves a mapped module's groups with an unrelated one therefore
# makes `AutoWeightsLoader._load_module` call that module's OWN
# `load_weights` MORE THAN ONCE per reload. This drives that exact
# recursion, over a REAL nested `nn.Module` tree, to prove: (1) both calls
# validate under the ONE shared transaction; (2) the earlier, unrelated
# `visual` mutation -- processed BETWEEN the two `language_model` runs --
# does not cause the second call to fall back to the pathless guard and
# falsely re-reject a reload this preflight already proved safe; and (3)
# the runner-equivalent transaction boundary removes both tokens afterward.
# --------------------------------------------------------------------------- #


class _RealInterleavedInnerCausalLM(nn.Module):
    """Real `nn.Module` standing in for the nested `Qwen4ExpForCausalLM`:
    TWO real parameters, so a weight stream can address it through two
    SEPARATE prefix runs. Wired to the real
    `ple_mmap.validate_reload_approval`, exactly like the production
    class."""

    def __init__(self, compilation_config: object) -> None:
        super().__init__()
        self.compilation_config = compilation_config
        self.first = nn.Parameter(torch.zeros(2))
        self.second = nn.Parameter(torch.zeros(2))
        self.load_weights_calls = 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.load_weights_calls += 1
        ple_mmap.validate_reload_approval(self, self.compilation_config)
        loaded: set[str] = set()
        for name, tensor in weights:
            if name == "first":
                self.first.data.copy_(tensor)
                loaded.add(name)
            elif name == "second":
                self.second.data.copy_(tensor)
                loaded.add(name)
        return loaded


class _RealInterleavedOuterConditionalGeneration(nn.Module):
    """Real `nn.Module` standing in for `Qwen4ExpForConditionalGeneration`:
    a real `visual` parameter that a checkpoint's weight stream visits
    BETWEEN the two `language_model`-prefixed runs below, and a real
    nested `language_model`. `load_weights` is the STOCK
    `AutoWeightsLoader(self).load_weights(weights)` -- no hand-rolled
    dispatch -- so this exercises the exact recursion the production
    classes depend on, not a faithful stand-in for it."""

    def __init__(self, compilation_config: object) -> None:
        super().__init__()
        self.compilation_config = compilation_config
        self.visual = nn.Parameter(torch.zeros(2))
        self.language_model = _RealInterleavedInnerCausalLM(compilation_config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def preflight_reload_weights(
        self, weights_path: str | None = None, is_checkpoint_format: bool = True
    ) -> None:
        ple_mmap.approve_reload(
            self,
            self.compilation_config,
            weights_path=weights_path,
            is_checkpoint_format=is_checkpoint_format,
            inner_model_attr="language_model",
        )

    def clear_reload_approval(self) -> None:
        ple_mmap.clear_reload_approval(self, inner_model=self.language_model)


def test_interleaved_language_model_groups_both_validate_under_one_transaction(
    tmp_path: Path,
) -> None:
    """Red-proof for the transaction-scoped fix: an interleaved weight
    stream (`language_model`, `visual`, `language_model`) makes the REAL
    `AutoWeightsLoader` call the nested module's `load_weights` TWICE
    inside one reload. Both calls must validate against the SAME
    transaction's approval -- the earlier `visual` mutation, processed
    between them, must not cause the second call to falsely reject a
    staged-but-tableless PLE layer this preflight already proved safe.
    Neutralizing the transaction-scoped fix (reverting
    `validate_reload_approval` to delete the token on a valid match, as the
    destructively single-use predecessor did) makes this test fail: the
    second `language_model.load_weights` call would find no token and
    raise "row-layout compatibility" instead of returning cleanly."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    staging = torch.zeros((4, 3, 2), dtype=torch.float8_e4m3fn)
    ple_layer = _fake_ple_layer(0, embedding, 2, mmap_staging=staging)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    outer = _RealInterleavedOuterConditionalGeneration(cc)

    outer.preflight_reload_weights(
        weights_path=str(tmp_path), is_checkpoint_format=True
    )
    assert hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
    assert hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)

    # Interleaved: TWO separate "language_model"-prefixed runs, split by an
    # intervening "visual" run. `AutoWeightsLoader._groupby_prefix` groups
    # by `itertools.groupby` over CONSECUTIVE keys, so this really does
    # call `outer.language_model.load_weights` twice, not once with
    # everything -- the same shape a real checkpoint's serialization order
    # produces when a mapped module's weights are not all contiguous.
    weights = [
        ("language_model.first", torch.tensor([1.0, 2.0])),
        ("visual", torch.tensor([9.0, 9.0])),
        ("language_model.second", torch.tensor([3.0, 4.0])),
    ]

    loaded = outer.load_weights(iter(weights))

    assert loaded == {"visual", "language_model.first", "language_model.second"}
    assert outer.language_model.load_weights_calls == 2
    assert torch.equal(outer.visual.data, torch.tensor([9.0, 9.0]))
    assert torch.equal(outer.language_model.first.data, torch.tensor([1.0, 2.0]))
    assert torch.equal(outer.language_model.second.data, torch.tensor([3.0, 4.0]))

    # Validating does not spend either token: both are still live after two
    # real `AutoWeightsLoader`-driven calls into the same transaction.
    assert hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
    assert hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)

    outer.clear_reload_approval()  # the runner-equivalent transaction boundary
    assert not hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
    assert not hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)


# --------------------------------------------------------------------------- #
# An initial multimodal load has no path-aware preflight
# ahead of it at all (nothing ever calls `preflight_reload_weights` for it),
# so there is no runner-granted token when `AutoWeightsLoader` reaches
# `language_model` -- unlike every reload test above. An interleaved
# checkpoint still makes the REAL `AutoWeightsLoader` call
# `language_model.load_weights` more than once, and the first call's own
# build_tables-equivalent attaches a table before the second call ever
# validates. Before scoped initial-load ownership, that second call found no
# token (nothing had ever minted one), fell back to the pathless guard, saw
# the just-attached table, and rejected -- AFTER real mutation. The fix:
# `validate_reload_approval` mints and OWNS a fresh token on the outer
# wrapper's first, token-less call, grants a copy to `language_model`, and
# the outer wrapper alone clears both in `finally`.
# --------------------------------------------------------------------------- #


class _InitialLoadInnerCausalLM(nn.Module):
    """Real nn.Module standing in for the nested `Qwen4ExpForCausalLM` on a
    TRUE initial load: wired to the real `ple_mmap.validate_reload_approval`
    / `clear_reload_approval`, exactly like the production class. Gates its
    build_tables-equivalent on `should_build_tables` -- False whenever an
    outer wrapper granted it a `_ROLE_DEFER` copy of the transaction (see
    `_InitialLoadOuterConditionalGeneration`), exactly like the real
    `Qwen4ExpForCausalLM.load_weights`."""

    def __init__(
        self, compilation_config: object, embedding: ple_mmap.MmapNgramEmbedding
    ) -> None:
        super().__init__()
        self.compilation_config = compilation_config
        self._embedding = embedding
        self.first = nn.Parameter(torch.zeros(2))
        self.second = nn.Parameter(torch.zeros(2))
        self.load_weights_calls = 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.load_weights_calls += 1
        owns, should_build_tables = ple_mmap.validate_reload_approval(
            self, self.compilation_config
        )
        try:
            loaded: set[str] = set()
            for name, tensor in weights:
                if name == "first":
                    self.first.data.copy_(tensor)
                    loaded.add(name)
                elif name == "second":
                    self.second.data.copy_(tensor)
                    loaded.add(name)
            if should_build_tables and self._embedding.table is None:
                self._embedding.table = SimpleNamespace()  # build_tables-equivalent
            return loaded
        finally:
            if owns:
                ple_mmap.clear_reload_approval(self)


class _InitialLoadOuterConditionalGeneration(nn.Module):
    """Real nn.Module standing in for `Qwen4ExpForConditionalGeneration` on
    a TRUE initial load -- no `preflight_reload_weights` call precedes it.
    `load_weights` is the STOCK `AutoWeightsLoader(self).load_weights`, the
    same recursion the production classes depend on. This wrapper's own
    token is always `_ROLE_ROOT`, so it is the one that applies the
    build_tables-equivalent, exactly once, after the whole recursion
    returns -- exactly like the real `Qwen4ExpForConditionalGeneration.
    load_weights`."""

    def __init__(
        self, compilation_config: object, embedding: ple_mmap.MmapNgramEmbedding
    ) -> None:
        super().__init__()
        self.compilation_config = compilation_config
        self.visual = nn.Parameter(torch.zeros(2))
        self.language_model = _InitialLoadInnerCausalLM(compilation_config, embedding)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        owns, should_build_tables = ple_mmap.validate_reload_approval(
            self, self.compilation_config, inner_model_attr="language_model"
        )
        try:
            loaded = AutoWeightsLoader(self).load_weights(weights)
            if should_build_tables and self.language_model._embedding.table is None:
                self.language_model._embedding.table = SimpleNamespace()
            return loaded
        finally:
            if owns:
                ple_mmap.clear_reload_approval(self, inner_model=self.language_model)


def test_initial_multimodal_load_survives_repeated_interleaved_inner_groups() -> None:
    """Making `should_build_tables` always true makes this fail: the first
    `language_model.load_weights` call, which
    only carries `first`, no PLE weight at all -- would attach a table from
    headers alone, and the second, interleaved `language_model.load_weights`
    call would then find that table and raise "already has a table attached
    from a previous load" -- AFTER `visual` and `first` already mutated, a
    partial-mutation failure mid-load. With the fix, the nested causal LM's
    token always carries `_ROLE_DEFER` under this outer wrapper, so BOTH of
    its calls skip building; only the outer wrapper, holding `_ROLE_ROOT`,
    applies the build_tables-equivalent, exactly once, after the whole
    recursion returns -- and both tokens clear only once, at that same
    return."""
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    # mmap_staging=None (the default): a true cold load has no staging yet
    # (V2 model state, which allocates it, is constructed AFTER load_weights).
    ple_layer = _fake_ple_layer(0, embedding, 2)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    outer = _InitialLoadOuterConditionalGeneration(cc, embedding)

    # Interleaved: TWO separate "language_model"-prefixed runs split by an
    # intervening "visual" run -- AutoWeightsLoader._groupby_prefix calls
    # language_model.load_weights twice, not once with everything.
    weights = [
        ("language_model.first", torch.tensor([1.0, 2.0])),
        ("visual", torch.tensor([9.0, 9.0])),
        ("language_model.second", torch.tensor([3.0, 4.0])),
    ]

    loaded = outer.load_weights(iter(weights))

    assert loaded == {"visual", "language_model.first", "language_model.second"}
    assert outer.language_model.load_weights_calls == 2
    assert torch.equal(outer.visual.data, torch.tensor([9.0, 9.0]))
    assert torch.equal(outer.language_model.first.data, torch.tensor([1.0, 2.0]))
    assert torch.equal(outer.language_model.second.data, torch.tensor([3.0, 4.0]))
    assert embedding.table is not None  # build_tables-equivalent attached

    # The outer wrapper owned the transaction it minted; both copies are
    # cleared at its own return, with no separate runner ever involved.
    assert not hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
    assert not hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)


def test_standalone_causal_lm_initial_load_owns_and_clears_its_own_token() -> None:
    """A standalone `Qwen4ExpForCausalLM` initial load (no multimodal
    wrapper, no preflight) has no outer wrapper to grant it a token --
    `validate_reload_approval` must mint and clear its OWN, exactly once."""
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    ple_layer = _fake_ple_layer(0, embedding, 2)
    cc = SimpleNamespace(static_forward_context={"a.ple": ple_layer})
    standalone = _InitialLoadInnerCausalLM(cc, embedding)

    loaded = standalone.load_weights(
        iter(
            [("first", torch.tensor([1.0, 2.0])), ("second", torch.tensor([3.0, 4.0]))]
        )
    )

    assert loaded == {"first", "second"}
    assert standalone.load_weights_calls == 1
    assert embedding.table is not None
    assert not hasattr(standalone, ple_mmap._RELOAD_APPROVAL_ATTR)


# --------------------------------------------------------------------------- #
# Full initial-load coverage uses the real
# `model_module.Qwen4ExpForCausalLM.load_weights` /
# `model_module.Qwen4ExpForConditionalGeneration.load_weights` (bound onto
# these stand-in nn.Modules, not reimplemented), the REAL
# `AutoWeightsLoader`, and the REAL `ple_mmap.build_tables` / `_attach_table`
# -- proving the exact production guard this fix depends on:
# `Qwen4ExpNGramEmbedding.load_weights` (nvidia/ple_layer.py) raises
# "already has a table attached" the instant a second call sees a non-None
# table, before touching its own weights at all. `_RealColdLoadPLEReceiver`
# reproduces ONLY that one guard plus the weight_scale/shard interception;
# table construction, shard discovery, and the scale cross-check are the
# real `ple_mmap` code, exercised through `static_forward_context` exactly
# like the production `Qwen4ExpPLELayer` does.
# --------------------------------------------------------------------------- #


class _RealColdLoadPLEReceiver(nn.Module):
    """Real nn.Module reproducing the ONE guard in
    `Qwen4ExpNGramEmbedding.load_weights` this red-proof depends on: reject
    re-entry once a table is attached, before touching the incoming weight
    at all. Registered as `language_model.ple_embedding` so the real
    `AutoWeightsLoader` delegates a `ple_embedding.*`-prefixed group to it,
    and as the `static_forward_context` entry's `ple_embedding` so the real
    `ple_mmap.build_tables` finds its `ngram_embedding` too."""

    def __init__(
        self, embedding: ple_mmap.MmapNgramEmbedding, split_ngram_parts: int
    ) -> None:
        super().__init__()
        self.ngram_embedding = embedding
        self.split_ngram_parts = split_ngram_parts

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.ngram_embedding.table is not None:
            raise RuntimeError(
                "PLE mmap: already has a table attached from a previous "
                "load; calling load_weights again on the same live module "
                "is unsupported"
            )
        loaded: set[str] = set()
        for name, tensor in weights:
            if name == "ngram_embedding.weight_scale":
                ple_mmap.set_weight_scale(
                    self.ngram_embedding, tensor, torch.device("cpu")
                )
                loaded.add(name)
            elif name.startswith("ngram_embedding.shard_") and name.endswith(".weight"):
                self.ngram_embedding.weights_streamed = True
                loaded.add(name)
        return loaded


class _RealColdLoadInnerCausalLM(nn.Module):
    """Real nn.Module standing in for `Qwen4ExpForCausalLM`: `load_weights`
    IS the real `model_module.Qwen4ExpForCausalLM.load_weights` (bound as a
    class attribute, ordinary Python method binding), not a hand-rolled
    stand-in of it -- this proves the actual production method's
    `should_build_tables` gating."""

    load_weights = model_module.Qwen4ExpForCausalLM.load_weights

    def __init__(
        self,
        model_config: object,
        embedding: ple_mmap.MmapNgramEmbedding,
        split_ngram_parts: int,
    ) -> None:
        super().__init__()
        self.hf_to_vllm_mapper = model_module.Qwen4ExpForCausalLM.hf_to_vllm_mapper
        self.model_config = model_config
        self.first = nn.Parameter(torch.zeros(2))
        self.ple_embedding = _RealColdLoadPLEReceiver(embedding, split_ngram_parts)


class _RealColdLoadOuterConditionalGeneration(nn.Module):
    """Real nn.Module standing in for `Qwen4ExpForConditionalGeneration`:
    `load_weights` IS the real
    `model_module.Qwen4ExpForConditionalGeneration.load_weights`."""

    load_weights = model_module.Qwen4ExpForConditionalGeneration.load_weights

    def __init__(
        self,
        model_config: object,
        embedding: ple_mmap.MmapNgramEmbedding,
        split_ngram_parts: int,
    ) -> None:
        super().__init__()
        self.hf_to_vllm_mapper = (
            model_module.Qwen4ExpForConditionalGeneration.hf_to_vllm_mapper
        )
        self.model_config = model_config
        self.language_model_only = False
        self.visual = nn.Parameter(torch.zeros(2))
        self.language_model = _RealColdLoadInnerCausalLM(
            model_config, embedding, split_ngram_parts
        )


def _real_cold_load_setup(
    tmp_path: Path, scale: float
) -> tuple[
    _RealColdLoadOuterConditionalGeneration,
    ple_mmap.MmapNgramEmbedding,
    SimpleNamespace,
]:
    """A well-formed on-disk checkpoint (layer 0, vocab=8, 2 shards, 2-wide,
    FP8) plus the real nested outer+inner module tree wired to it via
    `static_forward_context`, exactly like `build_tables` expects to find
    the production `Qwen4ExpPLELayer` composition."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=scale)
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    model_config = _model_config(tmp_path)
    outer = _RealColdLoadOuterConditionalGeneration(
        model_config, embedding, split_ngram_parts=2
    )
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": SimpleNamespace(
                layer_idx=0, ple_embedding=outer.language_model.ple_embedding
            )
        }
    )
    vllm_config = SimpleNamespace(compilation_config=cc)
    return outer, embedding, vllm_config


def test_real_production_cold_load_survives_interleaved_ple_shard_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Production-fidelity load order:
    `language_model.first` (a non-PLE weight) -> `visual` -> `language_model`'s
    PLE shard + scale -- the exact ordering QA identified: the FIRST
    `language_model.load_weights` call carries no PLE weight at all, so a
    table build right there would attach from headers alone, before the
    checkpoint's real PLE shard/scale ever streams.

    Making `should_build_tables` always True, as
    before roles existed -- see
    `test_real_production_cold_load_partially_fails_without_role_based_deferral`
    below for this exact scenario driven with that reversion) makes this
    fail the same way: the first `Qwen4ExpForCausalLM.load_weights` call
    attaches a table from headers alone, and the interleaved second call --
    carrying the real PLE shard + scale -- reaches
    `Qwen4ExpNGramEmbedding.load_weights`'s real guard and raises "already
    has a table attached from a previous load", AFTER `visual` and `first`
    have already mutated: a partial-mutation failure mid-load.

    With the fix, both nested calls hold a `_ROLE_DEFER` token and skip
    building; only the outer wrapper, holding `_ROLE_ROOT`, calls the real
    `ple_mmap.build_tables` -- exactly once, verified below via a spy on the
    real `ple_mmap._attach_table` -- after the whole recursion returns, and
    no approval token survives on either wrapper."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    scale = 0.25
    outer, embedding, vllm_config = _real_cold_load_setup(tmp_path, scale)

    attach_calls: list[int] = []
    real_attach_table = ple_mmap._attach_table

    def _counting_attach_table(*args: object, **kwargs: object) -> None:
        attach_calls.append(1)
        real_attach_table(*args, **kwargs)

    monkeypatch.setattr(ple_mmap, "_attach_table", _counting_attach_table)

    # Interleaved: `language_model.first` -> `visual` -> `language_model`'s
    # PLE shard + scale. AutoWeightsLoader._groupby_prefix calls
    # language_model.load_weights twice, not once with everything, and the
    # FIRST of those two calls carries no PLE weight at all.
    weights = [
        ("language_model.first", torch.tensor([1.0, 2.0])),
        ("visual", torch.tensor([9.0, 9.0])),
        (
            "language_model.ple_embedding.ngram_embedding.shard_0.weight",
            torch.zeros(4, 2, dtype=torch.float8_e4m3fn),
        ),
        (
            "language_model.ple_embedding.ngram_embedding.weight_scale",
            torch.tensor([scale], dtype=torch.bfloat16),
        ),
    ]

    with set_current_vllm_config(vllm_config):
        loaded = outer.load_weights(iter(weights))

    assert "visual" in loaded
    assert torch.equal(outer.visual.data, torch.tensor([9.0, 9.0]))
    assert torch.equal(outer.language_model.first.data, torch.tensor([1.0, 2.0]))
    assert embedding.table is not None  # the real build_tables really attached
    assert len(attach_calls) == 1  # exactly one final attachment, not two

    # No approval token survives past the outer wrapper's own return.
    assert not hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
    assert not hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)


def test_real_production_cold_load_partially_fails_without_role_based_deferral(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Forcing
    `should_build_tables=True` on every validated call -- the actual
    behavior before `_ROLE_DEFER` existed, when gating was only ever
    `compilation_config is not None` reproduces the partial-load failure.
    The FIRST `language_model.load_weights` call, carrying only `first`,
    attaches a table from headers alone via the real `ple_mmap.build_tables`;
    the interleaved SECOND call, carrying the real PLE shard + scale, then
    hits `Qwen4ExpNGramEmbedding.load_weights`'s real guard and raises --
    AFTER `visual` and `first` have already mutated."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    scale = 0.25
    outer, _embedding, vllm_config = _real_cold_load_setup(tmp_path, scale)

    real_validate_reload_approval = ple_mmap.validate_reload_approval

    def _should_build_whenever_enabled(
        model: object, compilation_config: object, **kwargs: object
    ) -> tuple[bool, bool]:
        owns, _role_based_should_build = real_validate_reload_approval(
            model, compilation_config, **kwargs
        )
        return owns, True  # pre-fix: gated only on compilation_config, never role

    monkeypatch.setattr(
        ple_mmap, "validate_reload_approval", _should_build_whenever_enabled
    )

    weights = [
        ("language_model.first", torch.tensor([1.0, 2.0])),
        ("visual", torch.tensor([9.0, 9.0])),
        (
            "language_model.ple_embedding.ngram_embedding.shard_0.weight",
            torch.zeros(4, 2, dtype=torch.float8_e4m3fn),
        ),
        (
            "language_model.ple_embedding.ngram_embedding.weight_scale",
            torch.tensor([scale], dtype=torch.bfloat16),
        ),
    ]

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="already has a table attached"),
    ):
        outer.load_weights(iter(weights))

    # Partial mutation: visual and first already landed before the raise.
    assert torch.equal(outer.visual.data, torch.tensor([9.0, 9.0]))
    assert torch.equal(outer.language_model.first.data, torch.tensor([1.0, 2.0]))

    # The outer wrapper's own finally still clears both tokens despite the
    # nested raise -- no leak even on the failure path.
    assert not hasattr(outer, ple_mmap._RELOAD_APPROVAL_ATTR)
    assert not hasattr(outer.language_model, ple_mmap._RELOAD_APPROVAL_ATTR)


# --------------------------------------------------------------------------- #
# Default-off inertness (invariant 2)
# --------------------------------------------------------------------------- #


def test_default_off_uses_the_stock_vocab_parallel_embedding() -> None:
    assert ple_mmap.enabled() is False
    config = _make_text_config()

    module = Qwen4ExpNGramEmbedding(
        config,
        8,
        0,
        16,
        4,
        "model.layers.1.ple.ple_embedding",
        "model.layers.1.ple",
        params_dtype=torch.float32,
    )

    assert isinstance(module.ngram_embedding, PLEVocabParallelEmbedding)
    assert not isinstance(module.ngram_embedding, ple_mmap.MmapNgramEmbedding)


def test_default_off_forward_never_calls_the_mmap_gather_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With VLLM_PLE_MMAP unset, forward() must take the exact stock branch
    (direct call + .flatten(-2)). There is no mmap gather op left to spy on
    -- test_mmap_gather_op_is_not_registered above pins its complete
    removal from torch.ops.vllm; this test pins that the stock branch is
    the one actually taken."""
    config = _make_text_config()
    module = Qwen4ExpNGramEmbedding(
        config,
        8,
        0,
        16,
        4,
        "model.layers.1.ple.ple_embedding",
        "model.layers.1.ple",
        params_dtype=torch.float32,
    )
    sentinel = torch.arange(2 * 4 * 2, dtype=torch.bfloat16).reshape(2, 4, 2)
    calls: list[torch.Tensor] = []

    def spy_forward(ids: torch.Tensor) -> torch.Tensor:
        calls.append(ids)
        return sentinel

    monkeypatch.setattr(module.ngram_embedding, "forward", spy_forward)
    # forward() still routes ID generation through the REGISTERED
    # qwen4_exp_compute_ple_ngram_ids op (upstream architecture, untouched by
    # this PR). That op only has a CUDA dispatch-key impl (matches upstream
    # test_ple.py, which drives it as a plain function for the same reason),
    # so on a CUDA-platform host it cannot run against CPU tensors through
    # torch.ops. Shadow the OpOverloadPacket with the real underlying
    # function, and stand in a no_compile_layers context resolving straight
    # to this module -- mirrors test_ple.py's
    # test_ple_ngram_ids_custom_op_uses_current_request_layout.
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            no_compile_layers={module.layer_name: SimpleNamespace(ple_embedding=module)}
        ),
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen4_exp_compute_ple_ngram_ids",
        ple_layer_module.qwen4_exp_compute_ple_ngram_ids,
        raising=False,
    )

    input_ids = torch.tensor([1, 2], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2], dtype=torch.long)
    ngram_context = torch.zeros((1, 4), dtype=torch.long)

    output = module.forward(input_ids, query_start_loc, ngram_context)

    assert len(calls) == 1  # the stock embedding was called directly
    assert torch.equal(output, sentinel.flatten(-2))


def test_default_off_load_weights_matches_the_stock_contract() -> None:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(module)
    module.split_ngram_parts = 2
    module.register_buffer("layer_multipliers", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_offsets", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_vocab_sizes", torch.zeros(1, dtype=torch.long))
    embedding = SimpleNamespace(
        org_vocab_size=8,
        embedding_dim=2,
        weight=torch.nn.Parameter(torch.full((4, 2), -1.0)),
        shard_indices=SimpleNamespace(org_vocab_start_index=2, org_vocab_end_index=6),
    )
    # Real PLEVocabParallelEmbedding construction attaches weight_loader to
    # the weight Parameter itself (see VocabParallelEmbedding.__init__ ->
    # quant_method.create_weights(..., weight_loader=self.weight_loader));
    # this fake embedding needs the same attribute for load_weights's
    # generic `embedding.weight.weight_loader(...)` dispatch to resolve.
    # Mirrors test_ple.py's _set_test_embedding_weight_loader.
    embedding.weight.weight_loader = partial(
        copy_ple_embedding_shard_,
        tp_start=embedding.shard_indices.org_vocab_start_index,
        tp_end=embedding.shard_indices.org_vocab_end_index,
    )
    module.ngram_embedding = embedding  # not MmapNgramEmbedding -> mmap_enabled=False

    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
        ]
    )

    assert loaded == {"ngram_embedding.weight"}
    expected = torch.cat((shard_0[2:4], shard_1[0:2]))
    torch.testing.assert_close(embedding.weight, expected)


# --------------------------------------------------------------------------- #
# Source-level oracles: the reconciliation onto #53896 must have fully
# removed the private hash helper and never construct the generic resident
# embedding directly — a bug here would still pass every behavioral test
# above by coincidence (e.g. a leftover dead `_hash_ngram_ids` method that
# nothing calls), so these inspect the actual source text/AST.
# --------------------------------------------------------------------------- #


def test_ple_layer_has_no_private_hash_helper() -> None:
    source = inspect.getsource(ple_layer_module)
    assert "_hash_ngram_ids" not in source


def test_ple_mmap_has_no_private_hash_helper() -> None:
    source = inspect.getsource(ple_mmap)
    assert "_hash_ngram_ids" not in source


def test_ple_layer_constructs_ple_vocab_embedding_once_in_env_off_arm() -> None:
    """`PLEVocabParallelEmbedding(` must appear exactly once, as the env-off
    construction — never a bare, generic `VocabParallelEmbedding(`."""
    source = inspect.getsource(ple_layer_module)
    assert source.count("PLEVocabParallelEmbedding(") == 1
    assert "= VocabParallelEmbedding(" not in source
