# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1 tests for the mmap-backed PLE table (VLLM_PLE_MMAP).

No GPU, no real checkpoint: synthetic fp8 safetensors fixtures stand in for
the checkpoint's PLE shards, and the custom op is
exercised through its CPU dispatch key.
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch
from torch import nn

import vllm.envs as envs
import vllm.forward_context as forward_context
import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp.nvidia.model as model_module
import vllm.models.qwen4_exp.nvidia.ple_mmap as ple_mmap
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.fp8_utils import is_fp8
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)

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


def _synthetic_weight(vocab: int, cols: int, layer_idx: int = 0) -> torch.Tensor:
    """Deterministic, layer-dependent fp8 values (never all-zero/uniform, and
    distinguishable across layers so per-layer-keying tests are meaningful).
    """
    raw = torch.arange(vocab * cols, dtype=torch.float32).reshape(vocab, cols)
    raw = torch.remainder(raw + layer_idx * 97, 6.0) - 3.0
    return raw.to(torch.float8_e4m3fn)


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
) -> torch.Tensor:
    """Write one PLE layer's shard + weight_scale tensors as synthetic
    safetensors files (no model.safetensors.index.json, matching the real
    checkpoint). Returns the full logical [vocab, cols] fp8 table.
    """
    prefix = (
        f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding"
    )
    shard_size = (vocab + parts - 1) // parts
    full = _synthetic_weight(vocab, cols, layer_idx)
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
    # warning_once: _record_warnings only intercepts warning_once.
    warnings: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        ple_mmap.logger,
        "warning",
        lambda msg, *args: warnings.append((msg, args)),
    )

    ids = np.array([0, 5, 12, 20, 31, 39], dtype=np.int64)
    got = torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn)

    assert torch.equal(got, full[ids])
    assert advised == []
    assert table._latencies_ms[-1][3] == 6
    assert len(warnings) == 1
    assert warnings[0][1] == (6, 2)
    table.close()


def test_bound_exceeded_warns_exactly_once_per_table_across_varying_run_counts(
    tmp_path: Path,
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

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    ple_mmap.logger.addHandler(handler)
    try:
        for ids in (
            np.array([0, 39], dtype=np.int64),  # 2 shards -> 2 runs
            np.array([0, 5, 39], dtype=np.int64),  # a different run count
            np.array([0, 5, 12, 39], dtype=np.int64),  # a different run count again
        ):
            table.gather(ids)
    finally:
        ple_mmap.logger.removeHandler(handler)
        table.close()

    bound_records = [r for r in records if "readahead skipped" in r.getMessage()]
    assert len(bound_records) == 1


def test_readahead_bound_skip_avoids_materializing_the_run_list(
    tmp_path: Path,
) -> None:
    """The count-only pre-pass must stay cheap even when the gather is large
    enough that materializing the (fd, offset, length) run list would be
    measurable: a bound-skipped gather's populate cost must stay close to
    the readahead=0 arm's (always exactly 0 — the pre-pass never even
    runs), not scale with row count the way the materializing path does.
    """
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    n_shards, rows_per_shard, cols = 128, 4, 4
    vocab = n_shards * rows_per_shard
    full = _synthetic_weight(vocab, cols)
    tensors: dict[str, torch.Tensor] = {
        f"{prefix}.weight_scale": torch.tensor([1.0], dtype=torch.bfloat16)
    }
    for shard_idx in range(n_shards):
        start = shard_idx * rows_per_shard
        tensors[f"{prefix}.shard_{shard_idx}.weight"] = full[
            start : start + rows_per_shard
        ]
    safetensors.torch.save_file(
        tensors, str(tmp_path / "model-ple-0-00000.safetensors")
    )
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    # One (non-adjacent) row from every shard: each segment coalesces to
    # exactly one run, so the run list genuinely has n_shards entries.
    ids = np.array(
        [shard_idx * rows_per_shard for shard_idx in range(n_shards)], dtype=np.int64
    )

    def _table(readahead: int) -> ple_mmap.MmapPleTable:
        return ple_mmap.MmapPleTable(
            layer_shards.shards,
            rows_per_shard,
            cols,
            torch.float8_e4m3fn,
            workers=1,
            chunk=8,
            model_path=str(tmp_path),
            readahead=readahead,
        )

    active = _table(n_shards)  # never exceeds the bound: materializes + advises
    skipped = _table(1)  # exceeds on every gather: count-only path only
    try:
        for _ in range(5):
            active.gather(ids)
            skipped.gather(ids)
        # min(), not mean(): isolates the cost this test cares about from
        # scheduler noise, which only ever pushes a sample up.
        active_populate = min(sample[1] for sample in active._latencies_ms)
        skipped_populate = min(sample[1] for sample in skipped._latencies_ms)

        # Measured ratio is a stable ~0.27 across 128-1024 shards (the
        # per-segment numpy count survives; only the coalesced (fd, offset,
        # length) list and the posix_fadvise calls are skipped) — half that
        # leaves comfortable margin against box-to-box noise without being
        # so loose the assertion stops meaning anything.
        assert skipped_populate < 0.5 * active_populate
    finally:
        active.close()
        skipped.close()


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
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    full = _synthetic_weight(9, 3)
    path = tmp_path / "model-ple-0-00000.safetensors"
    safetensors.torch.save_file(
        {
            f"{prefix}.shard_0.weight": full[0:3],
            f"{prefix}.shard_1.weight": full[3:6],
            f"{prefix}.shard_2.weight": full[6:9],
            f"{prefix}.weight_scale": torch.tensor([1.0], dtype=torch.bfloat16),
        },
        str(path),
    )
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
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    full = _synthetic_weight(9, 2)
    safetensors.torch.save_file(
        {
            f"{prefix}.shard_0.weight": full[0:3],
            f"{prefix}.shard_1.weight": full[3:6],
            f"{prefix}.shard_2.weight": full[6:9],
            f"{prefix}.weight_scale": torch.tensor([1.0], dtype=torch.bfloat16),
        },
        str(tmp_path / "model-ple-0-00000.safetensors"),
    )
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
    assert table._fds

    table.close()
    table.close()  # idempotent: must not raise (e.g. a double os.close())

    assert not table._fds
    assert all(mm is None for mm in table.mm)


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
# Custom op registration
# --------------------------------------------------------------------------- #


def test_op_is_registered_under_platform_default_and_cpu_dispatch_keys() -> None:
    assert hasattr(torch.ops.vllm, ple_mmap.OP_NAME)
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        ple_mmap.QUALIFIED_OP_NAME, "CPU"
    )
    if torch.cuda.is_available():
        assert torch._C._dispatch_has_kernel_for_dispatch_key(
            ple_mmap.QUALIFIED_OP_NAME, "CUDA"
        )
    # The output arg's alias annotation ("(a3!)") is what tells
    # torch.compile the write to `output` must survive functionalization —
    # without it (mutates_args=[]), a compiled graph can drop the write and
    # the caller reads back its own uninitialized new_empty buffer instead
    # of the gathered rows. Registration is module-global and sticky within
    # a pytest process (a second import cannot re-register), so this pins
    # the CURRENT registration's schema string rather than re-registering.
    schema = str(getattr(torch.ops.vllm, ple_mmap.OP_NAME).default._schema)
    assert "!) output" in schema, schema
    assert schema.endswith("-> ()")
    # Exercise the CPU key directly: this is what every other test below
    # relies on to run without a GPU. The widened op calls
    # ple_embedding_module._hash_ngram_ids THEN .ngram_embedding(...), so
    # the fake stands in for both.
    hash_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    gather_calls: list[torch.Tensor] = []

    class _FakePleEmbeddingModule:
        def _hash_ngram_ids(
            self,
            input_ids: torch.Tensor,
            query_start_loc: torch.Tensor,
            ngram_context: torch.Tensor,
        ) -> torch.Tensor:
            hash_calls.append((input_ids, query_start_loc, ngram_context))
            return torch.zeros((input_ids.reshape(-1).shape[0], 2), dtype=torch.long)

        def ngram_embedding(self, ngram_ids: torch.Tensor) -> torch.Tensor:
            gather_calls.append(ngram_ids)
            return torch.zeros((*ngram_ids.shape, 2), dtype=torch.float8_e4m3fn)

    fake_layer = SimpleNamespace(ple_embedding=_FakePleEmbeddingModule())
    ctx = SimpleNamespace(no_compile_layers={"layer0": fake_layer})
    input_ids = torch.zeros((2,), dtype=torch.long)
    query_start_loc = torch.tensor([0, 2], dtype=torch.long)
    ngram_context = torch.zeros((1, 4), dtype=torch.long)
    output = torch.empty((2, 4), dtype=torch.float8_e4m3fn)

    with forward_context.override_forward_context(ctx):
        torch.ops.vllm.qwen4_exp_ple_mmap_forward(
            input_ids, query_start_loc, ngram_context, output, "layer0"
        )

    assert len(hash_calls) == 1
    assert len(gather_calls) == 1
    assert torch.equal(output, torch.zeros_like(output))


def test_op_raises_named_error_when_layer_name_does_not_resolve() -> None:
    ctx = SimpleNamespace(no_compile_layers={"layer0": SimpleNamespace()})
    input_ids = torch.zeros((1,), dtype=torch.long)
    query_start_loc = torch.tensor([0, 1], dtype=torch.long)
    ngram_context = torch.zeros((1, 4), dtype=torch.long)
    output = torch.empty((1, 4), dtype=torch.float8_e4m3fn)

    with (
        forward_context.override_forward_context(ctx),
        pytest.raises(RuntimeError, match="does not resolve to a PLE layer"),
    ):
        torch.ops.vllm.qwen4_exp_ple_mmap_forward(
            input_ids, query_start_loc, ngram_context, output, "layer0"
        )


# --------------------------------------------------------------------------- #
# (c) CUDAGraph startup refusal, parametrized over every mode.
# --------------------------------------------------------------------------- #


def _compilation_config(
    *,
    mode: CompilationMode,
    cudagraph_mode: CUDAGraphMode,
    splitting_ops: list[str] | None,
) -> CompilationConfig:
    return CompilationConfig(
        mode=mode, cudagraph_mode=cudagraph_mode, splitting_ops=splitting_ops
    )


@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_check_cudagraph_safety_refuses_full_cudagraph_modes(
    cudagraph_mode: CUDAGraphMode,
) -> None:
    cc = _compilation_config(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=cudagraph_mode,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    # Asserted directly against the enum values, never through
    # has_full_cudagraphs() (a rebase-fragile one-liner).
    assert cudagraph_mode in (
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    with pytest.raises(RuntimeError, match="piecewise-only CUDA graphs"):
        ple_mmap.check_cudagraph_safety(cc)


def test_check_cudagraph_safety_accepts_piecewise_compiled_with_op_split() -> None:
    cc = _compilation_config(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    assert cc.cudagraph_mode is CUDAGraphMode.PIECEWISE

    ple_mmap.check_cudagraph_safety(cc)  # must not raise


def test_check_cudagraph_safety_refuses_non_compile_mode() -> None:
    """mode=NONE is enforce-eager: it does not fully suppress capture on this
    model and leaves splitting_ops empty."""
    cc = _compilation_config(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    assert cc.mode is CompilationMode.NONE

    with pytest.raises(RuntimeError, match="VLLM_COMPILE"):
        ple_mmap.check_cudagraph_safety(cc)


def test_check_cudagraph_safety_refuses_when_op_missing_from_splitting_ops() -> None:
    """Catches an operator-supplied -cc.splitting_ops list, or an
    attn-fusion reset, that silently drops our op."""
    cc = _compilation_config(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=["vllm::some_other_op"],
    )

    with pytest.raises(RuntimeError, match="splitting_ops"):
        ple_mmap.check_cudagraph_safety(cc)


def test_set_splitting_ops_for_v1_emits_the_new_op() -> None:
    cc = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    cc.set_splitting_ops_for_v1(all2all_backend="naive", data_parallel_size=1)

    assert ple_mmap.QUALIFIED_OP_NAME in cc.splitting_ops


def test_set_splitting_ops_for_v1_output_satisfies_the_cudagraph_guard() -> None:
    """(Ordering, L): a CompilationConfig built through its NORMAL init
    path (set_splitting_ops_for_v1), not hand-constructed with
    splitting_ops pre-set, must both contain our op AND satisfy
    check_cudagraph_safety — the membership assertion runs BEFORE the
    guard call, proving the two checks agree on the same real object."""
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.PIECEWISE
    )
    cc.set_splitting_ops_for_v1(all2all_backend="naive", data_parallel_size=1)

    assert ple_mmap.QUALIFIED_OP_NAME in cc.splitting_ops

    ple_mmap.check_cudagraph_safety(cc)  # must not raise


# --------------------------------------------------------------------------- #
# check_cudagraph_safety is unit-tested as a free function
# above, but its CALL from Qwen4ExpNGramEmbedding.__init__ (ple_layer.py)
# was never exercised — deleting that call left the whole suite green.
# Each case pins the OTHER two predicates to pass, so a failure here can
# only mean the one predicate under test.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_ngram_embedding_construction_refuses_full_cudagraph_modes(
    monkeypatch: pytest.MonkeyPatch, cudagraph_mode: CUDAGraphMode
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=cudagraph_mode,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    vllm_config = SimpleNamespace(compilation_config=cc, model_config=SimpleNamespace())

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="piecewise-only CUDA graphs"),
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


def test_ngram_embedding_construction_refuses_non_compile_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()
    cc = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    vllm_config = SimpleNamespace(compilation_config=cc, model_config=SimpleNamespace())

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="VLLM_COMPILE"),
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


def test_ngram_embedding_construction_refuses_missing_splitting_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=["vllm::some_other_op"],
    )
    vllm_config = SimpleNamespace(compilation_config=cc, model_config=SimpleNamespace())

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="splitting_ops"),
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
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    vllm_config = SimpleNamespace(
        compilation_config=cc, model_config=_model_config(tmp_path)
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


# --------------------------------------------------------------------------- #
# build_tables: construction hook, fail-closed validation, per-layer keying.
# --------------------------------------------------------------------------- #


def _fake_ple_layer(
    layer_idx: int, embedding: ple_mmap.MmapNgramEmbedding, split_ngram_parts: int
) -> SimpleNamespace:
    return SimpleNamespace(
        layer_idx=layer_idx,
        ple_embedding=SimpleNamespace(
            ngram_embedding=embedding, split_ngram_parts=split_ngram_parts
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
    """VLLM_PLE_MMAP_WORKERS/_CHUNK/_READAHEAD must reach the attached
    MmapPleTable's workers/chunk/readahead in the right order — a
    swapped-args regression would still construct a table, just with the
    wrong concurrency knobs, and nothing else would notice."""
    monkeypatch.setenv("VLLM_PLE_MMAP_WORKERS", "3")
    monkeypatch.setenv("VLLM_PLE_MMAP_CHUNK", "7")
    monkeypatch.setenv("VLLM_PLE_MMAP_READAHEAD", "11")
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})

    ple_mmap.build_tables(_model_config(tmp_path), cc)

    assert emb.table is not None
    assert emb.table.workers == 3
    assert emb.table.chunk == 7
    assert emb.table.readahead == 11


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
    """F8_E5M2 was dropped from _FP8_DTYPES because is_fp8() does not
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
    _write_ple_layer(
        tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25, write_scale=True
    )
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    shard_1 = shard_1.to(torch.float8_e4m3fn)

    module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
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
    warnings: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        ple_mmap.logger,
        "warning",
        lambda msg, *args: warnings.append((msg, args)),
    )

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
# (a) env-on vs env-off FORWARD equivalence, through the CPU dispatch key.
# The op boundary is the whole forward: env-on hashes AND gathers inside the
# op. Both arms call the SAME Qwen4ExpNGramEmbedding._hash_ngram_ids, so
# this test proves the env-on path loads the RIGHT weights and gathers and
# dequantizes them the same way the stock VocabParallelEmbedding path
# does — it does NOT independently verify the hashing math itself: a bug
# in _hash_ngram_ids would move both arms identically and cancel out here.
# Hashing correctness is pinned separately by
# test_hash_ngram_ids_matches_golden_ids below.
# --------------------------------------------------------------------------- #


def test_env_on_off_forward_equivalence_fp8_and_dequantized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """env-off: a real stock VocabParallelEmbedding under
    Qwen4ExpPLEFp8EmbeddingMethod (real FP8 weight + weight_scale
    Parameters, mirrors test_ple.py's _make_fp8_embedding_layer). env-on:
    an MmapNgramEmbedding placeholder attached to shard files holding the
    IDENTICAL weight values, driven through the REGISTERED widened op via
    its CPU dispatch key. Same input_ids/query_start_loc/ngram_context on
    both sides; compared byte-equal at fp8 AND through
    _dequantize_embeddings to bf16. Proves weight-loading/gather/dequant
    equivalence between the two paths, not hashing correctness (both
    arms share the same _hash_ngram_ids call, see module comment above).
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
    assert isinstance(stock.ngram_embedding, embedding_module.VocabParallelEmbedding)
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

    reference = stock.forward(input_ids, query_start_loc, ngram_context)
    assert reference.dtype == torch.float8_e4m3fn

    # --- env-on: mmap placeholder backed by shards holding the SAME
    # weight values, driven through the registered custom op. ---
    _write_ple_layer(
        tmp_path, layer_idx=1, vocab=vocab, parts=parts, cols=head_dim, scale=scale
    )
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    vllm_config = SimpleNamespace(
        compilation_config=cc, model_config=_model_config(tmp_path)
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

    fake_ple_layer = SimpleNamespace(ple_embedding=mmap_module)
    ctx = SimpleNamespace(no_compile_layers={mmap_module.layer_name: fake_ple_layer})
    with forward_context.override_forward_context(ctx):
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


# --------------------------------------------------------------------------- #
# _hash_ngram_ids golden pin. The equivalence test above drives BOTH
# arms through the same _hash_ngram_ids call, so it cannot catch a bug in
# the hashing math itself (xor chain / remainder / offset) — a mutation
# there moves both arms identically and cancels out. This test freezes the
# exact output of a fixed, small, real Qwen4ExpNGramEmbedding on fixed
# inputs, so a hashing regression has to change these hardcoded numbers.
# --------------------------------------------------------------------------- #


def test_hash_ngram_ids_matches_golden_ids() -> None:
    """Golden values captured by running this exact scenario once and
    hardcoding the result — they pin the xor-chain/remainder/offset math
    in _hash_ngram_ids (ngram_size=3, heads_per_ngram=2, seed=1234,
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

    ngram_ids = module._hash_ngram_ids(input_ids, query_start_loc, ngram_context)

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
# Qwen4ExpNGramEmbedding.forward's mmap branch allocates
# the output buffer in the TABLE's dtype, not params_dtype — zero prior
# coverage exercised this exact allocation through the real forward().
# --------------------------------------------------------------------------- #


def test_mmap_forward_allocates_an_fp8_output_buffer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A regression here (e.g. back to params_dtype bf16) would leave the
    model serving unscaled embeddings — is_fp8() would stop firing and
    Qwen4ExpPLELayer._dequantize_embeddings would silently skip
    dequantization — while every test that exercises only the custom op or
    the placeholder in isolation stays green."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()  # ngram_size=3, heads_per_ngram=2 -> 4 heads
    layer_name = "model.language_model.layers.1.ple"
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
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
        compilation_config=cc, model_config=unresolvable_config
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

    # ple_embedding must be the REAL module (not a bare embedding wrapper):
    # the widened op calls ple_embedding_module._hash_ngram_ids(...)
    # before the gather, and only Qwen4ExpNGramEmbedding provides that.
    fake_ple_layer = SimpleNamespace(ple_embedding=module)
    ctx = SimpleNamespace(no_compile_layers={layer_name: fake_ple_layer})

    input_ids = torch.tensor([1, 2], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2], dtype=torch.long)
    ngram_context = torch.zeros((1, 4), dtype=torch.long)

    with forward_context.override_forward_context(ctx):
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
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
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

    assert isinstance(module.ngram_embedding, embedding_module.VocabParallelEmbedding)
    assert not isinstance(module.ngram_embedding, ple_mmap.MmapNgramEmbedding)


def test_default_off_forward_never_calls_the_mmap_gather_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With VLLM_PLE_MMAP unset, forward() must take the exact stock branch
    (direct call + .flatten(-2)), never the custom op."""
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
    op_calls: list[object] = []
    monkeypatch.setattr(
        torch.ops.vllm,
        ple_mmap.OP_NAME,
        lambda *a, **k: op_calls.append((a, k)),
        raising=False,
    )

    input_ids = torch.tensor([1, 2], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2], dtype=torch.long)
    ngram_context = torch.zeros((1, 4), dtype=torch.long)

    output = module.forward(input_ids, query_start_loc, ngram_context)

    assert len(calls) == 1  # the stock embedding was called directly
    assert not op_calls  # the custom op was never reached
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
