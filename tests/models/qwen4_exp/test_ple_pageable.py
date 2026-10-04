# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-mapped (pageable-host) PLE storage."""

import glob
import json
import os
import struct
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.models.qwen4_exp.nvidia.ngram_embedding as ngram_embedding_module
from vllm.config.engram import EngramConfig
from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
)
from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
    Qwen4ExpPLEPageableHostEmbedding,
)
from vllm.models.qwen4_exp.nvidia.ple_pageable import (
    MappedTable,
    PagePrefetcher,
    PrefetchSource,
    discover_table_layout,
    require_pageable_access,
)

ROW = 160
PREFIX = "model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding"


def _shard_bytes(
    layer: int, shard: int, rows: int, width: int, salt: int = 0
) -> np.ndarray:
    base = np.arange(rows * width, dtype=np.int64) * 7 + shard * 131 + layer * 17
    base += salt
    return base.astype(np.uint8).reshape(rows, width)


def _write(path, tensors: dict[str, tuple[str, list[int], bytes]], pad: int = 7):
    """Write a safetensors file whose data section starts unaligned."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, blob) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        blobs.append(blob)
    raw = json.dumps(header).encode() + b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"".join(blobs))


def _checkpoint(tmp_path, shards_by_file, layer=1, dtype="F8_E4M3", width=ROW, salt=0):
    """shards_by_file: [{shard_index: rows}, ...], one dict per file."""
    for i, shards in enumerate(shards_by_file):
        tensors = {
            f"{PREFIX.format(layer=layer)}.shard_{s}.weight": (
                dtype,
                [rows, width // (2 if dtype == "BF16" else 1)],
                _shard_bytes(layer, s, rows, width, salt).tobytes(),
            )
            for s, rows in shards.items()
        }
        _write(tmp_path / f"model-{i:05d}.safetensors", tensors, pad=5 + i)
    return str(tmp_path)


def _files(model: str) -> list[str]:
    """What the loader hands over: every safetensors file, index-filtered."""
    return filter_duplicate_safetensors_files(
        sorted(glob.glob(os.path.join(model, "*.safetensors"))),
        model,
        "model.safetensors.index.json",
    )


def _reference(shards: dict[int, int], layer=1, width=ROW) -> np.ndarray:
    return np.concatenate(
        [_shard_bytes(layer, s, r, width) for s, r in sorted(shards.items())]
    )


# ---------------------------------------------------------------- discovery


def test_layout_spans_files_and_accepts_short_last_shard(tmp_path):
    model = _checkpoint(tmp_path, [{0: 3, 2: 3}, {1: 3, 3: 2}])
    layout = discover_table_layout(_files(model), 1, 11, ROW, torch.float8_e4m3fn, 4)
    assert layout.rows_per_shard == 3
    assert [s.rows for s in layout.shards] == [3, 3, 3, 2]
    assert len({s.path for s in layout.shards}) == 2


def test_layout_ignores_other_layers(tmp_path):
    _checkpoint(tmp_path, [{0: 3, 1: 3, 2: 3, 3: 3}], layer=5)
    model = str(tmp_path)
    (tmp_path / "model-00000.safetensors").rename(tmp_path / "other.safetensors")
    _checkpoint(tmp_path, [{0: 3, 1: 3, 2: 3, 3: 3}], layer=1)
    layout = discover_table_layout(_files(model), 1, 12, ROW, torch.float8_e4m3fn, 4)
    assert all("model-00000" in s.path for s in layout.shards)


@pytest.mark.parametrize(
    ("shards", "rows", "match"),
    [
        ({0: 3, 1: 3, 3: 3}, 12, "missing \\[2\\]"),
        ({0: 3, 1: 3, 2: 2}, 12, "missing \\[3\\]"),
        ({0: 3, 1: 3, 2: 3, 3: 3, 4: 3}, 12, "unexpected \\[4\\]"),
        ({0: 3, 1: 2, 2: 3, 3: 3}, 12, "shard 1 has shape"),
    ],
    ids=["missing", "short-coverage", "extra", "short-interior"],
)
def test_layout_refuses_inconsistent_shards(tmp_path, shards, rows, match):
    model = _checkpoint(tmp_path, [shards])
    with pytest.raises(ValueError, match=match):
        discover_table_layout(_files(model), 1, rows, ROW, torch.float8_e4m3fn, 4)


def test_layout_refuses_duplicate_shard(tmp_path):
    model = _checkpoint(tmp_path, [{0: 3, 1: 3}, {1: 3, 2: 3, 3: 3}])
    with pytest.raises(ValueError, match="appears twice"):
        discover_table_layout(_files(model), 1, 12, ROW, torch.float8_e4m3fn, 4)


def test_layout_refuses_dtype_mismatch(tmp_path):
    model = _checkpoint(tmp_path, [{0: 3, 1: 3, 2: 3, 3: 3}], dtype="BF16", width=320)
    with pytest.raises(ValueError, match="cannot convert"):
        discover_table_layout(_files(model), 1, 12, ROW, torch.float8_e4m3fn, 4)


def test_cpu_views_address_the_checkpoint_rows(tmp_path):
    shards = {0: 3, 1: 3, 2: 3, 3: 2}
    model = _checkpoint(tmp_path, [{0: 3, 3: 2}, {1: 3, 2: 3}])
    layout = discover_table_layout(_files(model), 1, 11, ROW, torch.float8_e4m3fn, 4)
    table = MappedTable(layout, torch.device("cpu"))
    ref = _reference(shards)
    rows = np.arange(11)
    got = np.stack([table.views[r // 3][r % 3] for r in rows])
    np.testing.assert_array_equal(got, ref)
    table.touch(rows, pool=None)  # CPU fault-in path must not raise


def _indexed_checkpoint_with_stale_file(tmp_path) -> str:
    """One indexed file plus a stale extra file holding the same four shards."""
    model = _checkpoint(tmp_path, [{0: 3, 1: 3, 2: 3, 3: 3}])
    _write(
        tmp_path / "model-stale.safetensors",
        {
            f"{PREFIX.format(layer=1)}.shard_{s}.weight": (
                "F8_E4M3",
                [3, ROW],
                _shard_bytes(1, s, 3, ROW).tobytes(),
            )
            for s in range(4)
        },
    )
    weight_map = {
        f"{PREFIX.format(layer=1)}.shard_{s}.weight": "model-00000.safetensors"
        for s in range(4)
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return model


def test_layout_follows_the_safetensors_index(tmp_path):
    """A stale extra file with duplicate shards is ignored when the index excludes it,
    exactly as the loader ignores it."""
    model = _indexed_checkpoint_with_stale_file(tmp_path)
    layout = discover_table_layout(_files(model), 1, 12, ROW, torch.float8_e4m3fn, 4)
    assert all(s.path.endswith("model-00000.safetensors") for s in layout.shards)
    all_files = sorted(glob.glob(os.path.join(model, "*.safetensors")))
    with pytest.raises(ValueError, match="appears twice"):
        discover_table_layout(all_files, 1, 12, ROW, torch.float8_e4m3fn, 4)


def test_resolve_checkpoint_files_through_the_real_loader(tmp_path):
    """The default loader's own preparation (not the test stand-in) resolves the
    files, so a change to its interface fails here rather than at serve time."""
    from vllm.config.load import LoadConfig

    model = _indexed_checkpoint_with_stale_file(tmp_path)
    model_config = SimpleNamespace(model=model, model_weights=None, revision=None)
    files = ngram_embedding_module.resolve_checkpoint_files(model_config, LoadConfig())
    assert [os.path.basename(f) for f in files] == ["model-00000.safetensors"]


def _rss_kib() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def test_zero_mapping_commits_no_memory_when_read():
    rows, width = (32 << 20) // ROW, ROW
    table = MappedTable.zeros(rows, width, torch.device("cpu"))
    before = _rss_kib()
    assert int(table.views[0][:: 4096 // width].sum()) == 0  # touch every page
    table.touch(np.arange(0, rows, 4096 // width), pool=None)
    grown_kib = _rss_kib() - before
    assert grown_kib < 4096, f"touching 32 MiB of zeros committed {grown_kib} KiB"


def _mapped_embedding(
    files_ref: dict, rows: int = 12, parts: int = 4
) -> Qwen4ExpPLEPageableHostEmbedding:
    """A CPU-only embedding with just the state binding needs."""
    emb = object.__new__(Qwen4ExpPLEPageableHostEmbedding)
    torch.nn.Module.__init__(emb)
    emb.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.empty(0, ROW, dtype=torch.float8_e4m3fn), requires_grad=False
        ),
    )
    emb.org_vocab_size, emb.embedding_dim, emb.layer_index = rows, ROW, 1
    emb.table, emb._rebind_pending = None, False
    emb._pending_table, emb._reload_error = None, None
    emb._prefetch_buffer = torch.empty(0)
    emb._load_format = "auto"
    emb._model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(split_ngram_parts=parts)
    )
    emb._load_config = None
    return emb


def _mapped_rows(emb) -> np.ndarray:
    t = emb.table
    return np.stack(
        [
            t.views[r // t.rows_per_shard][r % t.rows_per_shard]
            for r in range(emb.org_vocab_size)
        ]
    )


def test_reload_from_disk_remaps_to_the_new_checkpoint(tmp_path, monkeypatch):
    """Load A, then reload_weights(weights_path=B): the table must follow B."""
    shards = {0: 3, 1: 3, 2: 3, 3: 3}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _checkpoint(tmp_path / "a", [shards])
    b = _checkpoint(tmp_path / "b", [shards], salt=97)
    source = {"dir": a}
    monkeypatch.setattr(
        ngram_embedding_module,
        "resolve_checkpoint_files",
        lambda model_config, load_config: _files(source["dir"]),
    )
    emb = _mapped_embedding(source)
    emb.bind_storage_after_loading()  # first load: A
    np.testing.assert_array_equal(_mapped_rows(emb), _reference(shards))
    # The reload streams B's shards through load_weights, then rebinds.
    source["dir"] = b
    for s, rows in shards.items():
        raw = torch.from_numpy(_shard_bytes(1, s, rows, ROW, salt=97).copy())
        emb.record_incoming_shard(s * 3, raw.view(torch.float8_e4m3fn))
    assert emb._rebind_pending
    emb.bind_storage_after_loading()
    np.testing.assert_array_equal(
        _mapped_rows(emb),
        np.concatenate([_shard_bytes(1, s, 3, ROW, salt=97) for s in range(4)]),
    )
    assert not emb._rebind_pending


def _shard_tensor(shard: int, rows: int, salt: int = 0) -> torch.Tensor:
    raw = torch.from_numpy(_shard_bytes(1, shard, rows, ROW, salt).copy())
    return raw.view(torch.float8_e4m3fn)


def test_reload_from_memory_is_rejected(tmp_path, monkeypatch):
    """Weights delivered from memory cannot be mapped; never serve the old table."""
    shards = {0: 3, 1: 3, 2: 3, 3: 3}
    a = _checkpoint(tmp_path, [shards])
    monkeypatch.setattr(
        ngram_embedding_module,
        "resolve_checkpoint_files",
        lambda model_config, load_config: _files(a),
    )
    emb = _mapped_embedding({})
    emb.bind_storage_after_loading()
    old = emb.table
    with pytest.raises(ValueError, match="differs from the mapped checkpoint"):
        emb.record_incoming_shard(0, _shard_tensor(0, 3, salt=5))
    # A rejected reload stays rejected: binding again raises, and the previous
    # mapping is neither kept silently in use nor replaced.
    for _ in range(2):
        with pytest.raises(RuntimeError, match="rejected"):
            emb.bind_storage_after_loading()
    assert emb.table is old


def test_reload_rejects_a_single_changed_row(tmp_path, monkeypatch):
    """One changed row in the middle of a shard (no sample would hit it) is caught."""
    shards = {0: 10, 1: 10}
    a = _checkpoint(tmp_path, [shards])
    monkeypatch.setattr(
        ngram_embedding_module,
        "resolve_checkpoint_files",
        lambda model_config, load_config: _files(a),
    )
    emb = _mapped_embedding({}, rows=20, parts=2)
    emb.bind_storage_after_loading()
    incoming = _shard_tensor(0, 10).view(torch.uint8).clone()
    incoming[3, 17] ^= 0x5A
    with pytest.raises(ValueError, match="PLE row 3 received"):
        emb.record_incoming_shard(0, incoming.view(torch.float8_e4m3fn))


def test_reload_from_disk_after_a_rejection_recovers(tmp_path, monkeypatch):
    """A new load attempt after a rejection is verified from scratch."""
    shards = {0: 3, 1: 3, 2: 3, 3: 3}
    a = _checkpoint(tmp_path, [shards])
    monkeypatch.setattr(
        ngram_embedding_module,
        "resolve_checkpoint_files",
        lambda model_config, load_config: _files(a),
    )
    emb = _mapped_embedding({})
    emb.bind_storage_after_loading()
    with pytest.raises(ValueError):
        emb.record_incoming_shard(0, _shard_tensor(0, 3, salt=5))
    for s, rows in shards.items():  # reload from disk: the same files stream in
        emb.record_incoming_shard(s * 3, _shard_tensor(s, rows))
    emb.bind_storage_after_loading()
    np.testing.assert_array_equal(_mapped_rows(emb), _reference(shards))


# ------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "arch",
    ["DeepseekV41ForCausalLM"],
)
def test_config_rejects_architectures_without_mapped_storage(arch, monkeypatch):
    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms.current_platform, "is_cuda_alike", lambda: True, raising=False
    )
    model_config = SimpleNamespace(
        architecture=arch,
        hf_text_config=SimpleNamespace(engram_layer_ids=[1]),
    )
    EngramConfig(checkpoint_mapped=False).verify_model_config(model_config)
    with pytest.raises(ValueError, match="checkpoint_mapped is implemented for"):
        EngramConfig(checkpoint_mapped=True).verify_model_config(model_config)


@pytest.mark.parametrize("is_cuda", [True, False])
def test_config_rejects_mapping_off_cuda(is_cuda, monkeypatch):
    import vllm.platforms

    platform = vllm.platforms.current_platform
    monkeypatch.setattr(platform, "is_cuda_alike", lambda: True, raising=False)
    monkeypatch.setattr(platform, "is_cuda", lambda: is_cuda, raising=False)
    model_config = SimpleNamespace(
        architecture="Qwen4ExpForCausalLM",
        hf_text_config=SimpleNamespace(ple_layer_ids=[1]),
    )
    EngramConfig(checkpoint_mapped=False).verify_model_config(model_config)
    if is_cuda:
        EngramConfig(checkpoint_mapped=True).verify_model_config(model_config)
    else:
        with pytest.raises(ValueError, match="implemented for CUDA only"):
            EngramConfig(checkpoint_mapped=True).verify_model_config(model_config)


def test_config_rejects_shared_memory_with_mapping():
    with pytest.raises(ValueError, match="checkpoint_mapped"):
        EngramConfig(checkpoint_mapped=True, dp_shared_memory=True)


def test_config_rejects_embedding_across_dp_with_mapping():
    with pytest.raises(ValueError, match="embedding_across_dp"):
        EngramConfig(checkpoint_mapped=True, embedding_across_dp=True)


def test_config_does_not_default_shared_memory_when_mapped():
    config = EngramConfig(checkpoint_mapped=True)

    class _Parallel:
        data_parallel_size = 4
        enable_elastic_ep = False

    config.resolve_dp_shared_memory(_Parallel())
    assert config.dp_shared_memory is False


# ---------------------------------------------------------------------- GPU


def _pageable_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        require_pageable_access(0)
    except RuntimeError:
        return False
    return True


requires_pageable = pytest.mark.skipif(
    not _pageable_gpu(), reason="needs a GPU with pageable host-page-table access"
)


def _poison_allocator(rows: int, width: int) -> None:
    for _ in range(4):
        junk = torch.full((rows, width), 0xFF, dtype=torch.uint8, device="cuda")
        del junk


@requires_pageable
@pytest.mark.parametrize(
    ("dtype", "st_dtype", "width"),
    [(torch.float8_e4m3fn, "F8_E4M3", ROW), (torch.bfloat16, "BF16", 2 * ROW)],
    ids=["fp8", "bf16"],
)
def test_gather_is_bit_exact_with_etp_range(tmp_path, dtype, st_dtype, width):
    shards = {0: 3, 1: 3, 2: 3, 3: 2}
    model = _checkpoint(
        tmp_path, [{0: 3, 3: 2}, {1: 3, 2: 3}], dtype=st_dtype, width=width
    )
    layout = discover_table_layout(
        _files(model), 1, 11, width // dtype.itemsize, dtype, 4
    )
    table = MappedTable(layout, torch.device("cuda"))
    ref = _reference(shards, width=width)
    ids = torch.tensor([0, 2, 3, 5, 9, 10, 11, 12, -1, 10**9], device="cuda")
    _poison_allocator(ids.numel(), width)
    # This rank owns rows [3, 10): everything else must come back as zeros.
    out = torch.empty(ids.numel(), width, dtype=torch.uint8, device="cuda")
    out.fill_(0xAB)
    table.gather_into(ids, out, 3, 10)
    got = out.cpu().numpy()
    owned = [i for i, r in enumerate(ids.tolist()) if 3 <= r < 10]
    np.testing.assert_array_equal(got[owned], ref[ids.cpu().numpy()[owned]])
    assert not np.delete(got, owned, axis=0).any()


@requires_pageable
def test_graph_replay_zeroes_rows_that_become_invalid(tmp_path):
    model = _checkpoint(tmp_path, [{0: 3, 1: 3, 2: 3, 3: 2}])
    layout = discover_table_layout(_files(model), 1, 11, ROW, torch.float8_e4m3fn, 4)
    table = MappedTable(layout, torch.device("cuda"))
    ids = torch.tensor([0, 4, 8, 10], device="cuda")
    out = torch.empty(4, ROW, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        table.gather_into(ids, out, 0, 11)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        table.gather_into(ids, out, 0, 11)
    graph.replay()
    torch.accelerator.synchronize()
    assert out.any()
    ids.copy_(torch.tensor([11, -5, 10**12, 10], device="cuda"))
    graph.replay()
    torch.accelerator.synchronize()
    got = out.cpu().numpy()
    assert not got[:3].any()
    np.testing.assert_array_equal(got[3], _reference({0: 3, 1: 3, 2: 3, 3: 2})[10])


@requires_pageable
def test_full_graph_captures_prefetch_and_forward(tmp_path, monkeypatch):
    """start_prefetch + forward fit in one FULL cudagraph and replay new ids.

    The pinned backend joins its side stream in _finalize_prefetch; this
    backend looks up on the current stream, and joining the (uncaptured) side
    stream anyway fails the capture with cudaErrorStreamCaptureIsolation.
    """
    shards = {0: 3, 1: 3, 2: 3, 3: 3}
    model = _checkpoint(tmp_path, [shards])
    monkeypatch.setattr(
        ngram_embedding_module,
        "resolve_checkpoint_files",
        lambda model_config, load_config: _files(model),
    )
    emb = _mapped_embedding({"dir": model})
    heads, tokens = 2, 3
    emb._prefetch_buffer = torch.empty(
        4, heads, ROW, dtype=torch.float8_e4m3fn, device="cuda"
    )
    emb._output_dim = heads * ROW
    emb._prefetch_stream = torch.cuda.Stream()
    emb.tp_size, emb.etp_data_parallel_size = 1, 1
    emb.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=12)
    emb.bind_storage_after_loading()
    hidden = torch.empty(tokens, 8, device="cuda")
    ids = torch.tensor([[0, 5], [7, 11], [3, 3]], device="cuda")

    def step():
        emb.start_prefetch(hidden, ids)
        return emb.forward(hidden)

    ref = _reference(shards)
    eager = step().view(torch.uint8).cpu().numpy()
    np.testing.assert_array_equal(
        eager, ref[ids.cpu().numpy()].reshape(tokens, heads * ROW)
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    ids.copy_(torch.tensor([[1, 2], [4, 6], [9, 10]], device="cuda"))
    graph.replay()
    torch.accelerator.synchronize()
    np.testing.assert_array_equal(
        out.view(torch.uint8).cpu().numpy(),
        ref[ids.cpu().numpy()].reshape(tokens, heads * ROW),
    )


@requires_pageable
def test_zero_mapping_reads_zeros_without_committing_memory():
    table = MappedTable.zeros(1 << 20, ROW, torch.device("cuda"))
    ids = torch.tensor([0, 12345, (1 << 20) - 1], device="cuda")
    out = torch.full((3, ROW), 0xFF, dtype=torch.uint8, device="cuda")
    table.gather_into(ids, out, 0, 1 << 20)
    assert not out.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA events")
def test_prefetcher_staging_survives_short_then_long_steps():
    """The per-step input_ids view changes size; staging must be at capacity."""
    table = MappedTable.zeros(1000, ROW, torch.device("cuda"))
    source = PrefetchSource(
        lambda: table,
        (0, 1000),
        lambda: (lambda ids, qsl, ctx: torch.zeros(ids.shape[0], 16, dtype=torch.long)),
    )
    prefetcher = PagePrefetcher([source], torch.device("cuda"), 4096, 16, 2)
    qsl = torch.zeros(17, dtype=torch.int32, device="cuda")
    ctx = torch.zeros(16, 2, dtype=torch.int32, device="cuda")
    for num_tokens in (64, 4096, 7):
        ids = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
        qsl[1:] = num_tokens
        prefetcher.prepare(ids, qsl, ctx, 1, num_tokens)
        torch.accelerator.synchronize()
        deadline = time.monotonic() + 10
        while prefetcher.free.qsize() < prefetcher.SLOTS:  # worker released the slot
            assert time.monotonic() < deadline
            time.sleep(0.01)
    assert prefetcher.slots[0][0].shape[0] == 4096
    assert prefetcher.skipped == 0


# ---------------------------------------------------------------- readahead fill


def _table(tmp_path) -> MappedTable:
    model = _checkpoint(tmp_path, [{0: 3, 3: 2}, {1: 3, 2: 3}])
    layout = discover_table_layout(_files(model), 1, 11, ROW, torch.float8_e4m3fn, 4)
    return MappedTable(layout, torch.device("cpu"))


def test_fill_populates_the_rows(tmp_path):
    table = _table(tmp_path)
    rows = np.array([10, 0, 4, 7])
    assert table._fill([(table.views[r // 3], np.array([r % 3])) for r in rows])
    got = np.stack([table.views[r // 3][r % 3] for r in rows])
    np.testing.assert_array_equal(got, _reference({0: 3, 1: 3, 2: 3, 3: 2})[rows])


def test_touch_falls_back_when_populate_is_unsupported(tmp_path):
    """Before Linux 5.14 MADV_POPULATE_READ fails: touch the rows instead."""
    table = _table(tmp_path)

    class _NoPopulate:
        def madvise(self, addr, length, advice):
            return -1 if advice == 22 else 0

    table._libc = _NoPopulate()
    table.touch(np.arange(11), pool=None)
    assert not table._can_populate
    table.touch(np.arange(11), pool=None)  # the plain touch from now on
