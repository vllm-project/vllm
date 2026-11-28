# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for reading safetensors checkpoint tensors straight to the GPU."""

import json
import struct
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import safe_open, save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import default_loader, gpu_staging
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU staging needs CUDA"
)


def _tensors() -> dict[str, torch.Tensor]:
    """Tensors of every staged dtype, with odd byte sizes so that most of them
    start at offsets that are not aligned to their element size."""
    g = torch.Generator().manual_seed(0)

    def ints(dtype, *shape):
        return torch.randint(-100, 100, shape, generator=g).to(dtype)

    def floats(dtype, *shape):
        return torch.randn(shape, generator=g).to(dtype)

    return {
        "layers.0.bool": torch.tensor([True, False, True]),
        "layers.0.u8": ints(torch.uint8, 7),
        "layers.0.i8": ints(torch.int8, 3, 5),
        "layers.0.i16": ints(torch.int16, 9),
        "layers.0.u16": torch.randint(0, 2**16, (5,), generator=g).to(torch.uint16),
        "layers.0.i32": ints(torch.int32, 2, 3),
        "layers.0.u32": torch.randint(0, 2**31, (3,), generator=g).to(torch.uint32),
        "layers.0.i64": ints(torch.int64, 11),
        "layers.0.u64": torch.randint(0, 2**62, (2,), generator=g).to(torch.uint64),
        "layers.0.f16": floats(torch.float16, 13, 3),
        "layers.0.bf16": floats(torch.bfloat16, 3, 5),
        "layers.0.f32": floats(torch.float32, 257, 129),
        "layers.0.f64": floats(torch.float64, 4),
        "layers.0.scalar": torch.tensor(3.5),
        "layers.0.f8_e4m3": floats(torch.float8_e4m3fn, 33, 17),
        "layers.0.f8_e5m2": floats(torch.float8_e5m2, 19),
        "layers.0.f8_e8m0": ints(torch.uint8, 5, 9).view(torch.float8_e8m0fnu),
        "layers.0.empty": torch.empty(0, 4),
        "layers.1.weight": floats(torch.bfloat16, 512, 300),
        "layers.1.skipme": floats(torch.float32, 1000),
        "layers.2.weight": floats(torch.float16, 96, 1024),
        "layers.2.bias": floats(torch.float32, 1023),
    }


@pytest.fixture
def checkpoint(tmp_path) -> str:
    path = str(tmp_path / "model.safetensors")
    save_file(_tensors(), path)
    return path


def _skip(name: str) -> bool:
    return name.endswith(".skipme")


def _assert_same(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    assert got.dtype == ref.dtype, name
    assert got.shape == ref.shape, name
    assert torch.equal(
        got.cpu().reshape(-1).view(torch.uint8), ref.reshape(-1).view(torch.uint8)
    ), name


def _check_against_safe_open(
    path: str, yielded: list[tuple[str, torch.Tensor]], skip=_skip
) -> None:
    with safe_open(path, framework="pt") as st:
        expected = [name for name in st.keys() if not skip(name)]  # noqa: SIM118
        assert sorted(name for name, _ in yielded) == sorted(expected)
        for name, tensor in yielded:
            _assert_same(name, tensor, st.get_tensor(name))


def _write_header_only(path, header: dict) -> None:
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)


def test_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_staging, "_CHUNK_BYTES", 100)
    monkeypatch.setattr(gpu_staging, "_MAX_TENSOR_BYTES", 150)
    path = tmp_path / "plan.safetensors"
    # (name, dtype, nbytes), in file order; the header lists them shuffled.
    layout = [
        ("a", "F32", 40),
        ("b", "U8", 40),
        ("c", "BF16", 40),  # range would exceed _CHUNK_BYTES: new range
        ("skipme", "F32", 8),  # skipped: the next tensor starts a new range
        ("d", "F16", 120),  # over _CHUNK_BYTES on its own: a range of its own
        ("e", "F4", 2),  # unknown dtype: mapped
        ("f", "F32", 160),  # over _MAX_TENSOR_BYTES: mapped
        ("g", "F32", 0),  # empty: mapped
        ("h", "I8", 4),
    ]
    header, offset = {}, 0
    for name, dtype, nbytes in layout:
        header[name] = {
            "dtype": dtype,
            "shape": [nbytes],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header = dict(reversed(header.items()))
    header["__metadata__"] = {"format": "pt"}
    _write_header_only(path, header)

    items, ranges = gpu_staging._plan(str(path), lambda n: n == "skipme")
    assert items == [
        ("range", 0),
        ("range", 1),
        ("range", 2),
        ("map", "e"),
        ("map", "f"),
        ("map", "g"),
        ("range", 3),
    ]
    entries = [[(e[0], e[3], e[4]) for e in r[2]] for r in ranges]
    assert entries == [
        [("a", 0, 40), ("b", 40, 40)],
        [("c", 0, 40)],
        [("d", 0, 120)],
        [("h", 0, 4)],
    ]
    for start, end, ents in ranges:
        assert end - start == sum(e[4] for e in ents)


def _staging_config(monkeypatch, **extra_config) -> DefaultModelLoader:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return DefaultModelLoader(
        LoadConfig(model_loader_extra_config={"gpu_staging": True, **extra_config})
    )


def test_loader_config(monkeypatch):
    with pytest.raises(ValueError, match="must be a bool"):
        DefaultModelLoader(LoadConfig(model_loader_extra_config={"gpu_staging": "all"}))
    with pytest.raises(ValueError, match="gpu_staging needs"):
        _staging_config(monkeypatch, enable_multithread_load=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="gpu_staging needs"):
        DefaultModelLoader(
            LoadConfig(
                safetensors_load_strategy="eager",
                model_loader_extra_config={"gpu_staging": True},
            )
        )
    assert not DefaultModelLoader(LoadConfig())._use_gpu_staging()


@pytest.mark.parametrize(
    "is_cuda,extra_config,load_strategy,expected",
    [
        (True, {}, None, True),
        (True, {}, "prefetch", True),
        (True, {"gpu_staging": False}, None, False),
        (True, {"enable_multithread_load": True}, None, False),
        (True, {}, "eager", False),
        (False, {}, None, False),
    ],
)
def test_loader_default(monkeypatch, is_cuda, extra_config, load_strategy, expected):
    # On by default on CUDA; off, without an error, where it can't apply.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(default_loader.current_platform, "is_cuda", lambda: is_cuda)
    loader = DefaultModelLoader(
        LoadConfig(
            safetensors_load_strategy=load_strategy,
            model_loader_extra_config=extra_config,
        )
    )
    assert loader.gpu_staging is expected


@pytest.mark.parametrize(
    "tp,ep,expected", [(1, False, False), (2, True, False), (2, False, True)]
)
def test_loader_guard(monkeypatch, tp, ep, expected):
    loader = _staging_config(monkeypatch)
    parallel_config = SimpleNamespace(
        tensor_parallel_size=tp, enable_expert_parallel=ep
    )
    monkeypatch.setattr(
        default_loader,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(parallel_config=parallel_config),
    )
    assert loader._use_gpu_staging() is expected


@pytest.fixture
def released():
    yield
    if torch.cuda.is_available():
        gpu_staging.release_gpu_staging()


@requires_cuda
@pytest.mark.parametrize("chunk_bytes", [64 << 20, 4096])
def test_matches_safe_open(checkpoint, monkeypatch, released, chunk_bytes):
    # A small chunk size gives many ranges, so the staging buffers are reused
    # many times while the earlier tensors are still referenced.
    monkeypatch.setattr(gpu_staging, "_CHUNK_BYTES", chunk_bytes)
    # Split larger ranges across the read threads.
    monkeypatch.setattr(gpu_staging, "_READ_BYTES", 1000)
    yielded = list(gpu_staging.gpu_staged_safetensors_file(checkpoint, _skip))
    for name, tensor in yielded:
        assert tensor.is_cuda == (tensor.numel() > 0), name
    _check_against_safe_open(checkpoint, yielded)
    # Each staged tensor owns its memory, not a view into a staging buffer.
    for name, tensor in yielded:
        nbytes = tensor.numel() * tensor.element_size()
        assert tensor.untyped_storage().nbytes() == nbytes, name


@requires_cuda
def test_oversize_tensors_are_mapped(checkpoint, monkeypatch, released):
    max_bytes = 4096
    monkeypatch.setattr(gpu_staging, "_MAX_TENSOR_BYTES", max_bytes)
    yielded = list(gpu_staging.gpu_staged_safetensors_file(checkpoint, _skip))
    for name, tensor in yielded:
        nbytes = tensor.numel() * tensor.element_size()
        assert tensor.is_cuda == (0 < nbytes <= max_bytes), name
    assert any(not t.is_cuda and t.numel() > 0 for _, t in yielded)
    _check_against_safe_open(checkpoint, yielded)


@requires_cuda
def test_skipped_tensors_are_not_read(checkpoint, monkeypatch, released):
    read = []
    pread_range = gpu_staging._pread_range

    def recording_pread_range(fd, out, pos):
        read.append((pos, pos + out.numel()))
        pread_range(fd, out, pos)

    monkeypatch.setattr(gpu_staging, "_pread_range", recording_pread_range)
    monkeypatch.setattr(gpu_staging, "_CHUNK_BYTES", 4096)
    yielded = list(gpu_staging.gpu_staged_safetensors_file(checkpoint, _skip))
    assert "layers.1.skipme" not in dict(yielded)

    with open(checkpoint, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        info = json.loads(f.read(n))["layers.1.skipme"]
    start, end = (8 + n + o for o in info["data_offsets"])
    assert all(e <= start or s >= end for s, e in read)


@requires_cuda
def test_weights_iterator_skips_unused(checkpoint, released):
    def unused(name: str) -> bool:
        return name.startswith("layers.2.")

    yielded = list(
        safetensors_weights_iterator(
            [checkpoint], False, gpu_staging=True, is_unused_weight=unused
        )
    )
    _check_against_safe_open(checkpoint, yielded, skip=unused)
    assert any(t.is_cuda for _, t in yielded)
    # The staging buffers are released at the end of the load.
    assert not gpu_staging._rings


@requires_cuda
def test_read_error_is_raised(checkpoint, monkeypatch, released):
    def failing_pread_range(fd, out, pos):
        raise OSError("injected read error")

    monkeypatch.setattr(gpu_staging, "_pread_range", failing_pread_range)
    with pytest.raises(OSError, match="injected read error"):
        list(gpu_staging.gpu_staged_safetensors_file(checkpoint, _skip))


@requires_cuda
def test_early_close(checkpoint, monkeypatch, released):
    monkeypatch.setattr(gpu_staging, "_CHUNK_BYTES", 4096)
    files = gpu_staging.gpu_staged_safetensors_file(checkpoint, _skip)
    next(files)
    files.close()  # must not hang
    # The next file reuses the buffers.
    yielded = list(gpu_staging.gpu_staged_safetensors_file(checkpoint, _skip))
    _check_against_safe_open(checkpoint, yielded)
