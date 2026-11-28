# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDSOffloadingWorker with cuFile replaced by a POSIX-backed fake.

The fake turns each cuFile call into plain file I/O on the registered fd and
records it, so the tests cover the worker's own logic (direct-path probe,
temp-file publishing, cleanup) without GDS hardware.
"""

import os
from collections.abc import Iterator

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.gds import worker as gds_worker
from vllm.v1.kv_offload.fs.gds.cufile_bindings import CUfileBoolConfigParameter
from vllm.v1.kv_offload.fs.gds.worker import GDSOffloadingWorker
from vllm.v1.kv_offload.fs.worker import FileOp

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="GDS offloading requires CUDA"
)

_PAGE_SIZE = 4096
_HINT = "Run `gdscheck -p`"


class FakeCuFile:
    """POSIX stand-in for the cuFile bindings the worker uses.

    A "device pointer" is written to the file as `_PAGE_SIZE`-byte chunks
    of its low byte, so tests can tell which op landed where.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.fail: str | None = None
        self.short: str | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "cuFileSetParameterBool",
            "cuFileDriverOpen",
            "cuFileDriverClose",
            "cuFileHandleRegister",
            "cuFileHandleDeregister",
            "cuFileWrite",
            "cuFileRead",
        ):
            monkeypatch.setattr(gds_worker, name, getattr(self, name))
        # The real open_for_gds adds O_DIRECT, which tmpfs rejects.
        monkeypatch.setattr(
            gds_worker, "open_for_gds", lambda path, flags: os.open(path, flags, 0o644)
        )

    def _maybe_fail(self, name: str) -> None:
        if self.fail == name:
            raise RuntimeError(f"{name} failed: err=5008 (CU_FILE_IO_NOT_SUPPORTED)")

    def op_names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def cuFileSetParameterBool(self, param, value) -> None:
        self._maybe_fail("cuFileSetParameterBool")
        self.calls.append(("cuFileSetParameterBool", param, value))

    def cuFileDriverOpen(self) -> None:
        self._maybe_fail("cuFileDriverOpen")
        self.calls.append(("cuFileDriverOpen",))

    def cuFileDriverClose(self) -> None:
        self.calls.append(("cuFileDriverClose",))

    def cuFileHandleRegister(self, fd: int) -> int:
        return fd

    def cuFileHandleDeregister(self, handle: int) -> None:
        pass

    def cuFileWrite(self, fd: int, dev_ptr: int, size: int, file_offset: int) -> int:
        self._maybe_fail("cuFileWrite")
        self.calls.append(("cuFileWrite", dev_ptr, size, file_offset))
        written = os.pwrite(fd, bytes([dev_ptr & 0xFF]) * size, file_offset)
        return written - 1 if self.short == "cuFileWrite" else written

    def cuFileRead(self, fd: int, dev_ptr: int, size: int, file_offset: int) -> int:
        self._maybe_fail("cuFileRead")
        self.calls.append(("cuFileRead", dev_ptr, size, file_offset))
        read = len(os.pread(fd, size, file_offset))
        return read - 1 if self.short == "cuFileRead" else read


@pytest.fixture
def cufile(monkeypatch) -> FakeCuFile:
    fake = FakeCuFile()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def kv_caches() -> CanonicalKVCaches:
    tensor = torch.zeros(4, _PAGE_SIZE, dtype=torch.int8, device="cuda")
    return CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=_PAGE_SIZE)],
        group_data_refs=[],
    )


@pytest.fixture
def mapper(tmp_path) -> FileMapper:
    return FileMapper(
        root_dir=str(tmp_path / "kv"),
        model_name="test",
        tokens_per_hash=16,
        blocks_per_file=2,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        rank=0,
        dtype="float16",
    )


def _make_worker(mapper, kv_caches, root_dir) -> GDSOffloadingWorker:
    return GDSOffloadingWorker(
        file_mapper=mapper,
        blocks_per_file=2,
        device=kv_caches.tensors[0].tensor.device,
        kv_caches=kv_caches,
        root_dir=str(root_dir),
        extra_config={"gds_max_io_threads": 2},
    )


@pytest.fixture
def worker(cufile, mapper, kv_caches, tmp_path) -> Iterator[GDSOffloadingWorker]:
    w = _make_worker(mapper, kv_caches, tmp_path)
    cufile.calls.clear()
    yield w
    w.shutdown()


def _files_under(root) -> list[str]:
    return sorted(
        os.path.relpath(os.path.join(d, f), root)
        for d, _, files in os.walk(root)
        for f in files
    )


def test_init_disables_compat_mode_then_probes_one_kv_block(
    cufile, mapper, kv_caches, tmp_path
):
    """Compat mode is turned off before the driver opens, so the probe cannot
    pass via POSIX fallback. The probe writes and reads back KV block 0 under
    root_dir and leaves no file behind."""
    block = kv_caches.tensors[0].tensor
    w = _make_worker(mapper, kv_caches, tmp_path)

    probe = (block.data_ptr(), _PAGE_SIZE, 0)
    assert cufile.calls == [
        (
            "cuFileSetParameterBool",
            CUfileBoolConfigParameter.CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE,
            False,
        ),
        ("cuFileDriverOpen",),
        ("cuFileWrite", *probe),
        ("cuFileRead", *probe),
    ]
    assert _files_under(tmp_path) == []

    w.shutdown()
    assert cufile.op_names()[-1] == "cuFileDriverClose"


@pytest.mark.parametrize(
    "fail, short, error",
    [
        ("cuFileWrite", None, "CU_FILE_IO_NOT_SUPPORTED"),
        ("cuFileRead", None, "CU_FILE_IO_NOT_SUPPORTED"),
        (None, "cuFileWrite", "cuFileWrite short write"),
        (None, "cuFileRead", "cuFileRead short read"),
    ],
)
def test_probe_failure_raises_with_hint_and_cleans_up(
    cufile, mapper, kv_caches, tmp_path, fail, short, error
):
    """A failed or short probe transfer fails init with the cuFile error and
    the gdscheck hint, closes the driver, and removes the probe file."""
    cufile.fail, cufile.short = fail, short

    with pytest.raises(RuntimeError, match="direct-path probe failed") as exc:
        _make_worker(mapper, kv_caches, tmp_path)

    assert error in str(exc.value) and _HINT in str(exc.value)
    assert cufile.op_names()[-1] == "cuFileDriverClose"
    assert _files_under(tmp_path) == []


@pytest.mark.parametrize("fail", ["cuFileSetParameterBool", "cuFileDriverOpen"])
def test_driver_setup_failure_does_not_close_unopened_driver(
    cufile, mapper, kv_caches, tmp_path, fail
):
    """If compat mode cannot be disabled (e.g. libcufile too old) or the driver
    fails to open, init raises without probing or closing a driver that was
    never opened."""
    cufile.fail = fail

    with pytest.raises(RuntimeError, match=fail):
        _make_worker(mapper, kv_caches, tmp_path)

    assert not {"cuFileWrite", "cuFileDriverClose"} & set(cufile.op_names())


def test_write_block_publishes_ops_at_their_offsets(worker, cufile, tmp_path):
    """Each op is written at its file offset into a temp file that is renamed
    into place, so only the final file remains."""
    path = str(tmp_path / "kv" / "block.bin")
    ops = [
        FileOp(device_ptr=0x11, size_bytes=_PAGE_SIZE, file_offset=_PAGE_SIZE),
        FileOp(device_ptr=0x22, size_bytes=_PAGE_SIZE, file_offset=0),
    ]

    worker.write_block(path, ops)

    with open(path, "rb") as f:
        assert f.read() == b"\x22" * _PAGE_SIZE + b"\x11" * _PAGE_SIZE
    assert _files_under(tmp_path) == [os.path.join("kv", "block.bin")]


def test_write_block_skips_existing_file(worker, cufile, tmp_path):
    """File names are content hashes, so an existing file is not rewritten."""
    path = tmp_path / "block.bin"
    path.write_bytes(b"existing")

    worker.write_block(str(path), [FileOp(0x11, _PAGE_SIZE, 0)])

    assert path.read_bytes() == b"existing"
    assert "cuFileWrite" not in cufile.op_names()


@pytest.mark.parametrize("fail, short", [("cuFileWrite", None), (None, "cuFileWrite")])
def test_write_block_failure_leaves_no_file(worker, cufile, tmp_path, fail, short):
    """A failed or short write raises and removes the temp file, so no
    partial block is ever published."""
    cufile.fail, cufile.short = fail, short

    with pytest.raises(RuntimeError, match="cuFileWrite"):
        worker.write_block(str(tmp_path / "block.bin"), [FileOp(0x11, _PAGE_SIZE, 0)])

    assert _files_under(tmp_path) == []


def test_read_block_reads_ops_at_their_offsets(worker, cufile, tmp_path):
    path = tmp_path / "block.bin"
    path.write_bytes(bytes(2 * _PAGE_SIZE))
    ops = [FileOp(0x11, _PAGE_SIZE, _PAGE_SIZE), FileOp(0x22, _PAGE_SIZE, 0)]

    worker.read_block(str(path), ops)

    assert cufile.calls == [
        ("cuFileRead", 0x11, _PAGE_SIZE, _PAGE_SIZE),
        ("cuFileRead", 0x22, _PAGE_SIZE, 0),
    ]


def test_read_block_short_read_raises(worker, cufile, tmp_path):
    """A file shorter than the requested ops (e.g. truncated) fails the load."""
    path = tmp_path / "block.bin"
    path.write_bytes(bytes(_PAGE_SIZE))

    with pytest.raises(RuntimeError, match="cuFileRead short read"):
        worker.read_block(str(path), [FileOp(0x11, _PAGE_SIZE, _PAGE_SIZE)])
