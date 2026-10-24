# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path

import pytest

from vllm.distributed.ec_transfer.ec_connector.tiering import io


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    entry_path = tmp_path / "nested" / "entry.ec"
    temporary_path = entry_path.with_name(f".{entry_path.name}.tmp")
    return entry_path, temporary_path


def test_store_entry_publishes_all_allocated_block_bytes(tmp_path: Path) -> None:
    entry_path, temporary_path = _paths(tmp_path)
    sources = [memoryview(b"abcd"), memoryview(b"efXY")]

    io.store_entry(entry_path, temporary_path, sources, file_size_bytes=8)

    assert entry_path.read_bytes() == b"abcdefXY"
    assert not temporary_path.exists()


@pytest.mark.parametrize(
    ("sources", "file_size_bytes", "actual_size"),
    [
        ([memoryview(b"abcd"), memoryview(b"ef")], 8, 6),
        ([memoryview(b"abcd"), memoryview(b"efgh")], 6, 8),
    ],
)
def test_store_entry_rejects_any_buffer_size_mismatch(
    tmp_path: Path,
    sources: list[memoryview],
    file_size_bytes: int,
    actual_size: int,
) -> None:
    entry_path, temporary_path = _paths(tmp_path)

    with pytest.raises(
        ValueError,
        match=(
            f"buffer size is {actual_size} bytes, "
            f"expected exactly {file_size_bytes}"
        ),
    ):
        io.store_entry(
            entry_path,
            temporary_path,
            sources,
            file_size_bytes=file_size_bytes,
        )

    assert not entry_path.exists()
    assert not temporary_path.exists()


def test_store_entry_retries_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry_path, temporary_path = _paths(tmp_path)

    def partial_writev(fd: int, buffers: list[memoryview]) -> int:
        return os.write(fd, buffers[0][:2])

    monkeypatch.setattr(io.os, "writev", partial_writev)

    io.store_entry(
        entry_path,
        temporary_path,
        [memoryview(b"abc"), memoryview(b"def")],
        file_size_bytes=6,
    )

    assert entry_path.read_bytes() == b"abcdef"


def test_store_entry_cleans_temporary_file_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry_path, temporary_path = _paths(tmp_path)

    def failed_writev(fd: int, buffers: list[memoryview]) -> int:
        raise OSError("injected write failure")

    monkeypatch.setattr(io.os, "writev", failed_writev)

    with pytest.raises(OSError, match="injected write failure"):
        io.store_entry(
            entry_path,
            temporary_path,
            [memoryview(b"data")],
            file_size_bytes=4,
        )

    assert not entry_path.exists()
    assert not temporary_path.exists()


def test_load_entry_fills_all_destinations(
    tmp_path: Path,
) -> None:
    entry_path, _ = _paths(tmp_path)
    entry_path.parent.mkdir(parents=True)
    entry_path.write_bytes(b"abcdefgh")
    first = bytearray(b"xxxx")
    second = bytearray(b"yyyy")

    io.load_entry(
        entry_path,
        [memoryview(first), memoryview(second)],
        file_size_bytes=8,
    )

    assert first == b"abcd"
    assert second == b"efgh"


def test_load_entry_retries_partial_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry_path, _ = _paths(tmp_path)
    entry_path.parent.mkdir(parents=True)
    entry_path.write_bytes(b"abcdef")
    destination = bytearray(6)

    def partial_readv(fd: int, buffers: list[memoryview]) -> int:
        data = os.read(fd, 2)
        remaining = memoryview(data)
        for buffer in buffers:
            copied = min(len(buffer), len(remaining))
            buffer[:copied] = remaining[:copied]
            remaining = remaining[copied:]
            if not remaining:
                break
        return len(data)

    monkeypatch.setattr(io.os, "readv", partial_readv)

    io.load_entry(entry_path, [memoryview(destination)], file_size_bytes=6)

    assert destination == b"abcdef"


def test_load_entry_rejects_wrong_file_size_before_modifying_buffers(
    tmp_path: Path,
) -> None:
    entry_path, _ = _paths(tmp_path)
    entry_path.parent.mkdir(parents=True)
    entry_path.write_bytes(b"abc")
    destination = bytearray(b"xxxx")

    with pytest.raises(OSError, match="file size is 3 bytes, expected 4"):
        io.load_entry(
            entry_path, [memoryview(destination)], file_size_bytes=4
        )

    assert destination == b"xxxx"


def test_store_entry_requires_temporary_file_beside_final_file(
    tmp_path: Path,
) -> None:
    entry_path, _ = _paths(tmp_path)

    with pytest.raises(ValueError, match="temporary file must be beside"):
        io.store_entry(
            entry_path,
            tmp_path / "other" / "entry.tmp",
            [memoryview(b"data")],
            file_size_bytes=4,
        )


def test_remove_entry_is_idempotent(tmp_path: Path) -> None:
    entry_path, _ = _paths(tmp_path)
    entry_path.parent.mkdir(parents=True)
    entry_path.write_bytes(b"data")

    io.remove_entry(entry_path)
    io.remove_entry(entry_path)

    assert not entry_path.exists()
