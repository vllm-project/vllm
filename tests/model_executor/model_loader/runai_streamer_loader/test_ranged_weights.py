# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm.model_executor.model_loader.ranged_weights import (
    ObjectStorageRowReader,
    fill_range,
    param_byte_range,
    parse_header,
    read_tensor,
    select_ranged_files,
    split_parts,
    split_s3_uri,
    verify_rows,
)


class FakeS3:
    def __init__(self, blob: bytes):
        self.blob = blob
        self.ranges: list[tuple[int, int]] = []

    def get_object(self, Bucket, Key, Range):  # noqa: N803
        assert Bucket == "bucket" and Key == "model.safetensors"
        start, end = (int(x) for x in Range.removeprefix("bytes=").split("-"))
        self.ranges.append((start, end))
        return {"Body": io.BytesIO(self.blob[start : end + 1])}


def make_blob(rows: int = 6, cols: int = 4) -> tuple[bytes, int, dict]:
    header_bytes = json.dumps(
        {
            "weight": {
                "dtype": "U8",
                "shape": [rows, cols],
                "data_offsets": [0, rows * cols],
            }
        }
    ).encode()
    data = bytes(range(rows * cols))
    blob = len(header_bytes).to_bytes(8, "little") + header_bytes + data
    data_start, header = parse_header(
        len(header_bytes).to_bytes(8, "little"), header_bytes
    )
    return blob, data_start, header["weight"]


def test_split_s3_uri():
    assert split_s3_uri("s3://bucket/a/b.safetensors") == (
        "bucket",
        "a/b.safetensors",
    )
    for bad in ("gs://bucket/key", "s3://bucket", "s3:///key"):
        with pytest.raises(ValueError):
            split_s3_uri(bad)


def test_parse_header_and_offsets():
    _, data_start, entry = make_blob()
    offset, length = param_byte_range(
        data_start, entry, row_start=2, rows=2, row_bytes=4
    )
    assert offset == data_start + 8
    assert length == 8
    with pytest.raises(ValueError):
        param_byte_range(data_start, entry, row_start=5, rows=2, row_bytes=4)
    with pytest.raises(ValueError):
        param_byte_range(data_start, entry, row_start=0, rows=1, row_bytes=3)


def test_read_tensor():
    blob, _, _ = make_blob(rows=6, cols=4)
    data, dtype, shape = read_tensor(
        "s3://bucket/model.safetensors", "weight", client=FakeS3(blob)
    )
    assert data == bytes(range(24))
    assert dtype == "U8"
    assert shape == [6, 4]


def test_split_parts():
    assert split_parts(10, 4) == [(0, 4), (4, 4), (8, 2)]
    assert split_parts(0, 4) == []
    with pytest.raises(ValueError):
        split_parts(10, 0)


def test_select_ranged_files():
    weight_map = {
        "a.weight": "shard-1.safetensors",
        "a.scale": "shard-2.safetensors",
        "b.weight": "shard-3.safetensors",
    }
    names, files = select_ranged_files(weight_map, lambda name: name.startswith("a."))
    assert names == {
        "a.weight": "shard-1.safetensors",
        "a.scale": "shard-2.safetensors",
    }
    assert files == {"shard-1.safetensors", "shard-2.safetensors"}


def test_row_reader_copies_local_rows():
    blob, data_start, entry = make_blob(rows=6, cols=4)
    reader = ObjectStorageRowReader(
        FakeS3(blob),
        "bucket",
        "model.safetensors",
        data_start,
        entry,
        threads=2,
        part_bytes=4,
    )
    dst = bytearray(2 * 4)
    assert reader.copy_rows_into(dst, row_start=3) == 8
    assert bytes(dst) == bytes(range(12, 20))


def test_row_reader_verifies_rows():
    blob, data_start, entry = make_blob(rows=6, cols=4)
    client = FakeS3(blob)
    reader = ObjectStorageRowReader(
        client,
        "bucket",
        "model.safetensors",
        data_start,
        entry,
        threads=1,
        part_bytes=4,
        verify_rows=1,
    )
    reader.copy_rows_into(bytearray(2 * 4), row_start=3)
    # The verification read is a single row range.
    start, end = client.ranges[-1]
    assert end - start + 1 == 4


def test_verify_rows_detects_mismatch():
    blob, data_start, _ = make_blob()
    client = FakeS3(blob)
    dst = bytearray(range(8))
    verify_rows(
        client,
        "bucket",
        "model.safetensors",
        data_start,
        dst,
        row_bytes=4,
        num_rows=2,
        rng=random.Random(0),
    )
    dst[0] ^= 0xFF
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_rows(
            client,
            "bucket",
            "model.safetensors",
            data_start,
            dst,
            row_bytes=4,
            num_rows=2,
            rng=random.Random(0),
        )


def test_fill_range():
    blob, data_start, _ = make_blob()
    client = FakeS3(blob)
    dst = bytearray(24)
    with ThreadPoolExecutor(3) as pool:
        fill_range(
            client,
            "bucket",
            "model.safetensors",
            data_start,
            dst,
            part_bytes=8,
            pool=pool,
        )
    assert bytes(dst) == bytes(range(24))
