# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-range reads of checkpoint tensors from object storage.

Some checkpoints hold row-sharded tables (for example hash embeddings) whose
full tensor does not fit in host memory, while each rank only keeps a range of
rows. Streaming the whole tensor to every rank is wasteful or impossible;
`ObjectStorageRowReader` fetches just the rows a parameter asks for, directly
from the safetensors shard with parallel ranged GETs.

The reader is model-independent. A model opts in by declaring which checkpoint
tensors are row-sharded and by filling its parameters from the reader; see
`VllmModel.is_ranged_checkpoint_weight` and
`VllmModel.load_ranged_checkpoint_weights`.
"""

import json
import logging
import os
import random
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Tunables. Like other loader options, they are read when a read starts so
# they can be set by the deployment environment.
THREADS_ENV = "VLLM_RANGED_READ_THREADS"
PART_MB_ENV = "VLLM_RANGED_READ_PART_MB"
VERIFY_ROWS_ENV = "VLLM_RANGED_READ_VERIFY_ROWS"
DEFAULT_THREADS = 32
DEFAULT_PART_MB = 64


def threads_from_env() -> int:
    return int(os.environ.get(THREADS_ENV, DEFAULT_THREADS))


def part_bytes_from_env() -> int:
    return int(os.environ.get(PART_MB_ENV, DEFAULT_PART_MB)) << 20


def verify_rows_from_env() -> int:
    return int(os.environ.get(VERIFY_ROWS_ENV, "0"))


def split_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 URI: {uri}")
    bucket, _, key = uri[len("s3://") :].partition("/")
    if not bucket or not key:
        raise ValueError(f"incomplete s3 URI: {uri}")
    return bucket, key


def parse_header(len_bytes: bytes, header_bytes: bytes) -> tuple[int, dict]:
    """Parse a safetensors header; return (data start offset, header)."""
    size = int.from_bytes(len_bytes, "little")
    if len(header_bytes) != size:
        raise ValueError(f"header is {len(header_bytes)} bytes, want {size}")
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    header.pop("__metadata__", None)
    return 8 + size, header


def read_header(client, bucket: str, key: str) -> tuple[int, dict]:
    len_bytes = get_range(client, bucket, key, 0, 8)
    size = int.from_bytes(len_bytes, "little")
    return parse_header(len_bytes, get_range(client, bucket, key, 8, size))


def param_byte_range(
    data_start: int,
    entry: Mapping,
    row_start: int,
    rows: int,
    row_bytes: int,
) -> tuple[int, int]:
    """Absolute (offset, length) of ``rows`` rows starting at ``row_start``."""
    shape = entry["shape"]
    if len(shape) != 2:
        raise ValueError(f"row reads need a 2-D tensor, got shape {shape}")
    file_rows, file_cols = shape
    file_row_bytes = file_cols * _item_size(entry)
    if file_row_bytes != row_bytes:
        raise ValueError(
            f"row size mismatch: file {file_row_bytes} B, parameter {row_bytes} B"
        )
    if row_start < 0 or rows < 0 or row_start + rows > file_rows:
        raise ValueError(
            f"rows [{row_start}, {row_start + rows}) exceed file rows {file_rows}"
        )
    return (
        data_start + entry["data_offsets"][0] + row_start * row_bytes,
        rows * row_bytes,
    )


_ITEM_SIZE = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _item_size(entry: Mapping) -> int:
    try:
        return _ITEM_SIZE[entry["dtype"]]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype: {entry['dtype']}") from exc


def split_parts(length: int, part_bytes: int) -> list[tuple[int, int]]:
    """(relative start, size) parts of at most ``part_bytes`` covering length."""
    if part_bytes <= 0:
        raise ValueError(f"part size must be positive: {part_bytes}")
    return [
        (start, min(part_bytes, length - start))
        for start in range(0, length, part_bytes)
    ]


def get_range(
    client, bucket: str, key: str, start: int, length: int, attempts: int = 3
) -> bytes:
    """Ranged GET of exactly ``length`` bytes, retried on error or short read."""
    if length <= 0:
        return b""
    for attempt in range(attempts):
        try:
            response = client.get_object(
                Bucket=bucket,
                Key=key,
                Range=f"bytes={start}-{start + length - 1}",
            )
            data = response["Body"].read()
            if len(data) != length:
                raise OSError(f"short read: {len(data)} of {length} bytes")
            return data
        except Exception:
            if attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")


def fill_range(
    client,
    bucket: str,
    key: str,
    offset: int,
    dest,
    part_bytes: int,
    pool: ThreadPoolExecutor,
) -> None:
    """Fill the flat byte buffer ``dest`` from ``offset`` with parallel GETs."""
    view = memoryview(dest).cast("B")

    def fetch(part: tuple[int, int]) -> None:
        start, size = part
        view[start : start + size] = get_range(
            client, bucket, key, offset + start, size
        )

    futures = [pool.submit(fetch, part) for part in split_parts(len(view), part_bytes)]
    try:
        for future in futures:
            future.result()
    except BaseException:
        # Otherwise the error only surfaces after every queued GET has run its
        # retries, while the other ranks wait in collectives.
        for future in futures:
            future.cancel()
        raise


def verify_rows(
    client,
    bucket: str,
    key: str,
    offset: int,
    dest,
    row_bytes: int,
    num_rows: int,
    rng=None,
) -> None:
    """Re-read ``num_rows`` random rows independently and compare them."""
    view = memoryview(dest).cast("B")
    rows = len(view) // row_bytes
    for row in (rng or random).sample(range(rows), min(num_rows, rows)):
        fresh = get_range(
            client,
            bucket,
            key,
            offset + row * row_bytes,
            row_bytes,
        )
        without = view[row * row_bytes : (row + 1) * row_bytes].tobytes()
        if fresh != without:
            raise RuntimeError(f"ranged read mismatch at local row {row} of {key}")


def read_tensor(uri: str, name: str, client=None) -> tuple[bytes, str, list[int]]:
    """Read one whole tensor from an ``s3://`` safetensors shard.

    Returns ``(raw bytes, safetensors dtype, shape)``.
    """
    bucket, key = split_s3_uri(uri)
    client = client or _s3_client(threads_from_env())
    data_start, header = read_header(client, bucket, key)
    entry = header[name]
    begin, end = entry["data_offsets"]
    data = get_range(client, bucket, key, data_start + begin, end - begin)
    return data, entry["dtype"], list(entry["shape"])


def select_ranged_files(
    weight_map: Mapping[str, str],
    is_ranged: Callable[[str], bool],
) -> tuple[dict[str, str], set[str]]:
    """Split a checkpoint weight map into ranged tensors and their shards.

    Returns ``(name -> shard, shard basenames to exclude from streaming)``.
    """
    names = {name: shard for name, shard in weight_map.items() if is_ranged(name)}
    files = {os.path.basename(shard) for shard in names.values()}
    return names, files


def _s3_client(threads: int):
    import boto3
    import botocore.config

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("RUNAI_STREAMER_S3_ENDPOINT"),
        config=botocore.config.Config(
            s3={"addressing_style": "path"},
            max_pool_connections=threads,
            retries={"max_attempts": 10, "mode": "adaptive"},
        ),
    )


class ObjectStorageRowReader:
    """Reads one tensor's rows from an ``s3://`` safetensors shard on demand.

    A reader is bound to a tensor; `copy_rows_into` fills the local rows a
    parameter keeps, without materializing the full tensor.
    """

    def __init__(
        self,
        client,
        bucket: str,
        key: str,
        data_start: int,
        entry: Mapping,
        *,
        threads: int | None = None,
        part_bytes: int | None = None,
        verify_rows: int | None = None,
    ) -> None:
        self.client = client
        self.bucket = bucket
        self.key = key
        self.data_start = data_start
        self.entry = entry
        self.threads = threads if threads is not None else threads_from_env()
        self.part_bytes = (
            part_bytes if part_bytes is not None else part_bytes_from_env()
        )
        self.verify_rows = (
            verify_rows if verify_rows is not None else verify_rows_from_env()
        )

    @classmethod
    def from_uri(cls, uri: str, name: str, **kwargs) -> "ObjectStorageRowReader":
        bucket, key = split_s3_uri(uri)
        threads = kwargs.get("threads") or threads_from_env()
        client = _s3_client(threads)
        data_start, header = read_header(client, bucket, key)
        if name not in header:
            raise KeyError(f"{name} is not in {uri}")
        kwargs.setdefault("threads", threads)
        return cls(client, bucket, key, data_start, header[name], **kwargs)

    def copy_rows_into(self, dest, row_start: int) -> int:
        """Fill ``dest`` with rows ``[row_start, row_start + len(dest) rows)``.

        Returns the number of bytes read.
        """
        view = memoryview(dest).cast("B")
        _, file_cols = self.entry["shape"]
        row_bytes = file_cols * _item_size(self.entry)
        if row_bytes == 0 or len(view) % row_bytes:
            raise ValueError(
                f"destination of {len(view)} B is not a whole number of "
                f"{row_bytes} B rows"
            )
        rows = len(view) // row_bytes
        offset, length = param_byte_range(
            self.data_start, self.entry, row_start, rows, row_bytes
        )
        began = time.perf_counter()
        with ThreadPoolExecutor(self.threads) as pool:
            fill_range(
                self.client,
                self.bucket,
                self.key,
                offset,
                view,
                self.part_bytes,
                pool,
            )
        if self.verify_rows > 0:
            verify_rows(
                self.client,
                self.bucket,
                self.key,
                offset,
                view,
                row_bytes,
                self.verify_rows,
            )
        logger.debug(
            "Read %.2f GiB of %s in %.2fs",
            length / 2**30,
            os.path.basename(self.key),
            time.perf_counter() - began,
        )
        return length
