# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.abstract import (  # noqa: E501
    ExOffloadingStorageKVCacheConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.manager import (  # noqa: E501
    ExOffloadingStorageManager,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class ExKVCacheSegment:
    """
    Represents a segment of cached tokens and their corresponding KV cache location.

    Attributes:
        token_start: The starting token ID within the logical token sequence.
        token_length: The number of tokens in this segment.
        kv_uri: A URI string identifying the storage backend and path for the KV cache.
        kv_start: The starting position in the physical KV cache storage.
        kv_length: The length of the KV cache data in the storage (optional).
    """

    token_start: int
    token_length: int
    kv_uri: str
    kv_start: int
    kv_length: int | None = None

    def __post_init__(self):
        if self.token_start < 0:
            raise ValueError(
                f"token_start must be non-negative, got {self.token_start}"
            )
        if self.token_length <= 0:
            raise ValueError(f"token_length must be positive, got {self.token_length}")
        if self.kv_start < 0:
            raise ValueError(f"kv_start must be non-negative, got {self.kv_start}")
        if self.kv_length is not None and self.kv_length <= 0:
            raise ValueError(f"kv_length must be positive if set, got {self.kv_length}")

    @classmethod
    def from_dict(cls, d: dict) -> "ExKVCacheSegment":
        """Creates an instance from a dictionary."""
        required_fields = ["token_start", "token_length", "kv_uri", "kv_start"]
        missing_fields = [f for f in required_fields if f not in d]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in ExKVCacheSegment dictionary: "
                f"{missing_fields}. Required fields: {required_fields}"
            )
        return cls(
            token_start=d["token_start"],
            token_length=d["token_length"],
            kv_uri=d["kv_uri"],
            kv_start=d["kv_start"],
            kv_length=d.get("kv_length"),
        )

    def to_dict(self) -> dict:
        """Converts the instance to a dictionary."""
        return asdict(self)

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_length


class ExKVCacheContext:
    def __init__(
        self,
        segments: list[ExKVCacheSegment] | list[dict[str, Any]] | None = None,
        block_size: int = 0,
        block_ids: list[int] | None = None,
        kv_length_per_token: int | None = None,
    ):
        self._block_size: int = block_size
        self._block_ids: list[int] | None = block_ids
        self._kv_length_per_token: int | None = kv_length_per_token
        self._offset: int = 0

        if not segments:
            self._segments: tuple[ExKVCacheSegment, ...] = ()
            return

        processed: list[ExKVCacheSegment] = []
        for seg in segments:
            if isinstance(seg, dict):
                processed.append(ExKVCacheSegment.from_dict(seg))
            else:
                processed.append(seg)

        if not processed:
            self._segments = ()
            return

        self._segments = tuple(sorted(processed, key=lambda s: s.token_start))

        self._check_segments()

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, index: int) -> ExKVCacheSegment:
        return self._segments[index]

    def __iter__(self) -> Iterator[ExKVCacheSegment]:
        return iter(self._segments)

    def __repr__(self) -> str:
        return (
            f"ExKVCacheContext(segments={list(self._segments)}, _offset={self._offset})"
        )

    def _check_segments(self):
        prev_end = self._segments[0].token_start
        for i, seg in enumerate(self._segments):
            if seg.token_start % self._block_size != 0:
                raise ValueError(
                    f"Segment {i} token_start must be a multiple of block_size, "
                    f"got {seg.token_start}"
                )

            if seg.token_length % self._block_size != 0:
                raise ValueError(
                    f"Segment {i} token_length must be a multiple of block_size, "
                    f"got {seg.token_length}"
                )

            if seg.token_start != prev_end:
                raise ValueError(
                    f"Segments are not contiguous at index {i}: "
                    f"...[{self._segments[i - 1].token_start}, "
                    f"{self._segments[i - 1].token_end}], "
                    f"[{seg.token_start}, {seg.token_end}]..."
                    if i > 0
                    else ""
                )
            prev_end = seg.token_end

    @property
    def token_start(self) -> int:
        """The logical start token ID of the entire cache."""
        return self._segments[0].token_start + self._offset if self._segments else 0

    @property
    def token_end(self) -> int:
        """The logical end token ID (exclusive) of the entire cache."""
        return self._segments[-1].token_end if self._segments else 0

    @property
    def token_length(self) -> int:
        """The total logical token length covered by the cache."""
        return self.token_end - self.token_start

    @property
    def token_range(self) -> tuple[int, int] | None:
        """Returns the (start, end) logical token range, or None if empty."""
        if not self._segments:
            return None
        return (self.token_start, self.token_end)

    @property
    def real_token_start(self) -> int:
        """The real start token ID of the entire cache."""
        return self._segments[0].token_start if self._segments else 0

    @property
    def real_token_end(self) -> int:
        """The real end token ID (exclusive) of the entire cache."""
        return self.token_end

    @property
    def real_token_length(self) -> int:
        """The real token length including the internal offset."""
        return self.real_token_end - self.real_token_start

    @property
    def block_ids(self) -> list[int] | None:
        if not self._block_ids:
            return None

        return self._block_ids[
            self.token_start // self._block_size : self.token_end // self._block_size
        ]

    def result(self, tp_size: int = 1, use_mla: bool = False) -> list[dict]:
        if self._kv_length_per_token is None:
            raise ValueError("Missing kv_length_per_token")

        if use_mla:
            tp_size = 1

        result = []
        for seg in self._segments:
            kv_length = seg.token_length * self._kv_length_per_token * tp_size
            result.append(
                ExKVCacheSegment(
                    token_start=seg.token_start,
                    token_length=seg.token_length,
                    kv_uri=seg.kv_uri,
                    kv_start=seg.kv_start,
                    kv_length=kv_length,
                ).to_dict()
            )

        return result

    def bind_block_ids(self, block_ids: list[int]) -> "ExKVCacheContext":
        if not self._segments:
            return self

        self._block_ids = block_ids

        return self

    def reset(self) -> "ExKVCacheContext":
        self._segments = ()
        self._offset = 0
        self._block_ids = None
        return self

    def update_kv_layout(
        self,
        kv_length_per_token: int | None = None,
        tp_rank: int | None = None,
        replicates_kv_cache: bool = False,
    ) -> "ExKVCacheContext":
        if kv_length_per_token is not None:
            if self._kv_length_per_token is not None:
                raise ValueError("kv_length_per_token is already set")
            self._kv_length_per_token = kv_length_per_token

        if tp_rank is not None:
            if self._kv_length_per_token is None:
                raise ValueError("Setting tp_rank requires kv_length_per_token")

            new_segments = []
            for seg in self._segments:
                kv_length = seg.token_length * self._kv_length_per_token
                kv_start = seg.kv_start
                if not replicates_kv_cache:
                    kv_start = seg.kv_start + tp_rank * kv_length

                new_segments.append(
                    ExKVCacheSegment(
                        token_start=seg.token_start,
                        token_length=seg.token_length,
                        kv_uri=seg.kv_uri,
                        kv_start=kv_start,
                        kv_length=kv_length,
                    )
                )

            self._segments = tuple(new_segments)
            self._check_segments()

        return self

    def truncate_prefix(self, offset: int) -> "ExKVCacheContext":
        if not self._segments:
            return self

        if offset % self._block_size != 0:
            raise ValueError(
                f"offset ({offset}) must be aligned to block size ({self._block_size})"
            )

        if offset < self.token_start:
            raise ValueError(
                f"offset ({offset}) must be >= cache start ({self.token_start})"
            )

        if offset >= self.token_end:
            return self.reset()

        new_segments = [seg for seg in self._segments if seg.token_end > offset]
        if not new_segments:
            return self.reset()

        self._offset = offset - new_segments[0].token_start
        self._segments = tuple(new_segments)
        self._check_segments()

        return self

    def truncate_suffix(self, offset: int) -> "ExKVCacheContext":
        if not self._segments:
            return self

        if offset % self._block_size != 0:
            raise ValueError(
                f"offset ({offset}) must be aligned to block size ({self._block_size})"
            )

        if offset <= self.token_start:
            return self.reset()

        if offset >= self.token_end:
            return self

        new_segments = []
        for seg in self._segments:
            if seg.token_start >= offset:
                break
            elif seg.token_end <= offset:
                new_segments.append(seg)
            else:
                new_token_length = offset - seg.token_start
                new_kv_length = (
                    new_token_length * self._kv_length_per_token
                    if self._kv_length_per_token is not None
                    else None
                )
                new_seg = ExKVCacheSegment(
                    token_start=seg.token_start,
                    token_length=new_token_length,
                    kv_uri=seg.kv_uri,
                    kv_start=seg.kv_start,
                    kv_length=new_kv_length,
                )
                new_segments.append(new_seg)
                break

        if not new_segments:
            return self.reset()

        self._segments = tuple(new_segments)
        self._check_segments()

        return self

    async def _prefetch_seg(
        self, seg: ExKVCacheSegment, kvcache_config: ExOffloadingStorageKVCacheConfig
    ):
        assert self._block_ids is not None

        block_start = seg.token_start // self._block_size
        block_end = seg.token_end // self._block_size
        block_ids = self._block_ids[block_start:block_end]

        storage, path = ExOffloadingStorageManager.get_storage_by_uri(
            seg.kv_uri, kvcache_config
        )

        await storage.load(path, seg.kv_start, block_ids)

    async def prefetch(self, kvcache_config: ExOffloadingStorageKVCacheConfig):
        if self._block_ids is None:
            raise ValueError("block_ids are not bound")

        await asyncio.gather(
            *[self._prefetch_seg(seg, kvcache_config) for seg in self._segments]
        )

    async def backup(self, kvcache_config: ExOffloadingStorageKVCacheConfig):
        if self._block_ids is None:
            raise ValueError("block_ids are not bound")

        for seg in self._segments:
            block_start = seg.token_start // self._block_size
            block_end = seg.token_end // self._block_size
            block_ids = self._block_ids[block_start:block_end]

            storage, path = ExOffloadingStorageManager.get_storage_by_uri(
                seg.kv_uri, kvcache_config
            )

            await storage.save(path, seg.kv_start, block_ids)


@dataclass
class ExOffloadingRequestContext:
    id: str
    request_id: str
    exkvcache: ExKVCacheContext


@dataclass
class ExOffloadingConnectorMetadata(KVConnectorMetadata):
    load_req_ctx: list[ExOffloadingRequestContext]
    save_req_ctx: list[ExOffloadingRequestContext]
