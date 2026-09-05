# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from dataclasses import dataclass, field
from typing import Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.kv_cache_utils import resolve_block_hashes
from vllm.v1.request import Request


class AtomicCounter:
    """Thread-safe atomic counter for round-robin operations."""

    def __init__(self, n: int):
        assert n > 0, "Counter size must be positive"
        self._n = n
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        """Get next value in round-robin fashion."""
        with self._lock:
            current = self._value
            self._value = (current + 1) % self._n
            return current


def external_block_keys(
    request: Request,
    hash_block_size: int,
    block_size: int,
    num_blocks: int | None = None,
) -> list[str]:
    """External cache keys for a request's leading blocks, at ``block_size``.

    The engine already computes one hash per full block that folds in every
    dimension required to partition the KV cache keyspace -- multimodal identity,
    LoRA identity, ``cache_salt`` and prompt-embeds content -- see
    ``generate_block_hash_extra_keys``. It is built whenever a KV connector is
    configured, with prefix caching on or off, so it is always available here.

    Deriving the external key from the token ids alone drops those dimensions, and
    two requests that must not share blocks then collide on the same key.

    Those hashes are computed at ``hash_block_size``, which is not the scheduler
    block size the connector's block ids are indexed by whenever the model is
    hybrid or ``decode_context_parallel_size > 1``. ``resolve_block_hashes`` takes
    the view at ``block_size``; each hash is already chained over its whole prefix,
    so the last sub-hash of a block fingerprints that block's prefix exactly.
    """
    keys = [
        bytes(block_hash).hex()
        for block_hash in resolve_block_hashes(
            request.block_hashes, hash_block_size, block_size
        )
    ]
    return keys if num_blocks is None else keys[:num_blocks]


@dataclass
class LoadBlockInfo:
    """Operation for loading blocks from external storage."""

    num_computed_blocks: int
    num_blocks_to_load: int
    need_fetch_block_ids: list[int]


@dataclass
class SaveBlockInfo:
    """Operation for saving blocks to external storage."""

    skip_leading_blocks: int


@dataclass
class RequestSchedulingState:
    """Unified request scheduling state management."""

    request_id: str
    request: Request | None = None

    # Token and block tracking
    token_ids: list[int] = field(default_factory=list)
    allocated_block_ids: list[int] = field(default_factory=list)
    num_saved_blocks: int = 0

    # Load operation info
    load_op: LoadBlockInfo | None = None

    # Scheduling phase
    phase: str = "NEW"  # NEW -> WAITING_TO_LOAD -> ACTIVE -> FINISHED

    def needs_loading(self) -> bool:
        """Check if request needs loading."""
        return self.load_op is not None and self.load_op.num_blocks_to_load > 0

    def is_ready_to_load(self) -> bool:
        """Check if request is ready for loading."""
        return self.phase == "WAITING_TO_LOAD" and self.needs_loading()

    def update_tokens_and_blocks(self, new_token_ids: list[int], new_block_ids) -> None:
        """Update with new tokens and blocks."""
        if new_token_ids:
            self.token_ids.extend(new_token_ids)

        if new_block_ids is not None:
            normalized_block_ids = self._normalize_block_ids(new_block_ids)
            self.allocated_block_ids.extend(normalized_block_ids)

    def _normalize_block_ids(self, block_ids) -> list[int]:
        """Normalize block_ids to list format."""
        if not block_ids:
            return []
        if isinstance(block_ids, tuple):
            return block_ids[0] if block_ids else []
        if isinstance(block_ids, list):
            return block_ids
        return []


@dataclass
class HF3FSRequestMetadata:
    """Metadata for a single request in HF3FS connector."""

    request_id: str
    token_ids: list[int]
    block_ids: list[int]
    # External cache key per full block, indexed from block 0. Taken from the
    # engine's own block hashes so that every dimension required to partition the
    # KV keyspace reaches the external store.
    block_keys: list[str] = field(default_factory=list)
    load_block_op: LoadBlockInfo | None = None
    save_block_op: SaveBlockInfo | None = None

    @staticmethod
    def from_scheduling_state(
        state: "RequestSchedulingState",
        block_size: int,
        hash_block_size: int,
        load_op: LoadBlockInfo | None = None,
        skip_leading_blocks: int | None = None,
    ) -> Optional["HF3FSRequestMetadata"]:
        """Create request metadata from scheduling state."""
        assert state.request is not None
        token_count = len(state.token_ids)
        total_blocks = token_count // block_size

        skip_blocks = (
            state.num_saved_blocks
            if skip_leading_blocks is None
            else skip_leading_blocks
        )

        new_blocks_to_save = total_blocks - state.num_saved_blocks
        if new_blocks_to_save <= 0 and load_op is None:
            return None

        state.num_saved_blocks = total_blocks
        return HF3FSRequestMetadata(
            request_id=state.request_id,
            token_ids=state.token_ids,
            block_ids=state.allocated_block_ids,
            block_keys=external_block_keys(
                state.request, hash_block_size, block_size, total_blocks
            ),
            load_block_op=load_op,
            save_block_op=SaveBlockInfo(skip_leading_blocks=skip_blocks),
        )


class HF3FSConnectorMetadata(KVConnectorMetadata):
    """Container for HF3FS connector metadata."""

    def __init__(self):
        self.requests: list[HF3FSRequestMetadata] = []

    def add_request(self, request_metadata: HF3FSRequestMetadata) -> None:
        """Add request to metadata."""
        self.requests.append(request_metadata)
