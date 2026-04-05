import threading
from dataclasses import dataclass, field
from typing import List, Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
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


@dataclass
class LoadBlockInfo:
    """Operation for loading blocks from external storage."""

    num_computed_blocks: int
    num_blocks_to_load: int
    need_fetch_block_ids: List[int]


@dataclass
class SaveBlockInfo:
    """Operation for saving blocks to external storage."""

    skip_leading_blocks: int


@dataclass
class RequestSchedulingState:
    """Unified request scheduling state management."""

    request_id: str
    request: Optional["Request"] = None

    # Token and block tracking
    token_ids: List[int] = field(default_factory=list)
    allocated_block_ids: List[int] = field(default_factory=list)
    num_saved_blocks: int = 0

    # Load operation info
    load_op: Optional[LoadBlockInfo] = None

    # Scheduling phase
    phase: str = "NEW"  # NEW -> WAITING_TO_LOAD -> ACTIVE -> FINISHED

    def needs_loading(self) -> bool:
        """Check if request needs loading."""
        return self.load_op is not None and self.load_op.num_blocks_to_load > 0

    def is_ready_to_load(self) -> bool:
        """Check if request is ready for loading."""
        return self.phase == "WAITING_TO_LOAD" and self.needs_loading()

    def update_tokens_and_blocks(self, new_token_ids: List[int], new_block_ids) -> None:
        """Update with new tokens and blocks."""
        if new_token_ids:
            self.token_ids.extend(new_token_ids)

        if new_block_ids is not None:
            normalized_block_ids = self._normalize_block_ids(new_block_ids)
            self.allocated_block_ids.extend(normalized_block_ids)

    def _normalize_block_ids(self, block_ids) -> List[int]:
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
    token_ids: List[int]
    block_ids: List[int]
    load_block_op: Optional[LoadBlockInfo] = None
    save_block_op: Optional[SaveBlockInfo] = None

    @staticmethod
    def from_scheduling_state(
        state: "RequestSchedulingState",
        block_size: int,
        load_op: Optional[LoadBlockInfo] = None,
        skip_leading_blocks: Optional[int] = None,
    ) -> Optional["HF3FSRequestMetadata"]:
        """Create request metadata from scheduling state."""
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
            load_block_op=load_op,
            save_block_op=SaveBlockInfo(skip_leading_blocks=skip_blocks),
        )


class HF3FSConnectorMetadata(KVConnectorMetadata):
    """Container for HF3FS connector metadata."""

    def __init__(self):
        self.requests: List[HF3FSRequestMetadata] = []

    def add_request(self, request_metadata: HF3FSRequestMetadata) -> None:
        """Add request to metadata."""
        self.requests.append(request_metadata)