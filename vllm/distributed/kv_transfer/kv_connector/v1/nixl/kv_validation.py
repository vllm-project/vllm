# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV block content validation for the NIXL connector.

Every KV block has a metadata entry naming the content it holds: its
prefix-caching block hash. The producer invalidates an entry before the block
is rewritten and publishes the hash after the forward pass that filled it. The
consumer reads the entries after the KV data and compares them with the hashes
it computes from its own prompt. See docs/design/nixl_kv_checksum.md.
"""

import hashlib
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.core.kv_cache_utils import DEFAULT_NONE_HASH_SEED

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

KVBlockValidationMode = Literal["off", "log", "enforce"]
_MODES: tuple[KVBlockValidationMode, ...] = ("off", "log", "enforce")

ENTRY_SIZE = 64
_HEADER = struct.Struct("<II")  # state, hash length
_MAX_HASH_LEN = ENTRY_SIZE - _HEADER.size
_STATE_VALID = 1
_NON_CRYPTO_HASH_ALGOS = ("xxhash", "xxhash_cbor")


def get_kv_block_validation_mode(
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig",
    transfer_mode: str,
) -> KVBlockValidationMode:
    """Read `kv_block_validation` and reject unsupported deployments."""
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    mode = kv_transfer_config.get_from_extra_config("kv_block_validation", "off")
    if mode not in _MODES:
        raise ValueError(f"kv_block_validation must be one of {_MODES}, got {mode!r}.")
    if mode == "off":
        return mode

    parallel_config = vllm_config.parallel_config
    unsupported = {
        "push transfer mode": transfer_mode != "pull",
        "a kv cache with more than one group": (
            len(kv_cache_config.kv_cache_groups) != 1
        ),
        "mamba layers": kv_cache_config.has_mamba_layers,
        "kv_buffer_device='cpu'": kv_transfer_config.kv_buffer_device == "cpu",
        "bidirectional_kv_xfer": bool(
            kv_transfer_config.get_from_extra_config("bidirectional_kv_xfer", False)
        ),
        "decode/prefill context parallelism": (
            parallel_config.decode_context_parallel_size > 1
            or parallel_config.prefill_context_parallel_size > 1
        ),
        "pipeline parallelism": parallel_config.pipeline_parallel_size > 1,
    }
    if reasons := [name for name, hit in unsupported.items() if hit]:
        raise NotImplementedError(
            "kv_block_validation does not support " + ", ".join(reasons) + " yet."
        )
    if not vllm_config.cache_config.enable_prefix_caching:
        raise ValueError(
            "kv_block_validation identifies blocks by their prefix-caching "
            "hash; enable prefix caching."
        )
    if (
        vllm_config.cache_config.prefix_caching_hash_algo in _NON_CRYPTO_HASH_ALGOS
        and os.getenv("PYTHONHASHSEED") is None
    ):
        raise ValueError(
            "kv_block_validation with a non-cryptographic "
            "prefix_caching_hash_algo needs the same PYTHONHASHSEED on the "
            "producer and the consumer."
        )
    return mode


def get_block_hash_identity(vllm_config: "VllmConfig") -> str:
    """Digest of what makes block hashes comparable across engines.

    Both sides must hash with the same algorithm and NONE_HASH seed. The seed
    is digested rather than sent, since non-cryptographic hashes rely on it
    staying private.
    """
    algo = vllm_config.cache_config.prefix_caching_hash_algo
    seed = os.getenv("PYTHONHASHSEED", DEFAULT_NONE_HASH_SEED)
    return hashlib.sha256(f"{algo}:{seed}".encode()).hexdigest()


def make_entries(block_hashes: Sequence[bytes]) -> np.ndarray:
    """Metadata entries, `[len(block_hashes), ENTRY_SIZE]`, for valid blocks."""
    entries = np.zeros((len(block_hashes), ENTRY_SIZE), dtype=np.uint8)
    if not block_hashes:
        return entries
    lengths = {len(h) for h in block_hashes}
    if len(lengths) == 1:
        hash_len = min(lengths.pop(), _MAX_HASH_LEN)
        joined = np.frombuffer(b"".join(block_hashes), dtype=np.uint8)
        hashes = joined.reshape(len(block_hashes), -1)[:, :hash_len]
        entries[:, : _HEADER.size] = np.frombuffer(
            _HEADER.pack(_STATE_VALID, hash_len), dtype=np.uint8
        )
        entries[:, _HEADER.size : _HEADER.size + hash_len] = hashes
        return entries
    for row, block_hash in zip(entries, block_hashes):
        block_hash = block_hash[:_MAX_HASH_LEN]
        row[: _HEADER.size] = np.frombuffer(
            _HEADER.pack(_STATE_VALID, len(block_hash)), dtype=np.uint8
        )
        row[_HEADER.size : _HEADER.size + len(block_hash)] = np.frombuffer(
            block_hash, dtype=np.uint8
        )
    return entries


class KVBlockMetaTracker:
    """Scheduler side: which entries each step invalidates and publishes.

    A block is invalidated in the step that allocates it, before the forward
    pass writes it, and published in the step whose forward pass fills it.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self._requests: dict[str, Request] = {}
        self._block_tables: dict[str, list[int]] = {}

    def add_request(self, request: "Request") -> None:
        self._requests[request.request_id] = request

    def remove_request(self, req_id: str) -> None:
        self._requests.pop(req_id, None)
        self._block_tables.pop(req_id, None)

    def expected_hashes(self, req_id: str, first_block: int) -> list[bytes]:
        request = self._requests.get(req_id)
        if request is None:
            return []
        return list(request.block_hashes[first_block:])

    def collect(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[list[int], list[int], list[bytes]]:
        """Returns blocks to invalidate, and blocks to publish with hashes."""
        invalidate: list[int] = []
        publish_ids: list[int] = []
        publish_hashes: list[bytes] = []
        num_scheduled = scheduler_output.num_scheduled_tokens
        for new_req in scheduler_output.scheduled_new_reqs:
            table = list(new_req.block_ids[0])
            self._block_tables[new_req.req_id] = table
            num_computed = new_req.num_computed_tokens
            # Blocks holding computed tokens are prefix-cache hits or the
            # partial block a resumed request keeps: not new allocations.
            invalidate.extend(table[cdiv(num_computed, self.block_size) :])
            self._collect_filled(
                new_req.req_id,
                table,
                num_computed,
                num_scheduled.get(new_req.req_id, 0),
                publish_ids,
                publish_hashes,
            )
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            known_table = self._block_tables.get(req_id)
            new_block_ids = cached.new_block_ids[i]
            if new_block_ids is not None:
                invalidate.extend(new_block_ids[0])
                if known_table is not None:
                    known_table.extend(new_block_ids[0])
            if known_table is not None:
                self._collect_filled(
                    req_id,
                    known_table,
                    cached.num_computed_tokens[i],
                    num_scheduled.get(req_id, 0),
                    publish_ids,
                    publish_hashes,
                )
        return invalidate, publish_ids, publish_hashes

    def _collect_filled(
        self,
        req_id: str,
        table: list[int],
        num_computed: int,
        num_new: int,
        publish_ids: list[int],
        publish_hashes: list[bytes],
    ) -> None:
        request = self._requests.get(req_id)
        if request is None:
            return
        # Hashes only cover blocks full of real tokens, so blocks completed by
        # speculative tokens are left unpublished.
        end = min(
            (num_computed + num_new) // self.block_size,
            len(request.block_hashes),
            len(table),
        )
        for index in range(num_computed // self.block_size, end):
            publish_ids.append(table[index])
            publish_hashes.append(request.block_hashes[index])


class KVBlockMetaPool:
    """Worker side: one entry per logical KV block, next to the KV cache.

    Updates are issued on the current stream, so they are ordered with the
    forward passes that write the KV blocks.
    """

    def __init__(self, num_blocks: int, device: torch.device):
        self.device = device
        self.entries = torch.zeros(
            (num_blocks, ENTRY_SIZE), dtype=torch.uint8, device=device
        )

    @property
    def num_blocks(self) -> int:
        return self.entries.shape[0]

    def invalidate(self, block_ids: Sequence[int]) -> None:
        if not block_ids:
            return
        index = async_tensor_h2d(list(block_ids), self.device, torch.int64)
        self.entries.index_fill_(0, index, 0)

    def write(self, block_ids: Sequence[int], entries: np.ndarray) -> None:
        if not block_ids:
            return
        index = async_tensor_h2d(list(block_ids), self.device, torch.int64)
        source = async_tensor_h2d(entries, self.device)
        self.entries.index_copy_(0, index, source)


@dataclass
class _MetaRead:
    engine_id: str
    remote_rank: int
    source: int
    local_block_ids: list[int]
    remote_block_ids: list[int]
    expected: np.ndarray


@dataclass
class _Staging:
    buffer: torch.Tensor
    reg_descs: Any
    xfer_handle: int


class KVBlockValidator:
    """Consumer side: reads remote entries after the data and checks them.

    Each source rank reads into its own host staging area, indexed by the
    local block that received the data. With a local pool (an engine that also
    produces), validated blocks are published in it.
    """

    def __init__(
        self,
        nixl_wrapper: Any,
        backends: list[str],
        mode: KVBlockValidationMode,
        identity: str,
        num_blocks: int,
        remote_mem_type: str,
        pool: KVBlockMetaPool | None = None,
    ):
        self.nixl_wrapper = nixl_wrapper
        self.backends = backends
        self.mode = mode
        self.identity = identity
        self.num_blocks = num_blocks
        self.remote_mem_type = remote_mem_type
        self.pool = pool
        self._staging: list[_Staging] = []
        self._remote_handles: dict[tuple[str, int], int] = {}
        self._pending: dict[str, list[_MetaRead]] = {}
        self._posted: dict[str, list[_MetaRead]] = {}

    def add_remote(
        self,
        engine_id: str,
        remote_rank: int,
        agent_name: str,
        base_addr: int | None,
        num_entries: int,
        device_id: int,
        identity: str | None,
    ) -> None:
        if base_addr is None:
            logger.warning_once(
                "Remote engine %s does not publish KV block metadata; its KV "
                "blocks are not validated.",
                engine_id,
            )
            return
        if identity != self.identity:
            logger.error_once(
                "Remote engine %s hashes KV blocks differently (different "
                "prefix_caching_hash_algo or PYTHONHASHSEED); its KV blocks "
                "are not validated.",
                engine_id,
            )
            return
        descs = np.empty((num_entries, 3), dtype=np.int64)
        descs[:, 0] = base_addr + np.arange(num_entries, dtype=np.int64) * ENTRY_SIZE
        descs[:, 1] = ENTRY_SIZE
        descs[:, 2] = device_id
        self._remote_handles[(engine_id, remote_rank)] = (
            self.nixl_wrapper.prep_xfer_dlist(
                agent_name, descs, mem_type=self.remote_mem_type
            )
        )
        logger.info_once(
            "Validating KV blocks read from engine %s (kv_block_validation=%s).",
            engine_id,
            self.mode,
        )

    def remove_remote(self, engine_id: str) -> None:
        for key in [key for key in self._remote_handles if key[0] == engine_id]:
            self.nixl_wrapper.release_dlist_handle(self._remote_handles.pop(key))

    def supports(self, engine_id: str, remote_rank: int) -> bool:
        return (engine_id, remote_rank) in self._remote_handles

    def add_read(
        self,
        req_id: str,
        engine_id: str,
        remote_rank: int,
        source: int,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        expected_hashes: list[bytes],
    ) -> None:
        self._ensure_staging(source)
        self._pending.setdefault(req_id, []).append(
            _MetaRead(
                engine_id=engine_id,
                remote_rank=remote_rank,
                source=source,
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
                expected=make_entries(expected_hashes),
            )
        )

    def has_pending(self, req_id: str) -> bool:
        return req_id in self._pending

    def is_posted(self, req_id: str) -> bool:
        return req_id in self._posted

    def drop(self, req_id: str) -> None:
        self._pending.pop(req_id, None)
        self._posted.pop(req_id, None)

    def post(self, req_id: str) -> list[int]:
        """Read the remote entries of `req_id`'s blocks; returns the handles."""
        reads = self._pending.pop(req_id)
        handles: list[int] = []
        try:
            for read in reads:
                handle = self.nixl_wrapper.make_prepped_xfer(
                    "READ",
                    self._staging[read.source].xfer_handle,
                    read.local_block_ids,
                    self._remote_handles[(read.engine_id, read.remote_rank)],
                    read.remote_block_ids,
                )
                handles.append(handle)
                self.nixl_wrapper.transfer(handle)
        except Exception:
            for handle in handles:
                self.nixl_wrapper.release_xfer_handle(handle)
            raise
        self._posted[req_id] = reads
        return handles

    def finish(self, req_id: str) -> int:
        """Check `req_id`'s entries; returns the number of mismatched blocks."""
        reads = self._posted.pop(req_id)
        mismatched = 0
        for read in reads:
            staging = self._staging[read.source].buffer.numpy()
            received = staging[read.local_block_ids]
            staging[read.local_block_ids] = 0
            bad = np.flatnonzero((received != read.expected).any(axis=1))
            if bad.size:
                mismatched += bad.size
                first = int(bad[0])
                logger.warning(
                    "KV block validation failed for request %s: %d of %d "
                    "blocks from engine %s rank %d hold unexpected content "
                    "(first: local block %d, remote block %d, %s).",
                    req_id,
                    bad.size,
                    len(read.local_block_ids),
                    read.engine_id,
                    read.remote_rank,
                    read.local_block_ids[first],
                    read.remote_block_ids[first],
                    "invalid" if not received[first].any() else "other content",
                )
        if not mismatched and self.pool is not None:
            # The local blocks now hold the expected content.
            for read in reads:
                self.pool.write(read.local_block_ids, read.expected)
        return mismatched

    def _ensure_staging(self, source: int) -> None:
        while len(self._staging) <= source:
            num_blocks = self.num_blocks
            buffer = torch.zeros((num_blocks, ENTRY_SIZE), dtype=torch.uint8)
            base = buffer.data_ptr()
            reg_descs = self.nixl_wrapper.get_reg_descs(
                [(base, buffer.numel(), 0, "")], "DRAM"
            )
            self.nixl_wrapper.register_memory(reg_descs, backends=self.backends)
            descs = np.empty((num_blocks, 3), dtype=np.int64)
            descs[:, 0] = base + np.arange(num_blocks, dtype=np.int64) * ENTRY_SIZE
            descs[:, 1] = ENTRY_SIZE
            descs[:, 2] = 0
            xfer_handle = self.nixl_wrapper.prep_xfer_dlist(
                "NIXL_INIT_AGENT", descs, mem_type="DRAM"
            )
            self._staging.append(_Staging(buffer, reg_descs, xfer_handle))

    def shutdown(self) -> None:
        for key in list(self._remote_handles):
            self.nixl_wrapper.release_dlist_handle(self._remote_handles.pop(key))
        for staging in self._staging:
            self.nixl_wrapper.release_dlist_handle(staging.xfer_handle)
            self.nixl_wrapper.deregister_memory(staging.reg_descs)
        self._staging.clear()
        self._pending.clear()
        self._posted.clear()
