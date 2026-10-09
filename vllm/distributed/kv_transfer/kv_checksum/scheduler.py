# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side KV checksums: the blocks workers checksum, and verification
of loads against the producer's checksums."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import pybase64 as base64

from vllm.distributed.kv_transfer.kv_checksum.checksum import (
    KVChecksumBlocks,
    KVChecksumCarrier,
    KVChecksumGroup,
    KVChecksumRecord,
    KVChecksumScheduled,
    ReplicaKey,
    kv_checksum_enabled,
    num_valid_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheConfig, SlidingWindowSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

KV_TRANSFER_PARAMS_KEY = "kv_checksums"

_PAYLOAD_VERSION = 1


@dataclass
class KVChecksumPayload:
    """A producer's checksums of one request, as sent to its consumer.

    Attributes:
        num_tokens: Tokens whose KV the checksummed blocks hold.
        groups: Per KV cache group id, the group's fingerprint (see
            ``KVChecksumRecord.fingerprints``), the position of the first
            block, and one uint64 checksum per block position.

    """

    num_tokens: int
    groups: dict[int, tuple[int, int, np.ndarray]]

    def to_bytes(self) -> bytes:
        groups = [
            (group_id, fingerprint, start, values.astype("<u8").tobytes())
            for group_id, (fingerprint, start, values) in self.groups.items()
        ]
        return msgspec.msgpack.encode((_PAYLOAD_VERSION, self.num_tokens, groups))

    @classmethod
    def from_bytes(cls, data: bytes) -> "KVChecksumPayload":
        """Decode a payload.

        Raises:
            ValueError: If ``data`` is not a payload of this version.

        """
        try:
            version, num_tokens, groups = msgspec.msgpack.decode(
                data, type=tuple[int, int, list[tuple[int, int, int, bytes]]]
            )
        except msgspec.DecodeError as e:
            raise ValueError(f"malformed KV checksums: {e}") from e
        if version != _PAYLOAD_VERSION:
            raise ValueError(f"unsupported KV checksum version {version}")
        return cls(
            num_tokens,
            {
                group_id: (fingerprint, start, np.frombuffer(values, dtype="<u8"))
                for group_id, fingerprint, start, values in groups
            },
        )


class _Unverified(Exception):
    """A load, or one of its groups, that cannot be compared.

    ``reason`` is one of a few fixed messages, logged once each; ``detail``
    is logged per request at debug level.
    """

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass
class _Pending:
    """A request whose checksums are being collected from the workers."""

    blocks: KVChecksumBlocks
    expected: KVChecksumPayload | None = None
    """The producer's checksums; only for a load."""
    missing_reason: str = "the producer sent no checksums"
    """Why ``expected`` is None."""
    reports: dict[
        tuple[int, int], tuple[dict[int, KVChecksumGroup], KVChecksumRecord]
    ] = field(default_factory=dict)
    """Per (pp_rank, tp_rank): its checksums of the request per group, and
    its record."""

    def check_complete(self, num_workers: int) -> None:
        """Check that every worker reported the request.

        Raises:
            _Unverified: If not.

        """
        if len(self.reports) < num_workers:
            raise _Unverified(
                "not every worker reported its checksums",
                f"{len(self.reports)} of {num_workers} workers reported",
            )

    def skipped_groups(self) -> frozenset[int]:
        return frozenset().union(*(r.skipped_groups for _, r in self.reports.values()))

    def fingerprint(self, group_id: int) -> int:
        """The group's fingerprint for the whole model: one TP rank's per PP
        stage, summed."""
        stages = {pp_rank: record for (pp_rank, _), (_, record) in self.reports.items()}
        total = sum(record.fingerprints.get(group_id, 0) for record in stages.values())
        return total % 2**64

    def group_checksums(self, group_id: int) -> list[KVChecksumGroup]:
        """Every worker's checksums of the group's scheduled blocks.

        Raises:
            _Unverified: If a worker checksummed other blocks, e.g. those of
                an earlier prefill of a preempted request.

        """
        start, block_ids = self.blocks.groups[group_id]
        groups = [
            group
            for checksums, _ in self.reports.values()
            if (group := checksums.get(group_id)) is not None
        ]
        for group in groups:
            if group.start != start or len(group.values) != len(block_ids):
                raise _Unverified(
                    "a worker checksummed other blocks than scheduled",
                    f"group {group_id}",
                )
        return groups


class KVChecksumScheduler:
    """Schedules KV checksums and verifies loaded KV against the producer's.

    Only requests whose connector carries checksums for them
    (``KVConnectorBase_V1.kv_checksum_carrier``) are checksummed.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: KVCacheConfig,
        connector: KVConnectorBase_V1,
    ):
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        parallel_config = vllm_config.parallel_config
        self._connector = connector
        self._fail_closed = kv_transfer_config.kv_checksum_fail_closed
        self._num_workers = (
            parallel_config.tensor_parallel_size
            * parallel_config.pipeline_parallel_size
        )
        self._groups = {
            group_id: kv_cache_config.kv_cache_groups[group_id]
            for group_id in kv_cache_config.transfer_group_ids
        }
        self._sends: dict[str, _Pending] = {}
        self._recvs: dict[str, _Pending] = {}
        self._scheduled = KVChecksumScheduled()

    @classmethod
    def create(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_config: KVCacheConfig,
        connector: KVConnectorBase_V1 | None,
    ) -> "KVChecksumScheduler | None":
        if connector is None or not kv_checksum_enabled(vllm_config):
            return None
        if (
            type(connector).kv_checksum_carrier
            is KVConnectorBase_V1.kv_checksum_carrier
        ):
            message = f"{type(connector).__name__} does not carry KV checksums"
            kv_transfer_config = vllm_config.kv_transfer_config
            assert kv_transfer_config is not None
            if kv_transfer_config.kv_checksum_fail_closed:
                raise ValueError(f"{message}, so no load could be verified.")
            logger.warning("%s; KV checksums are disabled.", message)
            return None
        return cls(vllm_config, kv_cache_config, connector)

    def add_send(self, request: "Request", blocks: "KVCacheBlocks") -> None:
        """Checksum a request's KV in the step that finishes its prefill, for
        a consumer to verify after loading it."""
        if self._connector.kv_checksum_carrier(request, receiving=False) is None:
            return
        num_tokens = request.num_computed_tokens
        checksum_blocks = self._blocks(blocks, 0, num_tokens)
        # A new prefill (after a preemption) replaces an earlier one.
        self._sends[request.request_id] = _Pending(checksum_blocks)
        self._scheduled.reqs_to_send[request.request_id] = checksum_blocks

    def add_recv(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_local_computed_tokens: int,
    ) -> None:
        """Checksum the blocks a load starting this step writes, once it
        finishes.

        ``request.num_computed_tokens`` must already include the loaded
        tokens.
        """
        carrier = self._connector.kv_checksum_carrier(request, receiving=True)
        if carrier is None:
            return
        assert carrier == KVChecksumCarrier.KV_TRANSFER_PARAMS
        checksum_blocks = self._blocks(
            blocks, num_local_computed_tokens, request.num_computed_tokens
        )
        pending = _Pending(checksum_blocks)
        params = request.kv_transfer_params or {}
        if (encoded := params.get(KV_TRANSFER_PARAMS_KEY)) is not None:
            try:
                pending.expected = KVChecksumPayload.from_bytes(
                    base64.b64decode(encoded)
                )
            except (ValueError, TypeError) as e:
                pending.missing_reason = "the producer's checksums are invalid"
                logger.debug(
                    "Invalid KV checksums for request %s: %s", request.request_id, e
                )
        self._recvs[request.request_id] = pending
        self._scheduled.reqs_to_recv[request.request_id] = checksum_blocks

    def build_checksum_meta(self) -> KVChecksumScheduled | None:
        """The requests this step's workers checksum; resets them."""
        scheduled = self._scheduled
        if not scheduled.reqs_to_send and not scheduled.reqs_to_recv:
            return None
        self._scheduled = KVChecksumScheduled()
        return scheduled

    def update_from_output(
        self,
        kv_connector_output: "KVConnectorOutput | None",
        skip_req_ids: set[str],
    ) -> set[str]:
        """Collect the workers' checksums and verify the finished loads.

        Args:
            kv_connector_output: The step's aggregated connector output.
            skip_req_ids: Requests whose load already failed; not verified.

        Returns:
            The requests whose load fails verification.

        """
        if kv_connector_output is None:
            return set()
        if (output := kv_connector_output.kv_checksums) is not None:
            for record in output.finalize():
                worker = (record.pp_rank, record.tp_rank)
                for req_id, groups in record.checksums.items():
                    pending = self._sends.get(req_id) or self._recvs.get(req_id)
                    if pending is not None:
                        pending.reports[worker] = (groups, record)

        failed: set[str] = set()
        for req_id in kv_connector_output.finished_recving or ():
            pending = self._recvs.pop(req_id, None)
            if (
                pending is not None
                and req_id not in skip_req_ids
                and not self._verify(req_id, pending)
            ):
                failed.add(req_id)
        return failed

    def request_finished(
        self, request: "Request", kv_transfer_params: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Add the request's checksums to the kv_transfer_params it returns.

        Called for every finished request, which drops its state.
        """
        self._recvs.pop(request.request_id, None)
        pending = self._sends.pop(request.request_id, None)
        if pending is None or kv_transfer_params is None:
            return kv_transfer_params
        payload = self._payload(request.request_id, pending)
        if payload is not None:
            kv_transfer_params[KV_TRANSFER_PARAMS_KEY] = base64.b64encode(
                payload.to_bytes()
            ).decode()
        return kv_transfer_params

    def _blocks(
        self, blocks: "KVCacheBlocks", start_token: int, end_token: int
    ) -> KVChecksumBlocks:
        """The blocks of each transfer group holding tokens in
        ``[start_token, end_token)``.

        Leading null blocks (out of a sliding window) are left out, as is a
        group with a null block after a non-null one. Sliding-window groups
        keep only their last ``cdiv(window, block_size) + 1`` blocks, the
        blocks the NIXL and Mooncake connectors transfer.
        """
        groups: dict[int, tuple[int, list[int]]] = {}
        for group_id, group in self._groups.items():
            spec = group.kv_cache_spec
            block_size = spec.block_size
            first = start_token // block_size
            run = _non_null_run(
                blocks.blocks[group_id], first, cdiv(end_token, block_size)
            )
            if run is None:
                continue
            start, block_ids = run
            if isinstance(spec, SlidingWindowSpec):
                keep = cdiv(spec.sliding_window, block_size) + 1
                start += max(0, len(block_ids) - keep)
                block_ids = block_ids[-keep:]
            groups[group_id] = (start, block_ids)
        return KVChecksumBlocks(num_tokens=end_token, groups=groups)

    def _payload(self, req_id: str, pending: _Pending) -> KVChecksumPayload | None:
        """The sum of every worker's checksums, or None if some are missing."""
        try:
            pending.check_complete(self._num_workers)
            skipped = pending.skipped_groups()
            groups: dict[int, tuple[int, int, np.ndarray]] = {}
            for group_id, (start, block_ids) in pending.blocks.groups.items():
                if group_id in skipped:
                    continue
                values = np.zeros(len(block_ids), dtype=np.uint64)
                for group in pending.group_checksums(group_id):
                    values += group.values.sum(axis=1, dtype=np.uint64)
                groups[group_id] = (pending.fingerprint(group_id), start, values)
        except _Unverified as e:
            logger.warning_once("KV checksums not sent: %s.", e.reason)
            logger.debug("KV checksums of request %s not sent: %s", req_id, e.detail)
            return None
        return KVChecksumPayload(pending.blocks.num_tokens, groups)

    def _verify(self, req_id: str, pending: _Pending) -> bool:
        """Compare a finished load with the producer's checksums and log the
        outcome. Returns whether the load may proceed."""
        mismatches, unverified = self._compare(pending)
        if mismatches:
            logger.error(
                "KV checksum mismatch for request %s: %s.",
                req_id,
                "; ".join(
                    f"group {group_id} block positions {positions}"
                    for group_id, positions in mismatches.items()
                ),
            )
        elif unverified:
            for e in unverified:
                logger.warning_once("KV checksums not verified: %s.", e.reason)
            logger.debug(
                "KV checksums of request %s not verified: %s.",
                req_id,
                "; ".join(f"{e.reason} ({e.detail})" for e in unverified),
            )
        else:
            return True
        return not self._fail_closed

    def _compare(
        self, pending: _Pending
    ) -> tuple[dict[int, list[int]], list[_Unverified]]:
        """The mismatching block positions per group, and why the load or some
        of its groups could not be compared."""
        try:
            pending.check_complete(self._num_workers)
            if pending.expected is None:
                raise _Unverified(pending.missing_reason)
        except _Unverified as e:
            return {}, [e]
        mismatches: dict[int, list[int]] = {}
        unverified: list[_Unverified] = []
        skipped = pending.skipped_groups()
        for group_id in pending.blocks.groups:
            try:
                bad = self._compare_group(
                    pending, pending.expected, group_id, group_id in skipped
                )
            except _Unverified as e:
                unverified.append(e)
                continue
            if bad:
                mismatches[group_id] = bad
        compared = set(pending.blocks.groups) - skipped
        if not compared and not unverified:
            unverified.append(_Unverified("no loaded group can be checksummed"))
        return mismatches, unverified

    def _compare_group(
        self,
        pending: _Pending,
        expected: KVChecksumPayload,
        group_id: int,
        skipped: bool,
    ) -> list[int]:
        """The group's mismatching block positions.

        Raises:
            _Unverified: If the group cannot be compared.

        """
        sent = expected.groups.get(group_id)
        if skipped or sent is None:
            if skipped != (sent is None):
                raise _Unverified(
                    "a group is checksummed by one side only",
                    f"group {group_id} on the "
                    f"{'producer' if skipped else 'consumer'} only",
                )
            return []
        fingerprint, sent_start, sent_values = sent
        if fingerprint != pending.fingerprint(group_id):
            raise _Unverified(
                "a group holds other layers, block size or dtype on the producer",
                f"group {group_id}",
            )
        start, block_ids = pending.blocks.groups[group_id]
        block_size = self._groups[group_id].kv_cache_spec.block_size
        positions = [
            pos
            for pos in range(
                max(start, sent_start),
                min(start + len(block_ids), sent_start + len(sent_values)),
            )
            # Only the last block can hold fewer tokens on one side.
            if num_valid_tokens(pos, block_size, pending.blocks.num_tokens)
            == num_valid_tokens(pos, block_size, expected.num_tokens)
        ]
        if not positions:
            raise _Unverified("a group has no block in common", f"group {group_id}")
        index = np.array(positions)
        sent_values = sent_values[index - sent_start]
        bad = np.zeros(len(positions), dtype=bool)
        for values in _sum_replicas(pending.group_checksums(group_id), len(block_ids)):
            bad |= values[index - start] != sent_values
        return index[bad].tolist()


def _non_null_run(
    blocks: Sequence["KVCacheBlock"], first: int, end: int
) -> tuple[int, list[int]] | None:
    """The position and ids of the non-null blocks in ``blocks[first:end]``,
    if they are consecutive and any."""
    block_ids = [block.block_id for block in blocks[first:end]]
    nulls = [block.is_null for block in blocks[first:end]]
    try:
        offset = nulls.index(False)
    except ValueError:
        return None
    if any(nulls[offset:]):
        return None
    return first + offset, block_ids[offset:]


def _sum_replicas(
    groups: list[KVChecksumGroup], num_positions: int
) -> list[np.ndarray]:
    """The consumer's checksums of a group, one per check. Check ``k`` sums
    copy ``k % R`` of each layer replicated on ``R`` ranks, so every check
    covers each layer once and every copy is checked.

    Raises:
        _Unverified: If a check misses a copy.

    """
    totals: dict[ReplicaKey, np.ndarray] = {}
    for group in groups:
        for column, key in enumerate(group.replica_keys):
            total = totals.setdefault(key, np.zeros(num_positions, dtype=np.uint64))
            total += group.values[:, column]
    replica_counts = {replicas for replicas, _ in totals}
    checks = []
    for k in range(max(replica_counts, default=1)):
        check = np.zeros(num_positions, dtype=np.uint64)
        for replicas in replica_counts:
            if (total := totals.get((replicas, k % replicas))) is None:
                raise _Unverified(
                    "a replica was not checksummed",
                    f"replica {k % replicas} of {replicas}",
                )
            check += total
        checks.append(check)
    return checks
