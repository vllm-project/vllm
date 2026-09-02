# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution artifacts."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


@dataclass
class PackedBlockHashes:
    """Contiguous, self-contained block hashes for scheduler-worker IPC."""

    data: bytes
    item_size: int

    def __iter__(self) -> Iterator[bytes]:
        for start in range(0, len(self.data), self.item_size):
            yield self.data[start : start + self.item_size]


@dataclass
class ArtifactConnectorMetadata:
    generation: int
    requests: dict[str, int]
    block_hashes: dict[str, PackedBlockHashes]
    finished_requests: tuple[str, ...]


@dataclass
class ArtifactRequestOutput:
    token_start: int
    rows: np.ndarray


class ArtifactSchedulerConnector:
    """Build worker metadata without owning artifact payloads or stores."""

    def __init__(self) -> None:
        # Number of hashes already sent to the worker for each active request.
        self._sent_hash_counts: dict[str, int] = {}
        # Terminal events are delivered with the next connector metadata.
        self._finished_requests: dict[str, PackedBlockHashes | None] = {}
        self._generation = 0

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, Request],
    ) -> ArtifactConnectorMetadata:
        """Build one step's incremental worker metadata."""
        scheduled_requests: dict[str, int] = {}
        block_hashes_by_request: dict[str, PackedBlockHashes] = {}
        for request_id in scheduler_output.num_scheduled_tokens:
            num_sent = self._sent_hash_counts.setdefault(request_id, 0)
            request = requests[request_id]
            packed = self._pack_new_hashes(request.block_hashes, num_sent)
            if packed is not None:
                block_hashes_by_request[request_id] = packed
            self._sent_hash_counts[request_id] = len(request.block_hashes)
            assert request.sampling_params is not None
            scheduled_requests[request_id] = max(
                request.sampling_params.routed_experts_prompt_start,
                0 if request.num_output_tokens == 0 else request.num_tokens - 1,
            )
        # A settled token can complete a hash block after the next async schedule
        # was built. Send a hash-only update if the request was not rescheduled.
        if len(self._sent_hash_counts) != len(scheduled_requests):
            for request_id, num_sent in self._sent_hash_counts.items():
                if request_id in scheduled_requests:
                    continue
                request = requests[request_id]
                packed = self._pack_new_hashes(request.block_hashes, num_sent)
                if packed is None:
                    continue
                block_hashes_by_request[request_id] = packed
                self._sent_hash_counts[request_id] = len(request.block_hashes)
        # Sending transfers ownership of these one-shot events.
        finished_requests = tuple(self._finished_requests)
        block_hashes_by_request.update(
            (request_id, block_hashes)
            for request_id, block_hashes in self._finished_requests.items()
            if block_hashes is not None
        )
        self._finished_requests = {}
        return ArtifactConnectorMetadata(
            self._generation,
            scheduled_requests,
            block_hashes_by_request,
            finished_requests,
        )

    def take_output(
        self,
        request: Request,
        output: dict[str, ArtifactRequestOutput] | None,
    ) -> np.ndarray | None:
        """Return the accepted R3 rows for one scheduled request."""
        request_id = request.request_id
        assert output is not None and request_id in output, (
            f"artifact worker output is missing {request_id}"
        )
        request_output = output[request_id]
        token_end = request.num_tokens - 1
        local_end = token_end - request_output.token_start
        if local_end <= 0:
            assert not request.is_finished(), (
                "finished artifact output has no accepted token range: "
                f"request={request_id}, token_end={token_end}, "
                f"output_start={request_output.token_start}, "
                "output_end="
                f"{request_output.token_start + len(request_output.rows)}"
            )
            return None
        assert local_end <= len(request_output.rows), (
            "artifact worker output has an invalid token range: "
            f"request={request_id}, token_end={token_end}, "
            f"output_start={request_output.token_start}, "
            f"output_end={request_output.token_start + len(request_output.rows)}"
        )
        return request_output.rows[:local_end]

    def request_finished(self, request: Request) -> None:
        """Queue a request's terminal event and final block hashes."""
        request_id = request.request_id
        num_sent = self._sent_hash_counts.pop(request_id, None)
        if num_sent is None:
            return
        # The next metadata delivers this terminal event to the worker.
        self._finished_requests[request_id] = self._pack_new_hashes(
            request.block_hashes, num_sent
        )

    @staticmethod
    def _pack_new_hashes(
        block_hashes: Sequence[bytes], num_sent: int
    ) -> PackedBlockHashes | None:
        assert num_sent <= len(block_hashes), "KV block-hash history shrank"
        new_hashes = block_hashes[num_sent:]
        if not new_hashes:
            return None
        return PackedBlockHashes(b"".join(new_hashes), len(new_hashes[0]))

    def reset(self) -> None:
        """Start a new artifact namespace after a prefix-cache reset."""
        # The worker drops temporary state on generation changes; resend hashes.
        self._sent_hash_counts.clear()
        self._finished_requests.clear()
        self._generation += 1
