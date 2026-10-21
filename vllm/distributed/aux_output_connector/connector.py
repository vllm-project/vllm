# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
class AuxOutputConnectorMetadata:
    generation: int
    requests: dict[str, int]
    block_hashes: dict[str, PackedBlockHashes]
    finished_requests: tuple[str, ...]


@dataclass
class AuxRequestOutput:
    token_start: int
    """Logical token offset of the first row in both payloads."""
    routed_experts_rows: np.ndarray
    """Routed expert IDs per token, or zero-width rows when disabled."""
    indexer_rows: np.ndarray | None = None
    """Raw or encoded indexer selections per token, or None when disabled."""
    indexer_profile: dict[str, Any] | None = None
    """Metadata describing indexer sources and decoding, if enabled."""


class AuxOutputSchedulerConnector:
    """Build worker metadata without owning auxiliary output payloads or stores."""

    def __init__(
        self,
        hash_block_size: int,
        *,
        enable_routed_experts: bool = True,
        enable_indexer: bool = False,
    ) -> None:
        self._enable_routed_experts = enable_routed_experts
        self._enable_indexer = enable_indexer
        self._prompt_start_fields = tuple(
            field
            for enabled, field in (
                (enable_routed_experts, "routed_experts_prompt_start"),
                (enable_indexer, "indexer_topk_prompt_start"),
            )
            if enabled
        )
        self._hash_block_size = hash_block_size
        # Number of hashes already sent to the worker for each active request.
        self._sent_hash_counts: dict[str, int] = {}
        # Terminal events are delivered with the next connector metadata.
        self._finished_requests: dict[str, PackedBlockHashes | None] = {}
        self._generation = 0

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, Request],
    ) -> AuxOutputConnectorMetadata:
        """Build one step's incremental worker metadata."""
        scheduled_requests: dict[str, int] = {}
        block_hashes_by_request: dict[str, PackedBlockHashes] = {}
        for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            self._sent_hash_counts.setdefault(request_id, 0)
            request = requests[request_id]
            packed = self._take_new_hashes(
                request, request.num_computed_tokens + num_tokens
            )
            if packed is not None:
                block_hashes_by_request[request_id] = packed
            scheduled_requests[request_id] = self._get_request_start(request)
        # A settled token can complete a hash block after the next async schedule
        # was built. Send a hash-only update if the request was not rescheduled.
        if len(self._sent_hash_counts) != len(scheduled_requests):
            for request_id in self._sent_hash_counts:
                if request_id in scheduled_requests:
                    continue
                request = requests[request_id]
                packed = self._take_new_hashes(request, request.num_computed_tokens)
                if packed is not None:
                    block_hashes_by_request[request_id] = packed
        # Sending transfers ownership of these one-shot events.
        finished_requests = tuple(self._finished_requests)
        block_hashes_by_request.update(
            (request_id, block_hashes)
            for request_id, block_hashes in self._finished_requests.items()
            if block_hashes is not None
        )
        self._finished_requests = {}
        return AuxOutputConnectorMetadata(
            self._generation,
            scheduled_requests,
            block_hashes_by_request,
            finished_requests,
        )

    def _get_request_start(self, request: Request) -> int:
        assert request.sampling_params is not None
        prompt_start: int = min(
            getattr(request.sampling_params, field)
            for field in self._prompt_start_fields
        )
        # A P/D prefiller may cut the prompt short of the requested start.
        return max(
            min(prompt_start, request.num_prompt_tokens),
            0 if request.num_output_tokens == 0 else request.num_tokens - 1,
        )

    def take_output(
        self,
        request: Request,
        output: dict[str, AuxRequestOutput] | None,
    ) -> np.ndarray | None:
        """Return the accepted R3 rows for one scheduled request."""
        request_id = request.request_id
        assert output is not None and request_id in output, (
            f"auxiliary output worker output is missing {request_id}"
        )
        request_output = output[request_id]
        routed_experts_rows = request_output.routed_experts_rows
        token_end = request.num_tokens - 1
        local_end = token_end - request_output.token_start
        if local_end < 0:
            assert not request.is_finished(), (
                "finished auxiliary output output has no accepted token range: "
                f"request={request_id}, token_end={token_end}, "
                f"output_start={request_output.token_start}, "
                "output_end="
                f"{request_output.token_start + len(routed_experts_rows)}"
            )
            return None
        assert local_end <= len(routed_experts_rows), (
            "auxiliary output worker output has an invalid token range: "
            f"request={request_id}, token_end={token_end}, "
            f"output_start={request_output.token_start}, "
            "output_end="
            f"{request_output.token_start + len(routed_experts_rows)}"
        )
        if not self._enable_routed_experts:
            return None
        assert request.sampling_params is not None
        start = max(
            0,
            min(
                request.sampling_params.routed_experts_prompt_start,
                request.num_prompt_tokens,
            )
            - request_output.token_start,
        )
        return routed_experts_rows[start:local_end]

    def take_indexer_output(
        self,
        request: Request,
        output: dict[str, AuxRequestOutput] | None,
    ) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        if not self._enable_indexer:
            return None, None
        assert output is not None
        value = output[request.request_id]
        assert value.indexer_rows is not None and value.indexer_profile is not None
        assert request.sampling_params is not None
        start = max(
            0,
            min(
                request.sampling_params.indexer_topk_prompt_start,
                request.num_prompt_tokens,
            )
            - value.token_start,
        )
        end = request.num_tokens - 1 - value.token_start
        if end < 0:
            return None, None
        assert end <= len(value.indexer_rows), "Indexer output exceeds accepted range"
        return value.indexer_rows[start:end], value.indexer_profile

    def request_finished(self, request: Request) -> None:
        """Queue a request's terminal event and final block hashes."""
        request_id = request.request_id
        if request_id not in self._sent_hash_counts:
            return
        # The next metadata delivers this terminal event to the worker.
        self._finished_requests[request_id] = self._take_new_hashes(
            request, request.num_computed_tokens
        )
        del self._sent_hash_counts[request_id]

    def _take_new_hashes(
        self, request: Request, num_computed_tokens: int
    ) -> PackedBlockHashes | None:
        block_hashes = request.block_hashes
        num_sent = self._sent_hash_counts[request.request_id]
        assert num_sent <= len(block_hashes), "KV block-hash history shrank"
        num_keyed = min(len(block_hashes), num_computed_tokens // self._hash_block_size)
        if num_keyed <= num_sent:
            return None
        self._sent_hash_counts[request.request_id] = num_keyed
        new_hashes = block_hashes[num_sent:num_keyed]
        return PackedBlockHashes(b"".join(new_hashes), len(new_hashes[0]))

    def reset(self) -> None:
        """Start a new auxiliary output namespace after a prefix-cache reset."""
        # The worker drops temporary state on generation changes; resend hashes.
        self._sent_hash_counts.clear()
        self._finished_requests.clear()
        self._generation += 1
