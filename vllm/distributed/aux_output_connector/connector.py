# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import LogprobsLists, LogprobsTensors
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
    logprobs: dict[str, str] = field(default_factory=dict)
    prompt_logprobs: dict[str, str] = field(default_factory=dict)
    prompt_lens: dict[str, int] = field(default_factory=dict)
    logprob_block_hashes: dict[str, PackedBlockHashes] = field(default_factory=dict)
    logprob_boundary_token_ids: dict[str, tuple[int | None, ...]] = field(
        default_factory=dict
    )


@dataclass
class AuxRequestOutput:
    token_start: int
    rows: np.ndarray | None = None
    logprobs: LogprobsLists | None = None
    prompt_logprobs: LogprobsTensors | None = None


class AuxOutputSchedulerConnector:
    """Build worker metadata without owning auxiliary output payloads or stores."""

    def __init__(
        self,
        *,
        enable_routed_experts: bool = True,
        enable_logprobs: bool = False,
        enable_prompt_logprobs: bool = False,
        logprobs_mode: str = "raw_logprobs",
        hash_block_size: int = 1,
    ) -> None:
        # Number of hashes already sent to the worker for each active request.
        self._sent_hash_counts: dict[str, int] = {}
        self._sent_logprob_hash_counts: dict[str, int] = {}
        # Terminal events are delivered with the next connector metadata.
        self._finished_requests: dict[str, PackedBlockHashes | None] = {}
        self._finished_logprob_blocks: dict[
            str, tuple[PackedBlockHashes, tuple[int | None, ...]] | None
        ] = {}
        self._logprob_fingerprints: dict[str, str] = {}
        self._prompt_logprob_fingerprints: dict[str, str] = {}
        # Delivered once per Request object. Weak values keep the latch alive
        # across the request's final scheduler step without retaining finished
        # requests or conflating reused request IDs.
        self._prompt_logprobs_delivered: weakref.WeakValueDictionary[
            int, Request
        ] = weakref.WeakValueDictionary()
        self._generation = 0
        self._enable_logprobs = enable_logprobs
        self._enable_prompt_logprobs = enable_prompt_logprobs
        self._logprobs_mode = logprobs_mode
        self._enable_routed_experts = enable_routed_experts
        self._hash_block_size = hash_block_size

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, Request],
    ) -> AuxOutputConnectorMetadata:
        """Build one step's incremental worker metadata."""
        scheduled_requests: dict[str, int] = {}
        block_hashes_by_request: dict[str, PackedBlockHashes] = {}
        logprob_block_hashes: dict[str, PackedBlockHashes] = {}
        logprob_boundary_token_ids: dict[str, tuple[int | None, ...]] = {}
        for request_id in scheduler_output.num_scheduled_tokens:
            request = requests[request_id]
            assert request.sampling_params is not None
            replay_logprobs = self._request_replays(request, prompt=False)
            replay_prompt_logprobs = self._request_replays(request, prompt=True)
            if self._enable_logprobs and replay_logprobs:
                self._logprob_fingerprints[request_id] = self._fingerprint(
                    request, prompt=False
                )
            if self._enable_prompt_logprobs and replay_prompt_logprobs:
                self._prompt_logprob_fingerprints[request_id] = self._fingerprint(
                    request, prompt=True
                )
            replay_logprobs = (
                replay_logprobs and request_id in self._logprob_fingerprints
            )
            replay_prompt_logprobs = (
                replay_prompt_logprobs
                and request_id in self._prompt_logprob_fingerprints
            )
            if not self._enable_routed_experts and not (
                replay_logprobs or replay_prompt_logprobs
            ):
                continue
            scheduled_requests[request_id] = max(
                min(
                    request.sampling_params.routed_experts_prompt_start,
                    request.num_prompt_tokens,
                ),
                0 if request.num_output_tokens == 0 else request.num_tokens - 1,
            )
            num_sent = self._sent_hash_counts.setdefault(request_id, 0)
            packed = self._pack_new_hashes(request.block_hashes, num_sent)
            if packed is not None:
                block_hashes_by_request[request_id] = packed
            self._sent_hash_counts[request_id] = len(request.block_hashes)
            if replay_logprobs or replay_prompt_logprobs:
                self._add_logprob_hashes(
                    request,
                    logprob_block_hashes,
                    logprob_boundary_token_ids,
                )
        # A settled token can complete a hash block after the next async schedule
        # was built. Send a hash-only update if the request was not rescheduled.
        for request_id, num_sent in self._sent_hash_counts.items():
            if request_id in scheduled_requests:
                continue
            request = requests[request_id]
            packed = self._pack_new_hashes(request.block_hashes, num_sent)
            if packed is None:
                continue
            block_hashes_by_request[request_id] = packed
            self._sent_hash_counts[request_id] = len(request.block_hashes)
            if (
                request_id in self._logprob_fingerprints
                or request_id in self._prompt_logprob_fingerprints
            ):
                self._add_logprob_hashes(
                    request,
                    logprob_block_hashes,
                    logprob_boundary_token_ids,
                )
        # Sending transfers ownership of these one-shot events.
        finished_requests = tuple(self._finished_requests)
        block_hashes_by_request.update(
            (request_id, block_hashes)
            for request_id, block_hashes in self._finished_requests.items()
            if block_hashes is not None
        )
        for request_id, update in self._finished_logprob_blocks.items():
            if update is None:
                continue
            hashes, boundary_token_ids = update
            logprob_block_hashes[request_id] = hashes
            logprob_boundary_token_ids[request_id] = boundary_token_ids
        metadata = AuxOutputConnectorMetadata(
            self._generation,
            scheduled_requests,
            block_hashes_by_request,
            finished_requests,
            {
                request_id: self._logprob_fingerprints[request_id]
                for request_id in logprob_block_hashes
                if request_id in self._logprob_fingerprints
            }
            | {
                request_id: self._logprob_fingerprints[request_id]
                for request_id in scheduled_requests
                if request_id in self._logprob_fingerprints
            },
            {
                request_id: self._prompt_logprob_fingerprints[request_id]
                for request_id in logprob_block_hashes
                if request_id in self._prompt_logprob_fingerprints
            }
            | {
                request_id: self._prompt_logprob_fingerprints[request_id]
                for request_id in scheduled_requests
                if request_id in self._prompt_logprob_fingerprints
            },
            {
                request_id: requests[request_id].num_prompt_tokens
                for request_id in scheduled_requests
            },
            logprob_block_hashes,
            logprob_boundary_token_ids,
        )
        self._finished_requests = {}
        self._finished_logprob_blocks = {}
        for request_id in finished_requests:
            self._logprob_fingerprints.pop(request_id, None)
            self._prompt_logprob_fingerprints.pop(request_id, None)
        return metadata

    def take_output(
        self,
        request: Request,
        output: dict[str, AuxRequestOutput] | None,
    ) -> np.ndarray | None:
        """Return the accepted R3 rows for one scheduled request."""
        request_id = request.request_id
        if output is None or request_id not in output:
            raise RuntimeError(
                f"auxiliary output worker output is missing {request_id}"
            )
        request_output = output[request_id]
        if request_output.rows is None:
            return None
        token_end = request.num_tokens - 1
        local_end = token_end - request_output.token_start
        if local_end < 0:
            if request.is_finished():
                raise RuntimeError(
                    "finished auxiliary output has no accepted token range: "
                    f"request={request_id}, token_end={token_end}, "
                    f"output_start={request_output.token_start}, "
                    "output_end="
                    f"{request_output.token_start + len(request_output.rows)}"
                )
            return None
        if local_end > len(request_output.rows):
            raise RuntimeError(
                "auxiliary output worker output has an invalid token range: "
                f"request={request_id}, token_end={token_end}, "
                f"output_start={request_output.token_start}, "
                f"output_end={request_output.token_start + len(request_output.rows)}"
            )
        return request_output.rows[:local_end]

    @staticmethod
    def _request_replays(request: Request, *, prompt: bool) -> bool:
        params = request.sampling_params
        extra_args = None if params is None else getattr(params, "extra_args", None)
        if (
            params is None
            or not extra_args
            or not extra_args.get("aux_output_replay", False)
        ):
            return False
        return (
            (params.prompt_logprobs is not None)
            if prompt
            else (params.num_logprobs is not None)
        )

    def _fingerprint(self, request: Request, *, prompt: bool) -> str:
        params = request.sampling_params
        assert params is not None
        width = params.prompt_logprobs if prompt else params.num_logprobs
        token_ids = (
            params.prompt_logprob_token_ids if prompt else params.logprob_token_ids
        )
        token_ids = None if token_ids is None else list(token_ids)
        common_width = (
            max(
                value
                for value in (params.num_logprobs, params.prompt_logprobs)
                if value is not None
            )
            if token_ids is None
            else width
        )
        return repr((self._logprobs_mode, common_width, token_ids))

    def take_logprobs(
        self, request: Request, output: dict[str, AuxRequestOutput] | None
    ) -> LogprobsLists | None:
        if not self._request_replays(request, prompt=False):
            return None
        if output is None or request.request_id not in output:
            raise RuntimeError(
                f"auxiliary logprobs output is missing {request.request_id}"
            )
        value = output[request.request_id].logprobs
        if value is None:
            raise RuntimeError(
                f"auxiliary logprobs artifact is missing {request.request_id}"
            )
        return value

    def take_prompt_logprobs(
        self, request: Request, output: dict[str, AuxRequestOutput] | None
    ) -> LogprobsTensors | None:
        if not self._request_replays(request, prompt=True):
            return None
        # Prompt artifacts are complete once per request, but the scheduler
        # may call this on multiple visible steps (streaming and termination).
        # "take" semantics prevents the engine logprobs processor from
        # extending prompt_logprobs with the same rows more than once.
        if self._prompt_logprobs_delivered.get(id(request)) is request:
            return None
        if output is None or request.request_id not in output:
            raise RuntimeError(
                "auxiliary prompt logprobs output is missing "
                f"{request.request_id}"
            )
        request_id = request.request_id
        value = output[request_id].prompt_logprobs
        if value is None:
            raise RuntimeError(
                f"auxiliary prompt logprobs artifact is missing {request_id}"
            )
        self._prompt_logprobs_delivered[id(request)] = request
        return value

    def replays_prompt_logprobs(self, request: Request) -> bool:
        """True when this request's prompt logprobs come from the aux plane."""
        return self._request_replays(request, prompt=True)

    def release_request(self, request: Request) -> None:
        """Release scheduler-side prompt-logprob delivery state."""
        request_key = id(request)
        if self._prompt_logprobs_delivered.get(request_key) is request:
            self._prompt_logprobs_delivered.pop(request_key, None)

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
        logprob_sent = self._sent_logprob_hash_counts.pop(request_id, None)
        if logprob_sent is not None:
            self._finished_logprob_blocks[request_id] = self._logprob_hash_update(
                request, logprob_sent
            )

    def _add_logprob_hashes(
        self,
        request: Request,
        hashes_by_request: dict[str, PackedBlockHashes],
        boundaries_by_request: dict[str, tuple[int | None, ...]],
    ) -> None:
        request_id = request.request_id
        num_sent = self._sent_logprob_hash_counts.setdefault(request_id, 0)
        update = self._logprob_hash_update(request, num_sent)
        if update is None:
            return
        hashes, boundary_token_ids = update
        hashes_by_request[request_id] = hashes
        boundaries_by_request[request_id] = boundary_token_ids
        self._sent_logprob_hash_counts[request_id] = num_sent + len(boundary_token_ids)

    def _logprob_hash_update(
        self, request: Request, num_sent: int
    ) -> tuple[PackedBlockHashes, tuple[int | None, ...]] | None:
        num_hashes = len(request.block_hashes)
        packed = self._pack_new_hashes(request.block_hashes, num_sent)
        if packed is None:
            return None
        boundary_token_ids = tuple(
            (
                request.all_token_ids[(index + 1) * self._hash_block_size]
                if (index + 1) * self._hash_block_size < request.num_tokens
                else None
            )
            for index in range(num_sent, num_hashes)
        )
        return packed, boundary_token_ids

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
        """Start a new auxiliary output namespace after a prefix-cache reset."""
        # The worker drops temporary state on generation changes; resend hashes.
        self._sent_hash_counts.clear()
        self._sent_logprob_hash_counts.clear()
        self._finished_requests.clear()
        self._finished_logprob_blocks.clear()
        self._logprob_fingerprints.clear()
        self._prompt_logprob_fingerprints.clear()
        self._generation += 1
