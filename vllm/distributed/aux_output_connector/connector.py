# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

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
    ) -> None:
        # Number of hashes already sent to the worker for each active request.
        self._sent_hash_counts: dict[str, int] = {}
        # Terminal events are delivered with the next connector metadata.
        self._finished_requests: dict[str, PackedBlockHashes | None] = {}
        self._logprob_fingerprints: dict[str, str] = {}
        self._prompt_logprob_fingerprints: dict[str, str] = {}
        self._generation = 0
        self._enable_logprobs = enable_logprobs
        self._enable_prompt_logprobs = enable_prompt_logprobs
        self._logprobs_mode = logprobs_mode
        self._enable_routed_experts = enable_routed_experts

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, Request],
    ) -> AuxOutputConnectorMetadata:
        """Build one step's incremental worker metadata."""
        scheduled_requests: dict[str, int] = {}
        block_hashes_by_request: dict[str, PackedBlockHashes] = {}
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
        # Sending transfers ownership of these one-shot events.
        finished_requests = tuple(self._finished_requests)
        block_hashes_by_request.update(
            (request_id, block_hashes)
            for request_id, block_hashes in self._finished_requests.items()
            if block_hashes is not None
        )
        metadata = AuxOutputConnectorMetadata(
            self._generation,
            scheduled_requests,
            block_hashes_by_request,
            finished_requests,
            {
                request_id: self._logprob_fingerprints[request_id]
                for request_id in block_hashes_by_request
                if request_id in self._logprob_fingerprints
            }
            | {
                request_id: self._logprob_fingerprints[request_id]
                for request_id in scheduled_requests
                if request_id in self._logprob_fingerprints
            },
            {
                request_id: self._prompt_logprob_fingerprints[request_id]
                for request_id in block_hashes_by_request
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
        )
        self._finished_requests = {}
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
        assert output is not None and request_id in output, (
            f"auxiliary output worker output is missing {request_id}"
        )
        request_output = output[request_id]
        if request_output.rows is None:
            return None
        token_end = request.num_tokens - 1
        local_end = token_end - request_output.token_start
        if local_end < 0:
            assert not request.is_finished(), (
                "finished auxiliary output output has no accepted token range: "
                f"request={request_id}, token_end={token_end}, "
                f"output_start={request_output.token_start}, "
                "output_end="
                f"{request_output.token_start + len(request_output.rows)}"
            )
            return None
        assert local_end <= len(request_output.rows), (
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
        assert output is not None and request.request_id in output, (
            f"auxiliary logprobs output is missing {request.request_id}"
        )
        value = output[request.request_id].logprobs
        assert value is not None, (
            f"auxiliary logprobs artifact is missing {request.request_id}"
        )
        return value

    def take_prompt_logprobs(
        self, request: Request, output: dict[str, AuxRequestOutput] | None
    ) -> LogprobsTensors | None:
        if not self._request_replays(request, prompt=True):
            return None
        assert output is not None and request.request_id in output, (
            f"auxiliary prompt logprobs output is missing {request.request_id}"
        )
        return output[request.request_id].prompt_logprobs

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
        """Start a new auxiliary output namespace after a prefix-cache reset."""
        # The worker drops temporary state on generation changes; resend hashes.
        self._sent_hash_counts.clear()
        self._finished_requests.clear()
        self._logprob_fingerprints.clear()
        self._prompt_logprob_fingerprints.clear()
        self._generation += 1
