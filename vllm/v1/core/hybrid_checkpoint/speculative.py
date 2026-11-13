# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where a request's exact recurrent state is under speculative decoding.

Under speculative decoding a request's computed-token count does not say where
its exact state is. A verify step of `q` tokens that starts at device position
`s` writes the state after `s + t + 1` tokens into draft column `t`: block
`(s + q - 1) // B + t` of the request's table in every Mamba group, whose
convolution window lives in column 0 and is read `t` tokens ahead.

Under asynchronous scheduling a request may finish while one more step for it
is in flight. That step's first input token is the request's last token, so
its column 0 ends up holding the state after every token of the request;
`speculative_config.truncate_acceptance_at_stop` keeps it from accepting
anything past that token.

`SpeculativeFrontiers` keeps each request's last completed step, from
quantities the scheduler sees, and resolves where the exact state is when the
request finishes. It knows nothing about the drafter or the model.
"""

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.request import RequestStatus

# The stop token ids the GPU sampler holds per request
# (`vllm.v1.worker.gpu.sample.logit_bias.MAX_NUM_STOP_TOKEN_IDS`); a request
# with more gets no stop-id truncation on the device.
MAX_MIRRORED_STOP_TOKEN_IDS = 128

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request


def check_speculative_config(vllm_config: "VllmConfig") -> str | None:
    """The requirement that frontier checkpoints miss under this speculative
    decoding configuration, or None."""
    speculative = vllm_config.speculative_config
    assert speculative is not None
    if not speculative.truncate_acceptance_at_stop:
        return "speculative_config.truncate_acceptance_at_stop"
    if speculative.use_eagle_block_drop():
        # A frontier is resumed at its exact length, which the block drop would
        # cut back to a block boundary.
        return "speculative_config.disable_eagle_block_drop"
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        # More than one step of a request may be in flight.
        return "pipeline parallel size 1 under speculative decoding"
    if vllm_config.kv_transfer_config is not None:
        # A finished request's state may wait for a later step's draft-column
        # gather when a connector reads it.
        return "no KV connector under speculative decoding"
    return None


@dataclass(frozen=True, slots=True)
class _Step:
    # Device position of the step's first input token.
    start: int
    # The step's query length.
    num_scheduled: int
    # Whether the step verified draft tokens.
    verified: bool
    # Tokens the device sampled (accepted drafts plus one).
    num_sampled: int
    # Tokens the scheduler kept.
    num_kept: int


@dataclass(frozen=True, slots=True)
class ExactState:
    """Where a finished request's exact state is."""

    # Tokens the state has consumed.
    num_tokens: int
    # Device start and query length of the step that holds the state.
    step_start: int
    step_len: int
    # The draft column that holds the state.
    column: int

    def block_indices(self, block_size: int) -> tuple[int, int]:
        """Indices, in a Mamba group's block table, of the block to publish
        and of the block holding the recurrent state. They differ when the
        state is in a later draft column: its recurrent state and the first
        column's window, read `column` tokens ahead, are folded into the
        first column before anything reads it."""
        index = (self.step_start + self.step_len - 1) // block_size
        return index, index + self.column


class SpeculativeFrontiers:
    def __init__(self) -> None:
        self._last: dict[str, _Step] = {}
        # Why a finished request's state could not be resolved, by reason.
        self.declined: Counter[str] = Counter()

    def step_completed(
        self,
        request_id: str,
        start: int,
        num_scheduled: int,
        verified: bool,
        num_sampled: int,
        num_kept: int,
    ) -> None:
        self._last[request_id] = _Step(
            start, num_scheduled, verified, num_sampled, num_kept
        )

    def forget(self, request_id: str) -> None:
        self._last.pop(request_id, None)

    def resolve(self, request: "Request") -> ExactState | None:
        """Where the finished request's exact state is, or None when it is
        not known to be exact (counted in `declined`)."""
        resolved = self._resolve(request, self._last.pop(request.request_id, None))
        if isinstance(resolved, str):
            self.declined[resolved] += 1
            return None
        return resolved

    def _resolve(self, request: "Request", step: _Step | None) -> ExactState | str:
        if step is None:
            return "unknown_step"
        if step.num_kept != step.num_sampled:
            # A stop inside the accepted drafts that the device did not cut.
            return "stop_inside_accepted"
        in_flight = request.num_in_flight_tokens
        if in_flight:
            params = request.sampling_params
            stop_ids_mirrored = (
                params is None
                or len(params.stop_token_ids or ()) <= MAX_MIRRORED_STOP_TOKEN_IDS
            )
            if (
                request.status
                not in (
                    RequestStatus.FINISHED_STOPPED,
                    RequestStatus.FINISHED_LENGTH_CAPPED,
                )
                or not stop_ids_mirrored
            ):
                # A stop the device does not mirror (repetition detection, or
                # more stop token ids than it holds): the step in flight may
                # run past the request's end.
                return "not_mirrored_stop"
            start = step.start + (
                step.num_sampled if step.verified else step.num_scheduled
            )
            if start + 1 != request.num_tokens:
                return "unknown_step"
            return ExactState(start + 1, start, in_flight, 0)
        if step.verified:
            num_tokens = step.start + step.num_kept
            column = step.num_kept - 1
        else:
            num_tokens = step.start + step.num_scheduled
            column = 0
        if num_tokens != request.num_tokens - 1:
            return "unknown_step"
        return ExactState(num_tokens, step.start, step.num_scheduled, column)
