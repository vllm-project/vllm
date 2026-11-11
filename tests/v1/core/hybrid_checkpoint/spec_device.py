# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A stand-in for the GPU under speculative decoding, for CPU tests of hybrid
state checkpoints on the real scheduler.

`Device` decides each row's sampled tokens (a verify row accepts a chosen
number of drafts and, with truncation, stops at the request's stop exactly as
`speculative_config.truncate_acceptance_at_stop` does) and replays the
align-mode Mamba state handling of the worker in its order: state gathers,
zeroing, copy-on-write copies, the pre-copy across block boundaries, the
forward's reads and writes (one column per draft position), and the boundary
copy after sampling.

Every Mamba block carries the token position of the recurrent state it holds
(`rec`) and the position its convolution window's offset 0 stands for
(`conv`, valid for offsets `0..span`). Positions stand for state contents:
two states are equal exactly when their positions are. A read of a wrong state
and a cached state whose position differs from the one it is cached under are
violations.

`Engine` drives a scheduler with a `Device`; under asynchronous scheduling
it applies each step's output one step late, as the engine core does.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from vllm.model_executor.layers.mamba.checkpoint import (
    compute_mamba_prefill_checkpoints,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.hybrid_checkpoint.speculative import MAX_MIRRORED_STOP_TOKEN_IDS
from vllm.v1.core.kv_cache_utils import get_group_id, get_request_block_hasher
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request

EOS = 2
# A stop token id the target model samples only where a test asks for it.
STOP = 1000

Fault = Literal["skip_gather", "no_conv_shift"]


def token(request_id: str, position: int) -> int:
    """The token the target model samples at `position`."""
    return 3 + (sum(map(ord, request_id)) * 31 + position * 7919) % 997


@dataclass
class _Block:
    rec: int | None = None
    conv: int | None = None
    span: int = 0


@dataclass
class _Request:
    # Mamba group id -> block ids, as the worker's block table holds them.
    table: dict[int, list[int]]
    computed: int
    state_idx: int
    prefill_len: int
    tokens: list[int]
    num_accepted: int = 1


@dataclass
class Device:
    scheduler: Scheduler
    num_drafts: int
    truncate: bool
    # Drafts the target accepts in a verify step of request ``id`` at ``start``.
    accept: Callable[[str, int], int]
    fault: Fault | None = None
    eos_at: dict[str, int] = field(default_factory=dict)
    # Request -> the token sampled at `eos_at`, when it is not EOS.
    stop_token: dict[str, int] = field(default_factory=dict)
    blocks: dict[int, _Block] = field(default_factory=dict)
    requests: dict[str, _Request] = field(default_factory=dict)
    violations: list[dict] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        managers = self.scheduler.kv_cache_manager.coordinator.single_type_managers
        mamba = [m for m in managers if isinstance(m, MambaManager)]
        self.mamba_groups = [m.kv_cache_group_id for m in mamba]
        self.block_size = mamba[0].block_size
        self.hash_block_size = self.scheduler.hash_block_size

    def _block(self, block_id: int) -> _Block:
        return self.blocks.setdefault(block_id, _Block())

    def _violate(self, kind: str, **info) -> None:
        self.counts[f"violation_{kind}"] += 1
        self.violations.append(dict(kind=kind, **info))

    def _copy(self, src: int, dst: int, bias: int, rec_src: int) -> None:
        """The align copy: the recurrent state of `rec_src`, `src`'s
        window shifted `bias` tokens."""
        window = self._block(src)
        conv = None
        if window.conv is not None and bias <= window.span:
            conv = window.conv + bias
        self.blocks[dst] = _Block(
            rec=self._block(rec_src).rec, conv=conv, span=max(0, window.span - bias)
        )

    def admit(self, output: SchedulerOutput) -> None:
        for request_id in output.finished_req_ids:
            self.requests.pop(request_id, None)
        for request_id in output.preempted_req_ids or ():
            self.requests.pop(request_id, None)
        for new in output.scheduled_new_reqs:
            assert new.prefill_token_ids is not None
            tokens = list(new.prefill_token_ids)
            self.requests[new.req_id] = _Request(
                table={g: list(new.block_ids[g]) for g in self.mamba_groups},
                computed=new.num_computed_tokens,
                state_idx=(new.num_computed_tokens - 1) // self.block_size,
                prefill_len=len(tokens),
                tokens=tokens,
            )
        cached = output.scheduled_cached_reqs
        for i, request_id in enumerate(cached.req_ids):
            block_ids = cached.new_block_ids[i]
            if block_ids is not None:
                for g in self.mamba_groups:
                    self.requests[request_id].table[g].extend(block_ids[g])

    def _stop(self, request: Request, start: int, sampled: list[int]) -> int:
        """Sampled tokens kept by `truncate_acceptance_at_stop`."""
        params = request.sampling_params
        assert params is not None and params.max_tokens is not None
        max_len = min(
            request.num_prompt_tokens + params.max_tokens,
            self.scheduler.max_model_len,
        )
        # The device holds a bounded number of stop token ids; past that it
        # mirrors none of them.
        stop_ids = params.stop_token_ids or []
        stops = set(stop_ids) if len(stop_ids) <= MAX_MIRRORED_STOP_TOKEN_IDS else set()
        if params.eos_token_id is not None:
            stops.add(params.eos_token_id)
        first_input = self.requests[request.request_id].tokens[start]
        if start + 1 >= max_len or (
            start >= request.num_prompt_tokens and first_input in stops
        ):
            return min(len(sampled), 1)
        for i, sampled_token in enumerate(sampled):
            if sampled_token in stops or start + 2 + i >= max_len:
                return i + 1
        return len(sampled)

    def sample(self, output: SchedulerOutput) -> dict[str, list[int]]:
        """Each scheduled request's sampled tokens (none for a non-final
        prefill chunk)."""
        sampled: dict[str, list[int]] = {}
        for request_id, num_tokens in output.num_scheduled_tokens.items():
            request = self.scheduler.requests[request_id]
            state = self.requests[request_id]
            start = state.computed
            drafts = output.scheduled_spec_decode_tokens.get(request_id) or []
            if not drafts and start + num_tokens < state.prefill_len:
                sampled[request_id] = []
                continue
            num_sampled = 1
            if drafts:
                # A step that starts inside the prefill verifies placeholders.
                accepted = 0
                if start >= state.prefill_len:
                    accepted = min(self.accept(request_id, start), len(drafts))
                num_sampled = accepted + 1
            tokens = []
            for i in range(num_sampled):
                position = start + (1 + i if drafts else num_tokens)
                eos = self.eos_at.get(request_id) == position
                stop = self.stop_token.get(request_id, EOS)
                tokens.append(stop if eos else token(request_id, position))
            if self.truncate and drafts:
                tokens = tokens[: self._stop(request, start, tokens)]
            sampled[request_id] = tokens
        return sampled

    def apply(self, step: int, output: SchedulerOutput, sampled) -> None:
        block_size = self.block_size
        # 1. State gathers, before anything else.
        for gather in output.mamba_state_gathers or ():
            self.counts["gathers"] += 1
            if self.fault == "skip_gather":
                continue
            for dst, src in zip(gather.dst_block_ids, gather.src_block_ids):
                if self.fault == "no_conv_shift":
                    window = self._block(dst)
                    self.blocks[dst] = _Block(
                        rec=self._block(src).rec, conv=window.conv, span=window.span
                    )
                else:
                    self._copy(dst, dst, gather.column, rec_src=src)
        # 2. Zeroing; 3. copy-on-write copies.
        for block_id in output.new_block_ids_to_zero or ():
            self.blocks[block_id] = _Block()
        for copy in output.kv_cache_block_copies or ():
            src = self._block(copy.src_block_id)
            self.blocks[copy.dst_block_id] = _Block(src.rec, src.conv, src.span)
        # 4. Pre-copy, 5. forward, 6. after sampling; per request.
        for request_id, num_tokens in output.num_scheduled_tokens.items():
            state = self.requests[request_id]
            start = state.computed
            drafts = output.scheduled_spec_decode_tokens.get(request_id) or []
            first = (start + num_tokens - 1) // block_size
            for g in self.mamba_groups:
                table = state.table[g]
                if 0 <= state.state_idx != first:
                    bias = state.num_accepted - 1
                    if max(state.state_idx + bias, first) >= len(table):
                        self._violate("precopy_missing", step=step, req=request_id)
                        continue
                    self._copy(
                        table[state.state_idx],
                        table[first],
                        bias,
                        rec_src=table[state.state_idx + bias],
                    )
            if state.state_idx != first:
                state.num_accepted = 1
            state.state_idx = first
            column = state.num_accepted - 1 if drafts else 0
            if start > 0:
                for g in self.mamba_groups:
                    table = state.table[g]
                    rec = self._block(table[first + column]).rec
                    window = self._block(table[first])
                    conv = None
                    if window.conv is not None and column <= window.span:
                        conv = window.conv + column
                    if rec != start or conv != start:
                        self._violate(
                            "read", step=step, req=request_id, at=start, rec=rec
                        )
                self.counts["reads"] += 1
            tokens = sampled[request_id]
            if tokens:
                state.tokens.extend(tokens)
            if drafts:
                accepted = max(len(tokens), 1)
                self._verify(state, start, num_tokens, first, accepted)
                state.computed = start + accepted
            else:
                self._prefill(state, start, num_tokens, first)

    def _verify(
        self, state: _Request, start: int, num_tokens: int, first: int, accepted: int
    ) -> None:
        """One column per position, then the boundary copy of upstream's
        postprocess when the accepted tokens reached a block boundary."""
        block_size = self.block_size
        for g in self.mamba_groups:
            table = state.table[g]
            for t in range(num_tokens):
                if first + t < len(table):
                    self.blocks[table[first + t]] = _Block(
                        rec=start + t + 1,
                        conv=None if t else start + 1,
                        span=0 if t else num_tokens - 1,
                    )
        boundary = (start + accepted) // block_size * block_size
        if boundary >= start + 1:
            bias = boundary - start - 1
            dest = boundary // block_size - 1
            for g in self.mamba_groups:
                table = state.table[g]
                self._copy(table[first], table[dest], bias, rec_src=table[first + bias])
            if dest == first:
                accepted = 1
        state.num_accepted = accepted

    def _prefill(
        self, state: _Request, start: int, num_tokens: int, first: int
    ) -> None:
        end = start + num_tokens
        for g in self.mamba_groups:
            self.blocks[state.table[g][first]] = _Block(end, end, 0)
        offsets, columns = compute_mamba_prefill_checkpoints(
            [end],
            [num_tokens],
            hash_block_size=self.hash_block_size,
            mamba_block_size=self.block_size,
            checkpoint_alignment=16,
            drop_eagle_block=False,
            cache_hit_alignment_tokens=self.hash_block_size,
        )
        if offsets[0] > 0 and columns[0] >= 0:
            position = start + offsets[0]
            for g in self.mamba_groups:
                table = state.table[g]
                if columns[0] < len(table):
                    self.blocks[table[columns[0]]] = _Block(position, position, 0)
        state.num_accepted = 1
        state.computed = end

    def check_cached(self, step: int) -> None:
        """Every cached Mamba state holds the position it is cached under."""
        pool = self.scheduler.kv_cache_manager.block_pool
        for block in pool.blocks:
            if block.is_null or block.block_hash is None:
                continue
            keys = {
                block.block_hash,
                *pool.cached_block_hashes_by_block.get(block.block_id, ()),
            }
            if not any(get_group_id(key) in self.mamba_groups for key in keys):
                continue
            claimed = block.block_hash_num_tokens
            if claimed is None:
                continue
            self.counts["cached_checks"] += 1
            state = self._block(block.block_id)
            if state.rec != claimed or state.conv != claimed:
                self._violate(
                    "cached",
                    step=step,
                    block=block.block_id,
                    claimed=claimed,
                    rec=state.rec,
                    conv=state.conv,
                )


class Engine:
    """Steps one scheduler with a `Device`."""

    def __init__(self, scheduler: Scheduler, device: Device, async_scheduling: bool):
        self.scheduler = scheduler
        self.device = device
        self.async_scheduling = async_scheduling
        self.hasher = get_request_block_hasher(scheduler.hash_block_size, sha256)
        self.requests: dict[str, Request] = {}
        # Request -> the computed tokens of each admission.
        self.admissions: dict[str, list[int]] = {}
        self._pending: tuple[SchedulerOutput, ModelRunnerOutput] | None = None
        self.step_no = 0

    def add(
        self,
        request_id: str,
        tokens: list[int],
        max_tokens: int,
        eos_at: int | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> Request:
        """`eos_at`: the position at which the model samples EOS, or the
        first of `stop_token_ids` when given, which then stops the request;
        otherwise it runs to `max_tokens`."""
        params = SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=eos_at is None or stop_token_ids is not None,
            stop_token_ids=stop_token_ids,
        )
        params.update_from_generation_config({}, EOS)
        request = Request(
            request_id=request_id,
            prompt_token_ids=list(tokens),
            sampling_params=params,
            pooling_params=None,
            block_hasher=self.hasher,
        )
        self.requests[request_id] = request
        if eos_at is not None:
            self.device.eos_at[request_id] = eos_at
            if stop_token_ids:
                self.device.stop_token[request_id] = stop_token_ids[0]
        self.scheduler.add_request(request)
        return request

    def step(self) -> bool:
        output = self.scheduler.schedule()
        for new in output.scheduled_new_reqs:
            self.admissions.setdefault(new.req_id, []).append(new.num_computed_tokens)
        self.device.admit(output)
        sampled = self.device.sample(output)
        self.device.apply(self.step_no, output, sampled)
        self.device.check_cached(self.step_no)
        req_ids = list(output.num_scheduled_tokens)
        model_output = ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=[sampled[rid] for rid in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        if self.async_scheduling:
            previous, self._pending = self._pending, (output, model_output)
            if previous is not None:
                self.scheduler.update_from_output(*previous)
        else:
            self.scheduler.update_from_output(output, model_output)
            proposing = [
                rid
                for rid in req_ids
                if sampled[rid] and rid in self.scheduler.requests
            ]
            drafts = [
                [token(rid, 1_000_000 + i) for i in range(self.device.num_drafts)]
                for rid in proposing
            ]
            self.scheduler.update_draft_token_ids(DraftTokenIds(proposing, drafts))
        self.step_no += 1
        return bool(output.num_scheduled_tokens)

    def flush(self) -> None:
        if self._pending is not None:
            previous, self._pending = self._pending, None
            self.scheduler.update_from_output(*previous)

    def done(self, request_id: str) -> bool:
        return (
            self.requests[request_id].is_finished()
            and request_id not in self.scheduler.requests
        )

    def run(self, *request_ids: str, max_steps: int = 5000) -> None:
        idle = 0
        for _ in range(max_steps):
            if all(self.done(rid) for rid in request_ids):
                self.quiesce()
                return
            idle = 0 if self.step() else idle + 1
            if idle > 2:
                self.flush()
        raise AssertionError(f"requests did not finish: {request_ids}")

    def quiesce(self) -> None:
        """Run empty steps so queued gathers and copies are applied and the
        cached states are checked."""
        self.flush()
        for _ in range(2):
            self.step()
            self.flush()
