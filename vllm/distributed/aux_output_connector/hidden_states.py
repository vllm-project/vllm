# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-local, position-based capture of model output hidden states."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class HiddenStateCapturePlan:
    request_id: str
    prompt_len: int
    window_start_abs: int
    window_end_abs: int
    collection_id: str
    min_rows: int = 1
    aux_layer_ids: tuple[int, ...] = ()
    hidden_layout: str = "final"
    logprob_topk: int = 0

    def __post_init__(self) -> None:
        if (
            self.prompt_len < 1
            or self.window_start_abs < self.prompt_len - 1
            or self.window_start_abs >= self.window_end_abs
            or self.min_rows < 1
            or self.min_rows > self.window_end_abs - self.window_start_abs
        ):
            raise ValueError("Invalid hidden-state capture plan")
        if self.hidden_layout not in ("final", "aux_final", "dflash_aux"):
            raise ValueError("Invalid hidden-state layout")
        if bool(self.aux_layer_ids) != (self.hidden_layout != "final"):
            raise ValueError("Hidden-state layout does not match auxiliary layers")
        if len(set(self.aux_layer_ids)) != len(self.aux_layer_ids):
            raise ValueError("Duplicate auxiliary hidden-state layers")
        if not isinstance(self.logprob_topk, int) or self.logprob_topk < 0:
            raise ValueError("logprob_topk must be a nonnegative integer")

    @classmethod
    def from_window(
        cls,
        request_id: str,
        prompt_len: int,
        start: int,
        end: int,
        collection_id: str,
        *,
        coordinate: Literal["response", "absolute"] = "response",
        min_rows: int = 1,
        aux_layer_ids: tuple[int, ...] = (),
        hidden_layout: str | None = None,
        logprob_topk: int = 0,
    ) -> HiddenStateCapturePlan:
        if coordinate == "response":
            offset = prompt_len - 1
        elif coordinate == "absolute":
            offset = 0
        else:
            raise ValueError(f"Unknown hidden-state coordinate: {coordinate}")
        return cls(
            request_id=request_id,
            prompt_len=prompt_len,
            window_start_abs=start + offset,
            window_end_abs=end + offset,
            collection_id=collection_id,
            min_rows=min_rows,
            aux_layer_ids=aux_layer_ids,
            hidden_layout=hidden_layout or ("aux_final" if aux_layer_ids else "final"),
            logprob_topk=logprob_topk,
        )


@dataclass
class HiddenStateCaptureChunk:
    positions: np.ndarray
    hidden_states: torch.Tensor
    copied_bytes: int = 0
    copy_ms: float = 0.0
    logprobs: torch.Tensor | None = None
    logprob_token_ids: torch.Tensor | None = None

    def accepted(self, start: int, end: int) -> HiddenStateCaptureChunk:
        keep = (self.positions >= start) & (self.positions < end)
        return HiddenStateCaptureChunk(
            self.positions[keep],
            self.hidden_states[torch.from_numpy(keep)],
            self.copied_bytes,
            self.copy_ms,
            self.logprobs[torch.from_numpy(keep)]
            if self.logprobs is not None
            else None,
            self.logprob_token_ids[torch.from_numpy(keep)]
            if self.logprob_token_ids is not None
            else None,
        )


def drop_incompatible_aux_plans(
    plans: dict[str, HiddenStateCapturePlan],
    req_ids: list[str],
    available_layer_ids: tuple[int, ...],
    aux_hidden_states: list[torch.Tensor] | None,
) -> dict[str, str]:
    """Disable captures whose requested auxiliary layout cannot be produced."""
    errors: dict[str, str] = {}
    for req_id in req_ids:
        plan = plans.get(req_id)
        if plan is None or not plan.aux_layer_ids:
            continue
        if (
            aux_hidden_states is None
            or plan.aux_layer_ids != available_layer_ids
            or len(aux_hidden_states) != len(available_layer_ids)
        ):
            plans.pop(req_id)
            errors[req_id] = "aux_layers_unavailable"
    return errors


def capture_scheduled_hidden_states(
    plans: dict[str, HiddenStateCapturePlan],
    req_ids: list[str],
    num_scheduled_tokens: dict[str, int],
    num_computed_tokens: dict[str, int],
    hidden_states: torch.Tensor,
    aux_hidden_states: list[torch.Tensor] | None = None,
    query_start_loc: np.ndarray | None = None,
    *,
    logits: torch.Tensor | None = None,
    logits_indices: torch.Tensor | None = None,
    accepted_ends: dict[str, int] | None = None,
) -> dict[str, HiddenStateCaptureChunk]:
    """Snapshot only window rows before reusable model buffers are overwritten."""
    chunks: dict[str, HiddenStateCaptureChunk] = {}
    row_offset = 0
    for index, req_id in enumerate(req_ids):
        num_rows = num_scheduled_tokens[req_id]
        if query_start_loc is not None:
            row_offset = int(query_start_loc[index])
            if int(query_start_loc[index + 1]) - row_offset != num_rows:
                raise ValueError("Scheduled hidden-state row boundaries do not match")
        plan = plans.get(req_id)
        if plan is not None:
            start = num_computed_tokens[req_id]
            first = max(
                start, plan.window_start_abs, (accepted_ends or {}).get(req_id, 0)
            )
            end = min(start + num_rows, plan.window_end_abs)
            if first < end:
                first_row = row_offset + first - start
                end_row = row_offset + end - start
                if plan.aux_layer_ids:
                    if aux_hidden_states is None or len(aux_hidden_states) != len(
                        plan.aux_layer_ids
                    ):
                        raise ValueError(
                            "Requested auxiliary hidden layers are unavailable"
                        )
                    parts = [state[first_row:end_row] for state in aux_hidden_states]
                    if plan.hidden_layout != "dflash_aux":
                        parts.append(hidden_states[first_row:end_row])
                    states = torch.cat(parts, dim=-1)
                else:
                    states = hidden_states[first_row:end_row]
                values = token_ids = None
                if plan.logprob_topk:
                    if logits is None or logits_indices is None:
                        raise ValueError(
                            "Window logprobs require raw logits and indices"
                        )
                    rows = torch.arange(first_row, end_row, device=logits.device)
                    indices = torch.searchsorted(logits_indices.contiguous(), rows)
                    indices = indices.clamp(max=logits_indices.numel() - 1)
                    torch._assert_async(
                        (logits_indices[indices] == rows).all(),
                        "Captured hidden rows have no corresponding logits",
                    )
                    scores = logits[indices].float()
                    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

                    _, token_ids = scores.topk(plan.logprob_topk, dim=-1)
                    values = compute_token_logprobs(scores, token_ids)
                # Graph outputs are reused by the next forward; own selected rows.
                tensors = [states.detach().clone()]
                if values is not None:
                    tensors.extend([values, token_ids])
                chunks[req_id] = HiddenStateCaptureChunk(
                    positions=np.arange(first, end, dtype=np.int64),
                    hidden_states=tensors[0],
                    logprobs=tensors[1] if values is not None else None,
                    logprob_token_ids=tensors[2] if values is not None else None,
                    copied_bytes=sum(t.numel() * t.element_size() for t in tensors),
                )
        row_offset += num_rows
    return chunks


@dataclass
class HiddenStateCaptureResult:
    hidden_positions: np.ndarray
    hidden_states: torch.Tensor
    hidden_position_start: int
    hidden_position_end: int
    hidden_window_start: int
    hidden_window_end: int
    hidden_layout: str
    layer_ids: tuple[int, ...]
    includes_final_layer: bool
    request_id: str
    collection_id: str
    copied_bytes: int = 0
    copy_ms: float = 0.0
    logprobs: torch.Tensor | None = None
    logprob_token_ids: torch.Tensor | None = None


@dataclass
class HiddenStateCaptureBuffer:
    plan: HiddenStateCapturePlan
    chunks: list[HiddenStateCaptureChunk] = field(default_factory=list)
    copied_bytes: int = 0
    copy_ms: float = 0.0
    error: str | None = None

    def add(self, chunk: HiddenStateCaptureChunk) -> None:
        self.copied_bytes += chunk.copied_bytes
        self.copy_ms += chunk.copy_ms
        if chunk.positions.size:
            self.chunks.append(chunk)

    def finish(self) -> HiddenStateCaptureResult | None:
        if not self.chunks:
            return None
        positions = np.concatenate([chunk.positions for chunk in self.chunks])
        states = torch.cat([chunk.hidden_states for chunk in self.chunks])
        positions, unique_indices = np.unique(positions, return_index=True)
        states = states[torch.from_numpy(unique_indices)]
        logprobs = token_ids = None
        if self.plan.logprob_topk:
            if any(
                chunk.logprobs is None or chunk.logprob_token_ids is None
                for chunk in self.chunks
            ):
                raise ValueError("Captured window has missing logprob rows")
            logprobs = torch.cat([c.logprobs for c in self.chunks])[
                torch.from_numpy(unique_indices)
            ]
            token_ids = torch.cat([c.logprob_token_ids for c in self.chunks])[
                torch.from_numpy(unique_indices)
            ]
        if len(positions) < self.plan.min_rows:
            return None
        return HiddenStateCaptureResult(
            hidden_positions=positions,
            hidden_states=states,
            hidden_position_start=int(positions.min()),
            hidden_position_end=int(positions.max()) + 1,
            hidden_window_start=self.plan.window_start_abs,
            hidden_window_end=self.plan.window_end_abs,
            hidden_layout=self.plan.hidden_layout,
            layer_ids=self.plan.aux_layer_ids,
            includes_final_layer=self.plan.hidden_layout != "dflash_aux",
            request_id=self.plan.request_id,
            collection_id=self.plan.collection_id,
            copied_bytes=self.copied_bytes,
            copy_ms=self.copy_ms,
            logprobs=logprobs,
            logprob_token_ids=token_ids,
        )

    def update(
        self,
        chunk: HiddenStateCaptureChunk | None,
        accepted_start: int,
        accepted_end: int,
        *,
        stopped: bool,
        error: str | None = None,
    ) -> tuple[HiddenStateCaptureResult | None, str | None]:
        """Settle returned rows using accepted output progress."""
        if error is not None:
            self.error = error
        if self.error is not None:
            return None, self.error
        if chunk is not None:
            self.add(chunk.accepted(accepted_start, accepted_end))
        if not stopped and accepted_end < self.plan.window_end_abs:
            return None, None
        result = self.finish()
        if result is not None:
            return result, None
        reason = (
            "ended_before_window"
            if stopped and accepted_end <= self.plan.window_start_abs
            else "insufficient_rows"
        )
        return None, reason


CAPTURE_KEY = "hidden_state_capture"


def capture_plan(params, request_id: str, prompt_len: int):
    spec = (params.extra_args or {}).get(CAPTURE_KEY)
    if spec is None:
        return None
    spec = dict(spec)
    spec["request_id"] = request_id
    spec["prompt_len"] = prompt_len
    spec["aux_layer_ids"] = tuple(spec.get("aux_layer_ids", ()))
    return HiddenStateCapturePlan(**spec)


def validate_capture_request(
    params, prompt_len: int, config: VllmConfig, resumable: bool = False
) -> None:
    from vllm.exceptions import VLLMValidationError

    plan = capture_plan(params, "", prompt_len)
    if plan is None:
        return
    if not config.aux_output_config.enable_hidden_states:
        raise VLLMValidationError(
            "Hidden capture requires aux_output_config.enable_hidden_states"
        )
    if params.n != 1 or params.stop or resumable:
        raise VLLMValidationError(
            "Hidden capture requires n=1, no stop strings or streaming input"
        )
    if plan.logprob_topk > config.model_config.get_vocab_size():
        raise VLLMValidationError("logprob_topk exceeds the model vocabulary")


@dataclass
class PendingHiddenOutput:
    chunks: dict[str, HiddenStateCaptureChunk]
    errors: dict[str, str] = field(default_factory=dict)
    gpu_tensors: list[torch.Tensor] = field(default_factory=list)

    def enqueue_cpu_copy(self) -> None:
        for chunk in self.chunks.values():
            started = perf_counter()
            for name in ("hidden_states", "logprobs", "logprob_token_ids"):
                tensor = getattr(chunk, name)
                if tensor is not None:
                    self.gpu_tensors.append(tensor)
                    cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
                    cpu.copy_(tensor, non_blocking=True)
                    setattr(chunk, name, cpu)
            chunk.copy_ms = (perf_counter() - started) * 1000


class HiddenStateWorker:
    def __init__(self, model, config):
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            get_eagle3_aux_layers_from_config,
            requires_aux_hidden_states,
        )

        self.plans: dict[str, HiddenStateCapturePlan] = {}
        self.accepted_ends: dict[str, int] = {}
        self.pending: PendingHiddenOutput | None = None
        layers: tuple[int, ...] = ()
        if config.speculative_config is not None and requires_aux_hidden_states(
            config.speculative_config
        ):
            layers = (
                get_eagle3_aux_layers_from_config(config.speculative_config)
                or model.get_eagle3_default_aux_hidden_state_layers()
            )
        self.layer_ids: tuple[int, ...] = tuple(sorted(set(layers)))

    def begin_step(self, metadata) -> None:
        for request_id in (
            metadata.finished_requests + metadata.finished_hidden_requests
        ):
            self.plans.pop(request_id, None)
            self.accepted_ends.pop(request_id, None)
        self.plans.update(metadata.hidden_plans)
        self.accepted_ends.update(metadata.hidden_accepted_ends)

    def prepare(self, batch, hidden_states, aux_hidden_states, logits) -> None:
        if not self.plans:
            self.pending = None
            return
        errors = drop_incompatible_aux_plans(
            self.plans, batch.req_ids, self.layer_ids, aux_hidden_states
        )
        counts = {
            rid: int(batch.query_start_loc_np[i + 1] - batch.query_start_loc_np[i])
            for i, rid in enumerate(batch.req_ids)
        }
        computed = {
            rid: int(batch.num_computed_tokens_np[i])
            for i, rid in enumerate(batch.req_ids)
        }
        chunks = capture_scheduled_hidden_states(
            self.plans,
            batch.req_ids,
            counts,
            computed,
            hidden_states,
            aux_hidden_states,
            batch.query_start_loc_np,
            logits=logits,
            logits_indices=batch.logits_indices,
            accepted_ends=self.accepted_ends,
        )
        self.pending = PendingHiddenOutput(chunks) if chunks or errors else None
        if self.pending is not None:
            self.pending.errors = errors


class HiddenStateScheduler:
    def __init__(self):
        self.buffers: dict[str, HiddenStateCaptureBuffer] = {}
        self.done: set[str] = set()
        self.finished: set[str] = set()

    def build_metadata(self, scheduled, requests):
        plans = {}
        for request_id in scheduled:
            if request_id in self.done:
                continue
            request = requests[request_id]
            params = request.sampling_params
            if params is None or not (params.extra_args or {}).get(CAPTURE_KEY):
                continue
            if request_id not in self.buffers:
                plan = capture_plan(params, request_id, request.num_prompt_tokens)
                self.buffers[request_id] = HiddenStateCaptureBuffer(plan)
            plans[request_id] = self.buffers[request_id].plan
        finished, self.finished = tuple(self.finished), set()
        return plans, finished

    def take_output(self, request, output, num_new_tokens: int, stopped: bool):
        request_id = request.request_id
        buffer = self.buffers.get(request_id)
        if buffer is None:
            return None, None
        accepted_end = request.num_prompt_tokens - 1 + request.num_output_tokens
        chunk = None if output is None else output.hidden_chunk
        error = None if output is None else output.hidden_error
        result, reason = buffer.update(
            chunk,
            accepted_end - num_new_tokens,
            accepted_end,
            stopped=stopped,
            error=error,
        )
        if result is not None or reason is not None:
            self.done.add(request_id)
            self.finished.add(request_id)
            if num_new_tokens == 0 and not stopped:
                return None, None
            self.buffers.pop(request_id)
        return result, reason

    def request_finished(self, request):
        self.finished.add(request.request_id)
        if request.is_finished():
            self.buffers.pop(request.request_id, None)
            self.done.discard(request.request_id)


def verify_hidden_config(config: VllmConfig) -> None:
    if not config.use_v2_model_runner or config.device_config.device_type != "cuda":
        raise ValueError("Windowed hidden capture requires the CUDA Model Runner V2")
    if config.scheduler_config.async_scheduling:
        raise ValueError("Windowed hidden capture requires async_scheduling=False")
    parallel = config.parallel_config
    if (
        parallel.pipeline_parallel_size > 1
        or parallel.decode_context_parallel_size > 1
        or parallel.prefill_context_parallel_size > 1
        or parallel.enable_batch_sharded_sampling
    ):
        raise ValueError(
            "Windowed hidden capture does not support PP, CP or batch-sharded sampling"
        )
    model = config.model_config
    if model.is_encoder_decoder or model.is_diffusion:
        raise ValueError("Windowed hidden capture requires a causal decoder model")
    if (
        config.speculative_config
        and config.speculative_config.enable_adaptive_verification
    ):
        raise ValueError(
            "Windowed hidden capture does not support adaptive verification"
        )
