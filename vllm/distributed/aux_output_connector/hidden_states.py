# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-local, position-based capture of model output hidden states.

A capture plan names one absolute row range ``[start, end)``; row ``p`` is the
hidden state that predicts token ``p + 1``. Rows before ``prompt_len - 1`` form
the ``prompt`` segment (teacher scoring) and settle as soon as they are
computed. Later rows form the ``output`` segment (generation windows) and
settle with accepted tokens. Rows outside the range are never copied.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath, PureWindowsPath
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.aux_output_connector.connector import (
        AuxOutputConnectorMetadata,
        AuxRequestOutput,
    )
    from vllm.v1.request import Request
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_states.interface import ModelState

HIDDEN_STATES = "hidden_states"
LOGPROBS = "logprobs"
LOGPROB_TOKEN_IDS = "logprob_token_ids"


class InconsistentTensorFieldsError(ValueError):
    """Capture chunks disagree on their tensor fields."""


@dataclass(frozen=True)
class CaptureSegment:
    kind: Literal["prompt", "output"]
    start_abs: int
    end_abs: int


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
        for name in (
            "prompt_len",
            "window_start_abs",
            "window_end_abs",
            "min_rows",
            "logprob_topk",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if any(type(layer) is not int or layer < 0 for layer in self.aux_layer_ids):
            raise ValueError("aux_layer_ids must contain nonnegative integers")
        if (
            self.prompt_len < 1
            or self.window_start_abs < 0
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
        if self.logprob_topk < 0:
            raise ValueError("logprob_topk must be a nonnegative integer")
        if self.logprob_topk and self.window_start_abs < self.output_start:
            raise ValueError(
                "logprob_topk only supports output rows; use "
                "prompt_logprob_token_ids for prompt rows"
            )

    @property
    def output_start(self) -> int:
        return self.prompt_len - 1

    @property
    def segments(self) -> tuple[CaptureSegment, ...]:
        start, end = self.window_start_abs, self.window_end_abs
        split = self.output_start
        segments = []
        if start < split:
            segments.append(CaptureSegment("prompt", start, min(end, split)))
        if end > split:
            segments.append(CaptureSegment("output", max(start, split), end))
        return tuple(segments)

    @property
    def includes_final_layer(self) -> bool:
        return self.hidden_layout != "dflash_aux"

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
    tensors: dict[str, torch.Tensor]
    copied_bytes: int = 0
    copy_ms: float = 0.0
    logits_contract: dict[str, Any] | None = None

    @property
    def hidden_states(self) -> torch.Tensor:
        return self.tensors[HIDDEN_STATES]

    @property
    def logprobs(self) -> torch.Tensor | None:
        return self.tensors.get(LOGPROBS)

    @property
    def logprob_token_ids(self) -> torch.Tensor | None:
        return self.tensors.get(LOGPROB_TOKEN_IDS)

    def accepted(self, end: int) -> HiddenStateCaptureChunk:
        """Keep rows whose predicted token has been delivered."""
        keep = self.positions < end
        if keep.all():
            return self
        index = torch.from_numpy(keep)
        return replace(
            self,
            positions=self.positions[keep],
            tensors={name: value[index] for name, value in self.tensors.items()},
        )

    @classmethod
    def concat(cls, chunks: list[HiddenStateCaptureChunk]) -> HiddenStateCaptureChunk:
        if len(chunks) == 1:
            return chunks[0]
        names = set(chunks[0].tensors)
        if any(set(chunk.tensors) != names for chunk in chunks[1:]):
            raise InconsistentTensorFieldsError("tensor_fields_mismatch")
        return cls(
            positions=np.concatenate([chunk.positions for chunk in chunks]),
            tensors={
                name: torch.cat([chunk.tensors[name] for chunk in chunks])
                for name in names
            },
            copied_bytes=sum(chunk.copied_bytes for chunk in chunks),
            copy_ms=sum(chunk.copy_ms for chunk in chunks),
            logits_contract=chunks[-1].logits_contract,
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
    cu_num_logits: np.ndarray | None = None,
    settled_ends: dict[str, int] | None = None,
    errors: dict[str, str] | None = None,
) -> dict[str, HiddenStateCaptureChunk]:
    """Snapshot only window rows before reusable model buffers are overwritten.

    Rows below ``settled_ends`` were already delivered and are skipped when a
    preempted request recomputes its history.
    """
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
                start, plan.window_start_abs, (settled_ends or {}).get(req_id, 0)
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
                    if plan.includes_final_layer:
                        parts.append(hidden_states[first_row:end_row])
                    states = torch.cat(parts, dim=-1)
                else:
                    states = hidden_states[first_row:end_row]
                # Graph outputs are reused by the next forward; own selected rows.
                tensors = {HIDDEN_STATES: states.detach().clone()}
                if plan.logprob_topk:
                    if logits is None or cu_num_logits is None:
                        if errors is not None:
                            errors[req_id] = "logits_unavailable"
                        row_offset += num_rows
                        continue
                    logit_start, logit_end = map(int, cu_num_logits[index : index + 2])
                    # MRV2 places each request's logits on its query suffix.
                    query_logit_start = (
                        row_offset + num_rows - (logit_end - logit_start)
                    )
                    if first_row < query_logit_start or logit_end > logits.shape[0]:
                        if errors is not None:
                            errors[req_id] = "logits_rows_unavailable"
                        row_offset += num_rows
                        continue
                    selected_start = logit_start + first_row - query_logit_start
                    scores = logits[
                        selected_start : selected_start + end_row - first_row
                    ].float()
                    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

                    _, token_ids = scores.topk(plan.logprob_topk, dim=-1)
                    tensors[LOGPROBS] = compute_token_logprobs(scores, token_ids)
                    tensors[LOGPROB_TOKEN_IDS] = token_ids
                chunks[req_id] = HiddenStateCaptureChunk(
                    positions=np.arange(first, end, dtype=np.int64),
                    tensors=tensors,
                    copied_bytes=sum(
                        t.numel() * t.element_size() for t in tensors.values()
                    ),
                )
        row_offset += num_rows
    return chunks


@dataclass
class HiddenStateCaptureResult:
    hidden_positions: np.ndarray
    tensors: dict[str, torch.Tensor]
    hidden_position_start: int
    hidden_position_end: int
    hidden_window_start: int
    hidden_window_end: int
    hidden_layout: str
    layer_ids: tuple[int, ...]
    includes_final_layer: bool
    collection_id: str
    segments: tuple[CaptureSegment, ...] = ()
    copied_bytes: int = 0
    copy_ms: float = 0.0
    # LogitsProcessor parameters for rebuilding logits from the final layer:
    # logits = soft_cap(h @ W_head.T)[:, :vocab_size] * logit_scale.
    logits_contract: dict[str, Any] | None = None

    @property
    def hidden_states(self) -> torch.Tensor:
        return self.tensors[HIDDEN_STATES]

    @property
    def logprobs(self) -> torch.Tensor | None:
        return self.tensors.get(LOGPROBS)

    @property
    def logprob_token_ids(self) -> torch.Tensor | None:
        return self.tensors.get(LOGPROB_TOKEN_IDS)


@dataclass
class HiddenStateCaptureBuffer:
    plan: HiddenStateCapturePlan
    chunks: list[HiddenStateCaptureChunk] = field(default_factory=list)
    copied_bytes: int = 0
    copy_ms: float = 0.0
    error: str | None = None
    settled_end: int = 0
    logits_contract: dict[str, Any] | None = None

    def add(self, chunk: HiddenStateCaptureChunk) -> None:
        self.copied_bytes += chunk.copied_bytes
        self.copy_ms += chunk.copy_ms
        if chunk.logits_contract is not None:
            self.logits_contract = chunk.logits_contract
        if chunk.positions.size:
            self.chunks.append(chunk)
            self.settled_end = max(self.settled_end, int(chunk.positions.max()) + 1)

    def finish(self) -> HiddenStateCaptureResult | None:
        plan = self.plan
        prompt_end = min(plan.window_end_abs, plan.output_start)
        if not self.chunks:
            if plan.window_start_abs < prompt_end:
                self.error = "prompt_rows_unavailable"
            return None
        positions = np.concatenate([chunk.positions for chunk in self.chunks])
        unique_indices = None
        if np.any(positions[1:] <= positions[:-1]):
            positions, unique_indices = np.unique(positions, return_index=True)
        if plan.window_start_abs < prompt_end and not np.array_equal(
            positions[positions < prompt_end],
            np.arange(plan.window_start_abs, prompt_end),
        ):
            self.error = "prompt_rows_unavailable"
            return None
        names = set(self.chunks[0].tensors)
        if HIDDEN_STATES not in names or any(
            set(chunk.tensors) != names for chunk in self.chunks[1:]
        ):
            self.error = "tensor_fields_mismatch"
            return None
        if self.plan.logprob_topk and not {LOGPROBS, LOGPROB_TOKEN_IDS} <= names:
            self.error = "logprob_rows_unavailable"
            return None
        if len(positions) < self.plan.min_rows:
            return None
        tensors = (
            self.chunks[0].tensors
            if len(self.chunks) == 1
            else {
                name: torch.cat([chunk.tensors[name] for chunk in self.chunks])
                for name in names
            }
        )
        if unique_indices is not None:
            index = torch.from_numpy(unique_indices)
            tensors = {name: tensor[index] for name, tensor in tensors.items()}
        return HiddenStateCaptureResult(
            hidden_positions=positions,
            tensors={name: tensors[name] for name in sorted(names)},
            hidden_position_start=int(positions[0]),
            hidden_position_end=int(positions[-1]) + 1,
            hidden_window_start=plan.window_start_abs,
            hidden_window_end=plan.window_end_abs,
            hidden_layout=plan.hidden_layout,
            layer_ids=plan.aux_layer_ids,
            includes_final_layer=plan.includes_final_layer,
            collection_id=plan.collection_id,
            segments=plan.segments,
            copied_bytes=self.copied_bytes,
            copy_ms=self.copy_ms,
            logits_contract=self.logits_contract if plan.includes_final_layer else None,
        )

    def update(
        self,
        chunk: HiddenStateCaptureChunk | None,
        accepted_end: int,
        *,
        stopped: bool,
        error: str | None = None,
    ) -> tuple[HiddenStateCaptureResult | None, str | None]:
        """Settle returned rows; prompt rows are always below accepted_end."""
        if error is not None:
            self.error = error
        if self.error is not None:
            return None, self.error
        if chunk is not None:
            self.add(chunk.accepted(accepted_end))
        if not stopped and accepted_end < self.plan.window_end_abs:
            return None, None
        result = self.finish()
        if self.error is not None:
            return None, self.error
        if result is not None:
            return result, None
        reason = (
            "ended_before_window"
            if stopped and accepted_end <= self.plan.window_start_abs
            else "insufficient_rows"
        )
        return None, reason


CAPTURE_KEY = "hidden_state_capture"


def capture_plan(
    params: SamplingParams, request_id: str, prompt_len: int
) -> HiddenStateCapturePlan | None:
    spec = (params.extra_args or {}).get(CAPTURE_KEY)
    if spec is None:
        return None
    try:
        if not isinstance(spec, Mapping):
            raise ValueError("capture specification must be a mapping")
        spec = dict(spec)
        spec["aux_layer_ids"] = tuple(spec.get("aux_layer_ids", ()))
        if "start" in spec or "end" in spec:
            return HiddenStateCapturePlan.from_window(
                request_id=request_id, prompt_len=prompt_len, **spec
            )
        # Legacy asdict(plan) callers supplied their own prompt length.
        # Preserve response offsets when binding the actual engine prompt.
        client_prompt_len = spec.pop("prompt_len")
        spec.pop("request_id", None)
        spec["window_start_abs"] += prompt_len - client_prompt_len
        spec["window_end_abs"] += prompt_len - client_prompt_len
        return HiddenStateCapturePlan(
            request_id=request_id, prompt_len=prompt_len, **spec
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise VLLMValidationError(
            f"Invalid hidden capture specification: {exc}"
        ) from exc


def prompt_capture_start(params: SamplingParams, prompt_len: int) -> int | None:
    """First prompt row a capture needs, which must not be a prefix-cache hit."""
    if CAPTURE_KEY not in (params.extra_args or {}):
        return None
    plan = capture_plan(params, "", prompt_len)
    if plan is None or plan.window_start_abs >= plan.output_start:
        return None
    return plan.window_start_abs


def validate_capture_request(
    params: SamplingParams,
    prompt_len: int,
    config: VllmConfig,
    resumable: bool = False,
) -> None:
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
    if plan.aux_layer_ids and (
        config.speculative_config is None
        or not config.speculative_config.uses_aux_hidden_states
    ):
        raise VLLMValidationError(
            "Auxiliary hidden capture requires an auxiliary-layer speculator"
        )
    if plan.logprob_topk > config.model_config.get_vocab_size():
        raise VLLMValidationError("logprob_topk exceeds the model vocabulary")
    maximum = config.model_config.max_logprobs
    if maximum != -1 and plan.logprob_topk > maximum:
        raise VLLMValidationError("logprob_topk exceeds max_logprobs")
    rows = plan.window_end_abs - plan.window_start_abs
    if (
        plan.window_start_abs < plan.output_start
        and config.cache_config.kv_sharing_fast_prefill
    ):
        raise VLLMValidationError(
            "Prompt hidden capture is incorrect with --kv-sharing-fast-prefill; "
            "disable it for scoring"
        )
    if rows > config.model_config.max_model_len:
        raise VLLMValidationError("Hidden capture window exceeds max_model_len")
    hidden_width = config.model_config.get_hidden_size() * (
        len(plan.aux_layer_ids) + plan.includes_final_layer
    )
    element_size = torch.empty((), dtype=config.model_config.dtype).element_size()
    payload_bytes = rows * (hidden_width * element_size + plan.logprob_topk * 12)
    if payload_bytes * config.scheduler_config.max_num_seqs > (
        config.aux_output_config.max_hidden_capture_bytes
    ):
        raise VLLMValidationError(
            "Hidden capture exceeds max_hidden_capture_bytes budget"
        )


class HiddenCaptureCarry:
    """Hold copied rows until a step emits output for their request.

    The scheduler settles auxiliary outputs only on steps that emit tokens, so
    rows from intermediate prefill chunks wait here for the sampling chunk.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, list[HiddenStateCaptureChunk]] = {}
        self._errors: dict[str, str] = {}
        self._lock = Lock()

    def empty(self) -> bool:
        return not self._chunks and not self._errors

    def release(
        self,
        chunks: dict[str, HiddenStateCaptureChunk],
        errors: dict[str, str],
        emitted: Iterable[str],
        logits_contract: dict[str, Any] | None = None,
    ) -> dict[str, tuple[HiddenStateCaptureChunk | None, str | None]]:
        with self._lock:
            for request_id, chunk in chunks.items():
                self._chunks.setdefault(request_id, []).append(chunk)
            for request_id, error in errors.items():
                self._errors.setdefault(request_id, error)
            released: dict[str, tuple[HiddenStateCaptureChunk | None, str | None]] = {}
            for request_id in emitted:
                parts = self._chunks.pop(request_id, None)
                release_error = self._errors.pop(request_id, None)
                if parts is None and release_error is None:
                    continue
                released_chunk = None
                if parts is not None:
                    try:
                        released_chunk = HiddenStateCaptureChunk.concat(parts)
                    except InconsistentTensorFieldsError:
                        release_error = release_error or "tensor_fields_mismatch"
                    else:
                        released_chunk.logits_contract = logits_contract
                released[request_id] = (released_chunk, release_error)
            return released

    def discard(self, request_ids: Iterable[str]) -> None:
        with self._lock:
            for request_id in request_ids:
                self._chunks.pop(request_id, None)
                self._errors.pop(request_id, None)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._errors.clear()


@dataclass
class PendingHiddenOutput:
    owner: HiddenStateWorker
    req_ids: list[str]
    chunks: dict[str, HiddenStateCaptureChunk]
    errors: dict[str, str] = field(default_factory=dict)
    gpu_tensors: list[torch.Tensor] = field(default_factory=list)
    num_sampled: np.ndarray | None = None

    def enqueue_cpu_copy(self, num_sampled: np.ndarray) -> None:
        self.num_sampled = num_sampled
        for chunk in self.chunks.values():
            started = perf_counter()
            for name, tensor in chunk.tensors.items():
                self.gpu_tensors.append(tensor)
                cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
                cpu.copy_(tensor, non_blocking=True)
                chunk.tensors[name] = cpu
            chunk.copy_ms = (perf_counter() - started) * 1000

    def process_output(
        self,
    ) -> dict[str, tuple[HiddenStateCaptureChunk | None, str | None]]:
        assert self.num_sampled is not None, "hidden capture copy was not enqueued"
        emitted = [
            request_id
            for request_id, num_sampled in zip(
                self.req_ids, self.num_sampled, strict=True
            )
            if num_sampled > 0
        ]
        return self.owner.carry.release(
            self.chunks, self.errors, emitted, self.owner.logits_contract
        )


def _supports_logits_reconstruction(
    model_type: type[torch.nn.Module],
    processor_type: type,
    compute_logits: Callable[..., Any] | None,
    *,
    quantization: str | None,
    lora_enabled: bool,
    model_state_cls: type[ModelState],
) -> bool:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    return (
        model_type is Qwen3ForCausalLM
        and processor_type is LogitsProcessor
        and compute_logits is Qwen3ForCausalLM.compute_logits
        and quantization is None
        and not lora_enabled
        and model_state_cls.compute_sample_logits is ModelState.compute_sample_logits
    )


def _public_capture_model_name(name: str | list[str] | None) -> str | None:
    if isinstance(name, list):
        name = name[0] if name else None
    if name and (
        PurePosixPath(name).is_absolute()
        or PureWindowsPath(name).is_absolute()
        or name.startswith(".")
    ):
        return None
    return name


def _logits_contract(
    model: torch.nn.Module, config: VllmConfig
) -> dict[str, Any] | None:
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    from vllm.v1.worker.gpu.model_states import resolve_model_state_cls

    if type(model) is not Qwen3ForCausalLM:
        return None
    processor = getattr(model, "logits_processor", None)
    if not _supports_logits_reconstruction(
        type(model),
        type(processor),
        getattr(model.compute_logits, "__func__", None),
        quantization=config.model_config.quantization,
        lora_enabled=config.lora_config is not None,
        model_state_cls=resolve_model_state_cls(config, model),
    ):
        return None
    assert processor is not None
    hf_config = config.model_config.hf_text_config
    tied = bool(getattr(hf_config, "tie_word_embeddings", False))
    return {
        "served_model_name": _public_capture_model_name(
            config.model_config.served_model_name
        ),
        "revision": config.model_config.revision,
        "lm_head_weight": "model.embed_tokens.weight" if tied else "lm_head.weight",
        "vocab_size": int(processor.org_vocab_size),
        "logit_scale": float(processor.scale),
        "soft_cap": None if processor.soft_cap is None else float(processor.soft_cap),
        "tie_word_embeddings": tied,
    }


class HiddenStateWorker:
    def __init__(self, model: torch.nn.Module, config: VllmConfig) -> None:
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            get_eagle3_aux_layers_from_config,
        )

        self.plans: dict[str, HiddenStateCapturePlan] = {}
        self.settled_ends: dict[str, int] = {}
        self.pending: PendingHiddenOutput | None = None
        self.carry = HiddenCaptureCarry()
        self.logits_contract = _logits_contract(model, config)
        layers: tuple[int, ...] = ()
        if (
            config.speculative_config is not None
            and config.speculative_config.uses_aux_hidden_states
        ):
            layers = (
                get_eagle3_aux_layers_from_config(config.speculative_config)
                or model.get_eagle3_default_aux_hidden_state_layers()
            )
        self.layer_ids: tuple[int, ...] = tuple(sorted(set(layers)))

    def begin_step(self, metadata: AuxOutputConnectorMetadata) -> None:
        finished = metadata.finished_requests + metadata.finished_hidden_requests
        for request_id in finished:
            self.plans.pop(request_id, None)
            self.settled_ends.pop(request_id, None)
        self.carry.discard(finished)
        self.plans.update(metadata.hidden_plans)
        self.settled_ends.update(metadata.hidden_settled_ends)

    def prepare(
        self,
        batch: InputBatch,
        hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        logits: torch.Tensor | None,
    ) -> None:
        if not self.plans and self.carry.empty():
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
            cu_num_logits=batch.cu_num_logits_np,
            settled_ends=self.settled_ends,
            errors=errors,
        )
        self.pending = PendingHiddenOutput(self, list(batch.req_ids), chunks, errors)

    def take_pending(self) -> PendingHiddenOutput | None:
        pending, self.pending = self.pending, None
        return pending

    def close(self) -> None:
        self.plans.clear()
        self.settled_ends.clear()
        self.carry.clear()
        self.pending = None


class HiddenStateScheduler:
    def __init__(self) -> None:
        self.buffers: dict[str, HiddenStateCaptureBuffer] = {}
        self.done: set[str] = set()
        self.finished: set[str] = set()

    def build_metadata(
        self, scheduled: dict[str, int], requests: dict[str, Request]
    ) -> tuple[dict[str, HiddenStateCapturePlan], dict[str, int], tuple[str, ...]]:
        plans, settled_ends = {}, {}
        for request_id in scheduled:
            if request_id in self.done:
                continue
            request = requests[request_id]
            params = request.sampling_params
            if params is None or not (params.extra_args or {}).get(CAPTURE_KEY):
                continue
            if request_id not in self.buffers:
                plan = capture_plan(params, request_id, request.num_prompt_tokens)
                assert plan is not None
                self.buffers[request_id] = HiddenStateCaptureBuffer(plan)
            buffer = self.buffers[request_id]
            plans[request_id] = buffer.plan
            settled_ends[request_id] = max(
                buffer.settled_end, buffer.plan.window_start_abs
            )
        finished, self.finished = tuple(self.finished), set()
        return plans, settled_ends, finished

    def take_output(
        self, request: Request, output: AuxRequestOutput | None, stopped: bool
    ) -> tuple[HiddenStateCaptureResult | None, str | None]:
        """Settle rows on a step that emits output for the request."""
        request_id = request.request_id
        buffer = self.buffers.get(request_id)
        if buffer is None:
            return None, None
        result, reason = buffer.update(
            None if output is None else output.hidden_chunk,
            request.num_prompt_tokens - 1 + request.num_output_tokens,
            stopped=stopped,
            error=None if output is None else output.hidden_error,
        )
        if result is not None or reason is not None:
            self.done.add(request_id)
            self.finished.add(request_id)
            self.buffers.pop(request_id)
        return result, reason

    def request_finished(self, request: Request) -> None:
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
