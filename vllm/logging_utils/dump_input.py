# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import enum
import faulthandler
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.stats import SchedulerStats
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S = 300.0
ENGINE_EXECUTION_TIMEOUT_REQUEST_SAMPLE_LIMIT = 20
ENGINE_EXECUTION_TIMEOUT_REQUEST_ID_MAX_CHARS = 256
ENGINE_EXECUTION_TIMEOUT_SUMMARY_MAX_CHARS = 32_768
ENGINE_EXECUTION_TIMEOUT_WATCHDOG_STOP_TIMEOUT_S = 1.0
_ENGINE_TIMEOUT_MODEL_CONFIG_FIELDS = (
    "dtype",
    "enforce_eager",
    "max_model_len",
    "quantization",
    "runner_type",
)
_ENGINE_TIMEOUT_PARALLEL_CONFIG_FIELDS = (
    "data_parallel_size",
    "enable_expert_parallel",
    "pipeline_parallel_size",
    "tensor_parallel_size",
)
_ENGINE_TIMEOUT_SCHEDULER_CONFIG_FIELDS = (
    "async_scheduling",
    "enable_chunked_prefill",
    "long_prefill_token_threshold",
    "max_num_batched_tokens",
    "max_num_scheduled_tokens",
    "max_num_seqs",
    "policy",
    "prefill_schedule_interval",
    "runner_type",
)
_ENGINE_TIMEOUT_CACHE_CONFIG_FIELDS = (
    "block_size",
    "cache_dtype",
    "enable_prefix_caching",
    "gpu_memory_utilization",
)
_ENGINE_TIMEOUT_OFFLOAD_CONFIG_FIELDS = ("offload_backend",)
_ENGINE_TIMEOUT_UVA_OFFLOAD_CONFIG_FIELDS = ("cpu_offload_gb",)
_ENGINE_TIMEOUT_SPECULATIVE_CONFIG_FIELDS = (
    "draft_tensor_parallel_size",
    "max_model_len",
    "method",
    "num_speculative_tokens",
    "quantization",
)


@dataclass(frozen=True)
class EngineExecutionTimeoutSnapshot:
    scheduler_output_summary: dict[str, Any]
    scheduler_queue_summary: dict[str, Any]


def prepare_object_to_dump(obj) -> str:
    if isinstance(obj, str):
        return f"'{obj}'"  # Double quotes
    elif isinstance(obj, dict):
        dict_str = ", ".join(
            {f"{str(k)}: {prepare_object_to_dump(v)}" for k, v in obj.items()}
        )
        return f"{{{dict_str}}}"
    elif isinstance(obj, list):
        return f"[{', '.join([prepare_object_to_dump(v) for v in obj])}]"
    elif isinstance(obj, set):
        return f"[{', '.join([prepare_object_to_dump(v) for v in list(obj)])}]"
        # return [prepare_object_to_dump(v) for v in list(obj)]
    elif isinstance(obj, tuple):
        return f"[{', '.join([prepare_object_to_dump(v) for v in obj])}]"
    elif isinstance(obj, enum.Enum):
        return repr(obj)
    elif isinstance(obj, torch.Tensor):
        # We only print the 'draft' of the tensor to not expose sensitive data
        # and to get some metadata in case of CUDA runtime crashed
        return f"Tensor(shape={obj.shape}, device={obj.device},dtype={obj.dtype})"
    elif hasattr(obj, "anon_repr"):
        return obj.anon_repr()
    elif hasattr(obj, "__dict__"):
        items = obj.__dict__.items()
        dict_str = ", ".join(
            [f"{str(k)}={prepare_object_to_dump(v)}" for k, v in items]
        )
        return f"{type(obj).__name__}({dict_str})"
    else:
        # Hacky way to make sure we can serialize the object in JSON format
        try:
            return json.dumps(obj)
        except (TypeError, OverflowError):
            return repr(obj)


def dump_engine_exception(
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
):
    # NOTE: ensure we can log extra info without risking raises
    # unexpected errors during logging
    with contextlib.suppress(Exception):
        _dump_engine_execution_context(
            "exception", config, scheduler_output, scheduler_stats
        )


def dump_engine_execution_timeout(
    config: VllmConfig,
    snapshot: EngineExecutionTimeoutSnapshot,
    timeout_s: float,
    stage: str,
):
    try:
        logger.error(
            "V1 LLM engine stage '%s' has not completed after %.2f seconds "
            "(pid=%d). Dumping sanitized scheduler state and Python stack "
            "traces. "
            "Further dumps for this stage are throttled for %.0f seconds. "
            "Set VLLM_ENGINE_SLOW_STAGE_DUMP_S=0 to disable this diagnostic.",
            stage,
            timeout_s,
            os.getpid(),
            ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S,
        )
        _dump_engine_timeout_context(config, snapshot)
    except Exception:
        with contextlib.suppress(Exception):
            logger.exception("Failed to dump V1 engine timeout context")

    with contextlib.suppress(Exception):
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


def _dump_engine_timeout_context(
    config: VllmConfig,
    snapshot: EngineExecutionTimeoutSnapshot,
) -> None:
    summary = {
        "config": _make_engine_config_summary(config),
        **snapshot.scheduler_output_summary,
    }
    logger.error("Scheduler output summary: %s", _serialize_diagnostic(summary))
    if snapshot.scheduler_queue_summary:
        logger.error(
            "Scheduler queue summary: %s",
            _serialize_diagnostic(snapshot.scheduler_queue_summary),
        )


def make_engine_execution_timeout_snapshot(
    scheduler_output: SchedulerOutput,
    scheduler_state: dict[str, Any] | None,
) -> EngineExecutionTimeoutSnapshot:
    """Copy bounded timeout diagnostics off scheduler-owned objects."""
    scheduler_state = scheduler_state or {}
    cached_sampling_params = scheduler_state.get("cached_request_sampling_params", {})
    request_samples = _make_request_samples(
        scheduler_output,
        cached_sampling_params=cached_sampling_params,
    )
    num_scheduled_requests = (
        len(scheduler_output.scheduled_new_reqs)
        + scheduler_output.scheduled_cached_reqs.num_reqs
    )
    scheduler_output_summary: dict[str, Any] = {
        "has_kv_connector_metadata": (
            scheduler_output.kv_connector_metadata is not None
        ),
        "has_pending_structured_output_tokens": (
            scheduler_output.pending_structured_output_tokens
        ),
        "num_finished_reqs": len(scheduler_output.finished_req_ids),
        "num_preempted_reqs": len(scheduler_output.preempted_req_ids or ()),
        "num_scheduled_cached_reqs": scheduler_output.scheduled_cached_reqs.num_reqs,
        "num_scheduled_encoder_inputs": len(scheduler_output.scheduled_encoder_inputs),
        "num_scheduled_new_reqs": len(scheduler_output.scheduled_new_reqs),
        "num_scheduled_reqs": len(scheduler_output.num_scheduled_tokens),
        "num_scheduled_spec_decode_reqs": len(
            scheduler_output.scheduled_spec_decode_tokens
        ),
        "request_sample_limit": ENGINE_EXECUTION_TIMEOUT_REQUEST_SAMPLE_LIMIT,
        "request_samples": request_samples,
        "request_samples_truncated": num_scheduled_requests > len(request_samples),
        "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
    }
    queue_summary = {
        _bounded_diagnostic_string(str(key)): _diagnostic_scalar(value)
        for key, value in scheduler_state.items()
        if key != "cached_request_sampling_params"
    }
    return EngineExecutionTimeoutSnapshot(scheduler_output_summary, queue_summary)


def _make_engine_config_summary(config: VllmConfig) -> dict[str, Any]:
    model_config = config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    speculative_config = config.speculative_config
    summary = {
        "model": {
            **_select_diagnostic_fields(
                model_config,
                _ENGINE_TIMEOUT_MODEL_CONFIG_FIELDS,
            ),
            "architectures": list(getattr(hf_config, "architectures", None) or ()),
            "model_type": getattr(hf_config, "model_type", None),
        },
        "parallel": _select_diagnostic_fields(
            config.parallel_config,
            _ENGINE_TIMEOUT_PARALLEL_CONFIG_FIELDS,
        ),
        "scheduler": _select_diagnostic_fields(
            config.scheduler_config,
            _ENGINE_TIMEOUT_SCHEDULER_CONFIG_FIELDS,
        ),
        "cache": _select_diagnostic_fields(
            config.cache_config,
            _ENGINE_TIMEOUT_CACHE_CONFIG_FIELDS,
        ),
        "offload": {
            **_select_diagnostic_fields(
                config.offload_config,
                _ENGINE_TIMEOUT_OFFLOAD_CONFIG_FIELDS,
            ),
            **_select_diagnostic_fields(
                config.offload_config.uva,
                _ENGINE_TIMEOUT_UVA_OFFLOAD_CONFIG_FIELDS,
            ),
        },
        "speculative": {
            "enabled": speculative_config is not None,
            **_select_diagnostic_fields(
                speculative_config,
                _ENGINE_TIMEOUT_SPECULATIVE_CONFIG_FIELDS,
            ),
        },
    }
    return summary


def _select_diagnostic_fields(
    obj: object | None,
    field_names: tuple[str, ...],
) -> dict[str, Any]:
    if obj is None:
        return {}
    return {
        field_name: _diagnostic_scalar(value)
        for field_name in field_names
        if (value := getattr(obj, field_name, None)) is not None
    }


def _diagnostic_scalar(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, str):
        return _bounded_diagnostic_string(value)
    if value is None or isinstance(value, (bool, float, int)):
        return value
    return _bounded_diagnostic_string(str(value))


def _bounded_diagnostic_string(value: str) -> str:
    if len(value) <= ENGINE_EXECUTION_TIMEOUT_REQUEST_ID_MAX_CHARS:
        return value
    digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"{value[:160]}...{value[-64:]} [length={len(value)}, sha256={digest}]"


def _serialize_diagnostic(value: dict[str, Any]) -> str:
    serialized = json.dumps(value, sort_keys=True, default=str)
    if len(serialized) <= ENGINE_EXECUTION_TIMEOUT_SUMMARY_MAX_CHARS:
        return serialized

    metadata = {
        "diagnostic_output_truncated": True,
        "original_length": len(serialized),
        "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }
    low = 0
    high = min(len(serialized), ENGINE_EXECUTION_TIMEOUT_SUMMARY_MAX_CHARS)
    bounded = json.dumps({**metadata, "diagnostic_prefix": ""}, sort_keys=True)
    while low <= high:
        prefix_length = (low + high) // 2
        candidate = json.dumps(
            {**metadata, "diagnostic_prefix": serialized[:prefix_length]},
            sort_keys=True,
        )
        if len(candidate) <= ENGINE_EXECUTION_TIMEOUT_SUMMARY_MAX_CHARS:
            bounded = candidate
            low = prefix_length + 1
        else:
            high = prefix_length - 1
    return bounded


def _make_request_samples(
    scheduler_output: SchedulerOutput,
    cached_sampling_params: dict[str, dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    cached_requests = scheduler_output.scheduled_cached_reqs
    new_request_indices, cached_request_indices = (
        get_engine_timeout_request_sample_indices(
            len(scheduler_output.scheduled_new_reqs), cached_requests.num_reqs
        )
    )
    samples = [
        _make_new_request_sample(
            scheduler_output.scheduled_new_reqs[index], scheduler_output
        )
        for index in new_request_indices
    ]
    samples.extend(
        _make_cached_request_sample(
            index,
            scheduler_output,
            cached_sampling_params=cached_sampling_params,
        )
        for index in cached_request_indices
    )
    return samples


def get_engine_timeout_request_sample_indices(
    num_new_requests: int, num_cached_requests: int
) -> tuple[list[int], list[int]]:
    total_requests = num_new_requests + num_cached_requests
    sample_limit = min(ENGINE_EXECUTION_TIMEOUT_REQUEST_SAMPLE_LIMIT, total_requests)
    if num_new_requests == 0:
        return [], _evenly_spaced_indices(num_cached_requests, sample_limit)
    if num_cached_requests == 0:
        return _evenly_spaced_indices(num_new_requests, sample_limit), []

    new_request_limit = max(
        1,
        min(
            num_new_requests,
            sample_limit - 1,
            sample_limit * num_new_requests // total_requests,
        ),
    )
    cached_request_limit = sample_limit - new_request_limit
    return (
        _evenly_spaced_indices(num_new_requests, new_request_limit),
        _evenly_spaced_indices(num_cached_requests, cached_request_limit),
    )


def _evenly_spaced_indices(item_count: int, sample_count: int) -> list[int]:
    if sample_count >= item_count:
        return list(range(item_count))
    if sample_count == 1:
        return [item_count // 2]
    return [
        index * (item_count - 1) // (sample_count - 1) for index in range(sample_count)
    ]


def _make_new_request_sample(
    request: Any,
    scheduler_output: SchedulerOutput,
) -> dict[str, Any]:
    request_id = request.req_id
    prompt_embeds_shape = (
        tuple(request.prompt_embeds.shape)
        if request.prompt_embeds is not None
        else None
    )
    return {
        "has_lora": request.lora_request is not None,
        "has_pooling_params": request.pooling_params is not None,
        "num_computed_tokens": request.num_computed_tokens,
        "num_kv_blocks": _num_blocks(request.block_ids),
        "num_kv_cache_groups": len(request.block_ids),
        "num_mm_features": len(request.mm_features),
        "num_prefill_tokens": _optional_len(request.prefill_token_ids),
        "num_prompt_tokens": _optional_len(request.prompt_token_ids),
        "prompt_embeds_shape": prompt_embeds_shape,
        **_make_request_id_summary(request_id),
        "request_kind": "new",
        "sampling_params": make_sampling_params_summary(request.sampling_params),
        **_make_scheduled_request_summary(request_id, scheduler_output),
    }


def _make_cached_request_sample(
    index: int,
    scheduler_output: SchedulerOutput,
    cached_sampling_params: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    cached_requests = scheduler_output.scheduled_cached_reqs
    request_id = cached_requests.req_ids[index]
    new_block_ids = _item_at(cached_requests.new_block_ids, index)
    new_token_ids = _item_at(cached_requests.new_token_ids, index)
    return {
        "is_resumed": request_id in cached_requests.resumed_req_ids,
        "num_all_tokens": _optional_len(cached_requests.all_token_ids.get(request_id)),
        "num_computed_tokens": _item_at(cached_requests.num_computed_tokens, index),
        "num_new_blocks": _num_blocks(new_block_ids),
        "num_new_tokens": _optional_len(new_token_ids),
        "num_output_tokens": _item_at(cached_requests.num_output_tokens, index),
        **_make_request_id_summary(request_id),
        "request_kind": "cached",
        "sampling_params": (cached_sampling_params or {}).get(request_id),
        **_make_scheduled_request_summary(request_id, scheduler_output),
    }


def _make_request_id_summary(request_id: str) -> dict[str, Any]:
    if len(request_id) <= ENGINE_EXECUTION_TIMEOUT_REQUEST_ID_MAX_CHARS:
        return {"request_id": request_id}

    digest = hashlib.sha256(
        request_id.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    return {
        "request_id": f"{request_id[:160]}...{request_id[-64:]}",
        "request_id_length": len(request_id),
        "request_id_sha256": digest,
        "request_id_truncated": True,
    }


def _make_scheduled_request_summary(
    request_id: str,
    scheduler_output: SchedulerOutput,
) -> dict[str, int]:
    return {
        "num_encoder_inputs": len(
            scheduler_output.scheduled_encoder_inputs.get(request_id, ())
        ),
        "num_scheduled_tokens": scheduler_output.num_scheduled_tokens.get(
            request_id, 0
        ),
        "num_spec_tokens": len(
            scheduler_output.scheduled_spec_decode_tokens.get(request_id, ())
        ),
    }


def make_sampling_params_summary(sampling_params: Any | None) -> dict[str, Any] | None:
    if sampling_params is None:
        return None
    summary = _select_diagnostic_fields(
        sampling_params,
        (
            "detokenize",
            "flat_logprobs",
            "frequency_penalty",
            "ignore_eos",
            "include_stop_str_in_output",
            "logprobs",
            "max_tokens",
            "min_p",
            "min_tokens",
            "n",
            "output_kind",
            "presence_penalty",
            "prompt_logprobs",
            "repetition_penalty",
            "seed",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "temperature",
            "thinking_token_budget",
            "top_k",
            "top_p",
        ),
    )
    stop = getattr(sampling_params, "stop", None)
    summary.update(
        {
            "has_extra_args": getattr(sampling_params, "extra_args", None) is not None,
            "has_repetition_detection": (
                getattr(sampling_params, "repetition_detection", None) is not None
            ),
            "has_structured_outputs": (
                getattr(sampling_params, "structured_outputs", None) is not None
            ),
            "num_allowed_token_ids": _optional_len(
                getattr(sampling_params, "allowed_token_ids", None)
            ),
            "num_bad_words": _optional_len(getattr(sampling_params, "bad_words", None)),
            "num_logit_bias_entries": _optional_len(
                getattr(sampling_params, "logit_bias", None)
            ),
            "num_logprob_token_ids": _optional_len(
                getattr(sampling_params, "logprob_token_ids", None)
            ),
            "num_stop_strings": 1 if isinstance(stop, str) else _optional_len(stop),
            "num_stop_token_ids": _optional_len(
                getattr(sampling_params, "stop_token_ids", None)
            ),
        }
    )
    return summary


def _item_at(values: list[Any], index: int) -> Any | None:
    return values[index] if index < len(values) else None


def _optional_len(value: Any | None) -> int | None:
    return len(value) if value is not None else None


def _num_blocks(block_ids: Any | None) -> int | None:
    if block_ids is None:
        return None
    return sum(len(group) for group in block_ids)


def _dump_engine_execution_context(
    reason: str,
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
):
    logger.error(
        "Dumping input data for V1 LLM engine (v%s, reason=%s) with config: %s, ",
        VLLM_VERSION,
        reason,
        config,
    )
    try:
        dump_obj = prepare_object_to_dump(scheduler_output)
        logger.error("Dumping scheduler output for model execution: %s", dump_obj)
        if scheduler_stats:
            logger.error("Dumping scheduler stats: %s", scheduler_stats)
    except Exception:
        logger.exception("Error preparing object to dump")


@dataclass(frozen=True)
class _EngineExecutionTimeoutState:
    deadline_s: float
    generation: int
    snapshot: EngineExecutionTimeoutSnapshot
    stage: str


class EngineExecutionTimeoutWatchdog:
    """Dumps engine state when an armed execution stage exceeds its deadline."""

    def __init__(
        self,
        *,
        config: VllmConfig,
        timeout_s: float | None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.timeout_s = timeout_s
        self.time_fn = time_fn

        self._generation = 0
        self._last_dump_s_by_stage: dict[str, float] = {}
        self._lock = threading.Lock()
        self._state: _EngineExecutionTimeoutState | None = None
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._wake_event = threading.Event()

    @property
    def enabled(self) -> bool:
        return self.timeout_s is not None and self.timeout_s > 0 and not self._stopped

    def start(self) -> None:
        if not self.enabled:
            return

        start_error = None
        with self._lock:
            if self._stopped or self._thread is not None:
                return
            self._thread = self._create_thread()
            try:
                self._thread.start()
            except RuntimeError as err:
                self._thread = None
                self._stopped = True
                start_error = err

        if start_error is not None:
            logger.warning(
                "Unable to start the engine execution timeout watchdog; "
                "continuing with this diagnostic disabled: %s",
                start_error,
            )

    def _create_thread(self) -> threading.Thread:
        return threading.Thread(
            target=self._run,
            name="EngineExecutionTimeoutWatchdog",
            daemon=True,
        )

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._state = None
            thread = self._thread
        self._wake_event.set()
        if thread is not None and threading.current_thread() is not thread:
            thread.join(timeout=ENGINE_EXECUTION_TIMEOUT_WATCHDOG_STOP_TIMEOUT_S)
            if thread.is_alive():
                logger.warning(
                    "Engine execution timeout watchdog is still finishing "
                    "diagnostic output after shutdown"
                )

    def arm(
        self,
        snapshot: EngineExecutionTimeoutSnapshot,
        stage: str,
    ) -> int | None:
        timeout_s = self.timeout_s
        if timeout_s is None or timeout_s <= 0 or self._stopped:
            return None

        with self._lock:
            if self._stopped:
                return None
            self._generation += 1
            generation = self._generation
            previous_state = self._state
            deadline_s = self.time_fn() + timeout_s
            self._state = _EngineExecutionTimeoutState(
                deadline_s=deadline_s,
                generation=generation,
                snapshot=snapshot,
                stage=stage,
            )
            should_wake = (
                previous_state is None or deadline_s < previous_state.deadline_s
            )
        if should_wake:
            self._wake_event.set()
        return generation

    def disarm(self, generation: int | None) -> None:
        if generation is None:
            return
        with self._lock:
            if self._stopped:
                return
            state = self._state
            if state is None or state.generation != generation:
                return
            self._state = None

    def _run(self) -> None:
        while True:
            self._wake_event.clear()
            with self._lock:
                if self._stopped:
                    return
                state = self._state

            if state is None:
                self._wake_event.wait()
                continue

            remaining_s = max(0.0, state.deadline_s - self.time_fn())
            if self._wake_event.wait(remaining_s):
                continue

            with self._lock:
                current_state = self._state
                if (
                    current_state is None
                    or current_state.generation != state.generation
                    or current_state.deadline_s > self.time_fn()
                ):
                    continue
                self._state = None

            if not self._mark_dump_if_allowed(state.stage):
                continue
            dump_engine_execution_timeout(
                self.config,
                state.snapshot,
                self.timeout_s or 0,
                state.stage,
            )

    def _mark_dump_if_allowed(self, stage: str) -> bool:
        now_s = self.time_fn()
        with self._lock:
            last_dump_s = self._last_dump_s_by_stage.get(stage)
            if (
                last_dump_s is not None
                and now_s - last_dump_s < ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S
            ):
                return False
            self._last_dump_s_by_stage[stage] = now_s
            return True
