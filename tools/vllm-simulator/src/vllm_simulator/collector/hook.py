import gzip
import json
import os
import time
import torch
from typing import TYPE_CHECKING, Callable, Optional
from dataclasses import dataclass, field, asdict
from array import array

from vllm_simulator.hook import BaseHook
from vllm_simulator.simulation.vllm.utils import kv_hit_split
from vllm_simulator.utils.json import CustomJsonEncoder

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


# Environment variables:
#   VLLM_SIMULATOR_COLLECTOR_OUTPUT_DIR: export base directory (default: cwd)
#   VLLM_SIMULATOR_COLLECTOR_EXPORT_TOKEN_IDS: keep input_ids/output_ids in
#       requests export (default on)
#   VLLM_SIMULATOR_COLLECTOR_EXPORT_GZIP: write .jsonl.gz instead of .jsonl
#       (default on)
OUTPUT_DIR = os.getenv("VLLM_SIMULATOR_COLLECTOR_OUTPUT_DIR", os.getcwd())
EXPORT_TOKEN_IDS = _env_flag("VLLM_SIMULATOR_COLLECTOR_EXPORT_TOKEN_IDS", True)
EXPORT_GZIP = _env_flag("VLLM_SIMULATOR_COLLECTOR_EXPORT_GZIP", True)


@dataclass(slots=True)
class RequestInfos:
    rid: str = ""
    created_time: Optional[float] = None
    queue_start: float = 0.0
    queue_end: float = 0.0
    output_length: int = 0
    input_length: int = 0
    local_kv_hit_len: int = 0
    ext_kv_hit_len: int = 0
    input_ids: array = field(default_factory=lambda: array("i"))
    output_ids: array = field(default_factory=lambda: array("i"))


@dataclass(slots=True)
class BatchInfos:
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    forward_mode: int = 0
    # List of tuples (req_id, extend_input_len, past_kv_len, output_len)
    requests: list[tuple[str, int, int, int]] = field(default_factory=list)
    iter_latency: float = 0.0
    sample_tokens_latency: float = 0.0
    logprobs_req_count: int = 0
    logprobs_tokens: int = 0
    logprobs_n: int = 0


# Data is collected only between start_profile and stop_profile.  Worker
# and EngineCore may live in different processes, so each side keeps its
# own collecting flag, toggled by its own profile() call.
_COLLECTING_BATCHES = False
_COLLECTING_REQUESTS = False

BATCH_INFOS: list[BatchInfos] = []

REQUEST_INFOS: dict[str, RequestInfos] = {}


# Timing uses CUDA event pairs instead of torch.cuda.synchronize(), which
# would break async-scheduling overlap and skew the measurement.  Elapsed
# time is read after vLLM's own sync (AsyncModelRunnerOutput.get_output),
# so the hook never synchronizes by itself.


class _EventTimer:
    """A start/end pair of timing cuda events for one measured region."""

    __slots__ = ("_start", "_end")

    def __init__(self) -> None:
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def record_start(self) -> None:
        self._start.record()

    def record_end(self) -> None:
        self._end.record()

    def elapsed_s(self) -> Optional[float]:
        """Elapsed seconds, or None if an event has not completed (guards a
        rare race so it never crashes the engine)."""
        try:
            return self._start.elapsed_time(self._end) / 1000.0
        except RuntimeError:
            return None


# Hands the execute_model (RPC-1) finalizer to the same step's sample_tokens
# (RPC-2); one slot suffices since the two run back-to-back.
_PENDING_EXEC: Optional[Callable[[Optional[float]], None]] = None


def _finalize_on_get_output(output, finalize):
    """Run ``finalize`` once vLLM has synced this step's output: inside
    get_output() for async outputs, immediately otherwise."""
    original_get_output = getattr(output, "get_output", None)
    if original_get_output is None:
        finalize()
        return output

    def wrapped_get_output(*args, **kwargs):
        result = original_get_output(*args, **kwargs)
        finalize()
        return result

    output.get_output = wrapped_get_output
    return output


def _open_export(path: str):
    """Open a JSONL export file; gzip-compressed (level 1) when EXPORT_GZIP."""

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if EXPORT_GZIP:
        return gzip.open(path, "wt", compresslevel=1)
    return open(path, "w")


class C_VLLMEngineArgsHook(BaseHook):
    """Hook EngineArgs to default the profiler so /start_profile and
    /stop_profile (the collection window markers) are registered; the
    hooked Worker.profile never starts it."""

    HOOK_CLASS_NAME = "EngineArgs"
    HOOK_MODULE_NAME = "vllm.engine.arg_utils"

    @classmethod
    def hook(cls, target):
        original_post_init = target.__post_init__

        def wrapped_post_init(self):
            if self.profiler_config.profiler is None:
                self.profiler_config.profiler = "cuda"
            original_post_init(self)

        target.__post_init__ = wrapped_post_init


class C_WorkerHook(BaseHook):
    HOOK_MODULE_NAME = "vllm.v1.worker.gpu_worker"
    HOOK_CLASS_NAME = "Worker"

    @classmethod
    def hook(cls, target) -> None:

        original_execute_model = target.execute_model

        def wrapped_execute_model(self, scheduler_output: "SchedulerOutput"):
            if not _COLLECTING_BATCHES:
                return original_execute_model(self, scheduler_output)

            batch_req_infos = {}
            for req_id, sched_token in scheduler_output.num_scheduled_tokens.items():
                batch_req_infos[req_id] = {"extend_input_len": sched_token}
            
            for req_id, completed_token, output_token in zip(
                scheduler_output.scheduled_cached_reqs.req_ids, 
                scheduler_output.scheduled_cached_reqs.num_computed_tokens,
                scheduler_output.scheduled_cached_reqs.num_output_tokens
            ):
                batch_req_infos[req_id]["past_kv_len"] = completed_token
                batch_req_infos[req_id]["output_len"] = output_token
            
            for req in scheduler_output.scheduled_new_reqs:
                batch_req_infos[req.req_id]["past_kv_len"] = req.num_computed_tokens
                # V2 also sends resumed requests here, with prefill_token_ids
                # covering the prompt plus the outputs generated so far.
                batch_req_infos[req.req_id]["output_len"] = (
                    len(req.prefill_token_ids) - req.prompt_len
                )

            forward_mode = 2 if all([num_token == 1 for num_token in scheduler_output.num_scheduled_tokens.values()]) else 1

            # Logprobs features for compensation training.  Only new reqs
            # carry sampling_params, so decode steps are not counted.
            logprobs_req_count = 0
            logprobs_tokens = 0
            logprobs_n = 0
            for req in scheduler_output.scheduled_new_reqs:
                sp = getattr(req, "sampling_params", None)
                if sp is not None and getattr(sp, "logprobs", None):
                    logprobs_req_count += 1
                    logprobs_tokens += scheduler_output.num_scheduled_tokens.get(
                        req.req_id, 0
                    )
                    logprobs_n = max(logprobs_n, sp.logprobs)

            global _PENDING_EXEC

            exec_timer = _EventTimer()
            start_wall = time.time()
            exec_timer.record_start()
            ret = original_execute_model(self, scheduler_output)
            exec_timer.record_end()

            batch_info = None
            if len(batch_req_infos):
                batch_info = BatchInfos(
                    start_timestamp=start_wall,
                    # end_timestamp / iter_latency are set by _finalize_exec.
                    forward_mode=forward_mode,
                    # Keep only the last 12 chars of rid.
                    requests=[
                        (
                            rid[-12:] if len(rid) > 12 else rid,
                            req_info["extend_input_len"],
                            req_info["past_kv_len"],
                            req_info["output_len"],
                        )
                        for rid, req_info in batch_req_infos.items()
                    ],
                    logprobs_req_count=logprobs_req_count,
                    logprobs_tokens=logprobs_tokens,
                    logprobs_n=logprobs_n,
                )
                BATCH_INFOS.append(batch_info)

            def _finalize_exec(sample_latency: Optional[float] = None):
                if batch_info is None:
                    return
                iter_latency = exec_timer.elapsed_s()
                if iter_latency is None:
                    return
                batch_info.iter_latency = iter_latency
                batch_info.end_timestamp = start_wall + iter_latency
                if sample_latency is not None:
                    batch_info.sample_tokens_latency = sample_latency

            # Two-RPC vLLM: the output comes from sample_tokens; finalize
            # there.
            if ret is None:
                _PENDING_EXEC = _finalize_exec
                return ret

            return _finalize_on_get_output(ret, _finalize_exec)

        target.execute_model = wrapped_execute_model

        # RPC-2 window (sampler + MTP draft + bookkeeping D2H + output
        # build), paired 1:1 with the execute_model window.  Skipped on vLLM
        # without sample_tokens.
        original_sample_tokens = getattr(target, "sample_tokens", None)
        if original_sample_tokens is not None:

            def wrapped_sample_tokens(self, *args, **kwargs):
                global _PENDING_EXEC

                pending_exec = _PENDING_EXEC
                _PENDING_EXEC = None
                # Nothing pending: execute_model ran outside the window.
                if pending_exec is None:
                    return original_sample_tokens(self, *args, **kwargs)

                sample_timer = _EventTimer()
                sample_timer.record_start()
                ret = original_sample_tokens(self, *args, **kwargs)
                sample_timer.record_end()

                def _finalize():
                    pending_exec(sample_timer.elapsed_s())

                return _finalize_on_get_output(ret, _finalize)

            target.sample_tokens = wrapped_sample_tokens

        def override_profile(
            self, is_start: bool = True, profile_prefix: str | None = None
        ):
            global BATCH_INFOS, _COLLECTING_BATCHES

            # Only rank0 (the executor's output rank under TP) has its async
            # output synced via get_output(); other ranks would never get
            # their latencies filled, so they collect nothing.
            if self.rank != 0:
                return

            if is_start:
                BATCH_INFOS = []
                _COLLECTING_BATCHES = True
                return

            _COLLECTING_BATCHES = False
            batch_infos = BATCH_INFOS
            BATCH_INFOS = []

            path = f"{OUTPUT_DIR}/rank{self.rank}.schedule_batch.jsonl"
            if EXPORT_GZIP:
                path += ".gz"
            with _open_export(path) as f:
                for batch_info in batch_infos:
                    f.write(json.dumps(batch_info, cls=CustomJsonEncoder) + "\n")
            print(f"Schedule batch data has been saved to {path}")

        target.profile = override_profile



class C_EngineCoreHook(BaseHook):

    HOOK_MODULE_NAME = "vllm.v1.engine.core"
    HOOK_CLASS_NAME = "EngineCore"

    @classmethod
    def hook(cls, target) -> None:

        original_profile = target.profile

        def wrapped_profile(
            self, is_start: bool = True, profile_prefix: str | None = None
        ):
            global REQUEST_INFOS, _COLLECTING_REQUESTS

            if is_start:
                REQUEST_INFOS = {}
                _COLLECTING_REQUESTS = True
                return original_profile(self, is_start, profile_prefix)

            _COLLECTING_REQUESTS = False
            request_infos = REQUEST_INFOS
            REQUEST_INFOS = {}

            path = f"{OUTPUT_DIR}/requests.jsonl"
            if EXPORT_GZIP:
                path += ".gz"
            with _open_export(path) as f:
                for req_infos in request_infos.values():
                    row = asdict(req_infos)
                    if not EXPORT_TOKEN_IDS:
                        # Token ids dominate the payload; drop on request.
                        del row["input_ids"]
                        del row["output_ids"]
                    f.write(json.dumps(row, cls=CustomJsonEncoder) + "\n")
            print(f"Request data has been saved to {path}")

            return original_profile(self, is_start, profile_prefix)

        target.profile = wrapped_profile



class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "vllm.v1.core.sched.scheduler"

    @classmethod
    def hook(cls, target) -> None:

        original_add_request = target.add_request
        original_schedule = target.schedule
        original_free_request = target._free_request

        # Only requests arriving inside the window are tracked; schedule
        # and free skip ids absent from REQUEST_INFOS.
        def wrapped_add_request(self, request):
            if _COLLECTING_REQUESTS:
                REQUEST_INFOS[request.request_id] = RequestInfos(
                    rid=request.request_id,
                    created_time=request.arrival_time,
                    queue_start=time.time(),
                )
            return original_add_request(self, request)

        def wrapped_schedule(self, *args, **kwargs):
            scheduler_output = original_schedule(self, *args, **kwargs)

            if scheduler_output.scheduled_new_reqs:
                prefill_timestamp = time.time()
                for new_req_data in scheduler_output.scheduled_new_reqs:
                    req_id = new_req_data.req_id
                    req_info = REQUEST_INFOS.get(req_id)
                    request = self.requests.get(req_id)
                    if req_info is None or request is None:
                        continue
                    if req_info.queue_end == 0:
                        req_info.queue_end = prefill_timestamp
                        req_info.input_length = request.num_prompt_tokens
                        req_info.output_length = request.max_tokens

                        # The local/external hit split lives on prefill_stats.
                        req_info.local_kv_hit_len, req_info.ext_kv_hit_len = (
                            kv_hit_split(request, new_req_data.num_computed_tokens)
                        )

            return scheduler_output

        def wrapped_free_request(self, request, *args, **kwargs):
            # Token ids dominate REQUEST_INFOS memory; skip collecting them
            # entirely when they are not exported.
            req_info = REQUEST_INFOS.get(request.request_id)
            if req_info is not None and EXPORT_TOKEN_IDS and request.is_finished():
                req_info.input_ids = (
                    array("i", request.prompt_token_ids)
                    if request.prompt_token_ids is not None
                    else array("i")
                )
                req_info.output_ids = array("i", request.output_token_ids)
            return original_free_request(self, request, *args, **kwargs)

        target.add_request = wrapped_add_request
        target.schedule = wrapped_schedule
        target._free_request = wrapped_free_request