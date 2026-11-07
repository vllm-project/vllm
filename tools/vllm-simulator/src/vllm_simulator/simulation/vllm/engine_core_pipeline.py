"""
Engine-core hooks that drive one simulated engine step:

- C_VLLMEngineCoreHook: readiness-based request dispatch, BLOCKING-mode
  request stats, returning finished requests' stats on their outputs and
  resetting the round state once the engine drains.
- C_VLLMSchedulerHook: counts parked requests as unfinished so the busy
  loop keeps stepping.
- C_VLLMExecutorHook: inits the time predictor, accounts the predicted GPU
  span at execute_model, records queue/hit and per-token stats, and logs
  per-step iteration stats at DEBUG level.

The hooks share the module-level request stats.
"""

import heapq
import json
import math
import time
from dataclasses import asdict, dataclass, field

from vllm_simulator.hook import BaseHook
from vllm_simulator.simulation.manager import ConfigManager, Envs, StateManager
from vllm_simulator.simulation.types import (
    IterationStats,
    RequestStats,
    SimulationMode,
)
from vllm_simulator.simulation.vllm.utils import kv_hit_split
from vllm_simulator.time_predictor import ScheduleBatch, ScheduleRequest
from vllm_simulator.utils import get_logger

logger = get_logger()

# When set, overrides max_tokens (with ignore_eos) for every request.
_MAX_TOKENS = Envs.max_tokens()
_SIM_MODE = SimulationMode(Envs.simulation_mode())

# Per-request stats keyed by request id; one engine per process.
_REQUEST_STATS: dict[str, RequestStats] = {}


@dataclass(slots=True)
class _CPUOverheadStep:
    predictor_time: float = 0.0
    emitted: list[str] = field(default_factory=list)


def _new_request_stats(request, created_time, queue_start, last_event_time):
    """Initialize the RequestStats entry; hit fields are set at queue_end."""
    rid = request.request_id
    st = _REQUEST_STATS.setdefault(rid, RequestStats(rid=rid))
    st.created_time = created_time
    st.queue_start = queue_start
    st.queue_end = -1
    st.gen_token_latencies = []
    st.last_event_time = last_event_time
    st.input_length = request.num_prompt_tokens
    st.output_length = request.max_tokens
    return st


def _attach_request_stats(engine_core_outputs) -> None:
    """Hand finished requests' stats to the frontend.

    kv_transfer_params is the only free-form per-request field carried
    end-to-end (RequestOutput and non-streaming OpenAI responses).
    """
    for outputs in engine_core_outputs.values():
        for output in outputs.outputs:
            if not output.finished:
                continue
            st = _REQUEST_STATS.pop(output.request_id, None)
            if st is not None:
                output.kv_transfer_params = {
                    **(output.kv_transfer_params or {}),
                    "simulation": asdict(st),
                }


def _end_round_if_drained(scheduler) -> None:
    """Reset the per-round state once every received request has finished.

    Replaces an explicit round marker: the virtual clock restarts from 0 for
    the next round's created_times.  Skipped while a NON_BLOCKING round is
    still being received.
    """
    dispatcher = ReqDispatcher()
    if scheduler.has_unfinished_requests() or (
        dispatcher.seq_counter and not dispatcher.all_received
    ):
        return
    # Leftovers are aborted requests only.
    _REQUEST_STATS.clear()
    StateManager.reset()
    dispatcher.reset()


class ReqDispatcher:
    """Singleton holding the NON_BLOCKING replay state.

    Requests parked at EngineCore.add_request are released by dispatch()
    at queue_start when supplied, otherwise created_time (the TTFT origin).
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__class__._initialized:
            return
        # (dispatch_time, seq, request, request_wave, created_time).
        self.future_queue: list[tuple[float, int, object, int, float]] = []
        self.seq_counter = 0
        self.total_expected = float("inf")
        self.all_received = False
        # Captured once at hook install (single engine per process)
        self.original_core_add_request = None
        # Live Scheduler, read by C_VLLMExecutorHook.
        self.scheduler = None
        self.cpu_step: _CPUOverheadStep | None = None
        self.__class__._initialized = True

    def reset(self):
        self.future_queue.clear()
        self.seq_counter = 0
        self.total_expected = float("inf")
        self.all_received = False

    def __len__(self) -> int:
        return len(self.future_queue)

    def has_next(self) -> bool:
        return len(self.future_queue) > 0

    def next_req_dispatch_time(self) -> float:
        return self.future_queue[0][0]

    def add(
        self,
        request,
        request_wave: int,
        created_time: float,
        queue_start: float | None = None,
    ):
        """Park a request until its recorded engine-entry time.

        Pretokenized replay skips frontend preparation and input delivery.
        Using queue_start preserves that delay; dispatching at created_time
        would make the request eligible too early. This overhead can be tens
        of milliseconds: significant for short TTFTs, but often negligible
        when TTFT is on the order of seconds.
        created_time remains the TTFT origin and the dispatch fallback when
        queue_start is absent.
        """
        dispatch_time = created_time if queue_start is None else queue_start
        if (
            not math.isfinite(created_time)
            or not math.isfinite(dispatch_time)
            or dispatch_time < created_time
        ):
            raise ValueError("Request times must be finite and queue_start >= arrival")
        heapq.heappush(
            self.future_queue,
            (dispatch_time, self.seq_counter, request, request_wave, created_time),
        )
        self.seq_counter += 1

        if self.seq_counter >= self.total_expected:
            self.all_received = True
            logger.info(
                "All %d requests received. Starting simulation.",
                self.total_expected,
            )

    def dispatch(self, engine_core):
        """Re-submit due parked requests through the native add_request."""
        current_time = StateManager.get_global_clock()
        while self.future_queue and self.future_queue[0][0] <= current_time:
            _, _, request, request_wave, created_time = heapq.heappop(
                self.future_queue
            )
            _new_request_stats(
                request,
                created_time=created_time,
                queue_start=current_time,
                last_event_time=created_time,
            )
            self.original_core_add_request(engine_core, request, request_wave)


class C_VLLMSchedulerHook(BaseHook):
    """Count parked requests as unfinished.

    The busy loop queries the Scheduler directly, so without this it would
    go idle and never dispatch.  has_unfinished_requests / has_requests
    derive from get_num_unfinished_requests.
    """

    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "vllm.v1.core.sched.scheduler"

    @classmethod
    def hook(cls, target):
        original_get_num_unfinished = target.get_num_unfinished_requests
        target.get_num_unfinished_requests = lambda self: (
            original_get_num_unfinished(self) + len(ReqDispatcher())
        )


class C_VLLMEngineCoreHook(BaseHook):
    """Dispatch at supplied queue_start or created_time, retaining TTFT origins.

    Requests are parked ahead of Scheduler.add_request so no connector lookup
    or block allocation happens before the selected dispatch timestamp.

    Prefix block hashing still runs earlier in preprocess_add_request; it
    touches no cache state.
    """

    HOOK_CLASS_NAME = "EngineCore"
    HOOK_MODULE_NAME = "vllm.v1.engine.core"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_add_request = target.add_request
        original_step = target.step
        dispatcher = ReqDispatcher()
        dispatcher.original_core_add_request = original_add_request

        def wrapped_init(self, vllm_config, *args, **kwargs):
            """Reset replay state, stats and the virtual clock."""
            ReqDispatcher().reset()
            _REQUEST_STATS.clear()
            StateManager.reset()

            original_init(self, vllm_config, *args, **kwargs)
            self._sim_cpu_overhead_enabled = ConfigManager.cpu_overhead_enabled()
            ReqDispatcher().scheduler = self.scheduler

        def wrapped_add_request(self, request, request_wave=0):
            """Park NON_BLOCKING simulation requests; forward the rest."""
            if _MAX_TOKENS is not None:
                if request.sampling_params is not None:
                    request.sampling_params.max_tokens = _MAX_TOKENS
                    request.sampling_params.ignore_eos = True
                request.max_tokens = _MAX_TOKENS

            created_time = None
            queue_start = None
            sp = request.sampling_params
            if sp is not None and sp.extra_args:
                sim = sp.extra_args.get("simulation")
                # JSON-encoded by HTTP clients (see bench_serving).
                if isinstance(sim, str):
                    sim = json.loads(sim)
                if sim:
                    created_time = sim.get("created_time")
                    queue_start = sim.get("queue_start")
                    total = sim.get("total_request")
                    if total is not None:
                        dispatcher.total_expected = total

            if _SIM_MODE == SimulationMode.BLOCKING:
                now = time.time()
                _new_request_stats(
                    request,
                    created_time=request.arrival_time,
                    queue_start=now,
                    last_event_time=now,
                )
            if created_time is None or _SIM_MODE == SimulationMode.BLOCKING:
                original_add_request(self, request, request_wave)
                return

            dispatcher.add(request, request_wave, created_time, queue_start)

        def wrapped_step(self, *args, **kwargs):
            """Dispatch due future-queue requests before each engine step;
            return finished requests' stats on their outputs."""
            if _SIM_MODE == SimulationMode.NON_BLOCKING and dispatcher.all_received:
                dispatcher.dispatch(self)

                # Idle: jump the virtual clock to the next created_time.
                if (
                    not any(self.scheduler.get_request_counts())
                    and dispatcher.has_next()
                ):
                    StateManager.set_global_clock(
                        dispatcher.next_req_dispatch_time() + 1e-6
                    )
                    dispatcher.dispatch(self)

            if (
                _SIM_MODE != SimulationMode.NON_BLOCKING
                or not dispatcher.all_received
                or not getattr(self, "_sim_cpu_overhead_enabled", False)
            ):
                result = original_step(self, *args, **kwargs)
            else:
                result = step_with_cpu_overhead(self, *args, **kwargs)
            _attach_request_stats(result[0])
            _end_round_if_drained(self.scheduler)
            return result

        def step_with_cpu_overhead(self, *args, **kwargs):
            # Start after dispatch/idle waits; predictor work is excluded below.
            step = _CPUOverheadStep()
            dispatcher.cpu_step = step
            start = time.perf_counter()
            try:
                result = original_step(self, *args, **kwargs)
            finally:
                dispatcher.cpu_step = None
            cpu_overhead = max(time.perf_counter() - start - step.predictor_time, 0.0)
            StateManager.step_global_clock(cpu_overhead)
            logger.debug("[CPUOverhead] %.6fs", cpu_overhead)
            event_time = StateManager.get_global_clock()
            for req_id in step.emitted:
                st = _REQUEST_STATS.get(req_id)
                if st is not None:
                    st.gen_token_latencies.append(event_time - st.last_event_time)
                    st.last_event_time = event_time
            return result

        target.__init__ = wrapped_init
        target.add_request = wrapped_add_request
        target.step = wrapped_step


class C_VLLMExecutorHook(BaseHook):
    """Account the predicted GPU span at UniProcExecutor.execute_model.

    ExecutorWithExternalLauncher inherits the hooked method.
    """

    HOOK_CLASS_NAME = "UniProcExecutor"
    HOOK_MODULE_NAME = "vllm.v1.executor.uniproc_executor"

    @classmethod
    def hook(cls, target):
        original_init_executor = target._init_executor
        original_execute_model = target.execute_model

        def wrapped_init_executor(self):
            # Worker.init_device (inside) resolves the ConfigManager state.
            original_init_executor(self)
            self._sim_predictor = ConfigManager.get_inference_time_predictor()
            logger.info("%s initialized for vLLM.", type(self._sim_predictor).__name__)

        def wrapped_execute_model(self, scheduler_output, *args, **kwargs):
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            if not num_scheduled_tokens:
                return original_execute_model(self, scheduler_output, *args, **kwargs)

            # Clock before this step is accounted.
            queue_end_time = (
                time.time()
                if _SIM_MODE == SimulationMode.BLOCKING
                else StateManager.get_global_clock()
            )
            # Scheduled requests stay in Scheduler.requests until
            # update_from_output.
            requests = ReqDispatcher().scheduler.requests

            # Pre-step num_computed_tokens, i.e. past_kv_length.
            new_reqs = scheduler_output.scheduled_new_reqs
            num_computed = {r.req_id: r.num_computed_tokens for r in new_reqs}
            cached = scheduler_output.scheduled_cached_reqs
            num_computed.update(zip(cached.req_ids, cached.num_computed_tokens))

            for new_req_data in new_reqs:
                # Resumed requests also arrive here (V2 layout); queue and hit
                # stats belong to the first schedule only.
                st = _REQUEST_STATS.get(new_req_data.req_id)
                if st is not None and st.queue_end == -1:
                    st.queue_end = queue_end_time
                    st.local_kv_hit_len, st.ext_kv_hit_len = kv_hit_split(
                        requests[new_req_data.req_id],
                        new_req_data.num_computed_tokens,
                    )
                    logger.info(
                        "[HitRate] rid=%s input_len=%d local_hit=%d ext_hit=%d",
                        st.rid,
                        st.input_length,
                        st.local_kv_hit_len,
                        st.ext_kv_hit_len,
                    )

            # token_emitted is False for intermediate chunked-prefill steps;
            # passed to _ModelRunnerStub on scheduler_output.
            simulation_batch = ScheduleBatch(reqs=[])
            token_emitted: dict[str, bool] = {}
            logprobs_tokens = 0
            for req_id, num_tokens in num_scheduled_tokens.items():
                request = requests[req_id]
                past_kv_length = num_computed[req_id]
                # GPUModelRunnerV2 samples nothing while seq_len < prefill_len
                # (chunked prefill); num_tokens is prefill_len at (re)admission
                # and seq_len - 1 while decoding.
                token_emitted[req_id] = (
                    past_kv_length + num_tokens >= request.num_tokens
                )
                sp = request.sampling_params
                if sp is not None and sp.logprobs:
                    logprobs_tokens += num_tokens
                simulation_batch.reqs.append(
                    ScheduleRequest(
                        extend_length=num_tokens,
                        past_kv_length=past_kv_length,
                    )
                )
            scheduler_output._sim_token_emitted = token_emitted

            result = original_execute_model(self, scheduler_output, *args, **kwargs)

            cpu_step = ReqDispatcher().cpu_step
            predictor_start = time.perf_counter() if cpu_step is not None else 0.0
            predictor = self._sim_predictor
            predicted_latency, sample_tokens_latency = predictor.predict_step_time(
                simulation_batch
            )
            predicted_latency = float(predicted_latency)
            sample_tokens_latency = float(sample_tokens_latency)

            # Logprobs cost missing from a baseline collected without them;
            # active only with logprobs_cost_us_per_token > 0.
            logprobs_latency = predictor.logprobs_cost_s_per_token * logprobs_tokens

            # RPC-2 (sample_tokens) span, kept separate from the RPC-1-only
            # predicted_latency so both stay auditable in DEBUG logs.
            total_tokens = sum(num_scheduled_tokens.values())
            if cpu_step is not None:
                cpu_step.predictor_time += time.perf_counter() - predictor_start

            iter_stats = IterationStats(
                requests=simulation_batch.request_info(),
                forward_latency=predicted_latency,
                sample_tokens_latency=sample_tokens_latency,
                logprobs_latency=logprobs_latency,
                total_tokens=total_tokens,
            )
            logger.debug("[IterationStats] %s", iter_stats)
            full_step_latency = iter_stats.full_step_latency

            if _SIM_MODE == SimulationMode.BLOCKING:
                # Engine CPU overhead is spent by the real loop; sleep only
                # for the GPU span.
                time.sleep(full_step_latency)
                event_time = time.time()
            else:
                # Optional host overhead is charged separately by wrapped_step.
                StateManager.step_global_clock(full_step_latency)
                event_time = StateManager.get_global_clock()

            for req_id in num_scheduled_tokens:
                # Skip non-emitting chunks so the first latency is the full
                # TTFT.
                if not token_emitted[req_id]:
                    continue
                if cpu_step is not None:
                    cpu_step.emitted.append(req_id)
                else:
                    st = _REQUEST_STATS.get(req_id)
                    if st is not None:
                        st.gen_token_latencies.append(event_time - st.last_event_time)
                        st.last_event_time = event_time

            return result

        target._init_executor = wrapped_init_executor
        target.execute_model = wrapped_execute_model
        logger.info(
            "[vLLM Hijack] UniProcExecutor hook installed "
            "(simulated GPU span at execute_model)"
        )
