"""
Engine-core hooks that drive one simulated engine step:

- C_VLLMEngineCoreHook: created_time-based request dispatch, BLOCKING-mode
  request stats and profile()-driven stats export.
- C_VLLMSchedulerHook: counts parked requests as unfinished so the busy
  loop keeps stepping.
- C_VLLMExecutorHook: inits the time predictor, accounts the predicted GPU
  span at execute_model and records queue/hit, iteration and per-token stats.

The hooks share the module-level request/iteration stats.
"""

import heapq
import json
import os
import time
from dataclasses import asdict

from vllm_simulator.hook import BaseHook
from vllm_simulator.simulation.manager import ConfigManager, StateManager, Envs
from vllm_simulator.simulation.types import IterationStats, RequestStats, SimulationMode
from vllm_simulator.simulation.vllm.utils import kv_hit_split
from vllm_simulator.time_predictor import ScheduleBatch, ScheduleRequest
from vllm_simulator.utils import get_logger

logger = get_logger()

# When set, overrides max_tokens (with ignore_eos) for every request.
_MAX_DECODE_STEPS = int(v) if (v := os.environ.get("VLLM_SIMULATOR_MAX_DECODE_STEPS")) is not None else None
_SIM_MODE = SimulationMode(Envs.simulation_mode())

# Per-request stats keyed by request id; one engine per process.
_REQUEST_STATS: dict[str, RequestStats] = {}
# One record per forward step, appended by C_VLLMExecutorHook.
_ITERATION_STATS: list[IterationStats] = []

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


class ReqDispatcher:
    """Singleton holding the NON_BLOCKING replay state.

    Requests parked at EngineCore.add_request are released by dispatch()
    once global_clock >= created_time.
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
        # (created_time, seq, request, request_wave); seq breaks ties so
        # requests are never compared.
        self.future_queue: list[tuple[float, int, object, int]] = []
        self.seq_counter = 0
        self.total_expected = float("inf")
        self.all_received = False
        # Captured once at hook install (single engine per process)
        self.original_core_add_request = None
        # Live Scheduler, read by C_VLLMExecutorHook.
        self.scheduler = None
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

    def next_req_created_time(self) -> float:
        return self.future_queue[0][0]

    def add(self, request, request_wave: int, created_time: float):
        """Park a request fully inert until its created_time."""
        heapq.heappush(
            self.future_queue,
            (created_time, self.seq_counter, request, request_wave),
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
            created_time, _, request, request_wave = heapq.heappop(
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
    """Created_time-based request dispatch at EngineCore.

    Requests are parked in add_request, ahead of Scheduler.add_request, so
    no connector lookup or block allocation happens before created_time.
    step() dispatches them in created_time order.

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
            """Reset replay state and stats."""
            ReqDispatcher().reset()
            _REQUEST_STATS.clear()
            _ITERATION_STATS.clear()

            original_init(self, vllm_config, *args, **kwargs)
            ReqDispatcher().scheduler = self.scheduler

        def wrapped_add_request(self, request, request_wave=0):
            """Park NON_BLOCKING simulation requests; forward the rest."""
            if _MAX_DECODE_STEPS is not None:
                if request.sampling_params is not None:
                    request.sampling_params.max_tokens = _MAX_DECODE_STEPS
                    request.sampling_params.ignore_eos = True
                request.max_tokens = _MAX_DECODE_STEPS

            created_time = None
            sp = request.sampling_params
            if sp is not None and sp.extra_args:
                sim = sp.extra_args.get("simulation")
                # JSON-encoded by HTTP clients (see bench_serving).
                if isinstance(sim, str):
                    sim = json.loads(sim)
                if sim:
                    created_time = sim.get("created_time")
                    total = sim.get("total_request")
                    if total is not None:
                        dispatcher.total_expected = total

            if _SIM_MODE == SimulationMode.BLOCKING:
                now = time.time()
                _new_request_stats(
                    request,
                    created_time=created_time if created_time is not None else now,
                    queue_start=now,
                    last_event_time=now,
                )
            if created_time is None or _SIM_MODE == SimulationMode.BLOCKING:
                original_add_request(self, request, request_wave)
                return

            dispatcher.add(request, request_wave, created_time)

        def wrapped_step(self, *args, **kwargs):
            """Dispatch due future-queue requests before each engine step."""
            if _SIM_MODE == SimulationMode.NON_BLOCKING and dispatcher.all_received:
                dispatcher.dispatch(self)

                # Idle: jump the virtual clock to the next created_time.
                if (
                    not self.scheduler.waiting
                    and not self.scheduler.running
                    and dispatcher.has_next()
                ):
                    StateManager.set_global_clock(
                        dispatcher.next_req_created_time() + 1e-6
                    )
                    dispatcher.dispatch(self)

            return original_step(self, *args, **kwargs)

        target.__init__ = wrapped_init
        target.add_request = wrapped_add_request
        target.step = wrapped_step

        # profile() is the benchmark round separator: dump stats to
        # VLLM_SIMULATOR_OUTPUT_DIR and reset.  Merged here because only the
        # first hook matching a class is applied.  The signature must match
        # the native one: the utility RPC always sends (is_start,
        # profile_prefix).
        def wrapped_profile(
            self, is_start: bool = True, profile_prefix: str | None = None
        ):
            req_stats = list(_REQUEST_STATS.values())
            output_dir = Envs.output_dir()
            with open(os.path.join(output_dir, "request.jsonl"), "w") as f:
                for item in req_stats:
                    f.write(json.dumps(asdict(item), default=str) + "\n")
            with open(os.path.join(output_dir, "iteration.jsonl"), "w") as f:
                for item in _ITERATION_STATS:
                    f.write(json.dumps(asdict(item), default=str) + "\n")

            logger.info(
                "[ProfileHook] Exported %d requests, %d iterations to %s",
                len(req_stats),
                len(_ITERATION_STATS),
                output_dir,
            )

            _REQUEST_STATS.clear()
            _ITERATION_STATS.clear()
            # Runs in the engine-core process, the only place this state is
            # reachable under AsyncLLM.
            StateManager.reset()
            ReqDispatcher().reset()

        target.profile = wrapped_profile


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
            logger.info(
                "%s initialized for vLLM.", type(self._sim_predictor).__name__
            )

        def wrapped_execute_model(self, scheduler_output, *args, **kwargs):
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            if not num_scheduled_tokens:
                return original_execute_model(
                    self, scheduler_output, *args, **kwargs)

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

            result = original_execute_model(
                self, scheduler_output, *args, **kwargs)

            predictor = self._sim_predictor
            predicted_latency = float(
                predictor.predict_infer_time(simulation_batch)
            )

            # Logprobs cost missing from a baseline collected without them;
            # active only with logprobs_cost_us_per_token > 0.
            logprobs_latency = predictor.logprobs_cost_s_per_token * logprobs_tokens

            # RPC-2 (sample_tokens) span, kept separate from the RPC-1-only
            # predicted_latency so both stay auditable in iteration.jsonl.
            total_tokens = sum(num_scheduled_tokens.values())
            sample_tokens_latency = float(
                predictor.predict_sample_tokens_time(total_tokens)
            )

            iter_stats = IterationStats(
                requests=simulation_batch.request_info(),
                forward_latency=predicted_latency,
                sample_tokens_latency=sample_tokens_latency,
                logprobs_latency=logprobs_latency,
                total_tokens=total_tokens,
            )
            _ITERATION_STATS.append(iter_stats)
            full_step_latency = iter_stats.full_step_latency

            if _SIM_MODE == SimulationMode.BLOCKING:
                # Engine CPU overhead is spent by the real loop; sleep only
                # for the GPU span.
                time.sleep(full_step_latency)
                event_time = time.time()
            else:
                # Known diff, deliberately uncompensated: the engine residual
                # (~2.5ms/step of RPC + EngineCore bookkeeping, outside the
                # label) is not modelled, so NON_BLOCKING runs ~2.5ms/step
                # shorter than real.
                StateManager.step_global_clock(full_step_latency)
                event_time = StateManager.get_global_clock()

            for req_id in num_scheduled_tokens:
                # Skip non-emitting chunks so the first latency is the full
                # TTFT.
                if not token_emitted[req_id]:
                    continue
                st = _REQUEST_STATS.get(req_id)
                if st is not None:
                    st.gen_token_latencies.append(
                        event_time - st.last_event_time
                    )
                    st.last_event_time = event_time

            return result

        target._init_executor = wrapped_init_executor
        target.execute_model = wrapped_execute_model
        logger.info("[vLLM Hijack] UniProcExecutor hook installed "
                    "(simulated GPU span at execute_model)")
