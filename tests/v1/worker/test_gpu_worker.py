# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm.utils.jit_monitor as jit_monitor
import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.config.compilation import CUDAGraphMode
from vllm.config.parallel import ParallelConfig
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
)
from vllm.v1.worker import gpu_worker, startup_plan
from vllm.v1.worker.gpu_worker import Worker, maybe_rocm_profiling_fallback
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)
from vllm.v1.worker.utils import requires_persistent_attention_workspace_profiling


@pytest.mark.parametrize(
    ("parallel_config", "expected"),
    [
        pytest.param(ParallelConfig(), 1, id="single-ubatch-default"),
        pytest.param(ParallelConfig(ubatch_size=3), 3, id="manual-ubatch-size"),
        pytest.param(ParallelConfig(enable_dbo=True), 2, id="dbo"),
    ],
)
def test_parallel_config_num_ubatches_covers_all_configurations(
    parallel_config, expected
):
    assert max(1, parallel_config.num_ubatches) == expected


@pytest.mark.parametrize("profile_persistent_workspace", [False, True])
def test_initialize_kv_cache_finalizes_persistent_workspace(
    monkeypatch, profile_persistent_workspace
):
    events = []
    worker = object.__new__(Worker)
    worker.cache_config = SimpleNamespace(num_gpu_blocks=None)
    worker.vllm_config = object()
    worker.model_config = SimpleNamespace(enable_return_routed_experts=False)
    worker._maybe_get_memory_pool_context = lambda **kwargs: nullcontext()
    worker.model_runner = SimpleNamespace(
        initialize_kv_cache=lambda config, **kwargs: events.append(
            "initialize_kv_cache"
        ),
    )
    monkeypatch.setattr(
        gpu_worker,
        "reserve_persistent_attention_workspace",
        lambda runner: events.append("reserve_workspace"),
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=8,
        needs_kv_cache_zeroing=False,
        # The engine core resolves the layout; None keeps the worker on the
        # config it already has.
        kv_cache_layout=None,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "ensure_kv_transfer_initialized",
        lambda *args, **kwargs: events.append("initialize_connector"),
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "requires_persistent_attention_workspace_profiling",
        lambda config: profile_persistent_workspace,
    )

    worker.initialize_from_config(kv_cache_config)

    assert worker.cache_config.num_gpu_blocks == 8
    expected_events = [
        "initialize_connector",
        "initialize_kv_cache",
    ]
    if profile_persistent_workspace:
        expected_events.append("reserve_workspace")
    assert events == expected_events


def _record_capture(events: list[str]) -> int:
    events.append("capture")
    return 0


@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("profiling_support", [True, False])
@pytest.mark.parametrize("grows_during_warmup", [False, True])
@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_warmup_locks_the_workspace_when_profiling_is_enabled(
    monkeypatch,
    enforce_eager,
    profiling_support,
    grows_during_warmup,
    use_v2_model_runner,
):
    """The warmup-path lock follows the model's opt-in.

    capture_model() locks it when it captures, but it returns early when both
    capture modes are disabled and it is skipped entirely under enforce_eager,
    so a model that opted in needs the lock on the warmup path too. A model
    that did not opt in never had its workspace reserved, so locking the arena
    for it would turn a later lazy allocation into a hard failure instead of
    leaving it on the legacy path.
    """
    from vllm.config.compilation import CompilationMode

    events: list[str] = []
    worker = object.__new__(Worker)
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )
    worker.compilation_config = SimpleNamespace(
        mode=CompilationMode.NONE,
        backend="eager",
        compilation_time=0.0,
        encoder_compilation_time=0.0,
    )
    worker.model_config = SimpleNamespace(enforce_eager=enforce_eager, seed=0)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=1)
    worker.observability_config = SimpleNamespace(
        jit_monitor_mode="warn", jit_monitor_verbose=False
    )
    worker.device = torch.device("cpu")
    worker.use_v2_model_runner = use_v2_model_runner

    class _Manager:
        """The live arena; a warmup grows it the way a real wrapper would."""

        def __init__(self):
            self.sizes = (1024,)

        def grow(self):
            if grows_during_warmup:
                self.sizes = (2048,)

        def workspace_sizes_bytes(self):
            return self.sizes

        def assert_within(self, limits, phase):
            if any(c > lim for c, lim in zip(self.sizes, limits)):
                raise AssertionError(
                    f"Attention workspace arena exceeded its profiled size "
                    f"{phase}: profiled={limits}, current={self.sizes}."
                )

    manager = _Manager()
    monkeypatch.setattr(gpu_worker_module, "current_workspace_manager", lambda: manager)

    def _v1_sampler_warmup(hidden_states):
        # V1's eager sampler warmup is the last thing that can size a
        # workspace before the barrier.
        events.append("v1_sampler")
        manager.grow()

    worker.scheduler_config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1)
    worker.model_runner = SimpleNamespace(
        lora_config=None,
        maybe_remove_all_loras=lambda cfg: None,
        capture_model=lambda: _record_capture(events),
        _profiled_persistent_workspace_sizes=(1024,),
        is_pooling_model=False,
        _dummy_run=lambda **kwargs: (None, None),
        _dummy_sampler_run=_v1_sampler_warmup,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_worker,
        "reserve_persistent_attention_workspace",
        lambda runner: events.append("reserve"),
    )
    worker.execute_model = None
    worker.sample_tokens = None
    # capture_model() is wrapped in a profiler context that reads worker and
    # model-runner state this test has no reason to own. What is under test is
    # the order of reserve, warm up and lock around that call, so the context
    # is emptied rather than imitated; the profiler path has its own tests.
    worker._get_cudagraph_capture_context = lambda: nullcontext()

    monkeypatch.setattr(gpu_worker_module, "kernel_warmup", lambda w: None)
    monkeypatch.setattr(
        gpu_worker_module,
        "requires_persistent_attention_workspace_profiling",
        lambda config: profiling_support,
    )
    monkeypatch.setattr(
        gpu_worker_module, "is_workspace_manager_initialized", lambda: True
    )
    monkeypatch.setattr(
        gpu_worker_module, "lock_workspace", lambda: events.append("lock")
    )

    def _v2_warmup(*a, **k):
        events.append("warmup_kernels")
        manager.grow()

    monkeypatch.setattr(gpu_worker_module, "warmup_kernels", _v2_warmup)
    monkeypatch.setattr(gpu_worker_module, "set_random_seed", lambda seed: None)
    monkeypatch.setattr(gpu_worker_module, "freeze_gc_heap", lambda: None)
    monkeypatch.setattr(
        gpu_worker_module, "maybe_attach_gc_debug_callback", lambda: None
    )
    monkeypatch.setattr(gpu_worker_module, "enable_gpu_sync_check", lambda: None)
    monkeypatch.setattr(
        gpu_worker_module, "set_torch_threads_for_runtime", lambda: None
    )
    # compile_or_warm_up_model imports these lazily, so the patch has to land
    # on the defining module rather than on gpu_worker.
    monkeypatch.setattr(jit_monitor, "activate", lambda **kwargs: None)

    grower = "warmup_kernels" if use_v2_model_runner else "v1_sampler"
    if profiling_support and grows_during_warmup:
        # Growth after materialization leaves the reserved wrappers on a
        # replaced allocation, so the barrier has to fail before the lock.
        with pytest.raises(AssertionError, match="before workspace lock"):
            worker.compile_or_warm_up_model()
        assert "lock" not in events
        assert events.index("reserve") < events.index(grower)
        return

    worker.compile_or_warm_up_model()

    assert grower in events
    if enforce_eager:
        assert "capture" not in events
    if profiling_support:
        # Reserve first, then warm up, then lock: the warmup below the lock
        # executes real requests, so an attention wrapper can still grow the
        # workspace there.
        # reserve -> warm up -> assert ceiling -> lock.
        assert events.index("reserve") < events.index(grower)
        assert events.index("lock") == len(events) - 1
    else:
        # Nothing was reserved, so nothing may be locked here.
        assert "reserve" not in events
        assert "lock" not in events


@pytest.mark.parametrize(
    ("builder_support", "speculative", "elastic_ep", "expected"),
    [
        pytest.param(
            [True],
            False,
            False,
            True,
            id="single-required",
        ),
        pytest.param(
            [
                True,
                True,
            ],
            False,
            False,
            True,
            id="all-required",
        ),
        pytest.param(
            [
                True,
                False,
            ],
            False,
            False,
            True,
            id="required-with-neutral",
        ),
        pytest.param(
            [
                True,
                None,
            ],
            False,
            False,
            False,
            id="unsupported-vetoes-required",
        ),
        pytest.param(
            [False],
            False,
            False,
            False,
            id="neutral-only",
        ),
        pytest.param([object()], False, False, False, id="unknown-fails-closed"),
        pytest.param([], False, False, False, id="no-builders"),
        pytest.param(
            [True],
            True,
            False,
            False,
            id="speculative-fallback",
        ),
        pytest.param(
            [True],
            False,
            True,
            False,
            id="elastic-ep-fallback",
        ),
    ],
)
def test_persistent_workspace_profiling_supports_neutral_builders(
    monkeypatch, builder_support, speculative, elastic_ep, expected
):
    class Builder:
        def __init__(self, support):
            self.support = support

        def persistent_workspace_profiling_support(self, config, spec):
            return self.support

    class Backend:
        def __init__(self, support):
            self.builder = Builder(support)

        def get_builder_cls(self):
            return self.builder

    class Layer:
        def __init__(self, support):
            self.backend = Backend(support)

        def get_kv_cache_spec(self, config):
            return object()

        def get_attn_backend(self):
            return self.backend

    config = SimpleNamespace(
        speculative_config=object() if speculative else None,
        parallel_config=SimpleNamespace(enable_elastic_ep=elastic_ep),
    )
    layers = {
        f"layer-{index}": Layer(support)
        for index, support in enumerate(builder_support)
    }
    monkeypatch.setattr(
        "vllm.v1.worker.utils.get_layers_from_vllm_config",
        lambda config, layer_type: layers,
    )

    assert requires_persistent_attention_workspace_profiling(config) is expected


def test_gdn_builder_is_neutral_for_persistent_workspace_profiling():
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    assert (
        GDNAttentionMetadataBuilder.persistent_workspace_profiling_support(
            SimpleNamespace(), SimpleNamespace()
        )
        is False
    )


def test_unknown_builder_fails_closed_for_persistent_workspace_profiling():
    assert (
        AttentionMetadataBuilder.persistent_workspace_profiling_support(
            SimpleNamespace(), SimpleNamespace()
        )
        is None
    )


# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_rejects_stale_schema(plan_env, monkeypatch):
    worker = _plan_worker()
    fingerprint = startup_plan.compute_plan_fingerprint(
        worker.vllm_config, worker.rank, worker.parallel_config.world_size
    )
    maybe_save_startup_plan(worker, 50 * GiB_bytes)

    monkeypatch.setattr(
        startup_plan,
        "PLAN_SCHEMA_VERSION",
        startup_plan.PLAN_SCHEMA_VERSION + 1,
    )

    assert startup_plan._load_plan(fingerprint) is None


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


# Memory accounting of the profiling run (Worker.determine_available_memory).

# The fallback reads only the sign of the measured drop and this process's torch
# reservation; free memory is only logged, so no amount here is a device size.
ANY_FREE_MEMORY = 8 * GiB_bytes
MEASURED_DROP = 4 * GiB_bytes
TORCH_RESERVED = 3 * GiB_bytes
RELEASED_BY_OTHERS = 2 * GiB_bytes


def _snapshot(free_memory, torch_memory=0):
    return SimpleNamespace(free_memory=free_memory, torch_memory=torch_memory)


def _profile_result(consumed, reserved_before=0, reserved_after=0):
    """A result whose free-memory readings agree with `consumed`, which
    `memory_profiling` derives as the drop in free memory, negative when it grew."""
    return SimpleNamespace(
        total_consumed=consumed,
        transient_peak_headroom=0,
        before_create=_snapshot(ANY_FREE_MEMORY, reserved_before),
        after_profile=_snapshot(ANY_FREE_MEMORY - consumed, reserved_after),
    )


@pytest.fixture
def rocm(request):
    with patch.object(
        gpu_worker, "current_platform", SimpleNamespace(is_rocm=lambda: request.param)
    ):
        yield request.param


@pytest.mark.parametrize("rocm", [True, False], indirect=True)
def test_profiling_fallback_declines_when_free_memory_dropped(rocm):
    """The profiling measurement is kept as-is whenever free memory dropped."""
    result = _profile_result(consumed=MEASURED_DROP)

    assert maybe_rocm_profiling_fallback(result) is None


@pytest.mark.parametrize("rocm", [True], indirect=True)
def test_profiling_fallback_replaces_a_released_measurement(rocm):
    """A negative measurement describes the rest of the device, so it is replaced
    by this process's reservation, which the rest of the device cannot move."""
    result = _profile_result(
        consumed=-RELEASED_BY_OTHERS,
        reserved_after=TORCH_RESERVED,
    )

    assert maybe_rocm_profiling_fallback(result) == TORCH_RESERVED


@pytest.mark.parametrize("rocm", [True], indirect=True)
def test_profiling_fallback_never_returns_a_negative_amount(rocm):
    """A reservation that shrank across the run cannot become negative usage."""
    result = _profile_result(
        consumed=-RELEASED_BY_OTHERS,
        reserved_before=TORCH_RESERVED,
        reserved_after=0,
    )

    assert maybe_rocm_profiling_fallback(result) == 0


@pytest.mark.parametrize("rocm", [False], indirect=True)
def test_profiling_fallback_declines_off_rocm(rocm):
    """Platforms that account frees eagerly keep reporting the error, so the
    caller's assertion stays reachable there."""
    result = _profile_result(consumed=-RELEASED_BY_OTHERS)

    assert maybe_rocm_profiling_fallback(result) is None


class _OrderedHandle:
    """Send handle that logs when it is waited."""

    def __init__(self, log: list[str], name: str):
        self.log = log
        self.name = name

    def is_completed(self) -> bool:
        return True

    def wait(self) -> None:
        self.log.append(f"wait:{self.name}")


def test_execute_model_waits_previous_pp_send_before_forward(
    monkeypatch: pytest.MonkeyPatch,
):
    """Previous device handles are waited before the forward pass; the
    metadata handle is left to the GroupCoordinator's reaper."""
    import torch

    from vllm.sequence import IntermediateTensors

    log: list[str] = []
    previous_tensor_send = _OrderedHandle(log, "prev-tensor")
    metadata_handle = _OrderedHandle(log, "meta")
    tensor_handle = _OrderedHandle(log, "tensor")

    def isend_tensor_dict(tensors, all_gather_group=None, all_gather_tensors=None):
        log.append("isend")
        return [metadata_handle, tensor_handle]

    pp_group = SimpleNamespace(
        is_first_rank=True,
        is_last_rank=False,
        isend_tensor_dict=isend_tensor_dict,
    )
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: SimpleNamespace())

    def run_model(scheduler_output, intermediate_tensors):
        log.append("forward")
        return IntermediateTensors({"hidden_states": torch.zeros(1)})

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sp=False)
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=2, distributed_executor_backend="mp"
            ),
        ),
        use_v2_model_runner=False,
        model_runner=SimpleNamespace(execute_model=run_model),
        annotate_profile=lambda scheduler_output: nullcontext(),
        _pp_send_work=[previous_tensor_send],
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=4, num_scheduled_tokens={"r0": 4}
    )

    assert gpu_worker.Worker.execute_model(worker, scheduler_output) is None

    assert log == ["wait:prev-tensor", "forward", "isend"]
    assert worker._pp_send_work == [tensor_handle]


class _FakeProfilingBuilder:
    """A builder whose dedicated workspace is owned by nothing else."""

    def __init__(self) -> None:
        self.int_workspace = torch.empty(64, dtype=torch.uint8)


def test_profiling_lease_outlives_the_closing_memory_measurement(monkeypatch):
    """The dedicated workspace has to survive `memory_profiling`'s exit.

    The shared arena is held by the global workspace manager, but the int
    workspace a wrapper allocates is owned by that wrapper alone. Releasing
    the lease inside the profiling block frees it before the closing
    measurement that `total_consumed` is derived from, so KV cache sizing
    would go back to not knowing about it. It still has to be gone before the
    CUDA graph profiling that follows rebuilds the minimal KV state.
    """
    import gc
    import weakref
    from contextlib import contextmanager

    # The lease has to be the only owner, so the builder is handed over and
    # this scope keeps nothing but weak references to it.
    handover = [_FakeProfilingBuilder()]
    builder_ref = weakref.ref(handover[0])
    workspace_ref = weakref.ref(handover[0].int_workspace)
    alive_at: dict[str, bool] = {}

    def _alive() -> bool:
        gc.collect()
        return builder_ref() is not None and workspace_ref() is not None

    consumed = 4 * GiB_bytes

    @contextmanager
    def fake_memory_profiling(snapshot, weights_memory=0):
        result = _profile_result(consumed)
        result.non_kv_cache_memory = consumed
        yield result
        # memory_profiling measures here, after the block it wraps returns.
        alive_at["closing_measurement"] = _alive()

    worker = object.__new__(Worker)
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    )
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=0, gpu_memory_utilization=0.9
    )
    worker.model_config = SimpleNamespace(multimodal_config=None)
    worker.parallel_config = SimpleNamespace()
    worker.init_snapshot = SimpleNamespace(
        free_memory=ANY_FREE_MEMORY, total_memory=ANY_FREE_MEMORY
    )
    worker.requested_memory = ANY_FREE_MEMORY

    def profile_cudagraph_memory():
        alive_at["cudagraph_profiling"] = _alive()
        return 0

    worker.model_runner = SimpleNamespace(
        model_memory_usage=0,
        profile_run=lambda: None,
        profile_cudagraph_memory=profile_cudagraph_memory,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "prepare_profiling_workspace",
        lambda runner: [handover.pop()],
    )

    monkeypatch.setattr(gpu_worker_module, "maybe_apply_startup_plan", lambda w: None)
    monkeypatch.setattr(gpu_worker_module, "memory_profiling", fake_memory_profiling)
    monkeypatch.setattr(
        gpu_worker_module,
        "requires_persistent_attention_workspace_profiling",
        lambda config: True,
    )
    monkeypatch.setattr(
        gpu_worker_module, "maybe_rocm_profiling_fallback", lambda result: None
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "current_platform",
        SimpleNamespace(is_cuda_alike=lambda: True),
    )
    monkeypatch.setattr(gpu_worker_module, "reserve_mm_ipc_gpu_memory", lambda *args: 0)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)

    worker.determine_available_memory()

    assert alive_at["closing_measurement"] is True
    assert alive_at["cudagraph_profiling"] is False
