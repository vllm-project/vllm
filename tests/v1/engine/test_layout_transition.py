# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of collective admission through the real core and worker path."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.distributed.parallel_state as parallel_state
from vllm import SamplingParams
from vllm.config import DeviceConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.engine.layout_transition import (
    LayoutTransitionFailed,
    LayoutTransitionPhase,
    LayoutTransitionRejected,
    LayoutTransitionRequest,
)
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor.uniproc_executor import ExecutorWithExternalLauncher
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.cpu_test


class _DenseModelConfig(SimpleNamespace):
    verify_with_parallel_config = ModelConfig.verify_with_parallel_config


class _Receiver:
    packed = False

    def __init__(self):
        self.started = False

    def start_weight_update(self):
        self.started = True

    def update_weights(self, payload):
        self.payload = payload

    def finish_weight_update(self):
        self.finished = True

    def reset_weight_update_target(self):
        pass


def _make_core(rank: int) -> tuple[EngineCore, Worker]:
    """Omit device/model allocation while retaining production control flow."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.parallel_config = ParallelConfig(
        distributed_executor_backend="external_launcher",
        tensor_parallel_size=1,
        data_parallel_size=2,
        data_parallel_size_local=2,
    )
    config.model_config = _DenseModelConfig(
        enforce_eager=True,
        is_moe=False,
        dtype=torch.bfloat16,
        quantization=None,
        enable_sleep_mode=False,
        enable_cumem_allocator=False,
        multimodal_config=None,
        model_arch_config=SimpleNamespace(total_num_attention_heads=4),
        hf_config=SimpleNamespace(
            model_type="llama",
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=1,
            intermediate_size=16,
            vocab_size=32,
            num_hidden_layers=0,
            tie_word_embeddings=False,
        ),
    )
    config.weight_transfer_config = SimpleNamespace(backend="ipc")
    config.scheduler_config.async_scheduling = False

    worker = object.__new__(Worker)
    worker.vllm_config = config
    worker.rank = rank
    worker.use_v2_model_runner = True
    worker._layout_transition = None
    worker._layout_transition_target = None
    worker._layout_transition_phase = None
    worker._layout_checkpoint = None
    worker._weight_update_active = False
    worker._weight_update_is_draft = False
    worker.weight_transfer_engine = _Receiver()
    worker.model_runner = object()
    worker.synchronize_device = lambda: None

    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.worker = worker
    executor = object.__new__(ExecutorWithExternalLauncher)
    executor.driver_worker = wrapper
    executor.sleeping_tags = set()

    scheduler = object.__new__(Scheduler)
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.running = []
    scheduler.waiting = []
    scheduler.kv_holding_waiting = []
    scheduler.requests = {}
    scheduler.finished_req_ids = set()
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.connector = None
    scheduler.ec_connector = None

    core = object.__new__(EngineCore)
    core.vllm_config = config
    core.model_executor = executor
    core.scheduler = scheduler
    core.batch_queue = None
    core._layout_transition = None
    core._layout_transition_phase = None
    return core, worker


def _run_rank(rank: int, init_method: str, scenario: str):
    os.environ["RANK"] = str(rank)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    # A CPU world stands in for the persistent bootstrap group. Agreement itself
    # uses a real process group and must include the other independent DP rank.
    parallel_state._WORLD = SimpleNamespace(
        world_size=2, rank=rank, cpu_group=dist.group.WORLD
    )
    try:
        core, worker = _make_core(rank)
        old_parallel = core.vllm_config.parallel_config
        old_runner = worker.model_runner
        request = LayoutTransitionRequest("next-layout", 2, "next-weights")

        expected_error = None
        if scenario == "native_update":
            worker._weight_update_active = rank == 1
            expected_error = "rank 1: a native weight update is active"
        elif scenario == "paused_request":
            core.scheduler.set_pause_state(PauseState.PAUSED_ALL)
            if rank == 1:
                core.scheduler.waiting.append(object())
            assert not core.scheduler.has_requests()
            expected_error = "rank 1: scheduler"
        elif scenario == "finished_request":
            if rank == 1:
                core.scheduler.finished_req_ids.add("finished")
            expected_error = "rank 1: scheduler"
        elif scenario == "target_mismatch":
            request = LayoutTransitionRequest("next-layout", rank + 1, "next-weights")
            expected_error = "physical ranks disagree"
        elif scenario == "ubatching":
            worker.vllm_config.parallel_config.ubatch_size = 2 if rank == 1 else 0
            expected_error = "rank 1: microbatching"
        elif scenario == "fault_tolerance":
            worker.vllm_config.parallel_config.enable_fault_tolerance = rank == 1
            expected_error = "rank 1: microbatching and worker fault tolerance"
        elif scenario == "kv_partition":
            if rank == 1:
                mc = worker.vllm_config.model_config
                mc.model_arch_config.total_num_attention_heads = 6
                mc.hf_config.num_attention_heads = 6
                mc.hf_config.num_key_value_heads = 3
                mc.hf_config.hidden_size = 384
            expected_error = "KV heads"
        elif scenario == "mlp_partition":
            if rank == 1:
                worker.vllm_config.model_config.hf_config.intermediate_size = 17
            expected_error = "intermediate_size"
        elif scenario == "cumem_allocator":
            worker.vllm_config.model_config.enable_cumem_allocator = rank == 1
            expected_error = "CuMem"
        elif scenario == "launcher_env":
            os.environ["VLLM_DP_SIZE"] = "2"
            os.environ["VLLM_DP_RANK"] = str(rank)

        previous_pause = core.scheduler.pause_state
        if scenario.startswith("refit_"):
            _exercise_refit(core, worker, request, rank, scenario)
            dist.barrier()
            return
        if expected_error is not None:
            with pytest.raises(LayoutTransitionRejected, match=expected_error):
                core.prepare_layout_transition(request)
            assert core._layout_transition is None
            assert worker._layout_transition is None
            assert worker._layout_transition_target is None
            assert core.scheduler.pause_state == previous_pause
            assert worker._weight_update_active == (
                scenario == "native_update" and rank == 1
            )
            core._require_no_layout_transition()
        elif scenario == "sync_failure":
            if rank == 1:

                def fail_sync():
                    raise RuntimeError("injected device failure")

                worker.synchronize_device = fail_sync
            with pytest.raises(RuntimeError, match="synchronization failed"):
                core.prepare_layout_transition(request)
            assert core._layout_transition == request
            assert worker._layout_transition == request
            assert worker._layout_transition_target is None
            with pytest.raises(RuntimeError, match="layout transition"):
                core.add_request(object())
            with pytest.raises(RuntimeError, match="layout transition"):
                worker.start_weight_update()
            with pytest.raises(LayoutTransitionRejected, match="recovery is required"):
                core.cancel_layout_transition(request)
            assert core._layout_transition == request
            assert worker._layout_transition == request
        else:
            core.prepare_layout_transition(request)
            assert core.scheduler.pause_state == PauseState.PAUSED_ALL
            assert worker._layout_transition_target.tensor_parallel_size == 2
            assert worker._layout_transition_target.rank == rank
            assert worker._layout_transition_target.data_parallel_size == 1
            assert worker._layout_transition_target.data_parallel_rank == 0
            assert worker._layout_transition_target.data_parallel_index == 0
            assert worker._layout_transition_target.world_size == 2

            if scenario == "duplicate_prepare":
                with pytest.raises(LayoutTransitionRejected, match="already active"):
                    core.prepare_layout_transition(
                        LayoutTransitionRequest("duplicate", 1, "other-weights")
                    )
                assert core._layout_transition == request
                assert worker._layout_transition == request
                assert worker._layout_transition_target.tensor_parallel_size == 2

            for operation in (
                lambda: core.add_request(object()),
                core.step,
                core.resume_scheduler,
                core.execute_dummy_batch,
                worker.start_weight_update,
            ):
                with pytest.raises(RuntimeError, match="layout transition"):
                    operation()
            assert not worker.weight_transfer_engine.started

            core.cancel_layout_transition(request)
            assert core._layout_transition is None
            assert worker._layout_transition is None
            assert worker._layout_transition_target is None
            assert core.scheduler.pause_state == previous_pause
            assert core.step() == ({}, False)
            core.resume_scheduler()
            worker.start_weight_update()
            assert worker.weight_transfer_engine.started

        assert core.vllm_config.parallel_config is old_parallel
        assert worker.model_runner is old_runner
        # This also proves neither peer escaped its rejection path early.
        dist.barrier()
    finally:
        parallel_state._WORLD = None
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario",
    [
        "native_update",
        "paused_request",
        "finished_request",
        "target_mismatch",
        "ubatching",
        "fault_tolerance",
        "kv_partition",
        "mlp_partition",
        "cumem_allocator",
    ],
)
def test_collective_rejection_preserves_each_rank(tmp_path, monkeypatch, scenario):
    """A refusal or target mismatch reaches both ranks without releasing state."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.start_processes(
        _run_rank,
        args=(f"file://{tmp_path / 'world'}", scenario),
        nprocs=2,
        join=True,
        start_method="fork",
    )


@pytest.mark.parametrize("scenario", ["accepted", "duplicate_prepare", "launcher_env"])
def test_reservation_blocks_execution_until_cancel(tmp_path, monkeypatch, scenario):
    """Collective preparation excludes work; cancellation restores the old model."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.start_processes(
        _run_rank,
        args=(f"file://{tmp_path / 'world'}", scenario),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def test_frontend_rejects_before_registering_request():
    """Rejected admission must not leave a phantom frontend request behind."""
    core = object.__new__(EngineCore)
    core._layout_transition = LayoutTransitionRequest("reserved", 2, "next-weights")
    client = object.__new__(InprocClient)
    client.engine_core = core
    frontend = object.__new__(LLMEngine)
    frontend.engine_core = client
    registered = []
    frontend.output_processor = SimpleNamespace(
        add_request=lambda *args: registered.append(args)
    )

    with pytest.raises(RuntimeError, match="layout transition"):
        frontend.add_request("blocked", "Hello", SamplingParams(max_tokens=1))

    assert not registered


def test_synchronization_failure_keeps_all_ranks_reserved(tmp_path, monkeypatch):
    """A device error must not reopen admission on the rank whose sync succeeded."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.start_processes(
        _run_rank,
        args=(f"file://{tmp_path / 'world'}", "sync_failure"),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def _exercise_refit(core, worker, request, rank, scenario):
    """Retain real collective control flow, replace GPU allocations and kernels."""
    from vllm.model_executor.models.llama import LlamaForCausalLM

    cfg = core.vllm_config
    cfg.model_config.hf_config = SimpleNamespace(
        model_type="llama",
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=16,
        vocab_size=32,
        num_hidden_layers=0,
        tie_word_embeddings=False,
    )
    model = object.__new__(LlamaForCausalLM)
    torch.nn.Module.__init__(model)
    model.named_parameters = lambda: iter(
        (name, None)
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
    )
    worker.model_runner = SimpleNamespace(get_model=lambda: model)
    calls = []
    core.scheduler.shutdown = lambda: calls.append("scheduler release")

    def release():
        calls.append("model release")
        if scenario == "refit_release_failure" and rank == 1:
            raise RuntimeError("injected teardown failure")

    def load():
        calls.append("model load")
        worker.model_runner = SimpleNamespace(reset_lora_state=lambda: None)
        worker._weight_update_active = True

    def warmup(config):
        calls.append("warmup")
        assert worker.weight_transfer_engine.finished
        if scenario == "refit_warmup_failure" and rank == 1:
            raise RuntimeError("injected warmup failure")
        return None

    worker._release_layout_model = release
    worker._rebuild_layout_groups = lambda: calls.append("groups rebuild")
    worker._load_layout_model = load
    core._initialize_kv_caches = warmup
    core._create_scheduler = lambda config: (core.scheduler, 16)
    core._initialize_effective_attention_block_size = lambda: None
    cfg.cache_config.enable_prefix_caching = False

    core.prepare_layout_transition(request)
    if scenario == "refit_release_failure":
        with pytest.raises(LayoutTransitionFailed, match="injected teardown failure"):
            core.install_layout_transition(request)
        assert "groups rebuild" not in calls
    else:
        core.install_layout_transition(request)
        with pytest.raises(LayoutTransitionRejected, match="cancel"):
            core.cancel_layout_transition(request)
        with pytest.raises(LayoutTransitionRejected, match="Incomplete checkpoint"):
            core.finish_layout_transition(request)
        assert "warmup" not in calls
        assert not getattr(worker.weight_transfer_engine, "finished", False)
        payload = {
            "names": ["model.embed_tokens.weight", "lm_head.weight"],
            "dtype_names": ["bfloat16", "bfloat16"],
            "shapes": [[32, 8], [32, 8]],
            "ipc_handles": [{}, {}],
        }
        if scenario == "refit_metadata_retry":
            invalid = {**payload, "extra": 1} if rank == 1 else payload
            with pytest.raises(LayoutTransitionRejected, match="Unexpected IPC"):
                core.update_layout_weights(request, invalid)
            assert worker._layout_transition_phase == LayoutTransitionPhase.REFITTING
            assert core._layout_transition_phase == LayoutTransitionPhase.REFITTING
            assert not worker._layout_checkpoint.received
            assert not hasattr(worker.weight_transfer_engine, "payload")
        core.update_layout_weights(request, payload)
        with pytest.raises(LayoutTransitionRejected, match="Duplicate"):
            core.update_layout_weights(request, payload)
        payload = {
            "names": ["model.norm.weight"],
            "dtype_names": ["bfloat16"],
            "shapes": [[8]],
            "ipc_handles": [{}],
        }
        core.update_layout_weights(request, payload)
        with pytest.raises(RuntimeError, match="layout transition"):
            worker.finish_weight_update()
        if scenario == "refit_warmup_failure":
            with pytest.raises(LayoutTransitionFailed, match="injected warmup failure"):
                core.finish_layout_transition(request)
        else:
            core.finish_layout_transition(request)
            assert calls[-1] == "warmup"
            assert core.get_weight_version() == request.weight_version
            assert core._layout_transition is None
            assert worker._layout_transition is None
            assert core.step() == ({}, False)
            return

    assert core._layout_transition_phase == LayoutTransitionPhase.FAILED
    assert worker._layout_transition == request
    with pytest.raises(LayoutTransitionRejected):
        core.cancel_layout_transition(request)
    with pytest.raises(RuntimeError, match="layout transition"):
        core.step()
    with pytest.raises(RuntimeError, match="layout transition"):
        worker.start_weight_update()


@pytest.mark.parametrize(
    "scenario",
    [
        "refit_success",
        "refit_metadata_retry",
        "refit_release_failure",
        "refit_warmup_failure",
    ],
)
def test_refit_requires_complete_weights_and_collective_readiness(
    tmp_path, monkeypatch, scenario
):
    """Incomplete weights never reach warmup; a peer failure never reopens admission."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.start_processes(
        _run_rank,
        args=(f"file://{tmp_path / 'world'}", scenario),
        nprocs=2,
        join=True,
        start_method="fork",
    )


@pytest.mark.parametrize("extra_options", [False, True])
@pytest.mark.parametrize("load_fails", [False, True])
def test_transition_dummy_load_preserves_original_loader_config(
    monkeypatch, extra_options, load_fails
):
    """A temporary dummy load accepts disk-loader options and never poisons reload."""
    import vllm.v1.worker.gpu_worker as worker_module
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

    worker = object.__new__(Worker)
    worker.vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    load_config = worker.vllm_config.load_config
    load_config.load_format = "auto"
    original_extra = {"enable_multithread_load": True} if extra_options else {}
    load_config.model_loader_extra_config = original_extra
    worker.cache_config = worker.vllm_config.cache_config
    worker.device = torch.device("cpu")
    worker._init_model_runner = lambda: None
    worker.weight_transfer_engine = SimpleNamespace(
        parse_init_info=lambda payload: payload,
        init_transfer_engine=lambda info: None,
        start_weight_update=lambda: None,
    )
    monkeypatch.setattr(worker_module, "MemorySnapshot", lambda **kwargs: object())
    monkeypatch.setattr(worker_module, "request_memory", lambda *args: 0)

    def load_model(load_dummy_weights):
        assert load_dummy_weights
        # MRV2 selects the dummy loader by mutating this shared field.
        load_config.load_format = "dummy"
        assert isinstance(get_model_loader(load_config), DummyModelLoader)
        if load_fails:
            raise RuntimeError("injected model-load failure")

    worker.load_model = load_model
    if load_fails:
        with pytest.raises(RuntimeError, match="injected model-load failure"):
            worker._load_layout_model()
    else:
        worker._load_layout_model()
        assert worker._weight_update_active
    assert worker.vllm_config.load_config is load_config
    assert load_config.load_format == "auto"
    assert load_config.model_loader_extra_config is original_extra


def test_reserved_shutdown_continues_after_retirement_error(monkeypatch):
    """A diagnostic retirement failure must not skip the remaining cleanup."""
    import vllm.v1.worker.gpu_worker as worker_module
    from vllm.device_allocator.cumem import CuMemAllocator

    calls = []
    worker = object.__new__(Worker)
    worker.profiler = None
    worker._layout_transition = LayoutTransitionRequest("failed", 2, "next")
    worker.weight_transfer_engine = SimpleNamespace(
        shutdown=lambda: calls.append("receiver")
    )
    worker.model_runner = SimpleNamespace(shutdown=lambda: calls.append("runner"))
    worker.elastic_ep_executor = SimpleNamespace(
        shutdown=lambda: calls.append("executor")
    )

    def fail_retirement():
        raise RuntimeError("injected retained layer")

    worker._release_layout_model = fail_retirement
    monkeypatch.setattr(worker_module, "ensure_kv_transfer_shutdown", lambda: None)
    monkeypatch.setattr(worker_module, "ensure_ec_transfer_shutdown", lambda: None)
    monkeypatch.setattr(worker_module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(
        CuMemAllocator,
        "instance",
        SimpleNamespace(release_pools=lambda: calls.append("pools")),
    )
    worker.shutdown()
    assert calls == ["receiver", "executor", "runner", "pools"]
