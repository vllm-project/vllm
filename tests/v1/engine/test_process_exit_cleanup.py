# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import vllm.distributed.parallel_state as parallel_state
import vllm.platforms as platforms
import vllm.v1.engine.core as core_module
import vllm.v1.worker.gpu.model_runner as model_runner_v2_module
import vllm.v1.worker.gpu_model_runner as model_runner_v1_module
import vllm.v1.worker.gpu_worker as gpu_worker_module
import vllm.v1.worker.xpu_worker as xpu_worker_module
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase


def _engine_core(executor: Mock, scheduler: Mock) -> EngineCore:
    engine_core = EngineCore.__new__(EngineCore)
    engine_core.structured_output_manager = Mock()
    engine_core.model_executor = executor
    engine_core.scheduler = scheduler
    return engine_core


def _v1_model_runner() -> tuple[object, SimpleNamespace]:
    runner = object.__new__(model_runner_v1_module.GPUModelRunner)
    layer = SimpleNamespace(
        kv_cache=["cache"],
        impl=SimpleNamespace(_k_scale_cache=object(), _v_scale_cache=object()),
    )
    runner.kv_caches = ["cache"]
    runner.cross_layers_kv_cache = "cache"
    runner.cross_layers_attn_backend = "backend"
    runner.attn_groups = [Mock()]
    runner.kv_cache_config = Mock()
    runner.cache_config = SimpleNamespace(num_gpu_blocks=1)
    runner.compilation_config = SimpleNamespace(static_forward_context={"layer": layer})
    runner.model = Mock()
    return runner, layer


@pytest.mark.parametrize("process_exiting", [False, True])
def test_engine_core_shutdown_gc_lifecycle(monkeypatch, process_exiting: bool):
    executor = Mock()
    scheduler = Mock()
    cleanup = Mock()
    unfreeze = Mock()
    monkeypatch.setattr(core_module, "cleanup_dist_env_and_memory", cleanup)
    monkeypatch.setattr(core_module.gc, "unfreeze", unfreeze)

    _engine_core(executor, scheduler).shutdown(process_exiting=process_exiting)

    if process_exiting:
        executor.shutdown_for_process_exit.assert_called_once_with()
        executor.shutdown.assert_not_called()
        unfreeze.assert_not_called()
    else:
        executor.shutdown.assert_called_once_with()
        executor.shutdown_for_process_exit.assert_not_called()
        unfreeze.assert_called_once_with()
    scheduler.shutdown.assert_called_once_with()
    cleanup.assert_called_once_with(collect_gc=not process_exiting)


def test_engine_core_process_entry_marks_process_exiting(monkeypatch):
    shutdown = Mock()
    run_engine_core = core_module.EngineCoreProc.run_engine_core
    busy_loop_outcomes = [None, SystemExit, RuntimeError("engine failed")]

    class FakeEngineCoreProc:
        def __init__(self, *args, **kwargs):
            self.input_queue = Mock()
            self.shutdown_state = None
            self._send_engine_dead = Mock()

        def run_busy_loop(self):
            outcome = busy_loop_outcomes.pop(0)
            if outcome is not None:
                raise outcome

        def shutdown(self, **kwargs):
            shutdown(**kwargs)

    class FakeSignalCallback:
        def __init__(self, callback):
            self.callback = callback

        def stop(self):
            pass

    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        data_parallel_rank_local=0,
        data_parallel_index=0,
        numa_bind=False,
        model_config=SimpleNamespace(is_moe=False),
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=SimpleNamespace(is_moe=False),
        kv_transfer_config=None,
    )
    monkeypatch.setattr(core_module, "EngineCoreProc", FakeEngineCoreProc)
    monkeypatch.setattr(core_module, "SignalCallback", FakeSignalCallback)
    monkeypatch.setattr(core_module, "maybe_register_config_serialize_by_value", Mock())
    monkeypatch.setattr(core_module, "set_process_title", Mock())
    monkeypatch.setattr(core_module, "maybe_init_worker_tracer", Mock())
    monkeypatch.setattr(core_module, "decorate_logs", Mock())
    monkeypatch.setattr(core_module.signal, "signal", Mock())

    run_engine_core(vllm_config=vllm_config, executor_class=Mock)
    with pytest.raises(SystemExit):
        run_engine_core(vllm_config=vllm_config, executor_class=Mock)
    with pytest.raises(RuntimeError, match="engine failed"):
        run_engine_core(vllm_config=vllm_config, executor_class=Mock)

    assert shutdown.call_args_list == [
        call(process_exiting=True),
        call(process_exiting=True),
        call(process_exiting=True),
    ]


def test_dp_engine_core_shutdown_forwards_process_exiting(monkeypatch):
    parent_shutdown = Mock()
    destroy_process_group = Mock()

    def fake_parent_shutdown(self, *, process_exiting: bool = False) -> None:
        parent_shutdown(self, process_exiting=process_exiting)

    monkeypatch.setattr(core_module.EngineCoreProc, "shutdown", fake_parent_shutdown)
    monkeypatch.setattr(
        core_module,
        "stateless_destroy_torch_distributed_process_group",
        destroy_process_group,
    )
    regular_engine_core = object.__new__(core_module.DPEngineCoreProc)
    terminal_engine_core = object.__new__(core_module.DPEngineCoreProc)
    dp_group = object()
    terminal_engine_core.dp_group = dp_group

    regular_engine_core.shutdown()
    terminal_engine_core.shutdown(process_exiting=True)

    assert parent_shutdown.call_args_list == [
        call(regular_engine_core, process_exiting=False),
        call(terminal_engine_core, process_exiting=True),
    ]
    destroy_process_group.assert_called_once_with(dp_group)


def test_executor_process_exit_shutdown_falls_back_to_shutdown():
    shutdown = Mock()
    Executor.shutdown_for_process_exit(SimpleNamespace(shutdown=shutdown))
    shutdown.assert_called_once_with()


def test_uniproc_executor_process_exit_shutdown_uses_worker_helper():
    worker = Mock()
    UniProcExecutor.shutdown_for_process_exit(SimpleNamespace(driver_worker=worker))
    worker.shutdown_for_process_exit.assert_called_once_with()
    worker.shutdown.assert_not_called()


def test_worker_process_exit_shutdown_falls_back_for_non_gpu_runner():
    shutdown = Mock()
    worker = SimpleNamespace(model_runner=object(), shutdown=shutdown)

    Worker.shutdown_for_process_exit(worker)

    shutdown.assert_called_once_with()


def test_worker_base_process_exit_shutdown_falls_back_to_shutdown():
    shutdown = Mock()

    WorkerBase.shutdown_for_process_exit(SimpleNamespace(shutdown=shutdown))

    shutdown.assert_called_once_with()


def test_worker_wrapper_process_exit_shutdown_forwards_to_worker():
    worker = Mock()

    WorkerWrapperBase.shutdown_for_process_exit(SimpleNamespace(worker=worker))

    worker.shutdown_for_process_exit.assert_called_once_with()
    worker.shutdown.assert_not_called()


def test_xpu_worker_process_exit_shutdown_uses_full_fallback():
    shutdown = Mock()

    xpu_worker_module.XPUWorker.shutdown_for_process_exit(
        SimpleNamespace(shutdown=shutdown)
    )

    shutdown.assert_called_once_with()


@pytest.mark.parametrize("collect_gc", [True, False])
def test_gpu_worker_shutdown_gc_lifecycle(monkeypatch, collect_gc: bool):
    unfreeze = Mock()
    monkeypatch.setattr("vllm.v1.worker.gpu_worker.gc.unfreeze", unfreeze)
    connector = Mock()
    ec_connector = Mock()
    profiler = Mock()
    transfer = Mock()
    model_runner = Mock()
    cumem_allocator = Mock()
    worker = object.__new__(Worker)
    worker.profiler = profiler
    worker.weight_transfer_engine = transfer
    worker.model_runner = model_runner
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.ensure_kv_transfer_shutdown", connector
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.ensure_ec_transfer_shutdown", ec_connector
    )
    monkeypatch.setattr(
        gpu_worker_module.current_platform, "is_cuda_alike", lambda: True
    )
    monkeypatch.setattr(
        "vllm.device_allocator.cumem.CuMemAllocator.instance", cumem_allocator
    )

    if collect_gc:
        Worker.shutdown(worker)
    else:
        Worker.shutdown_for_process_exit(worker)

    assert unfreeze.call_count == int(collect_gc)
    connector.assert_called_once_with()
    ec_connector.assert_called_once_with()
    profiler.shutdown.assert_called_once_with()
    transfer.shutdown.assert_called_once_with()
    if collect_gc:
        model_runner.shutdown.assert_called_once_with()
    else:
        model_runner.shutdown.assert_called_once_with(collect_gc=False)
    cumem_allocator.release_pools.assert_called_once_with()


@pytest.mark.parametrize("collect_gc", [True, False])
def test_v2_model_runner_shutdown_gc_lifecycle(monkeypatch, collect_gc: bool):
    collect = Mock()
    empty_cache = Mock()
    synchronize = Mock()
    monkeypatch.setattr(model_runner_v2_module.gc, "collect", collect)
    monkeypatch.setattr(
        model_runner_v2_module.torch.accelerator, "empty_cache", empty_cache
    )
    monkeypatch.setattr(
        model_runner_v2_module.torch.accelerator, "synchronize", synchronize
    )
    monkeypatch.setattr(model_runner_v2_module, "free_before_shutdown", Mock())
    runner = SimpleNamespace(
        kv_caches=[],
        attn_groups=[],
        kv_cache_config=Mock(),
        vllm_config=Mock(),
        model=Mock(),
    )

    model_runner_v2_module.GPUModelRunner.shutdown(runner, collect_gc=collect_gc)

    assert collect.call_count == int(collect_gc)
    assert not hasattr(runner, "model")
    assert not hasattr(runner, "kv_cache_config")
    empty_cache.assert_called_once_with()


def test_v2_model_runner_process_exit_shutdown_forwards_collect_gc():
    shutdown = Mock()

    model_runner_v2_module.GPUModelRunner.shutdown_for_process_exit(
        SimpleNamespace(shutdown=shutdown)
    )

    shutdown.assert_called_once_with(collect_gc=False)


def test_v1_model_runner_profiling_cleanup_still_collects(monkeypatch):
    collect = Mock()
    empty_cache = Mock()
    synchronize = Mock()
    monkeypatch.setattr(model_runner_v1_module.gc, "collect", collect)
    monkeypatch.setattr(
        model_runner_v1_module.torch.accelerator, "empty_cache", empty_cache
    )
    monkeypatch.setattr(
        model_runner_v1_module.torch.accelerator, "synchronize", synchronize
    )
    runner, layer = _v1_model_runner()

    runner._cleanup_profiling_kv_cache()

    collect.assert_called_once_with()
    assert runner.cache_config.num_gpu_blocks is None
    assert runner.kv_caches == []
    assert runner.cross_layers_kv_cache is None
    assert runner.cross_layers_attn_backend is None
    assert runner.attn_groups == []
    assert not hasattr(runner, "kv_cache_config")
    assert layer.kv_cache == []
    assert layer.impl._k_scale_cache is None
    assert layer.impl._v_scale_cache is None
    empty_cache.assert_called_once_with()


@pytest.mark.parametrize("collect_gc", [True, False])
def test_v1_model_runner_shutdown_forwards_gc_lifecycle(monkeypatch, collect_gc: bool):
    collect = Mock()
    empty_cache = Mock()
    synchronize = Mock()
    reset_workspace_manager = Mock()
    monkeypatch.setattr(model_runner_v1_module.gc, "collect", collect)
    monkeypatch.setattr(
        model_runner_v1_module.torch.accelerator, "empty_cache", empty_cache
    )
    monkeypatch.setattr(
        model_runner_v1_module.torch.accelerator, "synchronize", synchronize
    )
    monkeypatch.setattr(
        "vllm.v1.worker.workspace.reset_workspace_manager", reset_workspace_manager
    )
    monkeypatch.setattr(
        model_runner_v1_module.current_platform, "is_rocm", lambda: False
    )
    monkeypatch.setattr(
        model_runner_v1_module.current_platform, "is_xpu", lambda: False
    )
    runner, layer = _v1_model_runner()

    runner.shutdown(collect_gc=collect_gc)

    assert collect.call_count == int(collect_gc)
    assert runner.kv_caches == []
    assert runner.cross_layers_kv_cache is None
    assert runner.cross_layers_attn_backend is None
    assert runner.attn_groups == []
    assert not hasattr(runner, "kv_cache_config")
    assert layer.kv_cache == []
    assert runner.compilation_config.static_forward_context == {}
    assert runner.model is None
    empty_cache.assert_called_once_with()
    reset_workspace_manager.assert_called_once_with()


def test_v1_model_runner_process_exit_shutdown_forwards_collect_gc():
    shutdown = Mock()

    model_runner_v1_module.GPUModelRunner.shutdown_for_process_exit(
        SimpleNamespace(shutdown=shutdown)
    )

    shutdown.assert_called_once_with(collect_gc=False)


@pytest.mark.parametrize("collect_gc", [True, False])
def test_distributed_cleanup_gc_lifecycle(monkeypatch, collect_gc: bool):
    unfreeze = Mock()
    collect = Mock()
    destroy_model_parallel = Mock()
    destroy_distributed_environment = Mock()
    disable_envs_cache = Mock()
    empty_cache = Mock()
    host_empty_cache = Mock()
    monkeypatch.setattr(parallel_state.gc, "unfreeze", unfreeze)
    monkeypatch.setattr(parallel_state.gc, "collect", collect)
    monkeypatch.setattr(parallel_state.envs, "disable_envs_cache", disable_envs_cache)
    monkeypatch.setattr(
        parallel_state, "destroy_model_parallel", destroy_model_parallel
    )
    monkeypatch.setattr(
        parallel_state,
        "destroy_distributed_environment",
        destroy_distributed_environment,
    )
    mock_platform = Mock()
    mock_platform.is_rocm.return_value = False
    mock_platform.is_cpu.return_value = False
    monkeypatch.setattr(platforms, "current_platform", mock_platform)
    monkeypatch.setattr(parallel_state.torch.accelerator, "empty_cache", empty_cache)
    monkeypatch.setattr(
        parallel_state.torch._C, "_host_emptyCache", host_empty_cache, raising=False
    )

    parallel_state.cleanup_dist_env_and_memory(collect_gc=collect_gc)

    assert unfreeze.call_count == int(collect_gc)
    assert collect.call_count == int(collect_gc)
    disable_envs_cache.assert_called_once_with()
    destroy_model_parallel.assert_called_once_with()
    destroy_distributed_environment.assert_called_once_with()
    empty_cache.assert_called_once_with()
    host_empty_cache.assert_called_once_with()
