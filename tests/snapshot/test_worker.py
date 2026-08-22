# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest


def _checkpoint_worker(worker_class, backend="uni"):
    worker = object.__new__(worker_class)
    worker._checkpoint_prepare_state = None
    worker.parallel_config = SimpleNamespace(distributed_executor_backend=backend)
    worker.vllm_config = SimpleNamespace(parallel_config=worker.parallel_config)
    return worker


def test_reload_weights_worker_lifecycle_order(monkeypatch):
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.worker.gpu_worker import Worker

    events: list[Any] = []

    class Allocator:
        def discard(self, tags):
            events.append(("discard", tags))

        def wake_up(self, tags):
            events.append(("wake_up", tags))

    worker = _checkpoint_worker(Worker)
    worker._validate_checkpoint_weight_reload = lambda: events.append("validate")
    worker._save_sleep_buffers = lambda: events.append("save_buffers")
    worker._restore_sleep_buffers = lambda: events.append("restore_buffers")
    worker.reload_weights = lambda: events.append("reload_weights")
    worker.model_runner = type(
        "ModelRunner",
        (),
        {"post_kv_cache_wake_up": lambda self: events.append("kv_hook")},
    )()
    monkeypatch.setattr(gpu_worker, "get_discard_mem_allocator_instance", Allocator)
    monkeypatch.setattr(gpu_worker, "get_mem_allocator_instance", Allocator)
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_prepare_distributed_state",
        lambda: events.append("communicator_prepare"),
    )
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_restore_distributed_state",
        lambda: events.append("communicator_restore"),
    )
    policy = {"weights": "discard", "kv": "discard", "runtime": "cuda_image"}

    prepare_timings = Worker.checkpoint_prepare(worker, policy)
    restore_timings = Worker.checkpoint_restore(worker, policy)

    assert events == [
        "validate",
        "save_buffers",
        "communicator_prepare",
        ("discard", ("weights", "kv_cache")),
        ("wake_up", ["weights", "kv_cache"]),
        "reload_weights",
        "restore_buffers",
        "kv_hook",
        "communicator_restore",
    ]
    assert prepare_timings["total_seconds"] >= 0
    assert restore_timings["total_seconds"] >= 0
    assert worker._checkpoint_prepare_state is None


def test_checkpoint_prepare_rejects_multiproc_before_side_effects(monkeypatch):
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.worker.gpu_worker import Worker

    events: list[Any] = []
    worker = _checkpoint_worker(Worker, backend="mp")
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_prepare_distributed_state",
        lambda: events.append("communicator_prepare"),
    )
    policy = {"weights": "cuda_image", "kv": "cuda_image", "runtime": "cuda_image"}

    with pytest.raises(RuntimeError, match="require UniProcExecutor"):
        Worker.checkpoint_prepare(worker, policy)

    assert events == []
    assert worker._checkpoint_prepare_state is None


def test_uniproc_worker_checkpoint_uses_communicator_hooks(monkeypatch):
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.worker.gpu_worker import Worker

    events: list[Any] = []
    worker = _checkpoint_worker(Worker, backend="uni")
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_prepare_distributed_state",
        lambda: events.append("communicator_prepare"),
    )
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_restore_distributed_state",
        lambda: events.append("communicator_restore"),
    )
    policy = {"weights": "cuda_image", "kv": "cuda_image", "runtime": "cuda_image"}

    Worker.checkpoint_prepare(worker, policy)
    Worker.checkpoint_restore(worker, policy)

    assert events == ["communicator_prepare", "communicator_restore"]


def test_checkpoint_prepare_rejects_unknown_executor_before_side_effects(monkeypatch):
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.executor import Executor
    from vllm.v1.worker.gpu_worker import Worker

    class UnsupportedExecutor(Executor):
        pass

    events: list[Any] = []
    worker = _checkpoint_worker(Worker, backend=UnsupportedExecutor)
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_prepare_distributed_state",
        lambda: events.append("communicator_prepare"),
    )
    policy = {"weights": "cuda_image", "kv": "cuda_image", "runtime": "cuda_image"}

    with pytest.raises(RuntimeError, match="require UniProcExecutor"):
        Worker.checkpoint_prepare(worker, policy)

    assert events == []
    assert worker._checkpoint_prepare_state is None


def test_reload_weights_worker_rolls_back_failed_discard(monkeypatch):
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.worker.gpu_worker import Worker

    events: list[Any] = []

    class Allocator:
        def discard(self, tags):
            events.append(("discard", tags))
            raise RuntimeError("discard failed")

        def wake_up(self, tags):
            events.append(("wake_up", tags))

    worker = _checkpoint_worker(Worker)
    worker._validate_checkpoint_weight_reload = lambda: events.append("validate")
    worker._save_sleep_buffers = lambda: events.append("save_buffers")
    worker._restore_sleep_buffers = lambda: events.append("restore_buffers")
    worker.reload_weights = lambda: events.append("reload_weights")
    worker.model_runner = type(
        "ModelRunner",
        (),
        {"post_kv_cache_wake_up": lambda self: events.append("kv_hook")},
    )()
    monkeypatch.setattr(gpu_worker, "get_discard_mem_allocator_instance", Allocator)
    monkeypatch.setattr(gpu_worker, "get_mem_allocator_instance", Allocator)
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_prepare_distributed_state",
        lambda: events.append("communicator_prepare"),
    )
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_restore_distributed_state",
        lambda: events.append("communicator_restore"),
    )
    policy = {"weights": "discard", "kv": "discard", "runtime": "cuda_image"}

    with pytest.raises(RuntimeError, match="discard failed"):
        Worker.checkpoint_prepare(worker, policy)

    assert events == [
        "validate",
        "save_buffers",
        "communicator_prepare",
        ("discard", ("weights", "kv_cache")),
        ("wake_up", ["weights", "kv_cache"]),
        "reload_weights",
        "restore_buffers",
        "kv_hook",
        "communicator_restore",
    ]
    assert worker._checkpoint_prepare_state is not None


def test_reload_weights_worker_checks_discard_capability_before_side_effects(
    monkeypatch,
):
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.worker.gpu_worker import Worker

    events: list[Any] = []
    worker = _checkpoint_worker(Worker)
    worker._validate_checkpoint_weight_reload = lambda: events.append("validate")
    worker._save_sleep_buffers = lambda: events.append("save_buffers")

    def unsupported_allocator():
        events.append("allocator")
        raise NotImplementedError("selective discard is unsupported")

    monkeypatch.setattr(
        gpu_worker, "get_discard_mem_allocator_instance", unsupported_allocator
    )
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_prepare_distributed_state",
        lambda: events.append("communicator_prepare"),
    )
    policy = {"weights": "discard", "kv": "discard", "runtime": "cuda_image"}

    with pytest.raises(NotImplementedError, match="selective discard is unsupported"):
        Worker.checkpoint_prepare(worker, policy)

    assert events == ["validate", "allocator"]
    assert worker._checkpoint_prepare_state is None


def test_reload_weights_worker_rejects_active_lora():
    from vllm.v1.worker.gpu_worker import Worker

    worker = object.__new__(Worker)
    worker.get_draft_model = lambda: None
    worker.vllm_config = type("Config", (), {"lora_config": object()})()
    worker.list_loras = lambda: {1}

    with pytest.raises(NotImplementedError, match="active LoRA adapters"):
        Worker._validate_checkpoint_weight_reload(worker)


def test_reload_weights_worker_rejects_online_weight_updates():
    from vllm.v1.worker.gpu_worker import Worker

    worker = object.__new__(Worker)
    worker.get_draft_model = lambda: None
    worker.vllm_config = type("Config", (), {"lora_config": None})()
    worker.weight_transfer_engine = object()

    with pytest.raises(NotImplementedError, match="online weight updates"):
        Worker._validate_checkpoint_weight_reload(worker)


def test_host_backup_worker_lifecycle_order(monkeypatch):
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.worker.gpu_worker import Worker

    events: list[Any] = []

    class Allocator:
        def sleep(self, offload_tags=None):
            events.append(("sleep", offload_tags))

        def wake_up(self, tags=None):
            events.append(("wake_up", tags))

        def allocation_diagnostics(self):
            return {"tags": {"weights": {"host_backup_bytes": 1024}}}

    worker = _checkpoint_worker(Worker)
    worker.model_runner = type(
        "ModelRunner",
        (),
        {"post_kv_cache_wake_up": lambda self: events.append("kv_hook")},
    )()
    monkeypatch.setattr(gpu_worker, "get_mem_allocator_instance", Allocator)
    monkeypatch.setattr(gpu_worker, "get_diagnostic_mem_allocator_instance", Allocator)
    monkeypatch.setattr(
        gpu_worker.torch._C,
        "_host_emptyCache",
        lambda: events.append("host_cache_clear"),
    )
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_prepare_distributed_state",
        lambda: events.append("communicator_prepare"),
    )
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_restore_distributed_state",
        lambda: events.append("communicator_restore"),
    )
    policy = {"weights": "host_backup", "kv": "discard", "runtime": "cuda_image"}

    prepare_timings = Worker.checkpoint_prepare(worker, policy)
    restore_timings = Worker.checkpoint_restore(worker, policy)

    assert events == [
        "communicator_prepare",
        ("sleep", ("weights",)),
        ("wake_up", None),
        "host_cache_clear",
        "kv_hook",
        "communicator_restore",
    ]
    assert prepare_timings["host_backup_bytes"] == 1024
    assert prepare_timings["total_seconds"] >= 0
    assert restore_timings["host_restore_seconds"] >= 0
    assert worker._checkpoint_prepare_state is None


def test_host_backup_worker_rolls_back_failed_backup(monkeypatch):
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.worker.gpu_worker import Worker

    events: list[Any] = []

    class Allocator:
        def sleep(self, offload_tags=None):
            events.append(("sleep", offload_tags))
            raise RuntimeError("backup failed")

        def wake_up(self, tags=None):
            events.append(("wake_up", tags))

    worker = _checkpoint_worker(Worker)
    worker.model_runner = type(
        "ModelRunner",
        (),
        {"post_kv_cache_wake_up": lambda self: events.append("kv_hook")},
    )()
    monkeypatch.setattr(gpu_worker, "get_mem_allocator_instance", Allocator)
    monkeypatch.setattr(
        gpu_worker.torch._C,
        "_host_emptyCache",
        lambda: events.append("host_cache_clear"),
    )
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_prepare_distributed_state",
        lambda: events.append("communicator_prepare"),
    )
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_restore_distributed_state",
        lambda: events.append("communicator_restore"),
    )
    policy = {"weights": "host_backup", "kv": "discard", "runtime": "cuda_image"}

    with pytest.raises(RuntimeError, match="allocator prepare state is indeterminate"):
        Worker.checkpoint_prepare(worker, policy)

    assert events == [
        "communicator_prepare",
        ("sleep", ("weights",)),
        ("wake_up", None),
        "host_cache_clear",
        "kv_hook",
        "communicator_restore",
    ]
    assert worker._checkpoint_prepare_state is not None


def test_checkpoint_restore_rejects_changed_policy(monkeypatch):
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.worker.gpu_worker import Worker

    class Allocator:
        def sleep(self, offload_tags=None):
            pass

        def wake_up(self, tags=None):
            pass

        def allocation_diagnostics(self):
            return {}

    worker = _checkpoint_worker(Worker)
    worker.model_runner = type(
        "ModelRunner", (), {"post_kv_cache_wake_up": lambda self: None}
    )()
    monkeypatch.setattr(gpu_worker, "get_mem_allocator_instance", Allocator)
    monkeypatch.setattr(gpu_worker, "get_diagnostic_mem_allocator_instance", Allocator)
    monkeypatch.setattr(
        gpu_worker, "checkpoint_prepare_distributed_state", lambda: None
    )
    monkeypatch.setattr(
        gpu_worker, "checkpoint_restore_distributed_state", lambda: None
    )
    l1_policy = {"weights": "host_backup", "kv": "discard", "runtime": "cuda_image"}
    l2_policy = {"weights": "discard", "kv": "discard", "runtime": "cuda_image"}

    Worker.checkpoint_prepare(worker, l1_policy)

    with pytest.raises(RuntimeError, match="policy changed"):
        Worker.checkpoint_restore(worker, l2_policy)

    assert worker._checkpoint_prepare_state is not None


def test_checkpoint_rollback_continues_after_allocator_failure(monkeypatch):
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.worker.gpu_worker import Worker

    events: list[Any] = []

    class Allocator:
        def discard(self, tags):
            events.append("discard")
            raise RuntimeError("discard failed")

        def wake_up(self, tags=None):
            events.append("wake_up")
            raise RuntimeError("wake failed")

    worker = _checkpoint_worker(Worker)
    worker._validate_checkpoint_weight_reload = lambda: events.append("validate")
    worker._save_sleep_buffers = lambda: events.append("save_buffers")
    worker._restore_sleep_buffers = lambda: events.append("restore_buffers")
    worker.reload_weights = lambda: events.append("reload_weights")
    worker.model_runner = type(
        "ModelRunner", (), {"post_kv_cache_wake_up": lambda self: events.append("kv")}
    )()
    monkeypatch.setattr(gpu_worker, "get_discard_mem_allocator_instance", Allocator)
    monkeypatch.setattr(gpu_worker, "get_mem_allocator_instance", Allocator)
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_prepare_distributed_state",
        lambda: events.append("communicator_prepare"),
    )
    monkeypatch.setattr(
        gpu_worker,
        "checkpoint_restore_distributed_state",
        lambda: events.append("communicator_restore"),
    )
    policy = {"weights": "discard", "kv": "discard", "runtime": "cuda_image"}

    with pytest.raises(RuntimeError, match="allocator restore: wake failed"):
        Worker.checkpoint_prepare(worker, policy)

    assert events == [
        "validate",
        "save_buffers",
        "communicator_prepare",
        "discard",
        "wake_up",
        "communicator_restore",
    ]
    assert worker._checkpoint_prepare_state is not None


def test_checkpoint_abort_retries_weight_reload_before_buffers(monkeypatch):
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.worker.gpu_worker import Worker

    events: list[Any] = []
    reload_attempts = 0

    class Allocator:
        def discard(self, tags):
            events.append("discard")

        def wake_up(self, tags=None):
            events.append("wake_up")

    def reload_weights():
        nonlocal reload_attempts
        reload_attempts += 1
        events.append("reload_weights")
        if reload_attempts == 1:
            raise RuntimeError("reload failed")

    worker = _checkpoint_worker(Worker)
    worker._validate_checkpoint_weight_reload = lambda: None
    worker._save_sleep_buffers = lambda: events.append("save_buffers")
    worker._restore_sleep_buffers = lambda: events.append("restore_buffers")
    worker.reload_weights = reload_weights
    worker.model_runner = type(
        "ModelRunner", (), {"post_kv_cache_wake_up": lambda self: events.append("kv")}
    )()
    monkeypatch.setattr(gpu_worker, "get_discard_mem_allocator_instance", Allocator)
    monkeypatch.setattr(gpu_worker, "get_mem_allocator_instance", Allocator)
    monkeypatch.setattr(
        gpu_worker, "checkpoint_prepare_distributed_state", lambda: None
    )
    monkeypatch.setattr(
        gpu_worker, "checkpoint_restore_distributed_state", lambda: None
    )
    policy = {"weights": "discard", "kv": "discard", "runtime": "cuda_image"}

    Worker.checkpoint_prepare(worker, policy)
    worker._checkpoint_prepare_state.allocator_prepare_failed = True

    with pytest.raises(RuntimeError, match="weight reload: reload failed"):
        Worker.checkpoint_abort(worker)

    assert "restore_buffers" not in events
    assert "kv" not in events

    with pytest.raises(RuntimeError, match="allocator prepare state is indeterminate"):
        Worker.checkpoint_abort(worker)

    assert events.count("wake_up") == 1
    assert events.count("reload_weights") == 2
    assert events.count("restore_buffers") == 1
    assert events.count("kv") == 1
