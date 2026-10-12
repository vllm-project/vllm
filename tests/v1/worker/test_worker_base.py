# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import vllm.platforms
import vllm.plugins
from vllm.v1.executor import multiproc_executor
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.worker import worker_base
from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase

pytestmark = pytest.mark.skip_global_cleanup


class Worker(WorkerBase):
    def __init__(self, vllm_config, **_kwargs):
        self.events = []
        self.fail_destroy = vllm_config.fail_destroy

    def init_device(self):
        self.events.append("device")

    def shutdown(self):
        self.events.append("shutdown")


class ExtensionA:
    events: list[str]
    fail_destroy: bool

    def init_worker_extension(self):
        self.events.append("init-a")

    def destroy_worker_extension(self):
        self.events.append("destroy-a")
        if self.fail_destroy:
            raise RuntimeError("destroy failed")


class ExtensionB: ...


@pytest.fixture
def make_wrapper(monkeypatch):
    classes = {
        "Worker": Worker,
        "ExtensionA": ExtensionA,
        "ExtensionB": ExtensionB,
    }
    monkeypatch.setattr(worker_base, "resolve_obj_by_qualname", classes.__getitem__)
    monkeypatch.setattr(
        worker_base,
        "worker_receiver_cache_from_config",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        worker_base, "set_current_vllm_config", lambda _config: nullcontext()
    )
    monkeypatch.setattr(vllm.plugins, "load_general_plugins", lambda: None)
    monkeypatch.setattr(
        vllm.platforms.current_platform,
        "register_triton_kernel_overrides",
        lambda: None,
    )

    def make(extension: str, fail_destroy: bool = False) -> WorkerWrapperBase:
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                worker_cls="Worker", worker_extension_cls=extension
            ),
            enable_trace_function_call_for_thread=lambda: None,
            fail_destroy=fail_destroy,
        )
        wrapper = WorkerWrapperBase()
        wrapper.init_worker([{"vllm_config": config}])
        wrapper.init_device()
        return wrapper

    return make


def test_worker_extension_hooks_follow_configured_class(make_wrapper):
    for extension, fail_destroy, expected in (
        ("ExtensionA", False, ["device", "init-a", "destroy-a", "shutdown"]),
        ("ExtensionB", False, ["device", "shutdown"]),
        ("ExtensionA", True, ["device", "init-a", "destroy-a", "shutdown"]),
    ):
        wrapper = make_wrapper(extension, fail_destroy)
        # A failing destroy hook is logged, not raised.
        wrapper.shutdown()
        assert wrapper.worker.events == expected


def test_failing_destroy_hook_does_not_skip_outer_teardown(make_wrapper, monkeypatch):
    events = []
    monkeypatch.setattr(
        multiproc_executor,
        "destroy_model_parallel",
        lambda: events.append("destroy_model_parallel"),
    )
    monkeypatch.setattr(
        multiproc_executor,
        "destroy_distributed_environment",
        lambda: events.append("destroy_distributed_environment"),
    )
    proc = object.__new__(WorkerProc)
    proc.rpc_broadcast_mq = None
    proc.worker_response_mq = None
    proc.worker = make_wrapper("ExtensionA", fail_destroy=True)

    proc.shutdown()

    assert proc.worker.worker.events[-2:] == ["destroy-a", "shutdown"]
    assert events == ["destroy_model_parallel", "destroy_distributed_environment"]
