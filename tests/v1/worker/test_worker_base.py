# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import vllm.platforms
import vllm.plugins
from vllm.v1.worker import worker_base
from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase

pytestmark = pytest.mark.skip_global_cleanup


def test_worker_extension_hooks_follow_configured_class(monkeypatch):
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

    for extension, fail_destroy, expected in (
        ("ExtensionA", False, ["device", "init-a", "destroy-a", "shutdown"]),
        ("ExtensionB", False, ["device", "shutdown"]),
        ("ExtensionA", True, ["device", "init-a", "destroy-a", "shutdown"]),
    ):
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
        if fail_destroy:
            with pytest.raises(RuntimeError, match="destroy failed"):
                wrapper.shutdown()
        else:
            wrapper.shutdown()
        assert wrapper.worker.events == expected
