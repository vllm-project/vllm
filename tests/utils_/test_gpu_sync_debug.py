# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import warnings

import pytest
import torch

import vllm.utils.gpu_sync_debug as gsd
from vllm.utils.gpu_sync_debug import (
    SYNC_ERROR_MESSAGE,
    gpu_sync_allowed,
    with_gpu_sync_check,
)

from ..utils import create_new_process_for_each_test

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _no_sync():
    # Pure on-GPU compute, no implicit CPU sync...
    x = torch.ones(4, device="cuda") + 1
    # ...plus a sync that we explicitly allow.
    with gpu_sync_allowed():
        return x.cpu()


def _causes_sync():
    x = torch.ones(4, device="cuda")
    # An allowed sync (suppressed)...
    with gpu_sync_allowed():
        x.cpu()
    # ...then an un-allowed sync that should trip the check.
    return x.cpu()


@pytest.mark.parametrize("mode", ["warn", "error"])
@create_new_process_for_each_test()
def test_with_env_set(monkeypatch, mode):
    # Env set + gate flipped on: the unguarded sync is detected.
    # `_SYNC_CHECK_MODE` is read from the env once at import, so patch the
    # module attribute rather than the environment.
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", mode)
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)

    # Guarded syncs always pass.
    with_gpu_sync_check(_no_sync)()

    if mode == "error":
        # "error" mode turns the stray sync into a RuntimeError.
        with pytest.raises(RuntimeError, match=SYNC_ERROR_MESSAGE):
            with_gpu_sync_check(_causes_sync)()
    else:
        # "warn" mode only warns, so the call still succeeds.
        with_gpu_sync_check(_causes_sync)()


@create_new_process_for_each_test()
def test_other_threads_are_not_policed(monkeypatch):
    """A background thread that syncs deliberately must not be broken by the
    check being armed on the thread running the decorated function.
    """
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)

    def sync_on_worker():
        failure: list[BaseException] = []

        def worker():
            try:
                torch.ones(4, device="cuda").cpu()
            except BaseException as exc:  # pragma: no cover - failure path
                failure.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert not failure, f"background thread raised: {failure[0]!r}"

    with_gpu_sync_check(sync_on_worker)()


@create_new_process_for_each_test()
def test_allow_on_other_thread_does_not_disarm(monkeypatch):
    """`gpu_sync_allowed()` on one thread must not suppress the check on
    another. It is scoped by ContextVar rather than torch's process-global
    sync debug mode, which a previous implementation mutated.
    """
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)

    def main_syncs_while_worker_allows():
        stop = threading.Event()

        def worker():
            with gpu_sync_allowed():
                while not stop.is_set():
                    torch.ones(4, device="cuda").cpu()

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            # Must still be reported despite the worker's open allow region.
            torch.ones(4, device="cuda").cpu()
        finally:
            stop.set()
            thread.join()

    with pytest.raises(RuntimeError, match=SYNC_ERROR_MESSAGE):
        with_gpu_sync_check(main_syncs_while_worker_allows)()


@create_new_process_for_each_test()
def test_suppressing_works_while_compiling(monkeypatch):
    """`_suppressing` wraps torch compile entry points, which run with
    `torch.compiler.is_compiling()` true. `gpu_sync_allowed()` deliberately
    no-ops in that state, so `_suppressing` must not route through it.
    """
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)
    # Emulate being inside a torch compile, as inductor passes are.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    suppressed = gsd._suppressing(lambda: torch.ones(4, device="cuda").cpu())
    with_gpu_sync_check(suppressed)()


@create_new_process_for_each_test()
def test_sync_debug_mode_restored_after_checked_call(monkeypatch):
    """The mode is armed only for the duration of a checked call. Leaving it
    on process-wide made every sync outside a checked region emit a
    `UserWarning` whenever our handler was not the installed one.
    """
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)

    before = torch.cuda.get_sync_debug_mode()

    def nested():
        # `execute_model` -> `sample_tokens` both carry the decorator.
        assert torch.cuda.get_sync_debug_mode() != 0, "armed inside"
        with_gpu_sync_check(lambda: None)()
        assert torch.cuda.get_sync_debug_mode() != 0, "still armed after inner"

    with_gpu_sync_check(nested)()
    assert torch.cuda.get_sync_debug_mode() == before

    # With the mode back to its original value, torch emits nothing for a
    # sync outside a checked region.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.ones(4, device="cuda").cpu()
    assert not [r for r in caught if gsd._TORCH_SYNC_WARNING in str(r.message)]


@create_new_process_for_each_test()
def test_without_env_set(monkeypatch):
    # Env unset: the decorator is a pass-through, no sync is detected.
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", None)
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)

    with_gpu_sync_check(_no_sync)()
    with_gpu_sync_check(_causes_sync)()
