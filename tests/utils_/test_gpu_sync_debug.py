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
    check being armed on the thread running the decorated function."""
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
    sync debug mode, which a previous implementation mutated."""
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
    no-ops in that state, so `_suppressing` must not route through it."""
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
    `UserWarning` whenever our handler was not the installed one."""
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


def _pageable_h2d_nonblocking():
    # Pageable source: the "async" copy is staged through pageable memory.
    torch.zeros(4).to("cuda", non_blocking=True)


def _pinned_h2d_nonblocking():
    # Pinned, contiguous, same dtype: genuinely asynchronous.
    torch.zeros(4).pin_memory().to("cuda", non_blocking=True)


def _noncontiguous_h2d_nonblocking():
    # Pinned but gapped layout (strided slice): staged through a pageable temp.
    torch.zeros(8).pin_memory()[::2].to("cuda", non_blocking=True)


def _transposed_h2d_nonblocking():
    # Dense permuted layout: copied with pitched cudaMemcpy2DAsync, stays async.
    torch.zeros(4, 4).pin_memory().t().to("cuda", non_blocking=True)


def _dtype_converting_h2d_nonblocking():
    # Pinned and contiguous: the conversion runs GPU-side, staying async.
    torch.zeros(4, dtype=torch.float64).pin_memory().to(
        "cuda", dtype=torch.float32, non_blocking=True
    )


def _pageable_h2d_via_cuda():
    torch.zeros(4).cuda(non_blocking=True)


def _pageable_h2d_via_copy_():
    torch.zeros(4, device="cuda").copy_(torch.zeros(4), non_blocking=True)


def _d2h_via_to():
    # Genuinely asynchronous: torch allocates the destination pinned.
    torch.zeros(4, device="cuda").to("cpu", non_blocking=True)


def _pageable_d2h_via_copy_():
    torch.zeros(4).copy_(torch.zeros(4, device="cuda"), non_blocking=True)


def _pinned_d2h_via_copy_():
    # Pinned, contiguous, same dtype: genuinely asynchronous.
    torch.zeros(4).pin_memory().copy_(torch.zeros(4, device="cuda"), non_blocking=True)


def _noncontiguous_d2h_via_copy_():
    # Pinned but gapped (strided slice) destination.
    torch.zeros(8).pin_memory()[::2].copy_(
        torch.zeros(4, device="cuda"), non_blocking=True
    )


def _transposed_d2h_via_copy_():
    # Dense permuted destination: pitched cudaMemcpy2DAsync, stays async.
    torch.zeros(4, 4).pin_memory().t().copy_(
        torch.zeros(4, 4, device="cuda"), non_blocking=True
    )


def _empty_h2d_nonblocking():
    # Empty (e.g. first-step penalties): no CUDA call is issued at all.
    torch.zeros(0).to("cuda", non_blocking=True)


def _dtype_converting_d2h_via_copy_():
    # Pinned and contiguous: the conversion runs GPU-side, staying async.
    torch.zeros(4).pin_memory().copy_(
        torch.zeros(4, dtype=torch.float64, device="cuda"), non_blocking=True
    )


@pytest.mark.parametrize("mode", ["warn", "error"])
@pytest.mark.parametrize(
    "fn",
    [
        _pageable_h2d_nonblocking,
        _noncontiguous_h2d_nonblocking,
        _pageable_h2d_via_cuda,
        _pageable_h2d_via_copy_,
        _pageable_d2h_via_copy_,
        _noncontiguous_d2h_via_copy_,
    ],
)
@create_new_process_for_each_test()
def test_implicit_copy_sync_detected(monkeypatch, mode, fn):
    """`non_blocking=True` CPU<->CUDA copies with a pageable or
    non-densely-laid-out CPU tensor may block the host without tripping
    torch's sync debug mode; the `Tensor.to`/`cuda`/`copy_` wrappers must
    flag them.
    """
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", mode)
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)
    gsd._install_copy_checkers()

    if mode == "error":
        with pytest.raises(RuntimeError, match="Implicit GPU<->CPU sync"):
            with_gpu_sync_check(fn)()
    else:
        with pytest.warns(UserWarning, match="Implicit GPU<->CPU sync"):
            with_gpu_sync_check(fn)()


@create_new_process_for_each_test()
def test_genuinely_async_transfers_pass(monkeypatch):
    """Pinned, densely laid out CPU tensors make `non_blocking=True` truly
    asynchronous in both directions -- even with a dtype conversion, which
    runs GPU-side, and even permuted (e.g. transposed), which uses pitched
    cudaMemcpy2D/3DAsync; D2H `Tensor.to` allocates a pinned destination.
    None of these may be flagged."""
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)
    gsd._install_copy_checkers()

    for fn in (
        _pinned_h2d_nonblocking,
        _pinned_d2h_via_copy_,
        _d2h_via_to,
        _dtype_converting_h2d_nonblocking,
        _dtype_converting_d2h_via_copy_,
        _transposed_h2d_nonblocking,
        _transposed_d2h_via_copy_,
        _empty_h2d_nonblocking,
    ):
        fn()  # Warm up outside the checked region.
        with_gpu_sync_check(fn)()


@create_new_process_for_each_test()
def test_implicit_copy_sync_can_be_allowed(monkeypatch):
    """`gpu_sync_allowed()` must exempt implicit copy syncs too."""
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)
    gsd._install_copy_checkers()

    def allowed_pageable_copies():
        with gpu_sync_allowed():
            _pageable_h2d_nonblocking()
            _pageable_d2h_via_copy_()

    with_gpu_sync_check(allowed_pageable_copies)()
