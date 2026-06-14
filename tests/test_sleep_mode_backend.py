# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the sleep-mode backend abstraction (RFC #34303).

These cover the registry/factory contract and capability flags. They do not
touch CUDA - the ``cumem`` suspend/resume path is exercised end-to-end on GPU
in ``tests/basic_correctness/test_cumem.py``.
"""

import pytest

from vllm.device_allocator.sleep_mode_backend import (
    CuMemBackend,
    CuMemTagBackend,
    SleepModeBackend,
    SleepModeBackendFactory,
)


def test_cumem_is_the_default_registered_backend():
    backend_cls = SleepModeBackendFactory.get_backend_class("cumem")
    assert backend_cls is CuMemBackend
    assert issubclass(backend_cls, SleepModeBackend)


def test_cumem_capability_flags():
    # cumem leaves NCCL untouched but does not preserve compiled artifacts,
    # graphs, or durable state - these flags are what the executor and /health
    # introspect to decide reinit / persistence behavior.
    assert CuMemBackend.is_supported() is True
    assert CuMemBackend.preserves_nccl() is True
    assert CuMemBackend.preserves_compiled_artifacts() is False
    assert CuMemBackend.preserves_graphs_with_nccl() is False
    assert CuMemBackend.supports_durable_storage() is False


def test_new_backend_starts_in_running_state():
    # Constructing a backend must not touch the GPU; only suspend/resume do.
    assert CuMemBackend().state() == "RUNNING"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unsupported sleep-mode backend"):
        SleepModeBackendFactory.get_backend_class("does-not-exist")


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        SleepModeBackendFactory.register_backend(
            "cumem",
            "vllm.device_allocator.sleep_mode_backend",
            "CuMemBackend",
        )


def test_third_party_backend_registration_and_resolution():
    """A plugin registers a backend by name; the factory resolves it lazily."""
    name = "_pytest_dummy_backend"
    try:
        SleepModeBackendFactory.register_backend(
            name,
            "tests.test_sleep_mode_backend",
            "DummyBackend",
        )
        resolved = SleepModeBackendFactory.get_backend_class(name)
        assert resolved is DummyBackend
        assert resolved.supports_durable_storage() is True
    finally:
        SleepModeBackendFactory._registry.pop(name, None)


def test_suspend_resume_state_transitions():
    """Lifecycle state advances RUNNING -> SUSPENDED -> RUNNING without GPU."""
    backend = DummyBackend()
    assert backend.state() == "RUNNING"
    backend.suspend(level=1)
    assert backend.state() == "SUSPENDED"
    backend.resume()
    assert backend.state() == "RUNNING"


class DummyBackend(SleepModeBackend):
    """A no-GPU backend used to exercise lifecycle + registration in CPU tests."""

    def suspend(self, level: int = 1, tags: tuple[str, ...] | None = None) -> None:
        del tags  # accepted to match the abstract signature; ignored here
        self._state = "SUSPENDED"

    def resume(self, tags: list[str] | None = None) -> None:
        self._state = "RUNNING"

    @classmethod
    def supports_durable_storage(cls) -> bool:
        return True


# ---------- CuMemTagBackend (tag-based selective offload) -------------------


def test_cumem_tag_is_registered():
    backend_cls = SleepModeBackendFactory.get_backend_class("cumem_tag")
    assert backend_cls is CuMemTagBackend
    assert issubclass(backend_cls, CuMemBackend)


def test_cumem_tag_capability_flags():
    # Inherits CuMemBackend's flags - same NCCL handling, same lack of
    # compiled-artifact / graph-with-nccl / durable-storage preservation -
    # and adds the selective-offload opt-in.
    assert CuMemTagBackend.is_supported() is True
    assert CuMemTagBackend.preserves_nccl() is True
    assert CuMemTagBackend.preserves_compiled_artifacts() is False
    assert CuMemTagBackend.preserves_graphs_with_nccl() is False
    assert CuMemTagBackend.supports_durable_storage() is False
    assert CuMemTagBackend.supports_selective_offload() is True


def test_default_suspend_tags_match_cumem_backend_level1():
    # CuMemTagBackend with no override must behave identically to
    # CuMemBackend at level 1: offload only the "weights" tag.
    assert CuMemTagBackend.DEFAULT_SUSPEND_TAGS_L1 == ("weights",)
    # Level 2 default is empty - matches CuMemBackend, which passes ``()`` to
    # the allocator when level != 1 (weights are discarded, not offloaded).
    assert CuMemTagBackend.DEFAULT_SUSPEND_TAGS_L2 == ()


def test_explicit_tags_override_defaults():
    # Construction-time tag override changes the *effective* tag set returned
    # by ``effective_suspend_tags()`` - the public contract callers depend on.
    # We assert on that public surface, not on the underlying attribute, so
    # this test stays valid if the storage layout changes.
    backend = CuMemTagBackend(suspend_tags=("weights", "kv_cache"))
    assert backend.effective_suspend_tags(level=1) == ("weights", "kv_cache")
    # Override beats the level-based default at level 2 too: the explicit
    # set is applied verbatim regardless of level.
    assert backend.effective_suspend_tags(level=2) == ("weights", "kv_cache")
    # The public ``suspend_tags`` attribute exposes the construction-time
    # override (None when unset).
    assert backend.suspend_tags == ("weights", "kv_cache")
    # The no-argument constructor leaves the override unset and therefore
    # falls back to the level-based defaults.
    default_backend = CuMemTagBackend()
    assert default_backend.suspend_tags is None
    assert default_backend.effective_suspend_tags(level=1) == ("weights",)
    assert default_backend.effective_suspend_tags(level=2) == ()


def test_unregister_removes_backend():
    """Plugin-author cleanup: unregister() drops a backend from the registry."""
    name = "_pytest_unregister_target"
    SleepModeBackendFactory.register_backend(
        name,
        "tests.test_sleep_mode_backend",
        "DummyBackend",
    )
    assert SleepModeBackendFactory.get_backend_class(name) is DummyBackend
    SleepModeBackendFactory.unregister(name)
    with pytest.raises(ValueError, match="Unsupported sleep-mode backend"):
        SleepModeBackendFactory.get_backend_class(name)


def test_unregister_idempotent_on_missing():
    """unregister() on a never-registered name is a no-op, not an error."""
    SleepModeBackendFactory.unregister("_pytest_never_registered")  # no raise


def test_factory_plumbs_backend_options_dict():
    """``sleep_mode_backend_options`` is ``**``-unpacked into the backend ctor."""

    class _Cfg:
        sleep_mode_backend = "cumem_tag"
        sleep_mode_backend_options = {"suspend_tags": ("weights", "kv_cache")}

    backend = SleepModeBackendFactory.create_backend(_Cfg())
    assert isinstance(backend, CuMemTagBackend)
    assert backend.suspend_tags == ("weights", "kv_cache")


def test_factory_empty_options_preserves_default_behavior():
    """Empty options dict yields a default-constructed backend - no behavior
    change for callers that don't opt into backend-specific tuning."""

    class _Cfg:
        sleep_mode_backend = "cumem_tag"
        sleep_mode_backend_options = {}

    backend = SleepModeBackendFactory.create_backend(_Cfg())
    assert isinstance(backend, CuMemTagBackend)
    assert backend.suspend_tags is None


def test_supports_selective_offload_base_class_default():
    # Base class default is False; backends that can't do selective offload
    # (e.g. an eventual cuda_checkpoint backend, which is all-or-nothing)
    # inherit it. CuMemBackend, the existing default, also inherits False.
    assert SleepModeBackend.supports_selective_offload() is False
    assert CuMemBackend.supports_selective_offload() is False


# ---------- CuMemTagBackend safety + plumbing (no GPU) ----------------------


def test_suspend_tags_list_input_normalizes_to_tuple():
    """matteso1's review nit: list-from-CLI/JSON inputs must round-trip to
    tuple so ``effective_suspend_tags()`` return-type stays stable regardless
    of config source. ``None`` sentinel is preserved."""
    backend = CuMemTagBackend(suspend_tags=["weights", "kv_cache"])
    assert backend.suspend_tags == ("weights", "kv_cache")
    assert isinstance(backend.suspend_tags, tuple)
    assert backend.effective_suspend_tags() == ("weights", "kv_cache")
    # None stays None - it's the "no construction-time override" sentinel,
    # not an empty tuple.
    assert CuMemTagBackend(suspend_tags=None).suspend_tags is None


def test_factory_normalizes_list_options_to_tuple():
    """The ``sleep_mode_backend_options`` dict typically arrives as JSON/YAML,
    so the value comes in as a list. The factory + ctor must accept that and
    produce the same shape as the in-process tuple form."""

    class _Cfg:
        sleep_mode_backend = "cumem_tag"
        sleep_mode_backend_options = {"suspend_tags": ["weights", "kv_cache"]}

    backend = SleepModeBackendFactory.create_backend(_Cfg())
    assert backend.suspend_tags == ("weights", "kv_cache")
    assert isinstance(backend.suspend_tags, tuple)


def test_suspended_tags_default_returns_none():
    """Backends without per-tag suspend tracking return None. Lets the GPU
    worker fall back to the pre-abstraction wake_up behavior (always run
    post_kv_cache_wake_up)."""
    backend = CuMemBackend()
    assert backend.suspended_tags() is None


# ---- The next group exercises the suspend/resume bookkeeping at the
# ---- Python level by stubbing the CuMemAllocator. They do NOT touch CUDA;
# ---- they prove the safety logic (KV-cache zero guard, L2 weight reload
# ---- guard) runs the way the docstring claims, which is what the morning
# ---- L2-AWQ smoke broke on.


class _FakeAllocator:
    """Stand-in for ``CuMemAllocator``. Records calls for assertion."""

    def __init__(self):
        self.sleep_calls: list[tuple] = []
        self.wake_calls: list = []

    def sleep(self, offload_tags=()):
        self.sleep_calls.append(tuple(offload_tags))

    def wake_up(self, tags=None):
        self.wake_calls.append(tuple(tags) if tags else None)


def _patch_allocator(monkeypatch):
    """Replace ``get_mem_allocator_instance`` with a fake; return the fake."""
    fake = _FakeAllocator()
    import vllm.device_allocator as alloc_pkg

    monkeypatch.setattr(alloc_pkg, "get_mem_allocator_instance", lambda: fake)
    return fake


def test_per_call_tags_override_reaches_allocator(monkeypatch):
    """Finding 1 from the audit: per-call ``tags=`` must propagate through
    to ``allocator.sleep(offload_tags=...)``. Previously the kwarg lived on
    the backend but no caller threaded it down, leaving the per-call
    override entirely unreachable."""
    fake = _patch_allocator(monkeypatch)
    backend = CuMemTagBackend()  # no construction-time override
    backend.suspend(level=1, tags=("weights", "kv_cache"))
    assert fake.sleep_calls == [("weights", "kv_cache")]
    # Bookkeeping is set so the wake-up guard knows what was suspended.
    assert backend.suspended_tags() == ("weights", "kv_cache")


def test_selective_suspend_then_full_wake_does_not_widen_suspend_set(
    monkeypatch,
):
    """Finding 2 from the audit: wake_up(tags=None) after a selective
    suspend(weights only) must not re-init the KV cache - those pages were
    never offloaded and re-running ``post_kv_cache_wake_up`` would zero
    live state.

    Backend-level evidence: after the selective suspend, ``suspended_tags()``
    reports only ("weights",). The worker uses that to skip the KV re-init
    when "kv_cache" is not in the set (see gpu_worker.wake_up). The resume
    itself succeeds without raising."""
    _patch_allocator(monkeypatch)
    backend = CuMemTagBackend()
    backend.suspend(level=1, tags=("weights",))
    assert backend.suspended_tags() == ("weights",)
    # wake_up(None) clamps to the recorded suspended set, preserving KV.
    backend.resume(tags=None)
    assert backend.state() == "RUNNING"
    assert backend.suspended_tags() is None


def test_resume_rejects_wake_of_non_suspended_tag(monkeypatch):
    """Finding 2 (defensive): explicitly waking a tag that was never
    suspended is loud, not silent. The GPU worker path treats kv_cache wake
    as a re-init that overwrites live state, so this check has to fire
    *before* the wake_up reaches the allocator."""
    _patch_allocator(monkeypatch)
    backend = CuMemTagBackend()
    backend.suspend(level=1, tags=("weights",))
    with pytest.raises(ValueError, match="not suspended"):
        backend.resume(tags=["kv_cache"])
    # Backend state is left in RESUMING (failed mid-resume); bookkeeping is
    # untouched so the caller can retry with a corrected tag set.
    assert backend.suspended_tags() == ("weights",)


def test_resume_l2_weights_raises(monkeypatch):
    """Finding 3 from the audit: L2 suspend discards the allocator pages -
    reloading weights is the *worker's* responsibility
    (``model_runner.reload_weights``), not the allocator's
    ``wake_up``. Waking L2-suspended weights through this path silently
    returns a "RUNNING" backend with garbage pages (HTTP 200 healthy, model
    emits garbage tokens - the AWQ smoke this morning).

    Until the worker-side reload is wired into this dispatch path, refuse
    the call so the failure surfaces at the API boundary."""
    _patch_allocator(monkeypatch)
    backend = CuMemTagBackend()
    backend.suspend(level=2, tags=("weights",))
    with pytest.raises(RuntimeError, match="suspend\\(level=2\\)"):
        backend.resume(tags=["weights"])
    # Bookkeeping intact so the executor can run the reload flow then retry.
    assert backend.suspended_tags() == ("weights",)


def test_resume_l1_weights_succeeds(monkeypatch):
    """L1 suspend keeps the bytes on the host; the allocator can put them
    back. This is the happy path that the L2 guard above must NOT block."""
    fake = _patch_allocator(monkeypatch)
    backend = CuMemTagBackend()
    backend.suspend(level=1, tags=("weights",))
    backend.resume(tags=["weights"])
    assert backend.state() == "RUNNING"
    assert fake.wake_calls == [("weights",)]
    assert backend.suspended_tags() is None


def test_partial_resume_clears_only_resumed_tags(monkeypatch):
    """Multi-tag suspend with a partial wake: only the resumed tags clear
    from the bookkeeping; the remaining tag stays in ``suspended_tags()``
    so a later check still knows it's offloaded."""
    _patch_allocator(monkeypatch)
    backend = CuMemTagBackend()
    backend.suspend(level=1, tags=("weights", "kv_cache"))
    backend.resume(tags=["weights"])
    assert backend.suspended_tags() == ("kv_cache",)
    backend.resume(tags=["kv_cache"])
    assert backend.suspended_tags() is None


# ---- Round-2 audit fixes: allocator-call clamping + bookkeeping atomicity --


def test_full_wake_clamps_allocator_call_to_suspended_set(monkeypatch):
    """Round-2 HIGH (a): ``resume(tags=None)`` after a selective suspend
    must NOT pass ``None`` through to ``allocator.wake_up``. The pre-fix
    code computed a clamped ``requested_set`` for its own validation but
    then handed the original (``None``) ``tags`` to the allocator, so a
    ``wake_up(None)`` call could wake more pages than were suspended -
    silently zeroing live GPU state owned by another caller in the same
    process (or by a prior selective suspend that this resume should not
    have touched). The allocator call must see the same clamped tag set
    the validation gate just approved."""
    fake = _patch_allocator(monkeypatch)
    backend = CuMemTagBackend()
    backend.suspend(level=1, tags=("weights",))
    backend.resume(tags=None)
    # Allocator was called with the clamped suspended set, NOT ``None``.
    assert fake.wake_calls == [("weights",)]
    assert backend.state() == "RUNNING"
    assert backend.suspended_tags() is None


def test_full_wake_clamps_allocator_call_with_multi_tag_suspend(monkeypatch):
    """Same clamping property holds when multiple tags are suspended:
    ``resume(tags=None)`` widens to the full suspended set but never
    beyond it. Order matches the suspend-time order so allocator-side
    iteration stays deterministic."""
    fake = _patch_allocator(monkeypatch)
    backend = CuMemTagBackend()
    backend.suspend(level=1, tags=("weights", "kv_cache"))
    backend.resume(tags=None)
    assert fake.wake_calls == [("weights", "kv_cache")]
    assert backend.suspended_tags() is None


def test_partial_wake_passes_clamped_tuple_not_caller_list(monkeypatch):
    """Selective ``resume(tags=[...])`` should pass a tuple to the
    allocator, derived from the validated ``requested_set``, not the
    caller's mutable list. This pins the allocator-call shape so future
    refactors don't accidentally route the raw list through."""
    fake = _patch_allocator(monkeypatch)
    backend = CuMemTagBackend()
    backend.suspend(level=1, tags=("weights", "kv_cache"))
    caller_list = ["weights"]
    backend.resume(tags=caller_list)
    # Recorded call shape is a tuple, not the original list.
    assert fake.wake_calls == [("weights",)]
    assert isinstance(fake.wake_calls[0], tuple)
    # Mutating the caller's list afterward must not retroactively change
    # the recorded allocator call (defensive copy via tuple).
    caller_list.append("kv_cache")
    assert fake.wake_calls == [("weights",)]


def test_suspend_bookkeeping_is_post_allocator_atomic(monkeypatch):
    """Round-2 HIGH (b): if ``allocator.sleep`` raises, the backend must
    NOT be left in a phantom SUSPENDED state with bookkeeping recording
    tags that were never actually offloaded. The pre-fix code wrote
    ``self._state = "SUSPENDED"`` and ``self._suspended_tags = ...``
    BEFORE calling the allocator - so an allocator-side OOM (the cumem
    L2/AWQ failure mode in our morning smoke) would leave the backend
    claiming to be suspended while the GPU pages were still live, and a
    subsequent ``resume()`` would then try to wake pages that were never
    sleep-prepared, masking the real failure with a confusing wake-up
    error.

    With the fix, allocator failure leaves the backend RUNNING and
    bookkeeping clean - matching the actual GPU state."""

    class _RaisingAllocator(_FakeAllocator):
        def sleep(self, offload_tags=()):
            self.sleep_calls.append(tuple(offload_tags))
            raise RuntimeError("simulated allocator OOM")

    raising = _RaisingAllocator()
    import vllm.device_allocator as alloc_pkg

    monkeypatch.setattr(
        alloc_pkg, "get_mem_allocator_instance", lambda: raising
    )
    backend = CuMemTagBackend()
    with pytest.raises(RuntimeError, match="simulated allocator OOM"):
        backend.suspend(level=1, tags=("weights",))
    # Backend stayed RUNNING - did NOT advance to SUSPENDED.
    assert backend.state() == "RUNNING"
    # Bookkeeping was NOT written - no phantom suspended set.
    assert backend.suspended_tags() is None
    # Allocator call was attempted (proves we got past the dispatch path
    # and the failure happened in the allocator, not before).
    assert raising.sleep_calls == [("weights",)]


def test_resume_allocator_failure_reverts_to_suspended(monkeypatch):
    """Round-2 HIGH (b) symmetric: if ``allocator.wake_up`` raises during
    a resume, the backend must NOT be left stuck in RESUMING. Either
    transition cleanly back to SUSPENDED (pages still offloaded, retry
    safe) or surface the failure to the caller - never wedge in an
    intermediate state. This is the executor-visible state the morning
    AWQ smoke conflated; pinning it down stops phantom-state loops."""

    class _ResumeFails(_FakeAllocator):
        def wake_up(self, tags=None):
            self.wake_calls.append(tuple(tags) if tags else None)
            raise RuntimeError("simulated wake_up OOM")

    fail = _ResumeFails()
    import vllm.device_allocator as alloc_pkg

    monkeypatch.setattr(
        alloc_pkg, "get_mem_allocator_instance", lambda: fail
    )
    backend = CuMemTagBackend()
    # Bookkeeping path: drive a successful suspend through a working
    # allocator first, then swap to the failing one for resume so we
    # exercise the resume-time failure isolated from the suspend.
    successful = _FakeAllocator()
    monkeypatch.setattr(
        alloc_pkg, "get_mem_allocator_instance", lambda: successful
    )
    backend.suspend(level=1, tags=("weights",))
    monkeypatch.setattr(
        alloc_pkg, "get_mem_allocator_instance", lambda: fail
    )
    with pytest.raises(RuntimeError, match="simulated wake_up OOM"):
        backend.resume(tags=None)
    # Backend reverted to SUSPENDED so the executor sees a state matching
    # the actual GPU situation (pages still offloaded). Bookkeeping
    # preserved so the caller can retry without re-suspending.
    assert backend.state() == "SUSPENDED"
    assert backend.suspended_tags() == ("weights",)


def test_validation_failure_leaves_state_unchanged(monkeypatch):
    """Validation-time errors (spurious tag, L2-weights wake) must NOT
    advance the backend to RESUMING. The pre-fix code set ``RESUMING``
    before validating, leaving the backend wedged in a transitional
    state that no other path could clear. With the fix, the backend
    stays SUSPENDED and bookkeeping is preserved for retry."""
    _patch_allocator(monkeypatch)
    backend = CuMemTagBackend()
    backend.suspend(level=1, tags=("weights",))
    assert backend.state() == "SUSPENDED"
    with pytest.raises(ValueError, match="not suspended"):
        backend.resume(tags=["kv_cache"])
    # State did NOT advance to RESUMING - validation gate fired before
    # any state mutation, so the backend stays cleanly SUSPENDED.
    assert backend.state() == "SUSPENDED"
    assert backend.suspended_tags() == ("weights",)
