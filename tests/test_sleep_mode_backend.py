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

    def suspend(self, level: int = 1) -> None:
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
