# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache.v1.platform.cpu.shm``.

Validates that the POSIX-SHM-backed wrapper can round-trip a CPU
tensor in-process: the constructed wrapper's ``to_tensor()`` view
sees writes made through the original tensor.
"""

# Standard
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.posix_shm import shm_unlink
from lmcache.v1.platform.cpu.shm import (
    CpuShmTensorWrapper,
    migrate_to_shm_and_wrap,
    shm_create_readwrite,
)


def test_shm_create_unlink_roundtrip():
    """``shm_create_readwrite`` succeeds and ``shm_unlink`` cleans up."""
    name = "/lmcache_test_%d" % os.getpid()
    addr = shm_create_readwrite(name, 4096)
    try:
        assert addr not in (0, None)
    finally:
        shm_unlink(name)


def test_migrate_to_shm_and_wrap_zero_copy_view():
    """After migrate, writes via the original tensor are visible via wrapper."""
    src = torch.zeros((2, 4, 4), dtype=torch.float32)
    wrapper = migrate_to_shm_and_wrap(src)
    try:
        assert isinstance(wrapper, CpuShmTensorWrapper)
        assert wrapper.shape == (2, 4, 4)
        assert wrapper.dtype == torch.float32
        # Mutate via the migrated source tensor; its storage is now the
        # SHM segment, so the wrapper's reconstructed view must see it.
        src.add_(7.0)
        view = wrapper.to_tensor()
        assert torch.equal(view, src)
    finally:
        shm_unlink(wrapper.shm_name)


def test_migrate_normalizes_layout_in_place_on_caller_tensor():
    """Layout normalization must not detach migration from the caller's tensor.

    ``attempt_permute_to_contiguous_view`` returns a *new view* object; the
    migration must still re-point the caller's tensor (its storage, and its
    shape/stride to the stride-truthful layout) and key the unlink finalizer
    on it. Regression for the premature-unlink bug where the finalizer was
    attached to the temporary view and the segment vanished on return.
    """
    # Torch-contiguous, but the size-1 dims hide the true inner extent:
    # the stride-truthful layout is (8, 32, 1, 1).
    src = torch.zeros(8 * 32, dtype=torch.float32).as_strided(
        (8, 1, 1, 32), (32, 1, 1, 1)
    )
    wrapper = migrate_to_shm_and_wrap(src)
    try:
        assert tuple(src.shape) == (8, 32, 1, 1)
        assert wrapper.shape == (8, 32, 1, 1)
        # Writes via the caller's tensor land in the SHM segment.
        src.add_(3.0)
        view = wrapper.to_tensor()
        assert torch.equal(view, src)
    finally:
        shm_unlink(wrapper.shm_name)


def test_migrate_handles_empty_tensor():
    """Empty tensors must not call ``mmap`` (length 0 is EINVAL).

    Regression for the case where ``nbytes == 0``: the wrapper carries
    an empty ``shm_name`` and ``to_tensor`` rebuilds the empty view in
    process without touching POSIX shared memory.
    """
    src = torch.empty((0, 4), dtype=torch.float32)
    wrapper = migrate_to_shm_and_wrap(src)
    assert isinstance(wrapper, CpuShmTensorWrapper)
    assert wrapper.shm_name == ""
    assert wrapper.nbytes == 0
    view = wrapper.to_tensor()
    assert view.shape == (0, 4)
    assert view.dtype == torch.float32


def test_migrate_is_idempotent_on_same_tensor():
    """Re-wrapping the same tensor reuses the existing SHM segment."""
    src = torch.zeros((3, 5), dtype=torch.float32)
    w1 = migrate_to_shm_and_wrap(src)
    try:
        w2 = migrate_to_shm_and_wrap(src)
        assert w1.shm_name == w2.shm_name
    finally:
        shm_unlink(w1.shm_name)


def test_rejects_non_cpu_tensor():
    """Construction rejects tensors that are not on CPU."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available; cannot synthesize a non-cpu tensor")
    src = torch.zeros((2, 2), device="mps")
    with pytest.raises(ValueError, match="CPU tensor"):
        CpuShmTensorWrapper(src, "/lmcache_test_should_not_exist")


def test_migrate_finalizer_unlinks_on_gc():
    """Once the migrated tensor is GC-ed, its SHM segment is unlinked."""
    # Standard
    import gc

    # First Party
    from lmcache.v1.platform.cpu.shm import shm_map_readwrite

    src = torch.zeros((2, 2), dtype=torch.float32)
    w = migrate_to_shm_and_wrap(src)
    name = w.shm_name
    nbytes = w.nbytes
    # Drop both references; the weakref.finalize hook should unlink.
    del src, w
    gc.collect()
    with pytest.raises(OSError):
        shm_map_readwrite(name, nbytes)


def test_shm_create_cleans_up_on_existing_name():
    """If ``shm_open(O_EXCL)`` fails the helper must not leave the fd open.

    We exercise the failure path by creating a segment, then asking
    ``shm_create_readwrite`` to recreate the same name -- it must
    raise without leaking the file descriptor it briefly held.
    """
    name = "/lmcache_test_excl_%d" % os.getpid()
    addr = shm_create_readwrite(name, 4096)
    try:
        with pytest.raises(OSError):
            shm_create_readwrite(name, 4096)
    finally:
        shm_unlink(name)
    # And after unlink, the name is reusable again.
    addr2 = shm_create_readwrite(name, 4096)
    assert addr2 not in (0, None)
    shm_unlink(name)
    _ = addr  # silence unused-variable hint


def test_to_tensor_view_carries_munmap_finalizer():
    """``to_tensor`` returns a tensor whose storage releases its mmap on GC."""
    # Standard
    import gc
    import weakref

    src = torch.zeros((2, 2), dtype=torch.float32)
    w = migrate_to_shm_and_wrap(src)
    try:
        view = w.to_tensor()
        ref = weakref.ref(view)
        del view
        gc.collect()
        assert ref() is None
    finally:
        del src
        gc.collect()
        shm_unlink(w.shm_name)


def test_to_tensor_view_survives_reshape_after_original_is_gced():
    """A reshape view must keep working after the original tensor is GC-ed.

    Regression for the CI ``cpu_e2e_validation (server-side copy)`` segfault
    where ``normalize_kv_and_discover_format`` reshapes the unwrapped 4D
    tensor into the canonical 5D ``[NB, NH, BS, 2, HS]`` and the original
    4D tensor goes out of scope. The reshape view shares the same storage
    but used to be left without a lifetime hook, so the next read after GC
    landed on an already-``munmap``-ed page and crashed the LMCache server.
    """
    # Standard
    import gc

    NB, NH, BS, HS = 8, 4, 16, 8
    src = torch.zeros((NB, NH, BS, 2 * HS), dtype=torch.bfloat16)
    w = migrate_to_shm_and_wrap(src)
    try:
        unwrapped = [w.to_tensor()]
        normalized = [
            layer.reshape(*layer.shape[:3], 2, layer.shape[3] // 2)
            for layer in unwrapped
        ]
        # Drop every reference to the 4D unwrapped tensor; only the 5D
        # reshape view keeps the storage alive now.
        del unwrapped
        gc.collect()

        # These ops would segfault before the fix because the SHM mapping
        # had been munmap-ed when the 4D tensor's finalizer ran.
        idx = torch.tensor([0, 3, 5], dtype=torch.long)
        gathered = normalized[0].index_select(0, idx)
        assert tuple(gathered.shape) == (3, NH, BS, 2, HS)
        # The SHM segment is freshly mmap'd zeros; just make sure the
        # bytes are addressable end-to-end so the kernel does not fault.
        assert gathered.float().abs().sum().item() == 0.0
    finally:
        del src
        gc.collect()
        shm_unlink(w.shm_name)


def test_to_tensor_replays_stride_and_storage_offset():
    """``to_tensor`` rebuilds the view via stride+offset (not reshape)."""
    src = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).contiguous()
    w = migrate_to_shm_and_wrap(src)
    try:
        view = w.to_tensor()
        assert tuple(view.stride()) == w.stride
        assert int(view.storage_offset()) == w.storage_offset
        assert torch.equal(view, src)
    finally:
        del src, view
        shm_unlink(w.shm_name)


def test_wrap_kv_caches_unlinks_partial_batch_on_failure(monkeypatch):
    """If wrapping the N-th tensor raises, earlier SHM names are unlinked.

    Drives :func:`wrap_kv_caches` with two CPU tensors and forces the
    second factory call to raise; the first iteration's SHM segment
    must be ``shm_unlink``-ed so the named segment does not outlive
    the failed batch.
    """
    # First Party
    from lmcache.v1.platform import kv_wrap
    from lmcache.v1.platform.cpu.shm import shm_map_readwrite

    real_wrap = kv_wrap.wrap_one_kv_cache
    state = {"n": 0, "first_name": None}

    def flaky_wrap(tensor):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("simulated migration failure")
        w = real_wrap(tensor)
        state["first_name"] = w.shm_name
        return w

    monkeypatch.setattr(kv_wrap, "wrap_one_kv_cache", flaky_wrap)

    t1 = torch.zeros((2, 2), dtype=torch.float32)
    t2 = torch.zeros((2, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="simulated migration failure"):
        kv_wrap.wrap_kv_caches({"a": t1, "b": t2})

    # The first iteration's SHM segment must no longer be openable.
    nbytes = t1.numel() * t1.element_size()
    with pytest.raises(OSError):
        shm_map_readwrite(state["first_name"], nbytes)


def test_migrate_ignores_stale_entry_from_id_reuse():
    """A cached entry whose weakref is dead must not be reused.

    Simulates CPython recycling an object id by injecting a stale
    ``(dead_ref, old_name)`` tuple keyed by the live tensor's id,
    then calling :func:`migrate_to_shm_and_wrap`. The factory must
    treat the dead entry as a miss and allocate a fresh SHM segment
    -- if it blindly reused the cached name, ``shm_create_readwrite``
    would crash with ``EEXIST`` (and even worse, the fresh tensor
    would be silently bound to the wrong SHM name).
    """
    # Standard
    import gc
    import weakref as _wr

    # First Party
    from lmcache.v1.platform.cpu.shm import inject_stale_cache_entry_for_test

    # Build a tensor we will let die so we have a guaranteed-dead ref.
    ghost = torch.zeros((1,), dtype=torch.float32)
    dead_ref = _wr.ref(ghost)
    del ghost
    gc.collect()
    assert dead_ref() is None

    live = torch.zeros((2, 2), dtype=torch.float32)
    stale_name = "/lmcache_test_stale_%d" % os.getpid()
    inject_stale_cache_entry_for_test(live, dead_ref, stale_name)

    w = migrate_to_shm_and_wrap(live)
    try:
        assert w.shm_name != stale_name
    finally:
        shm_unlink(w.shm_name)
