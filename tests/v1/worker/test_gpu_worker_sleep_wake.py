# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Worker.wake_up buffer restore tag-gating (level-2 sleep).

These reproduce the partial-wake hazard where the level-2 sleep
buffer-restore loop in ``Worker.wake_up`` fired regardless of ``tags``.
Model buffers are allocated under the ``"weights"`` tag (see
``Worker.load_model``), so their CUDA virtual addresses are only re-mapped
once the allocator wakes that tag. A partial wake that does not include
``"weights"`` (e.g. ``wake_up(["kv_cache"])``) must NOT copy into those
still-unmapped VAs, which on a real device raises
CUDA_ERROR_ILLEGAL_ADDRESS.

The real ``Worker.wake_up`` source is loaded via AST extraction and
exec'd into an isolated namespace so the genuine code under test runs
without vLLM's heavy (GPU/CUDA) import chain. CPU-only; no torch needed.
"""

import ast
import os

import pytest


def _find_gpu_worker_src():
    """Locate the real gpu_worker.py source.

    Walks up from this test file looking for vllm/v1/worker/gpu_worker.py so
    the test reads the genuine source regardless of CWD. Honors the
    GPU_WORKER_SRC env var as an override (used when the test file is run
    from a copied location outside the repo tree).
    """
    override = os.environ.get("GPU_WORKER_SRC")
    if override and os.path.exists(override):
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    rel = os.path.join("vllm", "v1", "worker", "gpu_worker.py")
    d = here
    for _ in range(8):
        cand = os.path.join(d, rel)
        if os.path.exists(cand):
            return cand
        d = os.path.dirname(d)
    raise FileNotFoundError("could not locate vllm/v1/worker/gpu_worker.py")


def _load_real_wake_up():
    """Extract and compile the real Worker.wake_up method from source.

    Binds ``get_mem_allocator_instance`` into the function's globals so the
    executed code is byte-identical to what ships in gpu_worker.py.
    """
    with open(os.path.normpath(_find_gpu_worker_src())) as f:
        src = f.read()
    mod = ast.parse(src)
    cls = next(
        n
        for n in mod.body
        if isinstance(n, ast.ClassDef) and n.name == "Worker"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "wake_up"
    )
    seg = ast.get_source_segment(src, fn)
    ns: dict = {"get_mem_allocator_instance": lambda: _ACTIVE_ALLOCATOR[0]}
    exec(seg, ns)
    return ns["wake_up"]


# Test-scoped handle the loaded wake_up resolves get_mem_allocator_instance to.
_ACTIVE_ALLOCATOR: list = [None]
wake_up = _load_real_wake_up()


class _FakeAllocator:
    """Tracks which tags are currently mapped (woken)."""

    def __init__(self):
        # After a level-2 sleep, no tag is mapped.
        self.mapped_tags: set[str] = set()

    def wake_up(self, tags):
        if tags is None:
            self.mapped_tags = {"weights", "kv_cache"}
        else:
            self.mapped_tags.update(tags)


class _GuardedBuffer:
    """Stands in for a model buffer living under a given allocator tag.

    ``data.copy_`` simulates a device copy: if the buffer's backing tag is
    not currently mapped, the copy targets an unmapped virtual address,
    which on real CUDA raises an illegal-address error. We surface that as
    a RuntimeError so the test asserts it never happens on a partial wake
    that excludes the buffer's tag.
    """

    def __init__(self, tag: str, allocator: _FakeAllocator):
        self._tag = tag
        self._allocator = allocator
        self.restored_from: object | None = None
        self.data = self  # so buffer.data.copy_(...) exercises the guard

    def copy_(self, src):
        if self._tag not in self._allocator.mapped_tags:
            raise RuntimeError(
                f"copy into unmapped VA for tag {self._tag!r} "
                "(simulated CUDA_ERROR_ILLEGAL_ADDRESS)"
            )
        self.restored_from = src


class _Saved:
    """Minimal stand-in for a saved CPU buffer with a ``.data`` attribute."""

    def __init__(self, token):
        self.data = token


class _Worker:
    """Minimal duck-typed Worker for the extracted wake_up to operate on."""

    def __init__(self, allocator, buffers):
        self._sleep_saved_buffers = {}
        self.post_calls = []
        self.model_runner = type(
            "MR",
            (),
            {
                "model": type(
                    "M",
                    (),
                    {"named_buffers": staticmethod(lambda: list(buffers.items()))},
                )(),
                "post_kv_cache_wake_up": lambda _self: self.post_calls.append(True),
            },
        )()


def _setup(allocator, buffers):
    _ACTIVE_ALLOCATOR[0] = allocator
    return _Worker(allocator, buffers)


def test_partial_wake_kv_cache_does_not_restore_weight_buffers():
    """wake_up(["kv_cache"]) must NOT touch weight-tagged buffers.

    Pre-fix: the restore loop fires unconditionally and copies into the
    still-unmapped "weights" VA -> RuntimeError. Buffers must stay pending.
    """
    allocator = _FakeAllocator()
    buf = _GuardedBuffer("weights", allocator)
    worker = _setup(allocator, {"rotary.cos": buf})
    worker._sleep_saved_buffers = {"rotary.cos": _Saved("orig")}

    wake_up(worker, ["kv_cache"])  # weights stay unmapped

    assert buf.restored_from is None  # no copy into unmapped weights VA
    assert "rotary.cos" in worker._sleep_saved_buffers  # still pending


def test_subsequent_weights_wake_restores_buffers_bit_for_bit():
    """After kv_cache wake, a following weights wake restores buffers."""
    allocator = _FakeAllocator()
    src = _Saved("bits")
    buf = _GuardedBuffer("weights", allocator)
    worker = _setup(allocator, {"rotary.cos": buf})
    worker._sleep_saved_buffers = {"rotary.cos": src}

    wake_up(worker, ["kv_cache"])  # weights still pending
    assert buf.restored_from is None
    assert "rotary.cos" in worker._sleep_saved_buffers

    wake_up(worker, ["weights"])  # now weights are mapped
    assert buf.restored_from == "bits"  # restored bit-for-bit
    assert worker._sleep_saved_buffers == {}  # cleared only after restore


def test_full_wake_restores_buffers_and_runs_post_kv_cache():
    """wake_up(None) (full wake) restores buffers and runs post_kv_cache."""
    allocator = _FakeAllocator()
    src = _Saved("bits")
    buf = _GuardedBuffer("weights", allocator)
    worker = _setup(allocator, {"rotary.cos": buf})
    worker._sleep_saved_buffers = {"rotary.cos": src}

    wake_up(worker, None)

    assert buf.restored_from == "bits"
    assert worker._sleep_saved_buffers == {}
    assert worker.post_calls == [True]


def test_partial_weights_wake_restores_buffers_no_post_kv_cache():
    """wake_up(["weights"]) restores buffers but does NOT run post_kv_cache."""
    allocator = _FakeAllocator()
    src = _Saved("bits")
    buf = _GuardedBuffer("weights", allocator)
    worker = _setup(allocator, {"rotary.cos": buf})
    worker._sleep_saved_buffers = {"rotary.cos": src}

    wake_up(worker, ["weights"])

    assert buf.restored_from == "bits"
    assert worker._sleep_saved_buffers == {}
    assert worker.post_calls == []
