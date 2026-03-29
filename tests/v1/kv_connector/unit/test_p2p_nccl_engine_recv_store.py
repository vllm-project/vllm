# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for recv_store memory management in P2pNcclEngine.

Regression tests for #38472: KV caches accumulated in recv_store (VRAM or
pinned TensorMemoryPool RAM) on the decode instance until the request finished,
causing OOM under high QPS with long outputs.

The two root-cause defects were:
  1. recv_tensor() read from recv_store without popping – the entry (and its
     GPU buffer quota) persisted until get_finished() was called at request
     completion.
  2. For pool-backed entries (TensorMemoryPool tuples), pool.free() was only
     called in get_finished(), not immediately after the tensor was loaded back
     to device in recv_tensor().

These tests isolate P2pNcclEngine's recv_store lifecycle logic without
requiring ZMQ, NCCL, real CUDA streams, or a VllmConfig, by constructing a
minimal engine object via object.__new__ and patching only what is needed.

The vllm package normally detects the CUDA platform at import time and loads
compiled CUDA extensions (vllm._C, vllm._C_stable_libtorch).  On CPU-only or
non-GPU CI machines those shared libraries are absent.  We pre-register mock
stubs into sys.modules *before* any vllm import so that the platform detection
code can proceed without the hardware being present.  The actual methods under
test do not call any CUDA extension code; all tensors in these tests are CPU
tensors.
"""

import sys
import threading
import unittest
from unittest.mock import MagicMock, call

import torch

# ---------------------------------------------------------------------------
# Pre-stub compiled CUDA extensions so the module can be imported on machines
# without a CUDA-enabled torch build (CPU CI, dev laptops, etc.).
# These stubs are registered *once* at module load time, before any vllm
# package code is imported by the helper functions below.
# ---------------------------------------------------------------------------
for _stub_name in ("vllm._C", "vllm._C_stable_libtorch"):
    sys.modules.setdefault(_stub_name, MagicMock())

# Import the class under test AFTER the stubs are registered.
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (  # noqa: E402
    P2pNcclEngine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(send_type: str = "PUT_ASYNC"):
    """Return a P2pNcclEngine instance with ONLY the attributes needed to
    exercise recv_tensor() and get_finished(), bypassing the full __init__
    which requires ZMQ sockets, NCCL, CUDA streams, etc.
    """
    engine = object.__new__(P2pNcclEngine)

    # State that recv_tensor / get_finished actually touch.
    engine.send_type = send_type
    engine.recv_store: dict = {}
    engine.recv_store_cv = threading.Condition()
    engine.recv_request_id_to_tensor_ids: dict = {}
    engine.send_request_id_to_tensor_ids: dict = {}
    engine.buffer_size = 0
    engine.buffer_size_threshold = 1e12  # effectively unlimited
    engine.pool = MagicMock()
    engine.rank = 0
    engine.device = torch.device("cpu")

    return engine


def _register_gpu_tensor(
    engine,
    tensor_id: str,
    tensor: torch.Tensor,
) -> None:
    """Simulate the listener thread placing a plain GPU tensor into recv_store
    (the path where tensor fits within buffer_size_threshold)."""
    size = tensor.element_size() * tensor.numel()
    engine.buffer_size += size
    engine.recv_store[tensor_id] = tensor
    req_id = tensor_id.split("#")[0]
    engine.recv_request_id_to_tensor_ids.setdefault(req_id, set()).add(tensor_id)


def _register_pool_tensor(
    engine,
    tensor_id: str,
    addr: int,
    dtype: torch.dtype,
    shape: tuple,
) -> None:
    """Simulate the listener thread placing a pool-backed (addr, dtype, shape)
    tuple into recv_store (the path where the tensor overflowed the GPU
    buffer_size_threshold and was spilled to TensorMemoryPool)."""
    engine.recv_store[tensor_id] = (addr, dtype, shape)
    req_id = tensor_id.split("#")[0]
    engine.recv_request_id_to_tensor_ids.setdefault(req_id, set()).add(tensor_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecvTensorPopsFromStore(unittest.TestCase):
    """recv_tensor must REMOVE the entry from recv_store immediately."""

    def _cpu_tensor(self, *shape):
        # Use CPU tensors for dict-management tests – we only need the bytes
        # count math to hold; we never move them to GPU here.
        return torch.zeros(*shape, dtype=torch.float16)

    def setUp(self):
        self.engine = _make_engine()

    def test_entry_absent_after_recv_tensor(self):
        """After recv_tensor returns, the tensor_id must no longer be in
        recv_store.  This is the core regression guard for #38472."""
        tensor = self._cpu_tensor(4, 8)
        tensor_id = "req-001#layer.0"

        _register_gpu_tensor(self.engine, tensor_id, tensor)
        self.assertIn(tensor_id, self.engine.recv_store)

        self.engine.recv_tensor(tensor_id)

        self.assertNotIn(
            tensor_id,
            self.engine.recv_store,
            "recv_tensor must pop the entry immediately, not leave it for "
            "get_finished() – leaving it caused KV cache accumulation (OOM).",
        )

    def test_buffer_size_decremented_after_recv_tensor(self):
        """buffer_size must be fully returned to zero once the tensor is
        consumed; it should NOT remain inflated until request completion."""
        tensor = self._cpu_tensor(16, 16)
        tensor_id = "req-002#layer.0"
        _register_gpu_tensor(self.engine, tensor_id, tensor)

        expected_size = tensor.element_size() * tensor.numel()
        self.assertEqual(self.engine.buffer_size, expected_size)

        self.engine.recv_tensor(tensor_id)

        self.assertEqual(
            self.engine.buffer_size,
            0,
            "buffer_size must be decremented immediately in recv_tensor, not "
            "deferred to get_finished().",
        )

    def test_multiple_layers_all_popped(self):
        """Each layer tensor for a request must be popped independently."""
        req_id = "req-003"
        num_layers = 8
        tensors = {}
        for i in range(num_layers):
            tid = f"{req_id}#layer.{i}"
            t = self._cpu_tensor(2, 4)
            _register_gpu_tensor(self.engine, tid, t)
            tensors[tid] = t

        total_size = sum(
            t.element_size() * t.numel() for t in tensors.values()
        )
        self.assertEqual(self.engine.buffer_size, total_size)

        for tid in tensors:
            self.engine.recv_tensor(tid)

        self.assertEqual(
            len(self.engine.recv_store),
            0,
            "All layer entries must be gone from recv_store after processing.",
        )
        self.assertEqual(
            self.engine.buffer_size,
            0,
            "buffer_size must be zero after all layer tensors are consumed.",
        )

    def test_recv_store_does_not_grow_under_sustained_load(self):
        """Simulate a high-QPS scenario: many requests arrive while previous
        ones are still running.  recv_store must stay empty after each tensor
        is consumed – it must never accumulate in-flight entries."""
        num_requests = 50
        num_layers = 4

        for req_idx in range(num_requests):
            req_id = f"req-{req_idx:04d}"
            # Simulate tensors arriving from prefill.
            for layer_idx in range(num_layers):
                tid = f"{req_id}#layer.{layer_idx}"
                _register_gpu_tensor(self.engine, tid, self._cpu_tensor(8, 8))

            # Immediately consume them (start_load_kv is called this step).
            for layer_idx in range(num_layers):
                tid = f"{req_id}#layer.{layer_idx}"
                self.engine.recv_tensor(tid)

        # After processing all requests, nothing should remain.
        self.assertEqual(
            len(self.engine.recv_store),
            0,
            "recv_store must be empty after all tensors are consumed; found "
            f"{len(self.engine.recv_store)} stale entries.",
        )
        self.assertEqual(self.engine.buffer_size, 0)


class TestRecvTensorPoolBackedEntries(unittest.TestCase):
    """recv_tensor must free the TensorMemoryPool slot immediately for
    pool-backed (addr, dtype, shape) entries."""

    def setUp(self):
        self.engine = _make_engine()

    def test_pool_free_called_immediately(self):
        """pool.free(addr) must be called right after pool.load_tensor(),
        not deferred to get_finished()."""
        fake_addr = 0xDEADBEEF
        dtype = torch.float16
        shape = (4, 8)
        tensor_id = "req-010#attn.0"

        # Teach the mock what load_tensor should return.
        self.engine.pool.load_tensor.return_value = torch.zeros(
            shape, dtype=dtype
        )

        _register_pool_tensor(self.engine, tensor_id, fake_addr, dtype, shape)
        self.engine.recv_tensor(tensor_id)

        self.engine.pool.free.assert_called_once_with(fake_addr)

    def test_pool_entry_removed_from_recv_store(self):
        """The pool-backed entry must be removed from recv_store right
        after the tensor has been loaded back to device."""
        fake_addr = 0xCAFEBABE
        dtype = torch.float32
        shape = (2, 2)
        tensor_id = "req-011#attn.0"

        self.engine.pool.load_tensor.return_value = torch.zeros(shape, dtype=dtype)
        _register_pool_tensor(self.engine, tensor_id, fake_addr, dtype, shape)

        self.assertIn(tensor_id, self.engine.recv_store)
        self.engine.recv_tensor(tensor_id)
        self.assertNotIn(
            tensor_id,
            self.engine.recv_store,
            "Pool-backed entry must be removed from recv_store immediately.",
        )

    def test_pool_free_not_called_for_none_tensor(self):
        """When the stored value is None (listener OOM path), pool.free must
        NOT be called – there is no pool allocation to release."""
        tensor_id = "req-012#attn.0"
        self.engine.recv_store[tensor_id] = None
        # Also put it in cv so recv_tensor doesn't block.

        self.engine.recv_tensor(tensor_id)

        self.engine.pool.free.assert_not_called()
        self.assertNotIn(tensor_id, self.engine.recv_store)

    def test_pool_free_called_even_if_load_tensor_raises(self):
        """pool.free(addr) must be called via try/finally even if
        pool.load_tensor() raises an exception, to prevent pinned-memory
        leaks.  Without the try/finally the addr would be permanently lost
        because the entry is already popped from recv_store before the
        load attempt."""
        fake_addr = 0xDEADC0DE
        tensor_id = "req-014#attn.0"

        self.engine.pool.load_tensor.side_effect = RuntimeError("simulated OOM")
        _register_pool_tensor(
            self.engine, tensor_id, fake_addr, torch.float16, (2, 4)
        )

        with self.assertRaises(RuntimeError):
            self.engine.recv_tensor(tensor_id)

        self.engine.pool.free.assert_called_once_with(fake_addr)
        self.assertNotIn(tensor_id, self.engine.recv_store)

    def test_pool_free_called_once_per_entry_not_twice(self):
        """pool.free must be called exactly once per pool-backed entry.
        A second call with the same addr would corrupt the pool's free-list."""
        fake_addr = 0xABCD1234
        dtype = torch.float16
        shape = (1, 4)
        req_id = "req-013"
        tensor_id = f"{req_id}#attn.0"

        self.engine.pool.load_tensor.return_value = torch.zeros(shape, dtype=dtype)
        _register_pool_tensor(self.engine, tensor_id, fake_addr, dtype, shape)

        # Consume via recv_tensor.
        self.engine.recv_tensor(tensor_id)

        # Simulate request finishing – get_finished should be a no-op for
        # already-consumed entries.
        self.engine.get_finished({req_id})

        self.assertEqual(
            self.engine.pool.free.call_count,
            1,
            "pool.free must be called exactly once; a second call would "
            "double-free the pool slot.",
        )


class TestGetFinishedStragglerCleanup(unittest.TestCase):
    """get_finished must clean up tensors that were received but never
    consumed via recv_tensor (e.g. a layer was skipped or an error occurred
    before start_load_kv).  This is the safety-net path."""

    def _cpu_tensor(self, *shape):
        return torch.zeros(*shape, dtype=torch.float16)

    def setUp(self):
        self.engine = _make_engine()

    def test_straggler_gpu_tensor_removed_and_buffer_decremented(self):
        """If a gpu tensor lingers in recv_store at request completion,
        get_finished must pop it *and* decrement buffer_size."""
        req_id = "req-strag-001"
        tensor = self._cpu_tensor(4, 4)
        tensor_id = f"{req_id}#layer.0"
        size = tensor.element_size() * tensor.numel()

        _register_gpu_tensor(self.engine, tensor_id, tensor)
        self.assertEqual(self.engine.buffer_size, size)

        # recv_tensor was NEVER called; call get_finished to trigger cleanup.
        self.engine.get_finished({req_id})

        self.assertNotIn(tensor_id, self.engine.recv_store)
        self.assertEqual(
            self.engine.buffer_size,
            0,
            "buffer_size must be decremented by get_finished for straggler "
            "GPU tensors.",
        )

    def test_straggler_pool_tensor_freed(self):
        """Pool-backed straggler must have pool.free() called."""
        req_id = "req-strag-002"
        fake_addr = 0x11223344
        tensor_id = f"{req_id}#layer.0"

        _register_pool_tensor(
            self.engine, tensor_id, fake_addr, torch.float16, (2, 4)
        )
        self.engine.get_finished({req_id})

        self.engine.pool.free.assert_called_once_with(fake_addr)
        self.assertNotIn(tensor_id, self.engine.recv_store)

    def test_multiple_straggler_layers_all_cleaned(self):
        """All layers of a straggler request must be cleaned up."""
        req_id = "req-strag-003"
        num_layers = 6
        total_size = 0

        for i in range(num_layers):
            tid = f"{req_id}#layer.{i}"
            t = self._cpu_tensor(4, 4)
            _register_gpu_tensor(self.engine, tid, t)
            total_size += t.element_size() * t.numel()

        self.assertEqual(self.engine.buffer_size, total_size)
        self.engine.get_finished({req_id})

        self.assertEqual(len(self.engine.recv_store), 0)
        self.assertEqual(self.engine.buffer_size, 0)

    def test_only_finished_request_cleaned_not_others(self):
        """get_finished with req_A must not touch tensors belonging to req_B
        that is still in-flight."""
        req_a = "req-A"
        req_b = "req-B"
        tid_a = f"{req_a}#layer.0"
        tid_b = f"{req_b}#layer.0"

        tensor_a = self._cpu_tensor(2, 2)
        tensor_b = self._cpu_tensor(2, 2)
        _register_gpu_tensor(self.engine, tid_a, tensor_a)
        _register_gpu_tensor(self.engine, tid_b, tensor_b)

        size_b = tensor_b.element_size() * tensor_b.numel()

        self.engine.get_finished({req_a})

        self.assertNotIn(tid_a, self.engine.recv_store)
        self.assertIn(
            tid_b,
            self.engine.recv_store,
            "tensor for in-flight req_B must NOT be removed.",
        )
        self.assertEqual(
            self.engine.buffer_size,
            size_b,
            "buffer_size must only reflect req_B's remaining tensor.",
        )

    def test_noop_when_tensors_already_consumed(self):
        """If recv_tensor already popped all entries (normal operation),
        get_finished must be a no-op: no crash, no double-free, buffer_size
        stays at 0."""
        req_id = "req-consumed"
        tensor = self._cpu_tensor(4, 4)
        tensor_id = f"{req_id}#layer.0"

        _register_gpu_tensor(self.engine, tensor_id, tensor)
        # Consume via recv_tensor (normal path).
        self.engine.recv_tensor(tensor_id)

        self.assertEqual(self.engine.buffer_size, 0)
        self.assertNotIn(tensor_id, self.engine.recv_store)

        # get_finished should not raise and must not corrupt buffer_size.
        self.engine.get_finished({req_id})

        self.assertEqual(self.engine.buffer_size, 0)
        self.engine.pool.free.assert_not_called()

    def test_send_tracking_dict_cleaned_by_get_finished(self):
        """send_request_id_to_tensor_ids must be cleaned up for the finished
        request regardless of whether recv_store had anything."""
        req_id = "req-send-track"
        tensor_id = f"{req_id}#layer.0"
        self.engine.send_request_id_to_tensor_ids[req_id] = {tensor_id}

        self.engine.get_finished({req_id})

        self.assertNotIn(
            req_id,
            self.engine.send_request_id_to_tensor_ids,
            "send_request_id_to_tensor_ids must be cleaned for finished reqs.",
        )

    def test_recv_tracking_dict_cleaned_by_get_finished(self):
        """recv_request_id_to_tensor_ids must be cleaned for finished reqs."""
        req_id = "req-recv-track"
        tensor = self._cpu_tensor(2, 2)
        tensor_id = f"{req_id}#layer.0"
        _register_gpu_tensor(self.engine, tensor_id, tensor)

        self.engine.get_finished({req_id})

        self.assertNotIn(
            req_id,
            self.engine.recv_request_id_to_tensor_ids,
            "recv_request_id_to_tensor_ids must be cleaned for finished reqs.",
        )


class TestGetFinishedUsesRecvTracking(unittest.TestCase):
    """Regression guard: get_finished() must use recv_request_id_to_tensor_ids
    (the set of actually-received tensor_ids) rather than iterating
    no_compile_layers.  The old implementation would silently skip all straggler
    cleanup for any request whose tensors were not reflected in that dict."""

    def setUp(self):
        self.engine = _make_engine()

    def test_straggler_found_via_recv_tracking_dict(self):
        """Stragglers are found via recv_request_id_to_tensor_ids regardless
        of which layer names exist elsewhere.  The old code iterated
        no_compile_layers and would miss an entry whose key wasn't in that
        dict (or miss everything when the dict was empty)."""
        req_id = "req-no-layers"
        tensor = torch.zeros(4, 4, dtype=torch.float16)
        size = tensor.element_size() * tensor.numel()
        tensor_id = f"{req_id}#attn.v_proj"

        self.engine.buffer_size += size
        self.engine.recv_store[tensor_id] = tensor
        self.engine.recv_request_id_to_tensor_ids[req_id] = {tensor_id}

        self.engine.get_finished({req_id})

        self.assertNotIn(tensor_id, self.engine.recv_store)
        self.assertEqual(self.engine.buffer_size, 0)


class TestPoolFreeCalledOnce(unittest.TestCase):
    """pool.free must be called exactly once per pool-backed tensor across
    the combined recv_tensor + get_finished lifecycle."""

    def setUp(self):
        self.engine = _make_engine()

    def test_combined_lifecycle_exactly_one_free(self):
        """Normal flow: recv_tensor consumes the pool entry, then the request
        finishes.  pool.free must have been called exactly once – by
        recv_tensor – not a second time by get_finished."""
        fake_addr = 0x55AA55AA
        dtype = torch.float16
        shape = (3, 5)
        req_id = "req-lifecycle"
        tensor_id = f"{req_id}#layer.0"

        self.engine.pool.load_tensor.return_value = torch.zeros(shape, dtype=dtype)
        _register_pool_tensor(self.engine, tensor_id, fake_addr, dtype, shape)

        # Step 1: decode step runs, consumes the tensor.
        self.engine.recv_tensor(tensor_id)

        # Step 2: request finishes (many decode steps later).
        self.engine.get_finished({req_id})

        self.assertEqual(
            self.engine.pool.free.call_count,
            1,
            "pool.free must be called exactly once across the full lifecycle.",
        )
        self.assertEqual(
            self.engine.pool.free.call_args,
            call(fake_addr),
            "pool.free must be called with the exact pool address.",
        )


if __name__ == "__main__":
    unittest.main()
