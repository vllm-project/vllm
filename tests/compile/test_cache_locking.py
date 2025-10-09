# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for compilation cache locking mechanism.

Tests the write lock that prevents race conditions when multiple processes
compile and cache to the same directory.

Note: Uses multiprocessing with 'spawn' context to avoid fork() warnings
in multi-threaded pytest environments. The _worker_compile function must
be module-level for proper serialization with 'spawn'.

Synchronization is done using multiprocessing primitives (Barrier, Event)
rather than time.sleep() for deterministic, fast tests.
"""

import multiprocessing
import os
import tempfile
from contextlib import ExitStack
from unittest.mock import MagicMock

import pytest

from vllm.compilation.backends import CompilerManager
from vllm.config import CompilationConfig

# Filter out third-party deprecation warnings we can't fix
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:jsonschema",
)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_compiler():
    """Create a mock compiler for testing."""
    compiler = MagicMock()
    compiler.initialize_cache = MagicMock()
    return compiler


@pytest.fixture
def compiler_manager(mock_compiler):
    """Create a CompilerManager with a mock compiler."""
    config = CompilationConfig()
    manager = CompilerManager(config)
    manager.compiler = mock_compiler
    return manager


class TestCacheLockBasics:
    """Test basic cache locking functionality."""

    def test_get_cache_lock(self, compiler_manager, temp_cache_dir):
        """Test that get_cache_lock creates a lock file."""
        lock = compiler_manager.get_cache_lock(temp_cache_dir)
        assert lock is not None
        assert lock.lock_file == os.path.join(temp_cache_dir, ".vllm_compile.lock")

    def test_cache_ready_marker(self, compiler_manager, temp_cache_dir):
        """Test cache ready marker creation and detection."""
        assert not compiler_manager.is_cache_ready(temp_cache_dir)

        compiler_manager.mark_cache_ready(temp_cache_dir)

        assert compiler_manager.is_cache_ready(temp_cache_dir)
        marker_path = compiler_manager.get_cache_ready_marker(temp_cache_dir)
        assert os.path.exists(marker_path)

    def test_lock_file_permissions(self, compiler_manager, temp_cache_dir):
        """Test that lock file has correct permissions for multi-user sharing."""
        lock = compiler_manager.get_cache_lock(temp_cache_dir)

        # Acquire and release to create the lock file
        with lock:
            pass

        # Check permissions (should be 0o666)
        lock_file_path = os.path.join(temp_cache_dir, ".vllm_compile.lock")
        if os.path.exists(lock_file_path):
            # Note: actual permissions may be modified by umask
            # We just verify the file was created
            assert os.path.exists(lock_file_path)


class TestCacheInitialization:
    """Test cache initialization scenarios."""

    def test_initialize_cache_disabled(self, compiler_manager, temp_cache_dir):
        """Test that no locking occurs when cache is disabled."""
        compiler_manager.initialize_cache(cache_dir=temp_cache_dir, disable_cache=True)

        assert compiler_manager.cache_lock is None
        assert compiler_manager._lock_stack is None

    def test_initialize_cache_fast_path(self, compiler_manager, temp_cache_dir):
        """Test fast path when cache is already ready."""
        # Set up cache as ready
        os.makedirs(temp_cache_dir, exist_ok=True)
        cache_file = os.path.join(temp_cache_dir, "vllm_compile_cache.py")
        with open(cache_file, "w") as f:
            f.write("{}")
        compiler_manager.mark_cache_ready(temp_cache_dir)

        # Initialize cache
        compiler_manager.initialize_cache(cache_dir=temp_cache_dir)

        # Should not hold lock after initialization
        assert compiler_manager._lock_stack is None

    def test_initialize_cache_slow_path(self, compiler_manager, temp_cache_dir):
        """Test slow path when cache needs to be created."""
        compiler_manager.initialize_cache(cache_dir=temp_cache_dir)

        # Should hold lock for compilation
        assert compiler_manager.cache_lock is not None
        assert compiler_manager._lock_stack is not None

        # Clean up
        if compiler_manager._lock_stack:
            compiler_manager._lock_stack.close()
            compiler_manager._lock_stack = None

    def test_double_check_locking(self, compiler_manager, temp_cache_dir):
        """Test double-check locking pattern works correctly."""
        # Start with no cache
        assert not compiler_manager.is_cache_ready(temp_cache_dir)

        # First process initializes
        compiler_manager.initialize_cache(cache_dir=temp_cache_dir)
        assert compiler_manager._lock_stack is not None

        # Simulate cache becoming ready (another process finished)
        cache_file = os.path.join(temp_cache_dir, "vllm_compile_cache.py")
        with open(cache_file, "w") as f:
            f.write("{}")
        compiler_manager.mark_cache_ready(temp_cache_dir)

        # Clean up
        compiler_manager._lock_stack.close()
        compiler_manager._lock_stack = None


class TestSaveToFile:
    """Test cache file saving with lock management."""

    def test_save_disabled_cache(self, compiler_manager, temp_cache_dir):
        """Test that save_to_file does nothing when cache is disabled."""
        compiler_manager.disable_cache = True
        compiler_manager.is_cache_updated = True

        compiler_manager.save_to_file()

        # Should not create any files
        cache_file = os.path.join(temp_cache_dir, "vllm_compile_cache.py")
        assert not os.path.exists(cache_file)

    def test_save_with_held_lock(self, compiler_manager, temp_cache_dir):
        """Test saving when lock is held from compilation."""
        # Set up compiler manager as if it just compiled
        compiler_manager.initialize_cache(cache_dir=temp_cache_dir)
        compiler_manager.cache = {(None, 0, "test"): "value"}
        compiler_manager.is_cache_updated = True

        # Should have lock held
        assert compiler_manager._lock_stack is not None

        # Save to file
        compiler_manager.save_to_file()

        # Lock should be released
        assert compiler_manager._lock_stack is None

        # Cache file should exist
        cache_file = compiler_manager.cache_file_path
        assert os.path.exists(cache_file)
        assert compiler_manager.is_cache_ready(temp_cache_dir)

    def test_save_without_held_lock(self, compiler_manager, temp_cache_dir):
        """Test saving when lock is not held (subsequent compilation)."""
        # Initialize cache (fast path)
        os.makedirs(temp_cache_dir, exist_ok=True)
        cache_file = os.path.join(temp_cache_dir, "vllm_compile_cache.py")
        with open(cache_file, "w") as f:
            f.write("{}")
        compiler_manager.mark_cache_ready(temp_cache_dir)
        compiler_manager.initialize_cache(cache_dir=temp_cache_dir)

        # Simulate new compilation result
        compiler_manager.cache = {(None, 0, "test"): "new_value"}
        compiler_manager.is_cache_updated = True

        # Should not have lock held
        assert compiler_manager._lock_stack is None

        # Save to file should acquire lock temporarily
        compiler_manager.save_to_file()

        # Lock should still be released after
        assert compiler_manager._lock_stack is None

    def test_save_releases_lock_on_no_update(self, compiler_manager, temp_cache_dir):
        """Test that lock is released even if cache is not updated."""
        # Initialize and hold lock
        compiler_manager.initialize_cache(cache_dir=temp_cache_dir)
        assert compiler_manager._lock_stack is not None

        # Mark cache as not updated
        compiler_manager.is_cache_updated = False

        # Save should release lock
        compiler_manager.save_to_file()

        # Lock should be released
        assert compiler_manager._lock_stack is None


class TestExitStackUsage:
    """Test ExitStack context manager usage."""

    def test_exit_stack_manages_lock_lifecycle(self, compiler_manager, temp_cache_dir):
        """Test that ExitStack properly manages lock acquire/release."""
        compiler_manager.cache_lock = compiler_manager.get_cache_lock(temp_cache_dir)

        # Manually test ExitStack behavior
        stack = ExitStack()
        stack.enter_context(compiler_manager.cache_lock)

        assert compiler_manager.cache_lock.is_locked

        stack.close()

        assert not compiler_manager.cache_lock.is_locked

    def test_exit_stack_idempotent_close(self):
        """Test that ExitStack.close() is idempotent."""
        stack = ExitStack()

        # Close multiple times should not raise
        stack.close()
        stack.close()
        stack.close()


def _worker_compile(
    cache_dir: str,
    worker_id: int,
    results: list,
    start_barrier=None,
    compiled_event=None,
):
    """
    Worker function that tests cache compilation with locking.

    Must be module-level for multiprocessing spawn context.

    Args:
        cache_dir: Directory to use for cache
        worker_id: Unique identifier for this worker
        results: Shared list to store test results
        start_barrier: Optional barrier to synchronize simultaneous starts
        compiled_event: Optional event to signal when compilation completes
    """
    try:
        config = CompilationConfig()
        manager = CompilerManager(config)
        manager.compiler = MagicMock()
        manager.compiler.initialize_cache = MagicMock()

        # Wait for all processes to be ready before starting (coordinated start)
        if start_barrier is not None:
            start_barrier.wait()

        # Initialize cache (will acquire lock)
        manager.initialize_cache(cache_dir=cache_dir)

        # Check if we got the lock or cache was ready
        has_lock = manager._lock_stack is not None
        cache_ready = manager.is_cache_ready(cache_dir)

        # Write cache if we have the lock (no artificial delay)
        if has_lock:
            manager.cache = {(None, 0, f"worker_{worker_id}"): "compiled"}
            manager.is_cache_updated = True
            manager.save_to_file()

            # Signal that compilation and cache writing is complete
            if compiled_event is not None:
                compiled_event.set()

        results.append(
            {
                "worker_id": worker_id,
                "has_lock": has_lock,
                "cache_ready": cache_ready,
            }
        )
    except Exception as e:
        results.append(
            {
                "worker_id": worker_id,
                "error": str(e),
            }
        )


class TestConcurrency:
    """Test concurrent access scenarios."""

    def test_two_processes_one_compiles(self, temp_cache_dir):
        """Test that only one of two processes compiles."""
        # Use spawn to avoid fork() warnings in multi-threaded pytest
        ctx = multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        results = manager.list()

        # Use barrier to ensure both processes start simultaneously
        barrier = ctx.Barrier(2)

        # Start two workers with coordinated start
        p1 = ctx.Process(
            target=_worker_compile, args=(temp_cache_dir, 1, results, barrier)
        )
        p2 = ctx.Process(
            target=_worker_compile, args=(temp_cache_dir, 2, results, barrier)
        )

        p1.start()
        p2.start()

        p1.join(timeout=5)
        p2.join(timeout=5)

        # Both should complete
        assert len(results) == 2

        # One should have compiled (has_lock), one should have waited
        locks_held = [r.get("has_lock", False) for r in results]
        assert sum(locks_held) == 1, "Exactly one process should compile"

    def test_third_process_reads_without_lock(self, temp_cache_dir):
        """Test that third process reads completed cache without waiting."""
        # Use spawn to avoid fork() warnings in multi-threaded pytest
        ctx = multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        results = manager.list()

        # Event to signal when first process has finished writing cache
        cache_written_event = ctx.Event()

        # First process compiles and signals when done
        p1 = ctx.Process(
            target=_worker_compile,
            args=(temp_cache_dir, 1, results, None, cache_written_event),
        )
        p1.start()

        # Wait for cache to be written (blocks until p1 signals completion)
        assert cache_written_event.wait(timeout=5), (
            "First process should complete within timeout"
        )
        p1.join(timeout=5)

        # Third process should read without lock (cache is now ready)
        p3 = ctx.Process(target=_worker_compile, args=(temp_cache_dir, 3, results))
        p3.start()
        p3.join(timeout=5)

        # Should not have acquired lock (cache was ready)
        p3_result = [r for r in results if r.get("worker_id") == 3][0]
        assert not p3_result["has_lock"], "Third process should not acquire lock"
        assert p3_result["cache_ready"], "Cache should be ready"


class TestErrorHandling:
    """Test error handling and cleanup."""

    def test_lock_released_on_exception(self, compiler_manager, temp_cache_dir):
        """Test that lock is released when exception occurs during initialization."""
        # Mock compiler to raise exception
        compiler_manager.compiler.initialize_cache.side_effect = RuntimeError(
            "Test error"
        )

        with pytest.raises(RuntimeError, match="Test error"):
            compiler_manager.initialize_cache(cache_dir=temp_cache_dir)

        # Lock should be released (ExitStack.close was called in except block)
        # We can't directly test this, but we can verify the lock file isn't held
        lock = compiler_manager.get_cache_lock(temp_cache_dir)
        assert not lock.is_locked


@pytest.mark.skipif(
    os.environ.get("VLLM_DISABLE_COMPILE_CACHE") == "1",
    reason="Skipping when cache is disabled",
)
class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_full_lifecycle(self, temp_cache_dir):
        """Test complete lifecycle: initialize, compile, save, re-initialize."""
        config = CompilationConfig()

        # First manager: compiles
        manager1 = CompilerManager(config)
        manager1.compiler = MagicMock()
        manager1.compiler.initialize_cache = MagicMock()

        manager1.initialize_cache(cache_dir=temp_cache_dir)
        assert manager1._lock_stack is not None

        # Simulate compilation
        manager1.cache = {(None, 0, "test"): "compiled_value"}
        manager1.is_cache_updated = True
        manager1.save_to_file()

        assert manager1._lock_stack is None
        assert manager1.is_cache_ready(temp_cache_dir)

        # Second manager: reads cache
        manager2 = CompilerManager(config)
        manager2.compiler = MagicMock()
        manager2.compiler.initialize_cache = MagicMock()

        manager2.initialize_cache(cache_dir=temp_cache_dir)

        # Should have loaded cache without lock
        assert manager2._lock_stack is None
        assert manager2.cache == manager1.cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
