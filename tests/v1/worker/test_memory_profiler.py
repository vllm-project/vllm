# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MemorySnapshotProfiler class."""

import os
import tempfile

import pytest
import torch

from vllm.utils.mem_utils import MemorySnapshotProfiler


class TestMemorySnapshotProfiler:
    """Tests for MemorySnapshotProfiler class."""

    def test_start_stop_creates_snapshot(self):
        """Test that start/stop creates a valid snapshot file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = MemorySnapshotProfiler(output_dir=tmpdir)
            profiler.start()

            # Allocate some memory to profile
            x = torch.zeros(1000, device="cuda")

            snapshot_file = profiler.stop()

            assert snapshot_file is not None
            assert os.path.exists(snapshot_file)
            assert snapshot_file.endswith(".pickle")
            # Verify file has content (memory snapshot format is specific
            # to torch.cuda.memory._dump_snapshot, not torch.save)
            assert os.path.getsize(snapshot_file) > 0

            del x

    def test_context_manager(self):
        """Test context manager usage creates snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with MemorySnapshotProfiler(output_dir=tmpdir) as profiler:
                x = torch.zeros(1000, device="cuda")
                assert profiler.is_recording

            # Snapshot should be saved after exit
            files = [f for f in os.listdir(tmpdir) if f.endswith(".pickle")]
            assert len(files) == 1

            del x

    def test_rank_in_filename(self):
        """Test that rank is included in the filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = MemorySnapshotProfiler(output_dir=tmpdir)
            profiler.set_rank(3)
            profiler.start()
            snapshot_file = profiler.stop()

            assert "rank3" in snapshot_file

    def test_filename_prefix(self):
        """Test that filename prefix is used correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = MemorySnapshotProfiler(
                output_dir=tmpdir,
                filename_prefix="load_model",
            )
            profiler.start()
            snapshot_file = profiler.stop()

            assert "load_model" in os.path.basename(snapshot_file)

    def test_stop_suffix(self):
        """Test that stop suffix is included in filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = MemorySnapshotProfiler(output_dir=tmpdir)
            profiler.start()
            snapshot_file = profiler.stop(suffix="custom_stage")

            assert "custom_stage" in snapshot_file

    def test_dump_on_exception(self):
        """Test that snapshot is dumped when exception occurs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                pytest.raises(ValueError),
                MemorySnapshotProfiler(
                    output_dir=tmpdir,
                    dump_on_exception=True,
                ),
            ):
                _x = torch.zeros(100, device="cuda")  # noqa: F841
                raise ValueError("test error")

            files = [f for f in os.listdir(tmpdir) if f.endswith(".pickle")]
            # Should have error snapshot + normal snapshot from __exit__
            assert len(files) >= 1
            assert any("error" in f for f in files)

    def test_double_start_warning(self):
        """Test that double start doesn't crash and logs warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = MemorySnapshotProfiler(output_dir=tmpdir)
            profiler.start()
            assert profiler.is_recording

            # Second start should not crash
            profiler.start()
            assert profiler.is_recording

            profiler.stop()

    def test_stop_without_start(self):
        """Test that stop without start returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = MemorySnapshotProfiler(output_dir=tmpdir)
            result = profiler.stop()
            assert result is None

    def test_is_recording_property(self):
        """Test is_recording property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = MemorySnapshotProfiler(output_dir=tmpdir)
            assert profiler.is_recording is False

            profiler.start()
            assert profiler.is_recording is True

            profiler.stop()
            assert profiler.is_recording is False

    def test_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "a", "b", "c")
            profiler = MemorySnapshotProfiler(output_dir=nested_dir)
            profiler.start()
            profiler.stop()

            assert os.path.exists(nested_dir)
