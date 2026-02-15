# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for AsyncSafetensorsWriter in distillation capture."""

import json
import os
import tempfile
import time
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import load_file
from safetensors import safe_open

from vllm.v1.spec_decode.capture.safetensors_writer import (
    AsyncSafetensorsWriter,
    ZSTD_AVAILABLE,
)


class TestAsyncSafetensorsWriterInitialization:
    """Test AsyncSafetensorsWriter initialization."""

    def test_initialization_creates_output_dir(self):
        """Test that initialization creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "new_subdir")
            assert not os.path.exists(output_dir)
            
            writer = AsyncSafetensorsWriter(
                output_dir=output_dir,
                queue_size=10,
            )
            
            assert os.path.exists(output_dir)
            writer.shutdown()

    def test_initialization_starts_threads(self):
        """Test that initialization starts background threads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
            )
            
            assert writer.checker_thread.is_alive()
            assert writer.writer_thread.is_alive()
            writer.shutdown()

    def test_initialization_with_compression(self):
        """Test initialization with zstd compression enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
                use_compression=True,
                compression_level=3,
            )
            
            assert writer.use_compression == ZSTD_AVAILABLE
            assert writer.compression_level == 3
            writer.shutdown()


class TestAsyncSafetensorsWriterQueueWrite:
    """Test AsyncSafetensorsWriter.queue_write method."""

    def test_queue_write_basic(self):
        """Test basic queue_write functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=False,
            )
            
            probs = torch.randn(1, 5, 10)
            indices = torch.randint(0, 1000, (1, 5, 10))
            input_ids = torch.randint(0, 1000, (1, 5))
            metadata = {
                'acceptance_length': 1.5,
                'timestamp': time.time(),
            }
            
            writer.queue_write(probs, indices, input_ids, metadata)
            
            time.sleep(0.5)
            writer.shutdown()
            
            # Check file was written
            files = os.listdir(tmpdir)
            st_files = [f for f in files if f.endswith('.safetensors')]
            assert len(st_files) >= 1

    def test_queue_write_with_hidden_states(self):
        """Test queue_write with hidden states tensor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=False,
            )
            
            probs = torch.randn(1, 5, 10)
            indices = torch.randint(0, 1000, (1, 5, 10))
            input_ids = torch.randint(0, 1000, (1, 5))
            hidden_states = torch.randn(1, 5, 2880)
            metadata = {
                'acceptance_length': 1.5,
                'timestamp': time.time(),
            }
            
            writer.queue_write(
                probs, indices, input_ids, metadata,
                hidden_states=hidden_states
            )
            
            time.sleep(0.5)
            writer.shutdown()
            
            files = os.listdir(tmpdir)
            st_files = [f for f in files if f.endswith('.safetensors')]
            assert len(st_files) >= 1

    def test_queue_write_with_cuda_event(self):
        """Test queue_write with CUDA event for synchronization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=False,
            )
            
            probs = torch.randn(1, 5, 10)
            indices = torch.randint(0, 1000, (1, 5, 10))
            input_ids = torch.randint(0, 1000, (1, 5))
            metadata = {'acceptance_length': 1.5, 'timestamp': time.time()}
            
            mock_event = Mock()
            mock_event.query.return_value = True
            
            writer.queue_write(
                probs, indices, input_ids, metadata,
                cuda_event=mock_event
            )
            
            time.sleep(0.3)
            writer.shutdown()


class TestAsyncSafetensorsWriterDtypePreservation:
    """Test that writer preserves tensor dtypes."""

    def test_preserves_float16(self):
        """Test that float16 tensors are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=False,
            )
            
            hidden_states = torch.randn(1, 5, 2880, dtype=torch.float16)
            probs = torch.randn(1, 5, 10, dtype=torch.float16)
            indices = torch.randint(0, 1000, (1, 5, 10))
            input_ids = torch.randint(0, 1000, (1, 5))
            metadata = {'acceptance_length': 1.5, 'timestamp': time.time()}
            
            writer.queue_write(
                probs, indices, input_ids, metadata,
                hidden_states=hidden_states
            )
            
            time.sleep(0.5)
            writer.shutdown()
            
            # Load and verify dtype
            files = [f for f in os.listdir(tmpdir) if f.endswith('.safetensors')]
            assert len(files) >= 1
            
            loaded = load_file(os.path.join(tmpdir, files[0]))
            assert loaded['hidden_states'].dtype == torch.float16
            assert loaded['probs'].dtype == torch.float16

    def test_preserves_bfloat16(self):
        """Test that bfloat16 tensors are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=False,
            )
            
            hidden_states = torch.randn(1, 5, 2880, dtype=torch.bfloat16)
            probs = torch.randn(1, 5, 10, dtype=torch.bfloat16)
            indices = torch.randint(0, 1000, (1, 5, 10))
            input_ids = torch.randint(0, 1000, (1, 5))
            metadata = {'acceptance_length': 1.5, 'timestamp': time.time()}
            
            writer.queue_write(
                probs, indices, input_ids, metadata,
                hidden_states=hidden_states
            )
            
            time.sleep(0.5)
            writer.shutdown()
            
            files = [f for f in os.listdir(tmpdir) if f.endswith('.safetensors')]
            loaded = load_file(os.path.join(tmpdir, files[0]))
            assert loaded['hidden_states'].dtype == torch.bfloat16

    def test_preserves_int8(self):
        """Test that int8 tensors are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=False,
            )
            
            # Simulate int8 quantized hidden states
            hidden_states = torch.randint(-128, 127, (1, 5, 2880), dtype=torch.int8)
            probs = torch.randn(1, 5, 10)
            indices = torch.randint(0, 1000, (1, 5, 10))
            input_ids = torch.randint(0, 1000, (1, 5))
            metadata = {'acceptance_length': 1.5, 'timestamp': time.time()}
            
            writer.queue_write(
                probs, indices, input_ids, metadata,
                hidden_states=hidden_states
            )
            
            time.sleep(0.5)
            writer.shutdown()
            
            files = [f for f in os.listdir(tmpdir) if f.endswith('.safetensors')]
            loaded = load_file(os.path.join(tmpdir, files[0]))
            assert loaded['hidden_states'].dtype == torch.int8


class TestAsyncSafetensorsWriterCompression:
    """Test zstd compression functionality."""

    @pytest.mark.skipif(not ZSTD_AVAILABLE, reason="zstandard not installed")
    def test_compression_creates_zst_files(self):
        """Test that compression creates .zst files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=True,
            )
            
            probs = torch.randn(1, 5, 10)
            indices = torch.randint(0, 1000, (1, 5, 10))
            input_ids = torch.randint(0, 1000, (1, 5))
            metadata = {'acceptance_length': 1.5, 'timestamp': time.time()}
            
            writer.queue_write(probs, indices, input_ids, metadata)
            
            time.sleep(0.5)
            writer.shutdown()
            
            files = os.listdir(tmpdir)
            zst_files = [f for f in files if f.endswith('.safetensors.zst')]
            assert len(zst_files) >= 1

    @pytest.mark.skipif(not ZSTD_AVAILABLE, reason="zstandard not installed")
    def test_compressed_files_are_smaller(self):
        """Test that compressed files are smaller than uncompressed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Large tensor to see compression effect
            hidden_states = torch.randn(10, 100, 2880)
            probs = torch.randn(10, 100, 10)
            indices = torch.randint(0, 1000, (10, 100, 10))
            input_ids = torch.randint(0, 1000, (10, 100))
            metadata = {'acceptance_length': 1.5, 'timestamp': time.time()}
            
            # Write uncompressed
            uncompressed_dir = os.path.join(tmpdir, "uncompressed")
            os.makedirs(uncompressed_dir)
            writer1 = AsyncSafetensorsWriter(
                output_dir=uncompressed_dir,
                queue_size=10,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=False,
            )
            writer1.queue_write(probs, indices, input_ids, metadata, hidden_states=hidden_states)
            time.sleep(0.5)
            writer1.shutdown()
            
            # Write compressed
            compressed_dir = os.path.join(tmpdir, "compressed")
            os.makedirs(compressed_dir)
            writer2 = AsyncSafetensorsWriter(
                output_dir=compressed_dir,
                queue_size=10,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=True,
            )
            writer2.queue_write(probs, indices, input_ids, metadata, hidden_states=hidden_states)
            time.sleep(0.5)
            writer2.shutdown()
            
            # Compare sizes
            uncompressed_size = sum(
                os.path.getsize(os.path.join(uncompressed_dir, f))
                for f in os.listdir(uncompressed_dir)
            )
            compressed_size = sum(
                os.path.getsize(os.path.join(compressed_dir, f))
                for f in os.listdir(compressed_dir)
            )
            
            assert compressed_size < uncompressed_size


class TestAsyncSafetensorsWriterBatching:
    """Test batching behavior."""

    def test_batch_write_on_size_threshold(self):
        """Test that batch is written when size threshold is reached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=100,
                batch_size=3,
                batch_timeout=60.0,
                use_compression=False,
            )
            
            for i in range(3):
                probs = torch.randn(1, 5, 10)
                indices = torch.randint(0, 1000, (1, 5, 10))
                input_ids = torch.randint(0, 1000, (1, 5))
                metadata = {'acceptance_length': 1.5, 'timestamp': time.time()}
                writer.queue_write(probs, indices, input_ids, metadata)
            
            time.sleep(0.5)
            writer.shutdown()
            
            files = [f for f in os.listdir(tmpdir) if 'safetensors' in f]
            assert len(files) >= 1

    def test_batch_write_on_timeout(self):
        """Test that batch is written when timeout is reached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=100,
                batch_size=100,
                batch_timeout=0.2,
                use_compression=False,
            )
            
            probs = torch.randn(1, 5, 10)
            indices = torch.randint(0, 1000, (1, 5, 10))
            input_ids = torch.randint(0, 1000, (1, 5))
            metadata = {'acceptance_length': 1.5, 'timestamp': time.time()}
            writer.queue_write(probs, indices, input_ids, metadata)
            
            time.sleep(0.5)
            writer.shutdown()
            
            files = [f for f in os.listdir(tmpdir) if 'safetensors' in f]
            assert len(files) >= 1


class TestAsyncSafetensorsWriterDataIntegrity:
    """Test data integrity of written files."""

    def test_written_data_matches_input(self):
        """Test that written data matches input tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=False,
            )
            
            probs = torch.tensor([[[0.5, 0.3, 0.2]]])
            indices = torch.tensor([[[10, 20, 30]]])
            input_ids = torch.tensor([[42]])
            hidden_states = torch.tensor([[[1.0, 2.0, 3.0]]])
            metadata = {'acceptance_length': 1.5, 'timestamp': 12345.0}
            
            writer.queue_write(
                probs, indices, input_ids, metadata,
                hidden_states=hidden_states
            )
            
            time.sleep(0.5)
            writer.shutdown()
            
            files = [f for f in os.listdir(tmpdir) if f.endswith('.safetensors')]
            loaded = load_file(os.path.join(tmpdir, files[0]))
            
            torch.testing.assert_close(loaded['probs'], probs)
            torch.testing.assert_close(loaded['indices'], indices)
            torch.testing.assert_close(loaded['input_ids'], input_ids)
            torch.testing.assert_close(loaded['hidden_states'], hidden_states)

    def test_metadata_is_stored(self):
        """Test that metadata is stored alongside tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=False,
            )
            
            probs = torch.randn(1, 5, 10)
            indices = torch.randint(0, 1000, (1, 5, 10))
            input_ids = torch.randint(0, 1000, (1, 5))
            metadata = {
                'acceptance_length': 1.5,
                'timestamp': 12345.0,
                'teacher_model': 'test-model',
            }
            
            writer.queue_write(probs, indices, input_ids, metadata)
            
            time.sleep(0.5)
            writer.shutdown()
            
            files = [f for f in os.listdir(tmpdir) if f.endswith('.safetensors')]
            
            with safe_open(os.path.join(tmpdir, files[0]), framework="pt") as f:
                raw_metadata = f.metadata()
                stored = json.loads(raw_metadata.get('metadata', '{}'))
            
            # Metadata is stored as a list (one per batch item)
            assert stored['metadata'][0]['acceptance_length'] == 1.5
            assert stored['metadata'][0]['teacher_model'] == 'test-model'


class TestAsyncSafetensorsWriterShutdown:
    """Test shutdown behavior."""

    def test_shutdown_stops_threads(self):
        """Test that shutdown stops background threads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
            )
            
            assert writer.checker_thread.is_alive()
            assert writer.writer_thread.is_alive()
            
            writer.shutdown(timeout=2.0)
            
            assert not writer.checker_thread.is_alive()
            assert not writer.writer_thread.is_alive()

    def test_shutdown_flushes_pending_data(self):
        """Test that shutdown flushes pending data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=100,
                batch_size=100,
                batch_timeout=60.0,
                use_compression=False,
            )
            
            for _ in range(5):
                probs = torch.randn(1, 3, 5)
                indices = torch.randint(0, 1000, (1, 3, 5))
                input_ids = torch.randint(0, 1000, (1, 3))
                metadata = {'acceptance_length': 1.5, 'timestamp': time.time()}
                writer.queue_write(probs, indices, input_ids, metadata)
            
            time.sleep(0.3)
            writer.shutdown(timeout=2.0)
            
            # Data should have been flushed
            files = [f for f in os.listdir(tmpdir) if 'safetensors' in f]
            assert len(files) >= 1


class TestAsyncSafetensorsWriterStats:
    """Test statistics tracking."""

    def test_get_stats_returns_dict(self):
        """Test that get_stats returns expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
            )
            
            stats = writer.get_stats()
            
            assert isinstance(stats, dict)
            assert 'total_writes' in stats
            assert 'total_batches' in stats
            assert 'total_drops' in stats
            
            writer.shutdown()
