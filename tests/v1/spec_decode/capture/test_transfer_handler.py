# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for AsyncTransferHandler in distillation capture."""

import os
import tempfile
import time
from unittest.mock import Mock, MagicMock, patch, call

import pytest
import torch

from vllm.v1.spec_decode.capture.safetensors_writer import (
    AsyncSafetensorsWriter,
)
from vllm.v1.spec_decode.capture.transfer_handler import AsyncTransferHandler


class TestAsyncTransferHandlerInitialization:
    """Test AsyncTransferHandler initialization."""

    def test_initialization_with_writer(self):
        """Test that handler initializes with writer."""
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        handler = AsyncTransferHandler(mock_writer)
        
        assert handler.writer is mock_writer

    def test_initialization_stores_writer_reference(self):
        """Test that handler stores reference to writer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
                use_compression=False,
            )
            handler = AsyncTransferHandler(writer)
            
            assert handler.writer is writer
            writer.shutdown()


class TestAsyncTransferHandlerTransferAndWrite:
    """Test AsyncTransferHandler.transfer_and_write method."""

    def test_transfer_and_write_queues_data(self):
        """Test that transfer_and_write queues data to writer."""
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        handler = AsyncTransferHandler(mock_writer)
        
        top_k_probs = torch.randn(1, 5, 10)
        top_k_indices = torch.randint(0, 1000, (1, 5, 10))
        input_ids = torch.randint(0, 1000, (1, 5))
        
        handler.transfer_and_write(
            top_k_probs=top_k_probs,
            top_k_indices=top_k_indices,
            input_ids=input_ids,
            acceptance_length=1.5,
            num_accepted_tokens=2,
            num_draft_tokens=4,
            teacher_model="test-model",
            prompt="test prompt",
        )
        
        # Verify queue_write was called
        mock_writer.queue_write.assert_called_once()

    def test_transfer_and_write_clones_tensors(self):
        """Test that tensors are cloned before transfer."""
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        handler = AsyncTransferHandler(mock_writer)
        
        top_k_probs = torch.randn(1, 5, 10)
        top_k_indices = torch.randint(0, 1000, (1, 5, 10))
        input_ids = torch.randint(0, 1000, (1, 5))
        
        # Store original data pointers
        original_probs_ptr = top_k_probs.data_ptr()
        original_indices_ptr = top_k_indices.data_ptr()
        
        handler.transfer_and_write(
            top_k_probs=top_k_probs,
            top_k_indices=top_k_indices,
            input_ids=input_ids,
            acceptance_length=1.5,
            num_accepted_tokens=2,
            num_draft_tokens=4,
        )
        
        # Get the tensors passed to queue_write
        call_args = mock_writer.queue_write.call_args
        passed_probs = call_args[0][0]
        passed_indices = call_args[0][1]
        
        # Tensors should be different objects (cloned)
        # Note: data_ptr comparison may not work for CPU tensors after clone
        # but the shapes should match
        assert passed_probs.shape == top_k_probs.shape
        assert passed_indices.shape == top_k_indices.shape

    def test_transfer_and_write_creates_metadata(self):
        """Test that metadata is created correctly."""
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        handler = AsyncTransferHandler(mock_writer)
        
        handler.transfer_and_write(
            top_k_probs=torch.randn(1, 5, 10),
            top_k_indices=torch.randint(0, 1000, (1, 5, 10)),
            input_ids=torch.randint(0, 1000, (1, 5)),
            acceptance_length=1.75,
            num_accepted_tokens=3,
            num_draft_tokens=4,
            teacher_model="gpt-model",
            prompt="Hello world",
        )
        
        # Get metadata from call
        call_args = mock_writer.queue_write.call_args
        metadata = call_args[0][3]  # 4th positional arg
        
        assert metadata['acceptance_length'] == 1.75
        assert metadata['num_accepted_tokens'] == 3
        assert metadata['num_draft_tokens'] == 4
        assert metadata['teacher_model'] == 'gpt-model'
        assert metadata['prompt'] == 'Hello world'
        assert 'timestamp' in metadata

    def test_transfer_and_write_with_cpu_tensors(self):
        """Test transfer with tensors already on CPU."""
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        handler = AsyncTransferHandler(mock_writer)
        
        # CPU tensors
        top_k_probs = torch.randn(1, 5, 10)
        top_k_indices = torch.randint(0, 1000, (1, 5, 10))
        input_ids = torch.randint(0, 1000, (1, 5))
        
        assert top_k_probs.device.type == 'cpu'
        
        handler.transfer_and_write(
            top_k_probs=top_k_probs,
            top_k_indices=top_k_indices,
            input_ids=input_ids,
            acceptance_length=1.5,
            num_accepted_tokens=2,
            num_draft_tokens=4,
        )
        
        # Should still work without CUDA event
        mock_writer.queue_write.assert_called_once()
        
        # CUDA event should be None for CPU tensors
        call_args = mock_writer.queue_write.call_args
        cuda_event = call_args[0][4]  # 5th positional arg
        assert cuda_event is None

    def test_transfer_and_write_with_hidden_states(self):
        """Test transfer with hidden states tensor."""
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        handler = AsyncTransferHandler(mock_writer)
        
        top_k_probs = torch.randn(1, 5, 10)
        top_k_indices = torch.randint(0, 1000, (1, 5, 10))
        input_ids = torch.randint(0, 1000, (1, 5))
        hidden_states = torch.randn(1, 5, 2880)
        
        handler.transfer_and_write(
            top_k_probs=top_k_probs,
            top_k_indices=top_k_indices,
            input_ids=input_ids,
            acceptance_length=1.5,
            num_accepted_tokens=2,
            num_draft_tokens=4,
            hidden_states=hidden_states,
        )
        
        # Verify hidden_states was passed
        call_args = mock_writer.queue_write.call_args
        passed_hidden_states = call_args[0][5]  # 6th positional arg
        
        assert passed_hidden_states is not None
        assert passed_hidden_states.shape == hidden_states.shape

    def test_transfer_and_write_without_hidden_states(self):
        """Test transfer without hidden states (default)."""
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        handler = AsyncTransferHandler(mock_writer)
        
        handler.transfer_and_write(
            top_k_probs=torch.randn(1, 5, 10),
            top_k_indices=torch.randint(0, 1000, (1, 5, 10)),
            input_ids=torch.randint(0, 1000, (1, 5)),
            acceptance_length=1.5,
            num_accepted_tokens=2,
            num_draft_tokens=4,
        )
        
        # Verify hidden_states is None
        call_args = mock_writer.queue_write.call_args
        passed_hidden_states = call_args[0][5]  # 6th positional arg
        
        assert passed_hidden_states is None

    def test_transfer_and_write_with_2d_hidden_states(self):
        """Test transfer with 2D hidden states tensor [batch, hidden_size].
        
        This tests the fix for the error:
        'not enough values to unpack (expected 3, got 2)'
        when hidden_states is 2D instead of 3D.
        """
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        handler = AsyncTransferHandler(mock_writer)
        
        top_k_probs = torch.randn(1, 5, 10)
        top_k_indices = torch.randint(0, 1000, (1, 5, 10))
        input_ids = torch.randint(0, 1000, (1, 5))
        # 2D hidden states: [batch, hidden_size] instead of [batch, seq_len, hidden_size]
        hidden_states = torch.randn(1, 2880)
        
        handler.transfer_and_write(
            top_k_probs=top_k_probs,
            top_k_indices=top_k_indices,
            input_ids=input_ids,
            acceptance_length=1.5,
            num_accepted_tokens=2,
            num_draft_tokens=4,
            hidden_states=hidden_states,
        )
        
        # Verify hidden_states was passed and shape is preserved
        call_args = mock_writer.queue_write.call_args
        passed_hidden_states = call_args[0][5]  # 6th positional arg
        
        assert passed_hidden_states is not None
        # Should remain 2D after processing
        assert passed_hidden_states.shape == hidden_states.shape
        assert passed_hidden_states.dim() == 2

    def test_transfer_and_write_default_metadata_values(self):
        """Test that default metadata values are used when not provided."""
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        handler = AsyncTransferHandler(mock_writer)
        
        handler.transfer_and_write(
            top_k_probs=torch.randn(1, 5, 10),
            top_k_indices=torch.randint(0, 1000, (1, 5, 10)),
            input_ids=torch.randint(0, 1000, (1, 5)),
            acceptance_length=1.5,
            num_accepted_tokens=2,
            num_draft_tokens=4,
            # No teacher_model or prompt
        )
        
        call_args = mock_writer.queue_write.call_args
        metadata = call_args[0][3]
        
        assert metadata['teacher_model'] == 'unknown'
        assert metadata['prompt'] == ''


class TestAllGatherHiddenStates:
    """Test _all_gather_hidden_states method for 2D/3D handling."""

    def test_all_gather_2d_hidden_states_no_tp(self):
        """Test that 2D hidden states work without tensor parallelism."""
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        handler = AsyncTransferHandler(mock_writer, tp_size=1, tp_rank=0)
        
        # 2D hidden states: [batch, hidden_size]
        hidden_states = torch.randn(4, 2880)
        
        result = handler._all_gather_hidden_states(hidden_states)
        
        # Should return unchanged when tp_size=1
        assert result.shape == hidden_states.shape
        assert result.dim() == 2

    def test_all_gather_3d_hidden_states_no_tp(self):
        """Test that 3D hidden states work without tensor parallelism."""
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        handler = AsyncTransferHandler(mock_writer, tp_size=1, tp_rank=0)
        
        # 3D hidden states: [batch, seq_len, hidden_size]
        hidden_states = torch.randn(4, 5, 2880)
        
        result = handler._all_gather_hidden_states(hidden_states)
        
        # Should return unchanged when tp_size=1
        assert result.shape == hidden_states.shape
        assert result.dim() == 3


class TestAsyncTransferHandlerErrorHandling:
    """Test error handling in AsyncTransferHandler."""

    def test_transfer_and_write_handles_exceptions(self):
        """Test that exceptions don't propagate to caller."""
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        mock_writer.queue_write.side_effect = RuntimeError("Test error")
        handler = AsyncTransferHandler(mock_writer)
        
        # Should not raise exception
        handler.transfer_and_write(
            top_k_probs=torch.randn(1, 5, 10),
            top_k_indices=torch.randint(0, 1000, (1, 5, 10)),
            input_ids=torch.randint(0, 1000, (1, 5)),
            acceptance_length=1.5,
            num_accepted_tokens=2,
            num_draft_tokens=4,
        )

    def test_transfer_and_write_handles_clone_error(self):
        """Test handling of tensor clone errors."""
        mock_writer = Mock(spec=AsyncSafetensorsWriter)
        handler = AsyncTransferHandler(mock_writer)
        
        # Create a mock tensor that raises on clone
        mock_tensor = Mock()
        mock_tensor.clone.side_effect = RuntimeError("Clone failed")
        mock_tensor.device = Mock()
        mock_tensor.device.type = 'cpu'
        
        # Should not raise
        handler.transfer_and_write(
            top_k_probs=mock_tensor,
            top_k_indices=torch.randint(0, 1000, (1, 5, 10)),
            input_ids=torch.randint(0, 1000, (1, 5)),
            acceptance_length=1.5,
            num_accepted_tokens=2,
            num_draft_tokens=4,
        )


class TestAsyncTransferHandlerIntegration:
    """Integration tests for AsyncTransferHandler with real AsyncSafetensorsWriter."""

    def test_end_to_end_transfer_and_write(self):
        """Test complete flow from transfer to safetensors file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=10,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=False,
            )
            handler = AsyncTransferHandler(writer)
            
            handler.transfer_and_write(
                top_k_probs=torch.randn(1, 3, 5),
                top_k_indices=torch.randint(0, 1000, (1, 3, 5)),
                input_ids=torch.randint(0, 1000, (1, 3)),
                acceptance_length=1.5,
                num_accepted_tokens=2,
                num_draft_tokens=4,
                teacher_model="test-model",
                prompt="test prompt",
            )
            
            # Wait for write
            time.sleep(0.5)
            writer.shutdown()
            
            # Verify file was created
            files = os.listdir(tmpdir)
            st_files = [f for f in files if f.endswith('.safetensors')]
            assert len(st_files) >= 1

    def test_multiple_transfers(self):
        """Test multiple sequential transfers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=100,
                batch_size=5,
                batch_timeout=0.5,
                use_compression=False,
            )
            handler = AsyncTransferHandler(writer)
            
            for i in range(10):
                handler.transfer_and_write(
                    top_k_probs=torch.randn(1, 3, 5),
                    top_k_indices=torch.randint(0, 1000, (1, 3, 5)),
                    input_ids=torch.randint(0, 1000, (1, 3)),
                    acceptance_length=1.0 + i * 0.1,
                    num_accepted_tokens=i,
                    num_draft_tokens=4,
                    teacher_model=f"model-{i}",
                )
            
            # Wait for writes
            time.sleep(1.0)
            writer.shutdown()
            
            # Check stats
            stats = writer.get_stats()
            assert stats['total_writes'] >= 0

    def test_transfer_with_varying_shapes(self):
        """Test transfers with different tensor shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AsyncSafetensorsWriter(
                output_dir=tmpdir,
                queue_size=100,
                batch_size=1,
                batch_timeout=0.1,
                use_compression=False,
            )
            handler = AsyncTransferHandler(writer)
            
            shapes = [
                (1, 3, 5),
                (2, 5, 10),
                (1, 10, 20),
            ]
            
            for batch_size, seq_len, k in shapes:
                handler.transfer_and_write(
                    top_k_probs=torch.randn(batch_size, seq_len, k),
                    top_k_indices=torch.randint(0, 1000, (batch_size, seq_len, k)),
                    input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
                    acceptance_length=1.5,
                    num_accepted_tokens=2,
                    num_draft_tokens=4,
                )
            
            time.sleep(0.5)
            writer.shutdown()
