# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Asynchronous GPU-to-CPU transfer handler for distillation capture."""

import time
from typing import Optional

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


class AsyncTransferHandler:
    """Manages non-blocking GPU-to-CPU tensor transfers.
    
    Handles cloning tensors, initiating async transfers, and queuing
    data for writing without blocking the inference pipeline.
    
    Features:
    - Dedicated CUDA stream for transfers (doesn't block compute)
    - Pinned memory for faster GPU→CPU transfers
    - Tensor parallel support (all_gather + rank 0 writes)
    """
    
    def __init__(self, writer, tp_size: int = 1, tp_rank: int = 0):
        """Initialize async transfer handler.
        
        Args:
            writer: Writer instance (AsyncSafetensorsWriter)
                   that implements queue_write() method.
            tp_size: Tensor parallel size.
            tp_rank: This process's tensor parallel rank.
        """
        self.writer = writer
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Create dedicated CUDA stream for transfers
        self._transfer_stream: Optional[torch.cuda.Stream] = None
        
        # Pinned memory buffers (lazily initialized)
        self._pinned_buffers: dict[str, torch.Tensor] = {}
    
    def _get_transfer_stream(self, device: torch.device) -> torch.cuda.Stream:
        """Get or create dedicated CUDA stream for transfers."""
        if self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream(device=device)
        return self._transfer_stream
    
    def _get_pinned_buffer(self, name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Get or create a pinned memory buffer for faster transfers."""
        key = f"{name}_{shape}_{dtype}"
        if key not in self._pinned_buffers:
            self._pinned_buffers[key] = torch.empty(
                shape, dtype=dtype, pin_memory=True
            )
        buffer = self._pinned_buffers[key]
        # Resize if needed
        if buffer.shape != shape:
            self._pinned_buffers[key] = torch.empty(
                shape, dtype=dtype, pin_memory=True
            )
        return self._pinned_buffers[key]
    
    def _all_gather_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """All-gather hidden states across tensor parallel ranks.
        
        Args:
            hidden_states: Local hidden states [batch, seq_len, hidden_size/tp_size]
                          or [batch, hidden_size/tp_size]
        
        Returns:
            Full hidden states [batch, seq_len, hidden_size] or [batch, hidden_size]
        """
        if self.tp_size <= 1:
            return hidden_states
        
        # Get the tensor parallel process group
        try:
            from vllm.distributed import get_tp_group
            tp_group = get_tp_group().device_group
        except Exception as e:
            logger.warning(f"Could not get TP group, skipping all_gather: {e}")
            return hidden_states
        
        # Handle both 2D and 3D hidden states
        original_shape = hidden_states.shape
        if hidden_states.dim() == 2:
            # [batch, hidden_size] -> [batch, 1, hidden_size]
            hidden_states = hidden_states.unsqueeze(1)
            was_2d = True
        else:
            was_2d = False
        
        # All-gather along the hidden dimension
        batch_size, seq_len, local_hidden = hidden_states.shape
        full_hidden = local_hidden * self.tp_size
        
        # Create output tensor
        gathered = torch.empty(
            (batch_size, seq_len, full_hidden),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        
        # Gather into list then concatenate
        gather_list = [
            torch.empty_like(hidden_states) for _ in range(self.tp_size)
        ]
        dist.all_gather(gather_list, hidden_states, group=tp_group)
        
        # Concatenate along hidden dimension
        gathered = torch.cat(gather_list, dim=-1)
        
        # Restore original shape if was 2D
        if was_2d:
            gathered = gathered.squeeze(1)
        
        logger.debug(
            f"All-gathered hidden states: {original_shape} -> {gathered.shape}"
        )
        return gathered
    
    def _transfer_to_cpu_async(
        self,
        tensor: torch.Tensor,
        name: str,
        stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        """Transfer tensor to CPU using pinned memory and dedicated stream.
        
        Args:
            tensor: GPU tensor to transfer
            name: Name for the pinned buffer
            stream: CUDA stream to use for transfer
        
        Returns:
            CPU tensor (transfer may still be in progress)
        """
        if tensor.device.type == 'cpu':
            return tensor.clone()
        
        with torch.cuda.stream(stream):
            # Get pinned buffer
            pinned = self._get_pinned_buffer(name, tensor.shape, tensor.dtype)
            # Copy to pinned memory (async within stream)
            pinned.copy_(tensor, non_blocking=True)
        
        return pinned
    
    def transfer_and_write(
        self,
        top_k_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
        input_ids: torch.Tensor,
        acceptance_length: float,
        num_accepted_tokens: int,
        num_draft_tokens: int,
        teacher_model: Optional[str] = None,
        prompt: Optional[str] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> None:
        """Initiate non-blocking GPU-to-CPU transfer and queue for writing.
        
        Returns immediately without waiting for transfer completion.
        Uses dedicated CUDA stream and pinned memory for efficiency.
        
        For tensor parallel: all-gathers hidden states, only rank 0 writes.
        
        Args:
            top_k_probs: Top-k probabilities tensor [batch_size, seq_len, k].
            top_k_indices: Top-k indices tensor [batch_size, seq_len, k].
            input_ids: Input token IDs tensor [batch_size, seq_len].
            acceptance_length: Acceptance length for this draft.
            num_accepted_tokens: Number of accepted tokens.
            num_draft_tokens: Number of draft tokens.
            teacher_model: Name of the target model (optional).
            prompt: Prompt text (optional).
            hidden_states: Hidden states from target model (optional).
        """
        try:
            # Only rank 0 writes to avoid duplicates in TP mode
            if self.tp_size > 1 and self.tp_rank != 0:
                # Still need to participate in all_gather if hidden_states provided
                if hidden_states is not None:
                    self._all_gather_hidden_states(hidden_states)
                return
            
            # Get device and transfer stream
            device = top_k_probs.device
            if device.type != 'cpu':
                stream = self._get_transfer_stream(device)
            else:
                stream = None
            
            # Clone tensors to avoid interference with inference
            probs_clone = top_k_probs.clone()
            indices_clone = top_k_indices.clone()
            input_ids_clone = input_ids.clone()
            
            logger.debug(
                f"Cloned tensors for async transfer. "
                f"probs shape: {probs_clone.shape}, device: {probs_clone.device}"
            )
            
            # Handle hidden states with tensor parallel support
            hidden_states_cpu = None
            if hidden_states is not None:
                hidden_states_clone = hidden_states.clone()
                
                # All-gather if using tensor parallelism
                if self.tp_size > 1:
                    hidden_states_clone = self._all_gather_hidden_states(hidden_states_clone)
                
                # Transfer to CPU
                if stream is not None:
                    hidden_states_cpu = self._transfer_to_cpu_async(
                        hidden_states_clone, "hidden_states", stream
                    )
                else:
                    hidden_states_cpu = hidden_states_clone
                
                logger.debug(
                    f"Hidden states transfer initiated. Shape: {hidden_states_cpu.shape}, "
                    f"dtype: {hidden_states_cpu.dtype}"
                )
            
            # Record CUDA event for synchronization
            cuda_event = None
            if stream is not None:
                # Transfer other tensors using dedicated stream
                probs_cpu = self._transfer_to_cpu_async(probs_clone, "probs", stream)
                indices_cpu = self._transfer_to_cpu_async(indices_clone, "indices", stream)
                input_ids_cpu = self._transfer_to_cpu_async(input_ids_clone, "input_ids", stream)
                
                # Record event on transfer stream
                cuda_event = torch.cuda.Event()
                with torch.cuda.stream(stream):
                    cuda_event.record()
                
                logger.debug("Recorded CUDA event on transfer stream")
            else:
                # Already on CPU
                probs_cpu = probs_clone
                indices_cpu = indices_clone
                input_ids_cpu = input_ids_clone
                logger.debug("Tensors already on CPU, no transfer needed")
            
            # Create metadata
            metadata = {
                'acceptance_length': acceptance_length,
                'timestamp': time.time(),
                'num_accepted_tokens': num_accepted_tokens,
                'num_draft_tokens': num_draft_tokens,
                'teacher_model': teacher_model or 'unknown',
                'prompt': prompt or '',
                'tp_size': self.tp_size,
            }
            
            logger.debug(
                f"Queueing for async write. Acceptance length: {acceptance_length:.2f}"
            )
            
            # Queue for async writing (non-blocking)
            self.writer.queue_write(
                probs_cpu,
                indices_cpu,
                input_ids_cpu,
                metadata,
                cuda_event,
                hidden_states_cpu,
            )
            
        except Exception as e:
            # Log error but don't propagate to avoid blocking inference
            logger.error(f"Error in distillation capture transfer: {e}")
