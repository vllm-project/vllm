# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TrainingManager: Monitors training losses from vLLM forward passes.

This module collects loss information computed during vLLM forward passes.
The actual training (backward pass, optimizer steps) happens outside vLLM
using a separate HuggingFace model with LoRA adapters and weight sharing.

This manager only:
- Collects training losses for monitoring
- Provides statistics about training progress
- Does NOT perform backward passes (vLLM kernels don't support autograd)
"""

from typing import TYPE_CHECKING, Optional, Dict

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class TrainingManager:
    """
    Monitors training losses from vLLM forward passes.
    
    This class collects and reports training loss statistics.
    Actual training happens outside vLLM using ColocatedManager with LoRA.
    """
    
    def __init__(self, model_runner: "GPUModelRunner"):
        """
        Initialize the TrainingManager.
        
        Args:
            model_runner: The GPUModelRunner instance
        """
        self.model_runner = model_runner
        
        # Store loss history for monitoring
        self._loss_history: list[Dict[str, float]] = []
        
        logger.info("TrainingManager initialized (monitoring mode only)")
    
    def collect_losses(self) -> Optional[Dict[str, float]]:
        """
        Collect training losses from vLLM forward passes for monitoring.
        
        Note: This does NOT perform backward passes or parameter updates.
        vLLM's custom CUDA kernels don't support autograd. Actual training
        happens in ColocatedManager using a separate HF model with LoRA.
        
        Returns:
            Dictionary with loss statistics, or None if no training losses
        """
        # Check if there are training losses to process
        if not hasattr(self.model_runner, '_training_losses'):
            return None
        
        training_losses = self.model_runner._training_losses
        
        if not training_losses:
            return None
        
        # Extract loss values (convert from tensors to floats)
        loss_values = {}
        total_loss_value = 0.0
        
        for req_id, loss_tensor in training_losses.items():
            loss_value = float(loss_tensor.item())
            loss_values[req_id] = loss_value
            total_loss_value += loss_value
        
        avg_loss = total_loss_value / len(training_losses)
        
        logger.debug(
            f"Collected {len(training_losses)} training losses: "
            f"avg={avg_loss:.4f}"
        )
        
        # Clear losses
        training_losses.clear()
        
        # Create batch statistics
        batch_stats = {
            "num_requests": len(loss_values),
            "avg_loss": avg_loss,
            "total_loss": total_loss_value,
            "individual_losses": loss_values,
        }
        
        # Store in history for monitoring
        self._loss_history.append(batch_stats)
        
        return batch_stats
    
    def get_loss_history(self) -> list[Dict[str, float]]:
        """Get the complete history of training losses."""
        return self._loss_history
    
    def get_latest_loss(self) -> Optional[Dict[str, float]]:
        """Get the most recent training loss statistics."""
        return self._loss_history[-1] if self._loss_history else None

