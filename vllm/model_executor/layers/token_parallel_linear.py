"""
Token Parallel Linear Layers for vLLM.

This module implements token parallel versions of linear layers used in attention.
In token parallelism, only the root rank (rank 0) in each token parallel group 
loads weights and computes projections, while other ranks focus on attention 
computation for their assigned token partitions.
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple, Union, List

from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.distributed.parallel_state import (
    get_tknp_rank, 
    get_tknp_world_size, 
    get_tknp_group, 
    is_tknp_initialized
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class TokenParallelQKVLinear(QKVParallelLinear):
    """
    Token parallel QKV projection layer.
    
    Only the root rank (rank 0) in each token parallel group loads weights
    and computes QKV projections. The results are then broadcast to other
    ranks in the token parallel group.
    
    This allows other ranks to focus on attention computation for their
    assigned token partitions while maintaining the same interface as
    the standard QKVParallelLinear layer.
    """
    
    def __init__(self, *args, **kwargs):
        # Only initialize weights on root rank
        if not is_tknp_initialized() or get_tknp_rank() == 0:
            super().__init__(*args, **kwargs)
            self.is_root_rank = True
        else:
            # Non-root ranks don't need weights, just store config
            self.is_root_rank = False
            # Store essential attributes for interface compatibility
            self.hidden_size = kwargs.get('hidden_size', args[0] if args else None)
            self.head_size = kwargs.get('head_size', args[1] if len(args) > 1 else None)
            self.total_num_heads = kwargs.get('total_num_heads', args[2] if len(args) > 2 else None)
            self.total_num_kv_heads = kwargs.get('total_num_kv_heads', args[3] if len(args) > 3 else None)
            self.bias = kwargs.get('bias', False)
            
        logger.debug(f"TokenParallelQKVLinear initialized on rank {get_tknp_rank()}, "
                    f"is_root_rank: {self.is_root_rank}")
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for token parallel QKV projection.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            QKV projections (same format as QKVParallelLinear)
        """
        if not is_tknp_initialized():
            # Fallback to standard behavior if token parallelism not initialized
            return super().forward(x)
        
        tknp_group = get_tknp_group()
        
        if self.is_root_rank:
            # Root rank computes QKV projections
            qkv_output = super().forward(x)
        else:
            # Non-root ranks prepare placeholder tensors
            # We need to know the output shape to create proper placeholders
            batch_size, seq_len = x.shape[:2]
            
            # Calculate output dimensions based on stored config
            if hasattr(self, 'num_heads'):
                # If we have access to the actual layer config
                total_hidden_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_size
            else:
                # Estimate based on typical QKV layout (this is a fallback)
                total_hidden_size = self.hidden_size + 2 * (self.hidden_size // 8)  # Assume GQA ratio
                
            qkv_output = torch.empty(
                (batch_size, seq_len, total_hidden_size),
                dtype=x.dtype,
                device=x.device
            )
        
        # Broadcast QKV results from root rank to all ranks in token parallel group
        dist.broadcast(qkv_output, src=0, group=tknp_group.device_group)
        
        return qkv_output
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        if hasattr(self, 'hidden_size'):
            return f"hidden_size={self.hidden_size}, is_root_rank={self.is_root_rank}"
        return f"is_root_rank={self.is_root_rank}"


class TokenParallelRowLinear(RowParallelLinear):
    """
    Token parallel output projection layer.
    
    Only the root rank (rank 0) in each token parallel group loads weights
    and computes output projections. The results are then broadcast to other
    ranks in the token parallel group.
    
    This complements TokenParallelQKVLinear to provide a complete token
    parallel attention implementation.
    """
    
    def __init__(self, *args, **kwargs):
        # Only initialize weights on root rank
        if not is_tknp_initialized() or get_tknp_rank() == 0:
            super().__init__(*args, **kwargs)
            self.is_root_rank = True
        else:
            # Non-root ranks don't need weights, just store config
            self.is_root_rank = False
            # Store essential attributes for interface compatibility
            self.input_size = kwargs.get('input_size', args[0] if args else None)
            self.output_size = kwargs.get('output_size', args[1] if len(args) > 1 else None)
            self.bias = kwargs.get('bias', True)
            self.reduce_results = kwargs.get('reduce_results', True)
            
        logger.debug(f"TokenParallelRowLinear initialized on rank {get_tknp_rank()}, "
                    f"is_root_rank: {self.is_root_rank}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for token parallel output projection.
        
        Args:
            x: Input tensor from attention output
            
        Returns:
            Output projection results
        """
        if not is_tknp_initialized():
            # Fallback to standard behavior if token parallelism not initialized
            return super().forward(x)
        
        tknp_group = get_tknp_group()
        
        if self.is_root_rank:
            # Root rank computes output projection
            output = super().forward(x)
        else:
            # Non-root ranks prepare placeholder tensor
            batch_size, seq_len = x.shape[:2]
            output = torch.empty(
                (batch_size, seq_len, self.output_size),
                dtype=x.dtype,
                device=x.device
            )
        
        # Broadcast output results from root rank to all ranks in token parallel group
        dist.broadcast(output, src=0, group=tknp_group.device_group)
        
        return output
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        if hasattr(self, 'input_size') and hasattr(self, 'output_size'):
            return (f"input_size={self.input_size}, output_size={self.output_size}, "
                   f"is_root_rank={self.is_root_rank}")
        return f"is_root_rank={self.is_root_rank}"


def create_token_parallel_qkv_linear(
    hidden_size: int,
    head_size: int, 
    total_num_heads: int,
    total_num_kv_heads: Optional[int] = None,
    bias: bool = True,
    **kwargs
) -> TokenParallelQKVLinear:
    """
    Factory function to create TokenParallelQKVLinear layer.
    
    Args:
        hidden_size: Hidden dimension size
        head_size: Size of each attention head
        total_num_heads: Total number of attention heads
        total_num_kv_heads: Total number of key-value heads (for GQA/MQA)
        bias: Whether to include bias
        **kwargs: Additional arguments
        
    Returns:
        TokenParallelQKVLinear instance
    """
    return TokenParallelQKVLinear(
        hidden_size=hidden_size,
        head_size=head_size,
        total_num_heads=total_num_heads,
        total_num_kv_heads=total_num_kv_heads,
        bias=bias,
        **kwargs
    )


def create_token_parallel_row_linear(
    input_size: int,
    output_size: int,
    bias: bool = True,
    reduce_results: bool = True,
    **kwargs
) -> TokenParallelRowLinear:
    """
    Factory function to create TokenParallelRowLinear layer.
    
    Args:
        input_size: Input dimension size
        output_size: Output dimension size  
        bias: Whether to include bias
        reduce_results: Whether to reduce results across tensor parallel groups
        **kwargs: Additional arguments
        
    Returns:
        TokenParallelRowLinear instance
    """
    return TokenParallelRowLinear(
        input_size=input_size,
        output_size=output_size,
        bias=bias,
        reduce_results=reduce_results,
        **kwargs
    ) 