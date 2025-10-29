"""
Token Parallel Linear Layers for vLLM.

This module implements token parallel versions of linear layers used in attention.
In token parallelism, only the root rank (rank 0) in each token parallel group 
loads weights and computes projections, while other ranks focus on attention 
computation for their assigned token partitions.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass, fields

from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.distributed.parallel_state import (
    get_tknp_rank, 
    get_tknp_world_size, 
    get_tknp_group, 
    is_tknp_initialized
)

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank
)

from vllm.forward_context import ForwardContext, get_forward_context

from vllm.logger import init_logger

logger = init_logger(__name__)

class TokenParallelQKVLinear(QKVParallelLinear):
    """
    QKV projection layer with Token Parallelism.
    
    This class extends the functionality of QKVParallelLinear to support
    Token Parallelism. In this scheme, only the root rank (rank 0) in each
    token parallel group holds the model weights and performs the linear
    projection. The resulting QKV tensor is then scattered across all ranks
    in the token parallel group along the batch/token dimension.

    This design allows non-root ranks to offload the memory burden of storing
    weights while still participating in the distributed computation. Each rank
    receives its own slice of the QKV tensor and can proceed with attention

    computation independently.
    """
    
    def __init__(self, *args, **kwargs):

        # We need to call the nn.Module constructor first.
        nn.Module.__init__(self)
        
        # Determine the rank's role in the token parallel group.
        self.is_tknp_enabled = is_tknp_initialized()
        self.tknp_rank = get_tknp_rank() if self.is_tknp_enabled else 0
        self.tknp_world_size = get_tknp_world_size() if self.is_tknp_enabled else 1
        self.is_root_rank = (self.tknp_rank == 0)
        self.tknp_group = get_tknp_group() if self.is_tknp_enabled else None
        self.pg = self.tknp_group.device_group if hasattr(self.tknp_group, 'device_group') else self.tknp_group
        self.device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')

        self.params_dtype = kwargs.get('params_dtype', None)
        if self.params_dtype is None:
            # Fallback to torch default or common model dtypes
            self.params_dtype = torch.get_default_dtype()

        if self.is_root_rank:
            # Root rank initializes the full QKVParallelLinear layer with weights.
            super().__init__(*args, **kwargs)
        else:
            # Non-root ranks do not have weights. They only store the configuration
            # needed to know the shape of the tensor they will receive.
            # We avoid calling super().__init__ to prevent weight allocation.
            
            # Extract configuration from args and kwargs
            hidden_size = kwargs.get('hidden_size', args[0] if args else None)
            head_size = kwargs.get('head_size', args[1] if args else None)
            total_num_heads = kwargs.get('total_num_heads', args[2] if args else None)
            total_num_kv_heads = kwargs.get('total_num_kv_heads', args[3] if args else None)

            self.hidden_size = hidden_size
            self.head_size = head_size
            self.total_num_heads = total_num_heads
            self.total_num_kv_heads = total_num_kv_heads
            
            # Get tensor parallel config
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_rank = get_tensor_model_parallel_rank()

            # Calculate the number of heads for this tensor parallel rank
            self.num_heads = self.total_num_heads // self.tp_size
            self.num_kv_heads = self.total_num_kv_heads // self.tp_size
            
            # The output dimension for this tensor parallel rank
            self.qkv_size_per_partition = (self.num_heads + 2 * self.num_kv_heads) * self.head_size
            
            # Register parameters as None since they don't exist on this rank
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        logger.debug(f"TokenParallelQKVLinear initialized on rank {self.tknp_rank}, "
                     f"is_root_rank: {self.is_root_rank}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Token Parallel QKV projection.

        Args:
            x: Input tensor of shape (num_tokens, hidden_size).

        Returns:
            A tensor of shape (num_tokens / tknp_world_size, qkv_size_per_partition)
            containing the QKV projections for the tokens assigned to the current rank.
        """
        # If token parallelism is not enabled, act as a standard QKVParallelLinear layer.
        if not self.is_tknp_enabled:
            return super().forward(x)

        # === Token Parallelism is ENABLED ===
        
        # 1. Validate input shape for even distribution
        # num_tokens, _ = x.shape
        if get_forward_context().tknp_metadata._dummy_run:
            num_reqs = get_forward_context().tknp_metadata.num_actual_tokens
        else:
            num_reqs = get_forward_context().tknp_metadata.num_reqs
            
        # TODO: we need to figure out if we are in prefill or decode 
        # if prefill, we send the all qkv data to respective ranks using their seq lens
        # if decode, we send only 1 token per rank to respective ranks
        
        assert num_reqs % self.tknp_world_size == 0, (
            f"Number of requests ({num_reqs}) must be divisible by "
            f"token parallel world size ({self.tknp_world_size})."
        )
        
        # 2. Prepare tensors for the scatter operation
        if self.is_root_rank:
            # Root rank computes the full QKV projection and chunks it.
            qkv_full, bias = super().forward(x)
            qkv_chunks = list(torch.chunk(qkv_full, self.tknp_world_size, dim=0))
            qkv_local = qkv_chunks[0]
        else:
            # Non-root ranks prepare an empty tensor to receive their partition.
            local_num_tokens = num_reqs // self.tknp_world_size
            qkv_local = torch.empty(
                (local_num_tokens, self.qkv_size_per_partition),
                dtype=self.params_dtype,
                device=self.device,
            )
            # The scatter_list is only needed on the source rank.
            qkv_chunks = None
            
        # 3. Distribute the QKV tensor from the root to all ranks
        dist.scatter(
            tensor=qkv_local,
            scatter_list=qkv_chunks,
            src=0,  # Root rank (0) in the token parallel group
            group=self.pg,
        )

        return qkv_local, None

    def extra_repr(self) -> str:
        """String representation for debugging."""
        if hasattr(self, 'hidden_size'):
            return f"hidden_size={self.hidden_size}, is_root_rank={self.is_root_rank}"
        return f"is_root_rank={self.is_root_rank}"

class TokenParallelRowLinear(RowParallelLinear):
    """
    Row-parallel linear layer with Token Parallelism.

    This class extends `RowParallelLinear` to support Token Parallelism. In this
    scheme, the input tensor is distributed along the token/batch dimension
    across multiple ranks in a token parallel group.

    The forward pass follows a gather-then-compute pattern:
    1.  The local input tensors from each rank are gathered onto the root rank (rank 0)
        of the token parallel group.
    2.  The root rank concatenates these tensors to reconstruct the full input tensor.
    3.  The root rank, which holds the actual model weights, performs the
        row-parallel linear projection on the full tensor. The base class's
        `forward` method handles the necessary all-reduce across the
        *tensor parallel* group.
    4.  The final output tensor is returned by the root rank, while non-root
        ranks return None.
    """

    def __init__(self, *args, **kwargs):
        # We need to call the nn.Module constructor first.
        nn.Module.__init__(self)

        # Determine the rank's role in the token parallel group.
        self.is_tknp_enabled = is_tknp_initialized()
        self.tknp_rank = get_tknp_rank() if self.is_tknp_enabled else 0
        self.tknp_world_size = get_tknp_world_size() if self.is_tknp_enabled else 1
        self.is_root_rank = (self.tknp_rank == 0)
        self.tknp_group = get_tknp_group() if self.is_tknp_enabled else None
        self.pg = self.tknp_group.device_group if hasattr(self.tknp_group, 'device_group') else self.tknp_group


        if self.is_root_rank:
            # Root rank initializes the full RowParallelLinear layer with weights.
            super().__init__(*args, **kwargs)
        else:
            # Non-root ranks do not have weights. They only participate in the
            # gather operation. We avoid calling super().__init__() to save memory.
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        logger.debug(f"TokenParallelRowLinear initialized on rank {self.tknp_rank}, "
                     f"is_root_rank: {self.is_root_rank}")


    def forward(
        self,
        x: torch.Tensor
    ) -> Optional[Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.nn.Parameter]]]]:
        """
        Forward pass for Token Parallel Row-Parallel projection.

        Args:
            x: The local input tensor of shape (num_local_tokens, hidden_size).

        Returns:
            - On the root rank: The final output tensor (or a tuple with bias)
              of shape (num_global_tokens, output_size).
            - On non-root ranks: None.
        """
        # If token parallelism is not enabled, act as a standard RowParallelLinear layer.
        if not self.is_tknp_enabled:
            return super().forward(x)

        # === Token Parallelism is ENABLED ===
        
        # 1. Prepare for the gather operation.
        if self.is_root_rank:
            # The root rank prepares a list of tensors to receive data from all ranks.
            gather_list = [torch.empty_like(x) for _ in range(self.tknp_world_size)]
        else:
            # Non-root ranks don't need a receive list.
            gather_list = None

        # 2. Gather all local input tensors onto the root rank.
        # This is a blocking operation. All ranks in the group must call it.
        dist.gather(
            tensor=x,
            gather_list=gather_list,
            dst=0,  # Destination is the root rank (0) in the token parallel group
            group=self.pg,
        )

        # 3. Compute and return the result on the root rank.
        if self.is_root_rank:
            # The gather_list on the root rank is now populated.
            # Concatenate the gathered tensors to form the global input.
            global_input = torch.cat(gather_list, dim=0)

            # Perform the forward pass on the complete tensor.
            # The base class's forward method will handle tensor-parallel all-reduce.
            output = super().forward(global_input)
            return output
        else:
            # Non-root ranks have sent their data and are done. They return None.
            return None, None
    
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
    
# A decorator for MLP, normalization, standard layers to be used in token parallelism.
def init_tknp_layer(cls_to_wrap: type) -> type:
    """
    A class decorator that replaces a module with an identity function
    on non-root ranks when token parallelism is enabled.
    """

    class TokenParallelWrapper(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

            # 1. Check for token parallel init and ranks
            self.is_tknp_enabled = is_tknp_initialized()
            self.tknp_rank = get_tknp_rank() if self.is_tknp_enabled else 0
            self.tknp_world_size = get_tknp_world_size() if self.is_tknp_enabled else 1
            self.is_root_rank = (self.tknp_rank == 0)
            # self.hidden_size 

            # 2. If self.is_root_rank is True, setup the regular class.
            if self.is_root_rank:
                # Instantiate the actual module we are wrapping (e.g., LlamaMLP)
                # and pass all arguments to it.
                self.module = cls_to_wrap(*args, **kwargs)
                # self.weight = self.module.weight if hasattr(self.module, 'weight') else None
                # self.bias = self.module.bias if hasattr(self.module, 'bias') else None
            # 3. If we are not the root rank, setup an identity function.
            else:
                # On non-root ranks, we just need a placeholder that
                # passes tensors through without any computation.
                self.module = nn.Identity()

            self.embedding_dim = kwargs.get('embedding_dim', args[1] if len(args) > 1 else None)
            self._return_tuple = None  # Cache to remember if forward returns tuple or single value
            if "RMSNorm" in cls_to_wrap.__name__ or "LayerNorm" in cls_to_wrap.__name__:
                self._return_tuple = True
            else:
                self._return_tuple = False
            # if self.embedding_dim:
            #     logger.info(f"TKNP RANK {self.tknp_rank}: Initialized TokenParallelWrapper for {cls_to_wrap.__name__} with embedding_dim={self.embedding_dim}")


        def forward(self, *args, **kwargs):
            # Delegate the forward call to the instantiated module.
            # This will be either the real module's forward or nn.Identity's forward.
            
            # Special handling for embedding layers on non-root ranks
            if self.embedding_dim and not self.is_root_rank:
                # On non-root ranks, if this is an embedding layer,
                # create a zero tensor with the correct shape.
                input_tensor = args[0] if args else kwargs.get('input_', None)
                if input_tensor is not None and input_tensor.dim() == 1:
                    # input_tensor has shape [num_tokens], need [num_tokens, embedding_dim]
                    batch_size = input_tensor.shape[0]
                    return torch.zeros(batch_size, self.embedding_dim,
                                    dtype=torch.bfloat16, # We might need to update this to use the same dtype as the model
                                    device=input_tensor.device)
            
            if not self.is_root_rank:
                # For other layers on non-root ranks, match return type to input args
                input_tensor = args[0] if args else kwargs.get('input_', None)
                
                # Return format based on number of arguments
                if len(args) == 1:
                    # Single argument -> return single tensor
                    return input_tensor
                elif len(args) == 2:
                    # Two arguments -> return tuple
                    return (input_tensor, args[1])
                else:
                    # Default: return single value for other cases
                    return input_tensor
            
            return self.module(*args, **kwargs)

        def __getattr__(self, name):
            # Forward attribute access to the wrapped module
            # This is called only when the attribute is not found in the wrapper itself
            try:
                return super().__getattr__(name)
            except AttributeError:
                # Forward to the wrapped module
                return getattr(self.module, name)
            
        def named_parameters(self, *args, **kwargs):
            """Forward named_parameters to the wrapped module."""
            return self.module.named_parameters(*args, **kwargs)
        
        def parameters(self, *args, **kwargs):
            """Forward parameters to the wrapped module."""
            return self.module.parameters(*args, **kwargs)
        
        def named_modules(self, *args, **kwargs):
            """Forward named_modules to the wrapped module."""
            return self.module.named_modules(*args, **kwargs)
        
        def named_buffers(self, *args, **kwargs):
            """Forward named_buffers to the wrapped module."""
            return self.module.named_buffers(*args, **kwargs)

    return TokenParallelWrapper