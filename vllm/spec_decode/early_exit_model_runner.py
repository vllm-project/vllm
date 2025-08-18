import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from vllm.sequence import IntermediateTensors
from vllm.worker.model_runner_base import ModelRunnerWrapperBase
from vllm.logger import init_logger

logger = init_logger(__name__)

class EarlyExitModelRunner(ModelRunnerWrapperBase):
    """ModelRunner wrapper that performs early exit at specified layer.
    
    This wraps the scorer's ModelRunner to exit early during forward passes,
    optionally using a learned LSQ projection head instead of the LM head.
    
    NOTE: This implementation does not support CUDA graphs. The early-exit
    forward pass runs in eager mode, which may result in ~15-20% lower decode
    throughput compared to graph-captured execution. CUDA graph support would
    require special handling of the dynamic exit behavior.
    """
    
    def __init__(self, base_runner, exit_layer: int, lsq_head: Optional[nn.Module] = None):
        # Call parent constructor with the base runner
        super().__init__(base_runner)
        self.exit_layer = exit_layer
        # LSQ head must be loaded externally due to initialization order constraints
        # The scorer's model isn't loaded when this runner is created
        self.lsq_head = lsq_head
        
        # Expose all attributes that workers/scheduler might access
        # ModelRunnerWrapperBase already delegates most via __getattr__
        # but we explicitly set critical ones for clarity
        self.model = base_runner.model
        self.sampler = base_runner.sampler
        self.logits_processor = base_runner.logits_processor
        self.device = base_runner.device
        self.model_config = base_runner.model_config
        self.device_config = base_runner.device_config
        self.return_hidden_states = base_runner.return_hidden_states
        self.vllm_config = getattr(base_runner, 'vllm_config', None)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
        sampling_metadata,  # Required by vLLM's ModelRunner interface
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[IntermediateTensors]]:
        """Forward pass that exits early at specified layer.
        
        This method is called by the base ModelRunner's execute_model().
        We intercept the forward pass to implement early exit behavior.
        
        NOTE: This implementation does not support CUDA graphs for the early-exit
        path. The model will run in eager mode, which may have ~15-20% lower
        decode throughput compared to graph-captured execution.
        """
        model = self.model
        
        # Handle different model architectures (e.g., LlamaForCausalLM has model.model)
        if hasattr(model, 'model'):
            embed_tokens = model.model.embed_tokens
            layers = model.model.layers
            norm = model.model.norm
        else:
            # Direct access for models without nested structure
            embed_tokens = model.embed_tokens
            layers = model.layers
            norm = model.norm
        
        # Get input embeddings
        hidden_states = embed_tokens(input_ids)
        
        # Process layers up to exit point
        # Handle pipeline parallelism: calculate global layer index
        pp_rank = getattr(self.device_config, 'pp_rank', 0)
        pp_world_size = getattr(self.device_config, 'pp_world_size', 1)
        total_num_layers = self.model_config.hf_config.num_hidden_layers
        layers_per_rank = total_num_layers // pp_world_size
        
        for i in range(len(layers)):
            # Calculate global layer index for pipeline parallel
            global_layer_idx = i + pp_rank * layers_per_rank
            
            # Stop if we've reached the exit layer (inclusive semantics)
            if global_layer_idx > self.exit_layer:
                break
                
            layer = layers[i]
            
            # Most transformer layers in vLLM follow this pattern
            layer_outputs = layer(
                positions=positions,
                hidden_states=hidden_states,
                kv_caches=kv_caches[i] if kv_caches else None,
                attn_metadata=attn_metadata,
                # Note: sampling_metadata is not used by transformer layers
                # but we keep it available for consistency
            )
            
            # Handle different return formats
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
        
        # Apply final layer norm
        hidden_states = norm(hidden_states)
        
        return hidden_states, None
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata
    ) -> torch.Tensor:
        """Compute logits using LSQ head if available, else use original LM head.
        
        This is called by the base ModelRunner after forward().
        """
        if self.lsq_head is not None:
            # Use LSQ projection head
            logits = self.logits_processor(
                self.lsq_head,
                hidden_states,
                sampling_metadata
            )
        else:
            # Fallback to original LM head
            logits = self.logits_processor(
                self.model.lm_head,
                hidden_states,
                sampling_metadata
            )
        return logits


def load_lsq_head(path: str, layer: int, device: torch.device, 
                  dtype: torch.dtype) -> Optional[nn.Module]:
    """Load LSQ projection head for early exit layer.
    
    Args:
        path: Directory containing h{layer}.pt files
        layer: Which layer's projection head to load
        device: Target device for the head
        dtype: Target dtype for the head
        
    Returns:
        nn.Linear module with loaded weights, or None if not found
    """
    if not path:
        return None
    
    import os
    from pathlib import Path
    
    head_file = Path(path) / f"h{layer}.pt"
    if not head_file.exists():
        logger.warning(f"LSQ head file not found: {head_file}")
        return None
    
    try:
        # Load weights to CPU first
        weight = torch.load(head_file, map_location="cpu")
        
        # Validate shape
        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight tensor, got {weight.dim()}D")
        
        vocab_size, hidden_size = weight.shape
        
        # Create linear layer without bias (standard for LM heads)
        head = nn.Linear(hidden_size, vocab_size, bias=False)
        head.weight.data = weight.to(device=device, dtype=dtype)
        head.weight.requires_grad = False
        
        logger.info(f"Loaded LSQ head from {head_file} "
                   f"(vocab_size={vocab_size}, hidden_size={hidden_size})")
        return head
        
    except Exception as e:
        logger.error(f"Failed to load LSQ head from {head_file}: {e}")
        return None