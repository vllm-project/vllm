import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from vllm.sequence import IntermediateTensors
from vllm.worker.model_runner_base import ModelRunnerWrapperBase
from vllm.logger import init_logger

logger = init_logger(__name__)

class EarlyExitModelRunner(ModelRunnerWrapperBase):
    """ModelRunner wrapper that performs early exit at specified layer."""
    
    def __init__(self, base_runner, exit_layer: int, lsq_head: Optional[nn.Module] = None):
        super().__init__(base_runner)
        self.exit_layer = exit_layer
        self.lsq_head = lsq_head
    
    @property
    def can_capture_graph(self) -> bool:
        return False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
        sampling_metadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Tuple[torch.Tensor, Optional[IntermediateTensors]]:
        """Forward pass that exits early at specified layer."""
        model = self.model
        
        if hasattr(model, 'model'):
            embed_tokens = model.model.embed_tokens
            layers = model.model.layers
            norm = model.model.norm
        else:
            embed_tokens = model.embed_tokens
            layers = model.layers
            norm = model.norm
        
        hidden_states = embed_tokens(input_ids)
        
        pp_rank = getattr(self.device_config, 'pp_rank', 0)
        pp_world_size = getattr(self.device_config, 'pp_world_size', 1)
        total_num_layers = self.model_config.hf_config.num_hidden_layers
        layers_per_rank = total_num_layers // pp_world_size
        
        for i in range(len(layers)):
            global_layer_idx = i + pp_rank * layers_per_rank
            if global_layer_idx > self.exit_layer:
                break
                
            layer = layers[i]
            layer_outputs = layer(
                positions=positions,
                hidden_states=hidden_states,
                kv_caches=kv_caches[i] if kv_caches else None,
                attn_metadata=attn_metadata,
            )
            
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
        
        hidden_states = norm(hidden_states)
        return hidden_states, None
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata
    ) -> torch.Tensor:
        """Compute logits using LSQ head if available, else use original LM head."""
        if self.lsq_head is not None:
            logits = self.logits_processor(
                self.lsq_head,
                hidden_states,
                sampling_metadata
            )
        else:
            logits = self.logits_processor(
                self.model.lm_head,
                hidden_states,
                sampling_metadata
            )
        return logits


def load_lsq_head(path: str, layer: int, device: torch.device, 
                  dtype: torch.dtype) -> Optional[nn.Module]:
    """Load LSQ projection head for early exit layer."""
    if not path:
        return None
    
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


class EarlyExitModule(nn.Module):
    """Module wrapper that runs only layers [0..exit_layer] inclusive."""
    
    def __init__(self, base_runner, exit_layer: int):
        super().__init__()
        self._base_runner = base_runner
        self._base_model = base_runner.model
        self.exit_layer = exit_layer

        self._inner_model = getattr(self._base_model, "model", self._base_model)

        decoder = getattr(self._inner_model, "decoder", None)
        if decoder is not None:
            self.embed_tokens = getattr(decoder, "embed_tokens", None)
            self._layers = getattr(decoder, "layers", None)
            self._final_norm = getattr(decoder, "final_layer_norm", None)
        else:
            self.embed_tokens = getattr(self._inner_model, "embed_tokens", None)
            self._layers = getattr(self._inner_model, "layers", None)
            self._final_norm = getattr(self._inner_model, "norm", None)
        
        if self.embed_tokens is None:
            self.embed_tokens = getattr(self._base_model, "embed_tokens", None)
        
        assert self.embed_tokens is not None, "embed_tokens not found"
        assert self._layers is not None, "transformer layers not found"  
        assert self._final_norm is not None, "final norm not found"

        self._pp_rank = getattr(self._base_runner.device_config, "pp_rank", 0)
        self._pp_world = getattr(self._base_runner.device_config, "pp_world_size", 1)
        num_total = getattr(self._base_runner.model_config.hf_config, "num_hidden_layers", None)
        assert num_total is not None, "num_hidden_layers missing on model_config.hf_config"
        self._num_total_layers = int(num_total)
        self._per_rank = max(1, self._num_total_layers // max(1, self._pp_world))
        
        self._early_exit_layers = self._create_early_exit_layers()
        self.model = self._create_model_wrapper()

    def _create_early_exit_layers(self):
        """Create a ModuleList containing only the layers we'll execute (0..exit_layer)."""
        layers_on_rank = []
        for i in range(len(self._layers)):
            global_idx = i + self._pp_rank * self._per_rank
            if global_idx <= self.exit_layer:
                layers_on_rank.append(self._layers[i])
            else:
                break
        return nn.ModuleList(layers_on_rank)
    
    def _create_model_wrapper(self):
        """Create a wrapper that exposes our limited layers in the expected structure."""
        
        class ModelWrapper:
            def __init__(self, early_exit_module):
                self._early_exit = early_exit_module
                
            def __getattr__(self, name):
                return getattr(self._early_exit._inner_model, name)
        
        wrapper = ModelWrapper(self)
        
        if hasattr(self._inner_model, "decoder"):
            class DecoderWrapper:
                def __init__(self, early_exit_module):
                    self._early_exit = early_exit_module
                    self.layers = early_exit_module._early_exit_layers
                    
                def __getattr__(self, name):
                    return getattr(self._early_exit._inner_model.decoder, name)
                    
            wrapper.decoder = DecoderWrapper(self)
        else:
            wrapper.layers = self._early_exit_layers
            
        return wrapper

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._base_model, name)

    def forward(self, input_ids: torch.Tensor = None, positions: torch.Tensor = None,
                kv_caches = None, attn_metadata=None) -> torch.Tensor:
        """Run layers [0..exit_layer] inclusive and return final hidden states."""
        x = self.embed_tokens(input_ids)

        for i in range(len(self._layers)):
            global_idx = i + self._pp_rank * self._per_rank
            if global_idx > self.exit_layer:
                break

            layer = self._layers[i]
            out = layer(
                positions=positions,
                hidden_states=x,
                kv_caches=kv_caches[i] if kv_caches is not None else None,
                attn_metadata=attn_metadata,
            )
            x = out[0] if isinstance(out, tuple) else out

        x = self._final_norm(x)
        return x