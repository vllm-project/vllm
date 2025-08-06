from typing import Optional, List, Dict, Any, Iterable, Set
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensor

class ClassifierConfig:
    def __init__(self, name: str, location: str, num_hidden_layers: int, hidden_dim=None):
        self.name = name
        self.location = location
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        
        if self.num_hidden_layers > 0 and self.hidden_dim is None:
            raise ValueError(f"hidden_dim is required for head '{self.name}' with num_hidden_layers > 0")
    
    @classmethod
    def from_dict(cls, config):
        return cls(
            name=config["name"],
            location=config["location"],
            num_hidden_layers=config["num_hidden_layers"],
            hidden_dim=config.get("hidden_dim")
        )


class WithAdditionalHeads:
    """
    Extends model to add multiple binary classification heads.
    The additional_heads_config should be provided in vllm_config.additional_config["additional_heads_config"].

    Example:
        [
            {
                "name": "self_harm",
                "location": "/path/to/self_harm_head.safetensors",
                "num_hidden_layers": 2,
                "hidden_dim": 4096
            },
            {
                "name": "violence",
                "location": "/path/to/violence_head.safetensors",
                "num_hidden_layers": 0
            }
        ]
    """

    def __init__(self, *args, **kwargs):
        self.additional_heads_config = self._find_additional_heads_config(*args, **kwargs)
        self.heads = [
            self._build_classification_head(config.num_hidden_layers, config.hidden_dim) 
            for config in self.additional_heads_config
        ]

    def _find_additional_heads_config(self, *args, **kwargs) -> List[ClassifierConfig]:
        """Find and validate additional_heads_config in the given args and kwargs."""
        from vllm.config import VllmConfig  # avoid circular import

        vllm_config = None
        args_values = list(args) + list(kwargs.values())
        for arg in args_values:
            if isinstance(arg, VllmConfig):
                vllm_config = arg
                break
        if vllm_config is None:
            raise ValueError("VllmConfig is required")
        additional_heads_config = vllm_config.additional_config.get("additional_heads_config")
        if additional_heads_config is None or len(additional_heads_config) == 0:
            raise ValueError("additional_heads_config is required in additional_config")
    
        return [
            ClassifierConfig.from_dict(config) for config in additional_heads_config
        ]
    
    def _build_classification_head(self, num_hidden_layers: int, hidden_dim: Optional[int]) -> nn.Sequential:
        """Build a binary classification head with the specified number of hidden layers."""        
        if num_hidden_layers == 0:
            return nn.Sequential(nn.Linear(self.config.hidden_size, 1), nn.Sigmoid())

        layers = [nn.Linear(self.config.hidden_size, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.extend([nn.Linear(hidden_dim, 1), nn.Sigmoid()])
        return nn.Sequential(*layers)
    
    def _load_classification_heads_from_files(self) -> Set[str]:
        """Load the classification heads. Returns a set of loaded parameter names."""
        loaded_params = set()

        for i, config in enumerate(self.additional_heads_config):
            try:
                head = self.heads[i]
                state_dict = load_safetensor(config.location)
                head.load_state_dict(state_dict)
                for param_name in head.state_dict().keys():
                    loaded_params.add(f"heads.{i}.{param_name}")
            except Exception as e:
                raise RuntimeError(f"Failed to load state dict for head '{config.name}': {e}")
        
        return loaded_params

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Set[str]:
        """
        Override of load_weights that also loads the additional heads.
        Expects load_weights to be implemented in the parent class.
        """
        original_cls = super(WithAdditionalHeads, self)
        if not hasattr(original_cls, 'load_weights'):
            raise RuntimeError(
                "The parent class does not implement load_weights. "
                "Ensure that the parent class is compatible with WithAdditionalHeads."
            )

        loaded_params = original_cls.load_weights(weights)
        head_loaded_params = self._load_classification_heads_from_files()
        return loaded_params.union(head_loaded_params)

    def compute_additional_head(
        self, hidden_states: torch.Tensor, additional_heads_extra_inputs: Optional[List[Dict[str, Any]]] = None
    ) -> torch.Tensor:
        """
        Compute outputs for all additional heads.
        Returns a tensor of shape [batch_size, num_heads] with outputs for each head.
        """
        head_outputs = []
        for head in self.heads:
            logits = head(hidden_states)
            head_outputs.append(logits.squeeze(-1))
        return torch.stack(head_outputs, dim=1)
