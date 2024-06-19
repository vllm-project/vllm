from dataclasses import dataclass, field
import torch
from typing import Dict, List

@dataclass
class ControlVector:
    model_type: str ##TODO: prolly need to specify model architecture or sth
    directions: Dict[int, torch.Tensor]
    num_layers: int = field(init=False)

    
    def __post_init__(self):
        self.num_layers = len(self.directions)
        if not self.directions:
            raise ValueError("Directions cannot be an empty dictionary")
        if any(not isinstance(tensor, torch.Tensor) for tensor in self.directions.values()):
            raise TypeError("All directions must be torch.Tensor instances")

    def add(self, other: "ControlVector") -> "ControlVector":
        if self.model_type != other.model_type:
            raise ValueError("Model types do not match, cannot add ControlVectors.")
        
        combined_directions = {layer: tensor.clone() for layer, tensor in self.directions.items()}
        for layer, tensor in other.directions.items():
            if layer in combined_directions:
                combined_directions[layer] += tensor
            else:
                combined_directions[layer] = tensor.clone()
        
        return ControlVector(model_type=self.model_type, directions=combined_directions)

    def get_vector(self, id: int) -> torch.Tensor:
        if id not in self.directions:
            raise ValueError(f"Layer {id} not found in the control vector directions.")
        return self.directions[id]

    def get_layer_ids(self) -> List[int]:
        return list(self.directions.keys())

    def __add__(self, other: "ControlVector") -> "ControlVector":
        return self.add(other)

    def __sub__(self, other: "ControlVector") -> "ControlVector":
        return self.add(-other)

    def __neg__(self):
        negated_directions = {layer: -tensor.clone() for layer, tensor in self.directions.items()}
        return ControlVector(model_type=self.model_type, directions=negated_directions)

    def __mul__(self, scalar: float):
        if not isinstance(scalar, (float, int)):
            raise TypeError("Scalar multiplier must be a float or int.")
        
        scaled_directions = {layer: tensor * scalar for layer, tensor in self.directions.items()}
        return ControlVector(model_type=self.model_type, directions=scaled_directions)

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)