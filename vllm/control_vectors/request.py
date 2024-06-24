import torch
from typing import List, Optional
from vllm.control_vectors.control import ControlVector
from dataclasses import dataclass, field

@dataclass
class ControlVectorRequest:
    name: str
    control_vector: ControlVector
    coefficient: float = 1.0
    normalize: bool = False

    def __post_init__(self):
        if not isinstance(self.coefficient, float):
            raise TypeError(f"coefficient must be a float, got {type(self.coefficient).__name__}")
        if not isinstance(self.normalize, bool):
            raise TypeError(f"normalize must be a bool, got {type(self.normalize).__name__}")

    def get_control_vector(self) -> Optional[torch.tensor]:
        if self.control_vector is not None:
            return self.control_vector
        else:
            return None

    def reset_coefficient(self):
        self.coefficient = 1.0


    def set_coefficient(self, new_coefficient: float) -> None:
        if not isinstance(new_coefficient, float):
            raise TypeError("New coefficient is not a float")

        self.coefficient = new_coefficient
    
    def __eq__(self, other):
        if isinstance(other, ControlVectorRequest):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)