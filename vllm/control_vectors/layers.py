from vllm.control_vectors.control import ControlVector
from vllm.control_vectors.request import ControlVectorRequest
import torch
import torch.nn as nn

class BaseLayerWithControlVector(nn.Module):
    pass



class MLPWithControlVector(BaseLayerWithControlVector):

    def __init__(self, base_layer) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.normalize = True
        self.cv_vector = None

    def set_cv_vector(self, cv_vector: torch.Tensor):
        self.cv_vector = cv_vector

    def get_cv_vector(self) -> torch.Tensor:
        return self.cv_vector

    def reset_cv_vector(self):
        self.cv_vector = None
    
    def forward(self, hidden_states, positions):
        hidden_states = self.base_layer.forward(hidden_states, positions)

        if self.cv_vector is not None:
            norm_pre = torch.norm(hidden_states, dim=-1, keepdim=True)

            control = self.cv_vector.to(hidden_states.device)

            if len(control.shape) == 1:
                control = control.view(1, -1).expand(hidden_states.shape[0], -1)

            if positions.ndim == 1:
                positions = positions.unsqueeze(0)

            zero_indices = (positions == 0).cumsum(1).argmax(1, keepdim=True)
            col_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device).unsqueeze(1)

            mask = (col_indices >= zero_indices).float().unsqueeze(-1) 
            mask = mask.expand(-1, -1, hidden_states.shape[1])

            control = control * mask.squeeze(1)
            hidden_states += control

            if self.normalize:
                hidden_states = hidden_states * (norm_pre / torch.norm(hidden_states, dim=-1, keepdim=True))
        
        return hidden_states

