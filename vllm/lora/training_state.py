from typing import Optional
import torch

class TrainingState:
    def __init__(self, grad_accumulation_steps: int = 1):
        self.grad_accumulation_steps: int = grad_accumulation_steps

        self._loss: Optional[torch.Tensor] = None
        self._steps: int = 0

    @property
    def loss(self) -> Optional[torch.Tensor]:
        return self._loss

    @property
    def steps(self) -> int:
        return self._steps

    def add_loss(self, loss: torch.Tensor):
        if self._loss is None:
            self._loss = loss
        else:
            self._loss += loss

    def step(self):
        self._steps += 1

    def reset(self):
        self._loss = None
        self._steps = 0