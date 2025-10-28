from typing import Optional
import torch

class TrainingState:
    def __init__(self, grad_accumulation_steps: int = 1):
        self.grad_accumulation_steps: int = grad_accumulation_steps

        self._loss: torch.Tensor = torch.tensor(0.0)
        self._total_steps: int = 0
        self._steps: int = 0

    @property
    def loss(self) -> float:
        return self._loss.item()

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def add_loss(self, loss: torch.Tensor):
        self._loss += loss.to(self._loss.device)

    def step(self):
        self._steps += 1
        self._total_steps += 1

    def reset_steps(self):
        self._steps = 0

    def reset_loss(self):
        self._loss.zero_()