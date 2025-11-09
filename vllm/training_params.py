# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

@dataclass
class TrainingParams:
    """Parameters for training requests."""
    is_eval: bool
    gradient_accumulation_steps: int
    num_training_steps: int
    num_warmup_steps: int
