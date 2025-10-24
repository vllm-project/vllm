# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional

# TODO(girfan): Check which ones we need and which can be removed
@dataclass
class TrainingParams:
    """Parameters for training requests.

    This allows a request to be used for training instead of inference.
    Training requests compute loss and cache outputs for backward pass.
    """
    # Whether to compute loss for this request
    compute_loss: bool = True

    # Loss function to use (currently only cross_entropy supported)
    loss_fn: str = "cross_entropy"

    # Whether to cache activations for backward pass
    cache_for_backward: bool = True

    # LoRA configuration for training
    # If provided, the model will use this LoRA adapter during training
    lora_request: Optional["LoRARequest"] = None

    # Whether this is an evaluation request (skips backward pass)
    is_eval: bool = False
