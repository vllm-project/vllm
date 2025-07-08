# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig


@dataclass
class LogitProcessorCtorArgs:
    vllm_config: VllmConfig
    device: torch.device
    is_pin_memory: bool
