# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch


@dataclass
class Speculation:
    # [num_reqs, num_speculative_steps]
    draft_tokens: torch.Tensor
    # [num_reqs, num_speculative_steps, vocab_size]
    draft_logits: torch.Tensor
