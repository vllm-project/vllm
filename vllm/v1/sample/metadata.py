# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SamplingMetadata:

    temperature: torch.Tensor

    top_p: Optional[torch.Tensor]
    top_k: Optional[torch.Tensor]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: Optional[int]
