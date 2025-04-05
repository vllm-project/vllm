# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PoolingMetadata:
    """Tensors for pooling."""

    prompt_lens: torch.Tensor
    prompt_token_ids: Optional[torch.Tensor]
