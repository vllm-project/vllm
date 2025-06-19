# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.pooling_params import PoolingParams


@dataclass
class PoolingMetadata:
    """Tensors for pooling."""

    prompt_lens: torch.Tensor
    prompt_token_ids: Optional[torch.Tensor]
    pooling_params: list[PoolingParams]
