# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch


@dataclass
class CommonAttentionMetadata:
    """
    Attention Metadata that are same for different layer types.
    """
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
