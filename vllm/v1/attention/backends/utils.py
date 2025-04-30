# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch


@dataclass
class CommonAttentionMetadata:
    """
    Attention metadata attributes that can be shared by layers in different KV
    cache groups and thus having different block table.
    """
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
