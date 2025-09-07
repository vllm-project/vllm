# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


@dataclass
class InputBatch:

    # batch_idx -> req_id
    req_ids: list[str]

    # req_id -> batch_idx
    req_id_to_batch_idx: dict[str, int]

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray

    # batch_idx -> num_scheduled_tokens
    num_scheduled_tokens: np.ndarray
    total_num_tokens: int
    max_query_len: int
    num_reqs: int

    attn_metadata: dict[str, Any]
    spec_decode_common_attn_metadata: Optional[Any]
    spec_decode_metadata: Optional[SpecDecodeMetadata]

    logits_indices: torch.Tensor
