from abc import ABC
from dataclasses import dataclass
from typing import Optional
import torch
from vllm.wde.core.layers.attention.abstract import AttentionMetadata, AttentionBackend, AttentionImpl, AttentionMetadataBuilder


class EncodeOnlyAttentionBackend(AttentionBackend, ABC):
    pass


class EncodeOnlyAttentionImpl(AttentionImpl, ABC):
    pass


@dataclass
class EncodeOnlyAttentionMetadata(AttentionMetadata):
    # Maximum query length in the batch.
    max_seq_len: int
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]


class EncodeOnlyAttentionMetadataBuilder(AttentionMetadataBuilder, ABC):
    pass