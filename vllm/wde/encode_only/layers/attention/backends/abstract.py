from abc import ABC
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic, List
import torch
from vllm.wde.core.layers.attention.abstract import AttentionMetadata, AttentionBackend, AttentionImpl, AttentionMetadataBuilder
from vllm.utils import is_pin_memory_available

pin_memory = is_pin_memory_available()


class EncodeOnlyAttentionBackend(AttentionBackend, ABC):
    pass


class EncodeOnlyAttentionImpl(AttentionImpl, ABC):
    pass


@dataclass
class EncodeOnlyAttentionMetadata(AttentionMetadata):
    max_seq_len: int
    seq_lens: list[int]

    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]


T = TypeVar("T", bound=AttentionMetadata)


class EncodeOnlyAttentionMetadataBuilder(AttentionMetadataBuilder, Generic[T]):

    def __init__(self):
        pass

    def __call__(self, seq_lens: List[int]):
        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.long,
                                       pin_memory=pin_memory,
                                       device="cpu")
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device="cpu")
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        return EncodeOnlyAttentionMetadata(seq_lens=seq_lens,
                                           max_seq_len=max(seq_lens),
                                           seq_start_loc=seq_start_loc)
