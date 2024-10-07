from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

import torch

from vllm.attention.backends.abstract import AttentionType
from vllm.utils import is_pin_memory_available

pin_memory = is_pin_memory_available()


class PrefillOnlyAttentionBackend(ABC):

    def __init__(self, attn_type: AttentionType):
        if attn_type == AttentionType.ENCODER_DECODER:
            raise NotImplementedError("Encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PrefillOnlyAttentionBackend")

        self._attn_type = attn_type

    @property
    def attn_type(self) -> AttentionType:
        return self._attn_type

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> Type["PrefillOnlyAttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    def get_metadata_cls() -> Type["PrefillOnlyAttentionMetadata"]:
        return PrefillOnlyAttentionMetadata

    @classmethod
    def make_metadata(cls, *args, **kwargs) -> "PrefillOnlyAttentionMetadata":
        return cls.get_metadata_cls()(*args, **kwargs)

    @staticmethod
    def get_builder_cls() -> Type["PrefillOnlyAttentionMetadataBuilder"]:
        return PrefillOnlyAttentionMetadataBuilder

    @classmethod
    def make_metadata_builder(
            cls, *args, **kwargs) -> "PrefillOnlyAttentionMetadataBuilder":
        return cls.get_builder_cls()(*args, **kwargs)


@dataclass
class PrefillOnlyAttentionMetadata:
    max_seq_len: int
    seq_lens: List[int]

    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    def to(self, device, non_blocking=False):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device, non_blocking=non_blocking)

        return self


T = TypeVar("T", bound=PrefillOnlyAttentionMetadata)


class PrefillOnlyAttentionMetadataBuilder(Generic[T]):

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

        return PrefillOnlyAttentionMetadata(seq_lens=seq_lens,
                                            max_seq_len=max(seq_lens),
                                            seq_start_loc=seq_start_loc)


class PrefillOnlyAttentionImpl(ABC):

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: T,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        raise NotImplementedError
