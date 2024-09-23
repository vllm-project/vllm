from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

import torch


class AttentionType(Enum):
    DECODER = auto()  # Decoder attention between previous layer Q/K/V
    ENCODER = auto()  # Encoder attention between previous layer Q/K/V
    ENCODER_DECODER = auto()  # Attention between dec. Q and enc. K/V

    @staticmethod
    def attn_type_name_to_enum(attn_type: str) -> "AttentionType":
        assert attn_type is not None

        attn_type_members = AttentionType.__members__
        if attn_type not in attn_type_members:
            raise ValueError(
                f"Invalid attn_type '{attn_type}'. "
                f"Available backends: {', '.join(attn_type_members)} "
                "(case-sensitive).")

        return AttentionType[attn_type]


class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    def __init__(self, attn_type: AttentionType):
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
    def get_impl_cls() -> Type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        raise NotImplementedError

    @classmethod
    def make_metadata(cls, *args, **kwargs) -> "AttentionMetadata":
        return cls.get_metadata_cls()(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @classmethod
    def make_metadata_builder(cls, *args,
                              **kwargs) -> "AttentionMetadataBuilder":
        return cls.get_builder_cls()(*args, **kwargs)


@dataclass
class AttentionMetadata:
    pass

    def to(self, device, non_blocking=False):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device, non_blocking=non_blocking)

        return self


T = TypeVar("T", bound=AttentionMetadata)


class AttentionMetadataBuilder(ABC, Generic[T]):
    """Abstract class for attention metadata builders."""

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs) -> T:
        raise NotImplementedError


class AttentionImpl(ABC, Generic[T]):

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
        attn_metadata: AttentionMetadata,
        kv_cache: Optional[torch.Tensor] = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.ENCODER,
    ) -> torch.Tensor:
        raise NotImplementedError
