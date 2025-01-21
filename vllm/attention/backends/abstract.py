from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import (TYPE_CHECKING, Any, Dict, Generic, List, Optional,
                    Protocol, Set, Tuple, Type, TypeVar)

import torch

from vllm.multimodal import MultiModalPlaceholderMap

if TYPE_CHECKING:
    from vllm.worker.model_runner_base import (ModelRunnerBase,
                                               ModelRunnerInputBase,
                                               ModelRunnerInputBuilderBase)


class AttentionType:
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """
    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Encoder attention between previous layer Q/K/V for encoder-decoder
    ENCODER = "encoder"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"
    # Attention between dec. Q and enc. K/V for encoder-decoder
    ENCODER_DECODER = "encoder_decoder"


class AttentionBackend(ABC):
    """Abstract class for attention backends."""
    # For some attention backends, we allocate an output tensor before
    # calling the custom op. When piecewise cudagraph is enabled, this
    # makes sure the output tensor is allocated inside the cudagraph.
    accept_output_buffer: bool = False

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

    @staticmethod
    @abstractmethod
    def get_state_cls() -> Type["AttentionState"]:
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

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def advance_step(self, model_input: "ModelRunnerInputBase",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int, num_seqs: int, num_queries: int) -> None:
        raise NotImplementedError


@dataclass
class AttentionMetadata:
    """Attention metadata for prefill and decode batched together."""
    # Total number of prefill requests.
    num_prefills: int
    # Number of prefill tokens.
    num_prefill_tokens: int
    # Number of decode tokens. Note that it is equivalent to the number of
    # decode requests.
    num_decode_tokens: int
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor

    # The index maps that relate multi-modal embeddings to the corresponding
    # placeholders.
    #
    # N.B. These aren't really related to attention and don't belong on this
    # type -- this is just a temporary solution to make them available to
    # `model_executable`.
    multi_modal_placeholder_index_maps: Optional[Dict[
        str, MultiModalPlaceholderMap.IndexMap]]

    @property
    @abstractmethod
    def prefill_metadata(self) -> Optional["AttentionMetadata"]:
        """Return the attention metadata that's required to run prefill
        attention."""
        pass

    @property
    @abstractmethod
    def decode_metadata(self) -> Optional["AttentionMetadata"]:
        """Return the attention metadata that's required to run decode
        attention."""
        pass

    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {
            field.name: getattr(self, field.name)
            for field in fields(self) if field.name not in skip_fields
        }


T = TypeVar("T", bound=AttentionMetadata)


class AttentionState(ABC, Generic[T]):
    """Holds attention backend-specific objects reused during the
    lifetime of the model runner."""

    @abstractmethod
    def __init__(self, runner: "ModelRunnerBase"):
        ...

    @abstractmethod
    @contextmanager
    def graph_capture(self, max_batch_size: int):
        """Context manager used when capturing CUDA graphs."""
        yield

    @abstractmethod
    def graph_clone(self, batch_size: int) -> "AttentionState[T]":
        """Clone attention state to save in CUDA graph metadata."""
        ...

    @abstractmethod
    def graph_capture_get_metadata_for_batch(
            self,
            batch_size: int,
            is_encoder_decoder_model: bool = False) -> T:
        """Get attention metadata for CUDA graph capture of batch_size."""
        ...

    @abstractmethod
    def get_graph_input_buffers(
            self,
            attn_metadata: T,
            is_encoder_decoder_model: bool = False) -> Dict[str, Any]:
        """Get attention-specific input buffers for CUDA graph capture."""
        ...

    @abstractmethod
    def prepare_graph_input_buffers(
            self,
            input_buffers: Dict[str, Any],
            attn_metadata: T,
            is_encoder_decoder_model: bool = False) -> None:
        """In-place modify input buffers dict for CUDA graph replay."""
        ...

    @abstractmethod
    def begin_forward(self, model_input: "ModelRunnerInputBase") -> None:
        """Prepare state for forward pass."""
        ...


class AttentionMetadataBuilder(ABC, Generic[T]):
    """Abstract class for attention metadata builders."""

    @abstractmethod
    def __init__(self, input_builder: "ModelRunnerInputBuilderBase") -> None:
        raise NotImplementedError

    @abstractmethod
    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int) -> T:
        """Build attention metadata with on-device tensors."""
        raise NotImplementedError


class AttentionLayer(Protocol):

    _k_scale: float
    _v_scale: float

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        ...


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
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
