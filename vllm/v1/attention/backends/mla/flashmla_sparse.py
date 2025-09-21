from vllm.attention.backends.abstract import AttentionMetadata, AttentionLayer
import torch
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonBackend, MLACommonDecodeMetadata, MLACommonImpl, MLACommonMetadata, MLACommonMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata, split_decodes_and_prefills
from dataclasses import dataclass
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import AttentionSpec
from typing import Optional

logger = init_logger(__name__)


class FlashMLASparseBackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE_VLLM_V1"

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return FlashMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashMLASparseMetadataBuilder"]:
        return FlashMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLASparseImpl"]:
        return FlashMLASparseImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        print("try running get_supported_dtypes")
        # TODO: verify this
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # TODO: verify this
        return [576]


class MLASparsePrefillMetadata:
    # NOTE(Chen): not call it "FlashMLASparsePrefillMetadata" because
    # the kernel is not from flashmla
    def __init__(self):
        pass


class FlashMLASparseDecodeMetadata(MLACommonDecodeMetadata):

    def __init__(self):
        pass


@dataclass
class FlashMLASparseMetadata(MLACommonMetadata[MLASparsePrefillMetadata]):
    pass


@dataclass
class FlashMLASparseMetadataBuilder(
        MLACommonMetadataBuilder[FlashMLASparseMetadata]):

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         FlashMLASparseMetadata)

    def _build_prefill(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> MLASparsePrefillMetadata:
        return MLASparsePrefillMetadata()

    def _build_decode(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlashMLASparseDecodeMetadata:
        return FlashMLASparseDecodeMetadata()

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> FlashMLASparseMetadata:
        logger.info(f"build FlashMLASparseMetadata")
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens =\
            split_decodes_and_prefills(common_attn_metadata,
                                       decode_threshold=self.reorder_batch_threshold)
        return FlashMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            prefill=self._build_prefill(common_attn_metadata),
            decode=self._build_decode(common_attn_metadata),
        )


@dataclass
class FlashMLASparseImpl(MLACommonImpl[FlashMLASparseMetadata]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        k_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty_like(q)

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        return torch.empty_like(q)

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return output.fill_(0)
