# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 attention layer on SM80.

DeepSeek-V4.1 attends each query token to two key sets: the last
`sliding_window` tokens of its request (the SWA cache, 528-byte MXFP8
records) and the `index_topk` compressed tokens its indexer selected (the
compressed cache, 288-byte NVFP4 records here). On SM90 and SM100 FlashMLA
runs that sparse MLA; on SM80 this layer does, with the same records:

* the SWA records are written by the stock fused Q/KV kernel, which runs on
  SM80;
* the compressor writes its NVFP4 records with `AmpereCompressor`;
* `forward_mqa` attends decode and prefill tokens alike, each over its own
  window and selected compressed keys, and writes the output already
  inverse-RoPE'd, so `_o_proj` is a BF16 grouped `wo_a` and `wo_b`.

The metadata is the stock one, minus FlashMLA's tile scheduler and DeepGEMM's
paged-logits schedule: the SWA and indexer builders here plan neither.
Prefill context parallelism is not supported.
"""

from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch

from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    DeepseekV4Indexer,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v41.common.ops import compute_global_topk_indices_and_lens
from vllm.models.deepseek_v41.compressor import DeepseekCompressor
from vllm.models.deepseek_v41.nvidia.flashmla import (
    DeepseekSparseSWAFlashMLABackend,
    DeepseekSparseSWAFlashMLAMetadataBuilder,
)
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.platforms.interface import DeviceCapability

from . import AMPERE_CAPABILITY, rope_table_f32
from .indexer import AmpereIndexer, AmpereIndexerCache
from .quantize import store_compressed_records
from .sparse_mla import sparse_mla

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class AmpereSWAMetadataBuilder(DeepseekSparseSWAFlashMLAMetadataBuilder):
    """SWA metadata without FlashMLA's tile-scheduler plans."""

    def build_tile_scheduler(self, num_decode_tokens: int) -> dict[str, Any]:
        """No plans: the base builds none for a step without decode tokens."""
        return super().build_tile_scheduler(0)


class AmpereSWABackend(DeepseekSparseSWAFlashMLABackend):
    """The SWA cache backend of the SM80 layer."""

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type[AmpereSWAMetadataBuilder]:
        return AmpereSWAMetadataBuilder


class AmpereSparseMLABackend(DeepseekV4FlashMLABackend):
    """The compressed-cache backend of the SM80 layer: FlashMLA's metadata
    and paging, with the NVFP4 compressed record."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "nvfp4_ds_mla",
    ]

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "AMPERE_SPARSE_MLA_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.to_int() == AMPERE_CAPABILITY


class AmpereCompressor(DeepseekCompressor):
    """The compressor, writing its 288-byte NVFP4 records on SM80."""

    def insert_cache(
        self,
        latent: torch.Tensor | None,
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        """Publish the group-boundary latents as NVFP4 compressed records."""
        if latent is None:
            return
        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict)
        slot_mapping = cast(Any, attn_metadata[self.k_cache_prefix]).slot_mapping
        kv_cache = self._static_forward_context[self.k_cache_prefix].kv_cache
        store_compressed_records(
            latent,
            positions,
            slot_mapping,
            rope_table_f32(rotary_emb),
            self.compress_ratio,
            kv_cache,
        )


class DeepseekV41AmpereAttention(DeepseekV4Attention):
    """DeepSeek-V4.1 attention on SM80 (A100)."""

    backend_cls = AmpereSparseMLABackend
    swa_backend_cls = AmpereSWABackend
    compressor_cls = AmpereCompressor

    @property
    def packed_kv_cache_dtype(self) -> CacheDType:
        """The NVFP4 compressed record, as on SM100's mega attention."""
        return "nvfp4_ds_mla"

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        """No padding: the kernel takes any head count."""
        return num_heads

    def _indexer_cls(
        self, k_cache: DeepseekV4IndexerCache | None
    ) -> type[DeepseekV4Indexer]:
        """The indexer with SM80 query quantization and logits."""
        return AmpereIndexer

    def _indexer_cache_cls(self) -> type[DeepseekV4IndexerCache]:
        """The index K cache with the SM80 indexer metadata."""
        return AmpereIndexerCache

    def _o_proj(
        self,
        attn_out: "torch.Tensor | QuantizedActivation",
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """`wo_a` per head group as a BF16 batched GEMM, then `wo_b`.

        `forward_mqa` already applied the inverse RoPE. `wo_a` is the
        MXFP8 emulation kernel's BF16 weight `[groups * o_lora_rank, K]`.
        """
        assert isinstance(attn_out, torch.Tensor)
        weight = self.wo_a.weight
        assert isinstance(weight, torch.Tensor)
        assert weight.dtype == torch.bfloat16, (
            "the SM80 wo_a needs the BF16 weight the MXFP8 emulation kernel "
            "dequantizes at load (VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=1)"
        )
        num_tokens = attn_out.shape[0]
        groups = self.n_local_groups
        o = attn_out.view(num_tokens, groups, -1).transpose(0, 1)
        z = torch.bmm(o, weight.view(groups, self.o_lora_rank, -1).transpose(1, 2))
        return self._wo_b_proj(z.transpose(0, 1).reshape(num_tokens, -1))

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Sparse MLA of every token over its window and selected keys."""
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            # Profile run: no metadata and nothing to read.
            output.zero_()
            return
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata", attn_metadata[self.swa_cache_layer.prefix]
        )
        compressed_metadata = (
            cast(
                DeepseekV4FlashMLAMetadata, attn_metadata[self.compressed_cache_prefix]
            )
            if self.compressed_cache_prefix is not None
            else None
        )
        num_decode = swa_metadata.num_decode_tokens
        num_prefill = swa_metadata.num_prefill_tokens
        output[num_decode + num_prefill :].zero_()
        segments = []
        if num_decode > 0:
            assert swa_metadata.decode_swa_indices is not None
            assert swa_metadata.decode_swa_lens is not None
            segments.append(
                (
                    0,
                    num_decode,
                    swa_metadata.decode_swa_indices.view(num_decode, -1),
                    swa_metadata.decode_swa_lens,
                )
            )
        if num_prefill > 0:
            assert swa_metadata.prefill_swa_indices is not None
            assert swa_metadata.prefill_swa_lens is not None
            segments.append(
                (
                    num_decode,
                    num_prefill,
                    swa_metadata.prefill_swa_indices.view(num_prefill, -1),
                    swa_metadata.prefill_swa_lens,
                )
            )
        for start, count, swa_indices, swa_lens in segments:
            rows = slice(start, start + count)
            compressed_cache = compressed_indices = compressed_lens = None
            if compressed_metadata is not None:
                assert self.topk_indices_buffer is not None
                assert swa_metadata.token_to_req_indices is not None
                assert swa_metadata.is_valid_token is not None
                compressed_indices, compressed_lens = (
                    compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[rows],
                        swa_metadata.token_to_req_indices[rows],
                        compressed_metadata.block_table,
                        compressed_metadata.block_size // self.compress_ratio,
                        swa_metadata.is_valid_token[rows],
                    )
                )
                compressed_cache = self._compressed_kv_cache()
            sparse_mla(
                q[rows],
                self.swa_cache_layer.kv_cache,
                swa_indices,
                swa_lens,
                compressed_cache,
                compressed_indices,
                compressed_lens,
                self.attn_sink[: self.n_local_heads],
                self.scale,
                positions[rows],
                rope_table_f32(self.rotary_emb),
                output[rows],
            )
