# SPDX-License-Identifier: Apache-2.0
# FlashInfer MLA backend - plan() in builder, run() in forward (SGLang pattern)

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
from flashinfer import BatchMLAPagedAttentionWrapper

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
    get_mla_dims,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024


@dataclass
class FlashInferMLADecodeMetadata(MLACommonDecodeMetadata):
    wrapper: Optional[BatchMLAPagedAttentionWrapper] = None


class FlashInferMLAMetadataBuilder(MLACommonMetadataBuilder["FlashInferMLAMetadata"]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device,
                 **kwargs):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         metadata_cls=FlashInferMLAMetadata, **kwargs)
        self._fi_workspace = torch.zeros(
            FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8, device=device,
        )
        self._fi_wrapper = BatchMLAPagedAttentionWrapper(
            self._fi_workspace, backend="auto"
        )
        mla_dims = get_mla_dims(self.model_config)
        self._kv_lora_rank = mla_dims.kv_lora_rank
        self._qk_rope_head_dim = mla_dims.qk_rope_head_dim
        self._page_size = kv_cache_spec.block_size

    def _build_decode(self, block_table_tensor, seq_lens_device,
                      max_seq_len, query_start_loc_cpu, query_start_loc_device,
                      num_decode_tokens, dcp_tot_seq_lens_device=None):
        """Override to build FlashInfer decode metadata with plan()."""
        num_reqs = seq_lens_device.shape[0]
        page_size = self._page_size
        dev = seq_lens_device.device

        # Build CSR indices
        num_pages_per_req = (seq_lens_device + page_size - 1) // page_size
        kv_indptr = torch.zeros(num_reqs + 1, dtype=torch.int32, device=dev)
        torch.cumsum(num_pages_per_req, dim=0, out=kv_indptr[1:])
        max_blk = block_table_tensor.shape[1]
        blk_idx = torch.arange(max_blk, device=dev)
        mask = blk_idx.unsqueeze(0) < num_pages_per_req.unsqueeze(1)
        kv_indices = block_table_tensor[mask].to(torch.int32)

        tpr = num_decode_tokens // num_reqs if num_reqs > 0 else 1
        qo_indptr = torch.arange(0, num_reqs * tpr + 1, tpr,
                                 dtype=torch.int32, device=dev)

        # Plan - runs outside CUDA graph (in build())
        self._fi_wrapper.plan(
            qo_indptr, kv_indptr, kv_indices,
            seq_lens_device.to(torch.int32),
            num_heads=self.num_heads,
            head_dim_ckv=self._kv_lora_rank,
            head_dim_kpe=self._qk_rope_head_dim,
            page_size=page_size,
            causal=False,
            sm_scale=1.0 / (self._kv_lora_rank + self._qk_rope_head_dim) ** 0.5,
            q_data_type=self.model_config.dtype,
            kv_data_type=self.model_config.dtype,
        )

        return FlashInferMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            wrapper=self._fi_wrapper,
        )


@dataclass
class FlashInferMLAMetadata(MLACommonMetadata[FlashInferMLADecodeMetadata]):
    pass


class FlashInferMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16, torch.bfloat16
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto", "bfloat16", "fp8", "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA"

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLAMetadataBuilder"]:
        return FlashInferMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major >= 10

    @classmethod
    def supports_combination(cls, head_size, dtype, kv_cache_dtype, block_size,
                             use_mla, has_sink, use_sparse,
                             device_capability) -> str | None:
        return None


class FlashInferMLAImpl(MLACommonImpl[FlashInferMLAMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(self, num_heads, head_size, scale, num_kv_heads,
                 alibi_slopes, sliding_window, kv_cache_dtype,
                 logits_soft_cap, attn_type, kv_sharing_target_layer_name,
                 **mla_args):
        super().__init__(
            num_heads, head_size, scale, num_kv_heads,
            alibi_slopes, sliding_window, kv_cache_dtype,
            logits_soft_cap, attn_type, kv_sharing_target_layer_name,
            **mla_args,
        )
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FlashInferMLAImpl does not support "
                "alibi/sliding_window/logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Only decoder attention supported")

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None
        assert attn_metadata.decode.wrapper is not None

        if isinstance(q, tuple):
            q_nope, q_pe = q
        else:
            q_nope = q[:, :, :self.kv_lora_rank]
            q_pe = q[:, :, self.kv_lora_rank:]

        # Split KV cache: (blocks, block_size, 576) -> ckv(512) + kpe(64)
        ckv_cache = kv_c_and_k_pe_cache[:, :, :self.kv_lora_rank].unsqueeze(2)
        kpe_cache = kv_c_and_k_pe_cache[:, :, self.kv_lora_rank:].unsqueeze(2)

        # Just run() - plan() was already called in build()
        wrapper = attn_metadata.decode.wrapper
        o, lse = wrapper.run(
            q_nope.contiguous(), q_pe.contiguous(),
            ckv_cache, kpe_cache,
            return_lse=True,
            kv_len=attn_metadata.decode.seq_lens.to(torch.int32),
            page_table=attn_metadata.decode.block_table.to(torch.int32),
        )

        v_scale = layer._v_scale_float
        if v_scale != 1.0:
            o = o * v_scale

        return o.contiguous(), lse
