# SPDX-License-Identifier: Apache-2.0
# FlashInfer MLA backend - plan() in builder, run() in forward (SGLang pattern)
# CG support: UNIFORM_BATCH — per-batch-size wrappers with use_cuda_graph=True
# ensure plan() copies into fixed-address buffers for safe CUDA graph replay.

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
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
MAX_PAGES_PER_REQ = 1024


def _create_wrapper(device, batch_size, tpr=1):
    """Create a BatchMLAPagedAttentionWrapper with use_cuda_graph=True
    and pre-allocated buffers sized for exactly batch_size requests."""
    workspace = torch.zeros(
        FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device=device)
    num_tokens = batch_size * tpr
    qo_indptr = torch.zeros(num_tokens + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(batch_size * MAX_PAGES_PER_REQ, dtype=torch.int32, device=device)
    kv_len_arr = torch.zeros(batch_size, dtype=torch.int32, device=device)
    return workspace, BatchMLAPagedAttentionWrapper(
        workspace,
        use_cuda_graph=True,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_len_arr,
        backend="auto",
    )


@dataclass
class FlashInferMLADecodeMetadata(MLACommonDecodeMetadata):
    wrapper: Optional[BatchMLAPagedAttentionWrapper] = None


class FlashInferMLAMetadataBuilder(MLACommonMetadataBuilder["FlashInferMLAMetadata"]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kwargs):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         metadata_cls=FlashInferMLAMetadata, **kwargs)

        mla_dims = get_mla_dims(self.model_config)
        self._kv_lora_rank = mla_dims.kv_lora_rank
        self._qk_nope_head_dim = mla_dims.qk_nope_head_dim
        self._qk_rope_head_dim = mla_dims.qk_rope_head_dim
        self._page_size = kv_cache_spec.block_size
        self._device = device

        # Per-batch-size wrappers: created lazily on first use.
        # Key = (num_reqs, tokens_per_req), value = (workspace, wrapper).
        self._wrappers: dict[tuple[int, int], tuple[torch.Tensor, BatchMLAPagedAttentionWrapper]] = {}

    def _get_wrapper(self, num_reqs, tpr=1):
        """Get or create a wrapper for the given batch size."""
        key = (num_reqs, tpr)
        if key not in self._wrappers:
            logger.info("Creating FlashInfer MLA wrapper for batch=%d tpr=%d",
                        num_reqs, tpr)
            self._wrappers[key] = _create_wrapper(self._device, num_reqs, tpr)
        return self._wrappers[key][1]

    def _build_decode(self, block_table_tensor, seq_lens_device,
                      max_seq_len, query_start_loc_cpu, query_start_loc_device,
                      num_decode_tokens, dcp_tot_seq_lens_device=None):
        num_reqs = seq_lens_device.shape[0]
        page_size = self._page_size
        dev = seq_lens_device.device

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

        wrapper = self._get_wrapper(num_reqs, tpr)
        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices,
            seq_lens_device.to(torch.int32),
            num_heads=self.num_heads,
            head_dim_ckv=self._kv_lora_rank,
            head_dim_kpe=self._qk_rope_head_dim,
            page_size=page_size, causal=False,
            sm_scale=1.0 / (self._qk_nope_head_dim + self._qk_rope_head_dim) ** 0.5,
            q_data_type=self.model_config.dtype,
            kv_data_type=self.model_config.dtype)

        return FlashInferMLADecodeMetadata(
            block_table=block_table_tensor, seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device, wrapper=wrapper)


@dataclass
class FlashInferMLAMetadata(MLACommonMetadata[FlashInferMLADecodeMetadata]):
    pass


class FlashInferMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto", "bfloat16", "fp8", "fp8_e4m3"]

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
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type, kv_sharing_target_layer_name,
                         **mla_args)
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FlashInferMLAImpl does not support "
                "alibi/sliding_window/logits_soft_cap")
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
        # No .contiguous() — SGLang also passes strided slices, flashinfer handles it
        ckv_cache = kv_c_and_k_pe_cache[:, :, :self.kv_lora_rank].unsqueeze(2)
        kpe_cache = kv_c_and_k_pe_cache[:, :, self.kv_lora_rank:].unsqueeze(2)

        wrapper = attn_metadata.decode.wrapper
        o, lse = wrapper.run(
            q_nope.contiguous(), q_pe.contiguous(),
            ckv_cache, kpe_cache, return_lse=True)

        v_scale = layer._v_scale_float
        if v_scale != 1.0:
            o = o * v_scale

        return o.contiguous(), lse
