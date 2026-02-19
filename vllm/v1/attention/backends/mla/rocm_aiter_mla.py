# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.v1.attention.backend import AttentionCGSupport, AttentionLayer, MultipleOf
from vllm.v1.kv_cache_interface import AttentionSpec


class AiterMLABackend(MLACommonBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_impl_cls() -> type["AiterMLAImpl"]:
        return AiterMLAImpl

    @staticmethod
    def get_builder_cls() -> type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder


@dataclass
class AiterMLADecodeMetadata(MLACommonDecodeMetadata):
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: torch.Tensor | None = None
    # The page indices of the paged kv cache
    paged_kv_indices: torch.Tensor | None = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: torch.Tensor | None = None
    # The query indptr, shape : [num_decode + 1]
    qo_indptr: torch.Tensor | None = None
    # The dtype of MLA out tensor
    attn_out_dtype: torch.dtype = torch.bfloat16
    # The max query output length: int
    max_qo_len: int | None = None


class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    pass


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    # TODO(luka, lucas): audit this as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, AiterMLAMetadata
        )

        self.compilation_config = vllm_config.compilation_config
        self.decode_attn_out_dtype = vllm_config.model_config.dtype
        # kernel block size is always 1.
        max_num_pages_per_req = vllm_config.model_config.max_model_len
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        # Preparing persistent buffers
        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.

        # paged_kv_last_page_len is always 1s (kernel block size is always 1),
        # so we create it once and reuse slices in both eager and cudagraph modes.
        self.paged_kv_last_page_len = torch.ones(
            max_num_reqs, dtype=torch.int32, device=device
        )

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )
            self.paged_kv_indices = torch.zeros(
                max_num_pages, dtype=torch.int32, device=device
            )

            self.qo_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> AiterMLADecodeMetadata:
        # kernel block size is always 1, although the kv block size is not 1.
        device = self.device
        num_reqs = seq_lens_device.size(0)

        mask = torch.arange(
            block_table_tensor.size(1), dtype=block_table_tensor.dtype, device=device
        ).unsqueeze(0) < seq_lens_device.unsqueeze(1)
        paged_kv_indices = block_table_tensor[mask]

        # kernel block size is always 1, so each page has exactly 1 token.
        # last_page_len is always 1 - just slice the pre-initialized buffer.
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

        paged_kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=seq_lens_device.dtype, device=device),
                seq_lens_device.cumsum(dim=0, dtype=torch.int32),
            ]
        )
        qo_len = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_qo_len = qo_len.max().item()

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            num_actual_pages = paged_kv_indices.size(0)

            self.paged_kv_indices[:num_actual_pages].copy_(
                paged_kv_indices, non_blocking=True
            )
            self.paged_kv_indices[num_actual_pages:].fill_(-1)
            paged_kv_indices = self.paged_kv_indices[:num_actual_pages]

            self.paged_kv_indptr[: 1 + num_reqs].copy_(
                paged_kv_indptr, non_blocking=True
            )
            self.paged_kv_indptr[1 + num_reqs :].fill_(paged_kv_indptr[-1])
            paged_kv_indptr = self.paged_kv_indptr[: 1 + num_reqs]

            # paged_kv_last_page_len already uses the pre-initialized buffer slice
            # (set above), so no copy needed - buffer is always 1s.

            self.qo_indptr[: 1 + num_reqs].copy_(
                query_start_loc_device, non_blocking=True
            )
            self.qo_indptr[1 + num_reqs :] = query_start_loc_device[-1]
            qo_indptr = self.qo_indptr[: 1 + num_reqs]

        else:
            qo_indptr = torch.arange(
                0, num_reqs + 1, step=1, dtype=torch.int32, device=device
            )

        attn_metadata = AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            max_qo_len=max_qo_len,
            attn_out_dtype=self.decode_attn_out_dtype,
        )

        return attn_metadata


class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        assert num_heads == 16 or num_heads == 128, (
            f"Aiter MLA only supports 16 or 128 number of heads.\n"
            f"Provided {num_heads} number of heads.\n"
            "Try adjusting tensor_parallel_size value."
        )
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "Aiter MLA does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        from aiter import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        output = self.flash_attn_varlen_func(  # type: ignore[call-arg]
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_softmax_lse,
            **kwargs,
        )

        return output

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None
        assert attn_metadata.decode.max_qo_len is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        o = torch.zeros(
            B,
            self.num_heads,
            self.kv_lora_rank,
            dtype=attn_metadata.decode.attn_out_dtype,
            device=q.device,
        )

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        rocm_aiter_ops.mla_decode_fwd(
            q,
            kv_buffer,
            o,
            self.scale,
            attn_metadata.decode.qo_indptr,
            attn_metadata.decode.max_qo_len,
            attn_metadata.decode.paged_kv_indptr,
            attn_metadata.decode.paged_kv_indices,
            attn_metadata.decode.paged_kv_last_page_len,
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
        )

        return o, None
