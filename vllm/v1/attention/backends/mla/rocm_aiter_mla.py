# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.attention.backends.abstract import AttentionLayer, MultipleOf
from vllm.config import VllmConfig
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.distributed.parallel_state import get_dcp_group, is_global_first_rank
from vllm.attention.ops.common import cp_lse_ag_out_rs
from vllm.platforms import current_platform
from typing import ClassVar, Generic, TypeVar

M = TypeVar("M", bound=MLACommonMetadata)

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


class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    pass


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    # TODO(luka, lucas): audit this as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

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
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )
            self.paged_kv_indices = torch.zeros(
                max_num_pages, dtype=torch.int32, device=device
            )
            self.paged_kv_last_page_len = torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            )

            self.qo_indptr = torch.arange(
                0, max_num_reqs + 1, dtype=torch.int32, device=device
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_lens_device: torch.Tensor,
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

        paged_kv_last_page_len = torch.where(seq_lens_device == 0, 1, seq_lens_device)

        paged_kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=seq_lens_device.dtype, device=device),
                seq_lens_device.cumsum(dim=0, dtype=torch.int32),
            ]
        )

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

            self.paged_kv_last_page_len[:num_reqs].copy_(
                paged_kv_last_page_len, non_blocking=True
            )
            self.paged_kv_last_page_len[num_reqs:].fill_(1)
            paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

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
        self.is_aiter_triton_fp4_bmm_enabled = rocm_aiter_ops.is_fp4bmm_enabled()
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
        output = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_softmax_lse,
            **kwargs,
        )

        return output

    def _forward_decode(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
        mla_output_zeros: torch.Tensor | None = None,
        decode_q_cat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if decode_q_cat is not None:
            q = decode_q_cat
        elif type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        if mla_output_zeros is not None:
            o = mla_output_zeros
            assert o.shape[0] == B, f"{o.shape[0]=} {B=}"
            assert o.shape[1] == self.num_heads, f"{o.shape[1]=} {self.num_heads=}"
            assert o.shape[2] == self.kv_lora_rank, f"{o.shape[2]=} {self.kv_lora_rank=}"
        else:
            o = torch.zeros(
                B,
                self.num_heads,
                self.kv_lora_rank,
                dtype=attn_metadata.decode.attn_out_dtype,
                device=q.device,
            )

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        # max_seqlen_qo must be 1 except for MTP
        # TODO: Find the best value for MTP
        max_seqlen_qo = 1
        rocm_aiter_ops.mla_decode_fwd(
            q,
            kv_buffer,
            o,
            self.scale,
            attn_metadata.decode.qo_indptr,
            max_seqlen_qo,
            attn_metadata.decode.paged_kv_indptr,
            attn_metadata.decode.paged_kv_indices,
            attn_metadata.decode.paged_kv_last_page_len,
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
        )

        return o, None

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: M,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for MLACommonImpl"
            )

        if attn_metadata is None:
            # During the profile run try to simulate to worse case output size
            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
            # since this can be large
            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )

            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        if self.dcp_world_size is None:
            self.dcp_world_size = get_dcp_group().world_size

        fp8_attention = self.kv_cache_dtype.startswith("fp8")

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        mla_output_zeros = None
        decode_q_cat = None
        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            if positions is not None:
                # positions is not None entails that Q and K are not RoPE embedded yet, therefore, fused_qk_rope_cat_and_cache_mla is called
                assert hasattr(self, "rotary_emb"), f"rotary_emb not found in {self}"
                from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
                cos, sin = self.rotary_emb.cos_sin_cache.chunk(2, dim = -1)
                is_neox = self.rotary_emb.is_neox_style
                q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
                q_out_dtype = current_platform.fp8_dtype() if fp8_attention else q.dtype
                if self.is_aiter_triton_fp4_bmm_enabled or self.is_aiter_triton_fp8_bmm_enabled:
                    decode_q_cat = torch.empty((num_decode_tokens, self.num_heads, self.W_K.shape[1] + self.qk_rope_head_dim), dtype = q_out_dtype, device=q.device)
                if fp8_attention:
                    kv_cache_og_dtype = kv_cache.dtype
                    kv_cache = kv_cache.view(q_out_dtype)          
                q, _, k_pe, mla_output_zeros = fused_qk_rope_cat_and_cache_mla(
                    q_nope,
                    q_pe,
                    k_c_normed.unsqueeze(1),
                    k_pe,
                    kv_cache,
                    attn_metadata.slot_mapping.flatten(),
                    positions,
                    cos,
                    sin,
                    layer._k_scale,
                    is_neox,
                    num_decode_toks_for_zeros=num_decode_tokens,
                    apply_scale=(k_pe.dtype != kv_cache.dtype),
                    q_out=None,
                    decode_q_pe_out = decode_q_cat[... , -self.qk_rope_head_dim:] if self.is_aiter_triton_fp4_bmm_enabled or self.is_aiter_triton_fp8_bmm_enabled else None,
                    k_pe_out=k_pe,
                )
                if fp8_attention:
                    kv_cache = kv_cache.view(kv_cache_og_dtype)
            else:
                ops.concat_and_cache_mla(
                    k_c_normed,
                    k_pe.squeeze(1),
                    kv_cache,
                    attn_metadata.slot_mapping.flatten(),
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=layer._k_scale,
                )

        decode_q = q[:num_decode_tokens]

        prefill_q = q[num_decode_tokens:]
        prefill_k_pe = k_pe[num_decode_tokens:]
        prefill_k_c_normed = k_c_normed[num_decode_tokens:]

        if fp8_attention:
            kv_cache = kv_cache.view(current_platform.fp8_dtype())

        if has_prefill:
            self._forward_prefill(
                prefill_q,
                prefill_k_c_normed,
                prefill_k_pe,
                kv_cache,
                attn_metadata,
                layer._k_scale,
                output=output[num_decode_tokens:],
            )

        if has_decode:
            assert attn_metadata.decode is not None

            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )

            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)

            if self.q_pad_num_heads is not None:
                B, N, L = decode_q_pe.shape
                decode_pe_padded = decode_q_pe.new_empty((B, self.q_pad_num_heads, L))
                decode_pe_padded.resize_((B, N, L))
                decode_pe_padded.copy_(decode_q_pe)
                decode_q_pe = decode_pe_padded

            if self.is_aiter_triton_fp4_bmm_enabled:
                #x = x.view(-1, self.num_heads, self.kv_lora_rank)
                decode_ql_nope = decode_q_cat[... , :self.W_K.shape[1]] if (kv_cache.numel() > 0 and positions is not None) else None
                decode_ql_nope = rocm_aiter_ops.triton_fp4_bmm(
                    decode_q_nope,
                    self.W_K,
                    self.W_K_scale,
                    YQ=decode_ql_nope,
                    transpose_bm=True,
                    y_scale=layer._q_scale if fp8_attention else None,
                )
                # decode_ql_nope = decode_ql_nope.transpose(0, 1)
            elif self.is_aiter_triton_fp8_bmm_enabled:
                # Multiply+Transpose (N, B, P)x(N, P, L)->(N, B, L)->(B, N, L)
                decode_ql_nope = decode_q_cat[... , :self.W_K.shape[1]] if (kv_cache.numel() > 0 and positions is not None) else None
                decode_ql_nope = rocm_aiter_ops.triton_fp8_bmm(
                    decode_q_nope,
                    self.W_K,
                    self.W_K_scale,
                    group_size=128,
                    YQ=decode_ql_nope,
                    transpose_bm=True,
                )
            else:
                # Pads the head_dim if necessary (for the underlying kernel)
                N, B, P = decode_q_nope.shape
                _, _, L = self.W_UK_T.shape

                if self.q_pad_num_heads is not None:
                    decode_ql_nope = decode_q_nope.new_empty(
                        (self.q_pad_num_heads, B, L)
                    )
                    decode_ql_nope.resize_((N, B, L))
                else:
                    decode_ql_nope = decode_q_nope.new_empty((N, B, L))

                # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
                torch.bmm(decode_q_nope, self.W_UK_T, out=decode_ql_nope)

                # Convert from (N, B, L) to (B, N, L)
                decode_ql_nope = decode_ql_nope.transpose(0, 1)

            if fp8_attention and not (self.is_aiter_triton_fp4_bmm_enabled or self.is_aiter_triton_fp8_bmm_enabled):
                ql_nope_shape = decode_ql_nope.shape
                decode_ql_nope, _ = ops.scaled_fp8_quant(
                    decode_ql_nope.reshape(
                        [ql_nope_shape[0], ql_nope_shape[1] * ql_nope_shape[2]]
                    ),
                    layer._q_scale,
                )
                decode_ql_nope = decode_ql_nope.reshape(ql_nope_shape)
                q_pe_shape = decode_q_pe.shape
                decode_q_pe, _ = ops.scaled_fp8_quant(
                    decode_q_pe.reshape([q_pe_shape[0], q_pe_shape[1] * q_pe_shape[2]]),
                    layer._q_scale,
                )
                decode_q_pe = decode_q_pe.reshape(q_pe_shape)

            decode_q = (decode_ql_nope, decode_q_pe)
            if self.dcp_world_size > 1:
                assert not fp8_attention, "DCP not support fp8 kvcache now."
                # concatenate decode_ql_nope and decode_q_pe -> (B, N, L + P)
                decode_q = torch.cat(decode_q, dim=-1)
                # decode_q do allgather in head dim.
                decode_q = get_dcp_group().all_gather(decode_q, dim=1)

            # call decode attn
            attn_out, lse = self._forward_decode(
                decode_q, kv_cache, attn_metadata, layer, mla_output_zeros=mla_output_zeros, decode_q_cat=decode_q_cat
            )

            # correct dcp attn_out with lse.
            if self.dcp_world_size > 1:
                attn_out = cp_lse_ag_out_rs(
                    attn_out,
                    lse,
                    get_dcp_group(),
                    is_lse_base_on_e=not getattr(self, "_use_fi_prefill", False),
                )

            # v_up projection
            self._v_up_proj(attn_out, out=output[:num_decode_tokens])
        return output_padded
