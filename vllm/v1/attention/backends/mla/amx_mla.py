# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMX-only, high-performance MLA backend for DeepSeek V2/V3/R1 on CPU.

Built on the AMX decode/extend/bmm kernels vendored under
``csrc/cpu/sgl-kernels/``, plugged into vLLM's ``MLACommonBackend``/
``MLACommonImpl`` abstraction the same way every other concrete MLA backend
(``TritonMLAImpl``, etc.) does.

This is a separate backend from the reference ``CPUMLABackend``
(``vllm/v1/attention/backends/mla/cpu_mla.py``): that one targets every CPU
(any dtype, block_size=16, ``mla_decode_kvcache`` decode kernel + inherited
SDPA-style prefill) as a functional/CI reference, not performance. This
backend instead requires AMX (bf16 only, block_size a multiple of 32) and is
selected by the platform layer in preference to the reference backend
whenever the host supports it -- see ``CpuPlatform.get_attn_backend_cls``.

Two points where this backend differs structurally from the GPU backends,
both explained in the CPU MLA design plan:

- ``forward_mha`` is fully overridden (not inherited from
  ``MLACommonBaseImpl``): the GPU "compute-friendly" prefill path depends on
  a pluggable ``MLAPrefillBackend`` and CUDA-only chunked-context gather ops,
  neither of which exist on CPU. Instead, this attends directly in
  latent-MQA space via the ``extend_attention_cpu`` kernel, which handles
  cached-prefix continuation and fresh prefill in one causal pass (the KV
  cache already contains the new tokens by the time this runs, since
  ``do_kv_cache_update`` executes first).
- Weight absorption for the prefill path is done with this impl's own
  VNNI-packed copies of ``W_UK``/``W_UV`` (computed in
  ``process_weights_after_loading``, using ``bmm_cpu``), rather than reading
  the generic ``layer.W_UK_T``/``layer.W_UV``/``layer._v_up_proj`` the way
  GPU backends do, since ``forward_mha``'s abstract signature has no
  ``layer`` parameter (only ``forward_mqa`` does) and that signature is left
  untouched. The decode path needs no such packing: ``MLAAttention.forward_impl``
  already absorbs/de-absorbs Q around the ``forward_mqa`` call using the
  generic (unpacked) ``layer.W_UK_T``/``layer._v_up_proj``, so
  ``forward_mqa`` here only has to invoke the decode kernel.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionLayer, AttentionType, MultipleOf

_MIN_WORK_PER_SPLIT = 512
_SPLIT_OCCUPANCY_MULTIPLIER = 2


def _compute_num_kv_splits(max_seq_len: int, num_threads: int) -> int:
    """Mirrors TritonMLAImpl's _compute_num_kv_splits, using the CPU thread
    count in place of SM count."""
    ideal_splits = 1
    while ideal_splits < max(1, max_seq_len // _MIN_WORK_PER_SPLIT):
        ideal_splits *= 2
    max_splits = num_threads * _SPLIT_OCCUPANCY_MULTIPLIER
    return min(ideal_splits, max_splits)


def _expand_block_table(block_table: torch.Tensor, block_size: int) -> torch.Tensor:
    """Adapter: vLLM's block-table paging -> the flat per-(request, position)
    physical row index the decode/extend kernels expect. Called once per step
    from ``AMXMLAMetadataBuilder.build``, not per layer.
    """
    offsets = torch.arange(
        block_size, device=block_table.device, dtype=block_table.dtype
    )
    flat = block_table.unsqueeze(-1) * block_size + offsets
    return flat.reshape(block_table.size(0), -1).contiguous()


class AMXMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # kv_lora_rank(512) + qk_rope_head_dim(64), DeepSeek V2/V3/R1/V3.2.
        return [576]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(32)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 32 == 0

    @staticmethod
    def get_name() -> str:
        return "AMX_MLA"

    @staticmethod
    def get_impl_cls() -> type[AMXMLAImpl]:
        return AMXMLAImpl

    @staticmethod
    def get_builder_cls() -> type[AMXMLAMetadataBuilder]:
        return AMXMLAMetadataBuilder


class AMXMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        attn_metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        block_size = self.kv_cache_spec.block_size
        if attn_metadata.decode is not None:
            # Built once per step and reused by every layer, instead of
            # recomputing the same tensor arithmetic in every forward_mqa call.
            decode = attn_metadata.decode
            decode.req_to_token = _expand_block_table(  # type: ignore[attr-defined]
                decode.block_table, block_size
            )
            decode.seq_lens_i64 = decode.seq_lens.to(torch.int64)  # type: ignore[attr-defined]
            decode.req_pool_indices = torch.arange(  # type: ignore[attr-defined]
                decode.block_table.size(0),
                dtype=torch.int64,
                device=decode.block_table.device,
            )
            decode.num_kv_splits = _compute_num_kv_splits(  # type: ignore[attr-defined]
                attn_metadata.max_seq_len, current_platform.num_compute_units()
            )
        if attn_metadata.prefill is not None:
            # cpu_seq_lens: total per-request context length (prefix + new
            # tokens), the one thing the extend kernel needs that isn't
            # already on the shared MLACommonPrefillMetadata.
            prefill = attn_metadata.prefill
            num_decodes = attn_metadata.num_decodes
            num_prefills = attn_metadata.num_prefills
            prefill.cpu_seq_lens = common_attn_metadata.seq_lens[  # type: ignore[attr-defined]
                num_decodes : num_decodes + num_prefills
            ].to(torch.int64)
            prefill.req_to_token = _expand_block_table(  # type: ignore[attr-defined]
                prefill.block_table, block_size
            ).to(torch.int64)
            prefill.req_pool_indices = torch.arange(  # type: ignore[attr-defined]
                prefill.block_table.size(0),
                dtype=torch.int64,
                device=prefill.block_table.device,
            )
            query_start_loc_i64 = prefill.query_start_loc.to(torch.int64)
            extend_seq_lens = query_start_loc_i64[1:] - query_start_loc_i64[:-1]
            prefill.extend_seq_lens = extend_seq_lens  # type: ignore[attr-defined]
            prefill.extend_start_loc = query_start_loc_i64[:-1]  # type: ignore[attr-defined]
            prefill.max_len_extend = int(extend_seq_lens.max().item())  # type: ignore[attr-defined]
        return attn_metadata


class AMXMLAImpl(MLACommonImpl[MLACommonMetadata]):
    # Tells MLAAttention (mla_attention.py) to use this impl's own packed
    # W_UK/W_UV + bmm_cpu for the shared decode absorb/de-absorb, mirroring
    # is_aiter_triton_fp8_bmm_enabled/is_aiter_triton_fp4_bmm_enabled.
    uses_amx_bmm = True

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
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "AMXMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for AMXMLAImpl"
            )
        self._w_uk_packed: torch.Tensor | None = None
        self._w_uv_packed: torch.Tensor | None = None

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        # forward_mha (unlike forward_mqa) receives no `layer` argument, so it
        # cannot read the generic layer.W_UK_T/W_UV/._v_up_proj that
        # MLAAttention.process_weights_after_loading already computes. This
        # impl derives its own copies from self.kv_b_proj (already available
        # via the constructor) and VNNI-packs them once, ahead of time, for
        # use with bmm_cpu in forward_mha.
        kv_b_proj_weight = get_and_maybe_dequant_weights(
            self.kv_b_proj, out_dtype=act_dtype
        ).T
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        w_uk, w_uv = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # bmm_cpu computes out[b] = mat1[b] @ mat2[b]^T (Linear-weight
        # convention: mat2[b] is (OUT, IN)).
        #   absorb:    ql_nope(N,B,L) = q_nope(N,B,P) @ w_uk_for_bmm(N,L,P)^T
        #   de-absorb: out(N,B,V)     = attn_out(N,B,L) @ w_uv_for_bmm(N,V,L)^T
        w_uk_for_bmm = w_uk.permute(1, 0, 2).contiguous()  # (N, L, P)
        w_uv_for_bmm = w_uv.permute(1, 2, 0).contiguous()  # (N, V, L)
        self._w_uk_packed = torch.ops._C.convert_weight_packed(w_uk_for_bmm)
        self._w_uv_packed = torch.ops._C.convert_weight_packed(w_uv_for_bmm)

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        # Overrides the default (CUDA-only concat_and_cache_mla).
        if kv_cache.numel() == 0:
            return
        assert kv_cache_dtype == "auto", (
            "AMXMLAImpl only supports an unquantized (bf16) KV cache; fp8 "
            "KV cache for CPU MLA is not yet implemented."
        )
        ops.amx_mla_concat_and_cache(
            kv_c_normed, k_pe.squeeze(1), kv_cache, slot_mapping.flatten()
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # q arrives already absorbed (tensor or (ql_nope, q_pe) tuple) --
        # MLAAttention.forward_impl does the Q-absorption bmm generically
        # before calling forward_mqa, using layer.W_UK_T. De-absorption is
        # likewise done generically afterwards via layer._v_up_proj. Nothing
        # left for this impl to do beyond invoking the decode kernel.
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        assert attn_metadata.decode is not None

        num_tokens = q.shape[0]
        kv_cache_flat = kv_c_and_k_pe_cache.view(-1, 1, self.head_size)
        v_buffer = kv_cache_flat[..., : self.kv_lora_rank]

        seq_lens = attn_metadata.decode.seq_lens_i64  # type: ignore[attr-defined]
        req_to_token = attn_metadata.decode.req_to_token  # type: ignore[attr-defined]
        req_pool_indices = attn_metadata.decode.req_pool_indices  # type: ignore[attr-defined]

        num_kv_splits = attn_metadata.decode.num_kv_splits  # type: ignore[attr-defined]
        o = torch.zeros(
            num_tokens,
            self.num_heads,
            self.kv_lora_rank,
            dtype=q.dtype,
            device=q.device,
        )
        attn_logits = torch.zeros(
            num_tokens,
            self.num_heads,
            num_kv_splits,
            self.kv_lora_rank + 1,
            dtype=torch.float32,
            device=q.device,
        )

        ops.cpu_mla_decode(
            q,
            kv_cache_flat,
            v_buffer,
            o,
            None,
            None,
            None,  # loc: only meaningful when key/value are given; we
            # already wrote the new K/V via do_kv_cache_update before this.
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            self.scale,
            0.0,
            False,
            0,
            None,
            None,
        )
        return o, None

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        assert output_scale is None, (
            "AMXMLAImpl.forward_mha does not support fused output quantization"
        )
        prefill = attn_metadata.prefill
        assert prefill is not None
        assert self._w_uk_packed is not None and self._w_uv_packed is not None

        num_tokens = q.shape[0]
        num_heads = self.num_heads

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Absorb: (N, B, P) x (N, L, P)^T -> (N, B, L)
        # bmm_cpu tolerates non-contiguous mat1/out as long as their last dim
        # is contiguous, so q_nope's transpose and ql_nope's transposed view
        # can be passed directly -- no copy needed for either.
        q_nope_t = q_nope.transpose(0, 1)
        ql_nope = torch.empty(
            num_tokens, num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        ops.bmm_cpu(ql_nope.transpose(0, 1), q_nope_t, self._w_uk_packed, True, None)
        mqa_q = torch.cat([ql_nope, q_pe], dim=-1)

        kv_cache_flat = kv_c_and_k_pe_cache.view(-1, 1, self.head_size)
        v_buffer = kv_cache_flat[..., : self.kv_lora_rank]

        req_to_token = prefill.req_to_token  # type: ignore[attr-defined]
        req_pool_indices = prefill.req_pool_indices  # type: ignore[attr-defined]

        seq_lens = prefill.cpu_seq_lens  # type: ignore[attr-defined]
        extend_seq_lens = prefill.extend_seq_lens  # type: ignore[attr-defined]
        extend_start_loc = prefill.extend_start_loc  # type: ignore[attr-defined]
        max_len_extend = prefill.max_len_extend  # type: ignore[attr-defined]

        # k_extend/v_extend: the new tokens' own latent K/V, aliased in one
        # 576-wide buffer (v_extend is a view of k_extend's first
        # kv_lora_rank columns) so the kernel can gather both from one read,
        # mirroring the cache's own layout.
        k_extend = torch.cat(
            [kv_c_normed, k_pe.reshape(num_tokens, self.qk_rope_head_dim)], dim=-1
        ).unsqueeze(1)
        v_extend = k_extend[..., : self.kv_lora_rank]

        attn_out = torch.empty(
            num_tokens, num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        ops.cpu_mla_extend(
            mqa_q,
            k_extend,
            v_extend,
            attn_out,
            kv_cache_flat,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
            max_len_extend,
            self.scale,
            0.0,
            False,
            0,
            None,
            None,
            None,
        )

        # De-absorb directly into `output` (view, no extra copy on either side):
        # (N, B, L) x (N, V, L)^T -> (N, B, V)
        attn_out_t = attn_out.transpose(0, 1)
        output_view = output.view(num_tokens, num_heads, self.v_head_dim).transpose(
            0, 1
        )
        ops.bmm_cpu(output_view, attn_out_t, self._w_uv_packed, True, None)
