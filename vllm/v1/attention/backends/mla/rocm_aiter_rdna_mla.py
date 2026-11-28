# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionLayer, MultipleOf
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend, TritonMLAImpl


class RocmAiterRdnaMLABackend(TritonMLABackend):
    """TRITON_MLA with AITER's tuned Triton MLA decode kernel on gfx1100."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # The AITER kernel's tile is the KV page; the gfx1100 tuning targets 16.
        return [16]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_RDNA_MLA"

    @staticmethod
    def get_impl_cls() -> type["RocmAiterRdnaMLAImpl"]:
        return RocmAiterRdnaMLAImpl

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.platforms.rocm import on_gfx1100

        if not on_gfx1100():
            return "ROCM_AITER_RDNA_MLA requires gfx1100"
        # Untuned gfx1100 launches can exceed GPU resources during graph capture.
        if not rocm_aiter_ops.is_triton_mla_decode_tuned():
            return "AITER has no gfx1100 BF16 MLA decode tuning"
        return None


class RocmAiterRdnaMLAImpl(TritonMLAImpl):
    """Runs AITER for causal single-token decode, else TRITON_MLA."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from aiter.ops.triton.attention.mla import mla_decode_fwd

        self._aiter_mla_decode = mla_decode_fwd
        self._use_aiter_mla_decode = not self.need_to_return_lse_for_decode and all(
            dim > 0 and dim & (dim - 1) == 0
            for dim in (self.kv_lora_rank, self.qk_rope_head_dim)
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        B = q.shape[0]
        if not (
            self._use_aiter_mla_decode
            and attn_metadata.causal
            and B > 0
            and attn_metadata.decode.seq_lens.shape[0] == B
            and kv_c_and_k_pe_cache.shape[1] == 16
        ):
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)

        # Padded seq_len=0 rows may be undefined; only active rows are consumed.
        out = q.new_empty((B, q.shape[1], self.kv_lora_rank))
        self._aiter_mla_decode(
            q=q,
            kv_buffer=kv_c_and_k_pe_cache.unsqueeze(2),
            out=out,
            cu_seqlens_q=attn_metadata.query_start_loc[: B + 1],
            seqused_k=attn_metadata.decode.seq_lens,
            max_seqlen_kv=attn_metadata.max_seq_len,
            block_tables=attn_metadata.decode.block_table,
            softmax_scale=self.scale,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            causal=True,
            q_descale=None,
            kv_descale=None,
            skip_reduce=False,
        )
        return out, None
