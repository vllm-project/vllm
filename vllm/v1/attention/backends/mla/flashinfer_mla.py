# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
from flashinfer.rope import mla_rope_quantize_fp8

from vllm.attention.backends.abstract import (
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.v1.attention.backends.utils import AttentionCGSupport, KVCacheLayoutType

logger = init_logger(__name__)

FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024


class FlashInferMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM


class FlashInferMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [32, 64]

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
        return capability.major == 10

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        return "HND"


g_fi_workspace = torch.zeros(
    FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE,
    dtype=torch.uint8,
    device="cuda",
)


class FlashInferMLAImpl(MLACommonImpl[MLACommonMetadata]):
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

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferMLAImpl"
            )

        self._workspace_buffer = g_fi_workspace
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None

        # Enable fused RoPE+quant for FP8 attention
        self.use_fused_rope_quant = kv_cache_dtype.startswith("fp8")

    def _fused_rope_quant(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        positions: torch.Tensor,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Fused RoPE + FP8 quantization for decode Q.

        Uses flashinfer.rope.mla_rope_quantize_fp8 to apply RoPE and quantize
        in a single fused kernel for better performance.

        Args:
            ql_nope: Projected q_nope. Shape: [B, N, L] where L = kv_lora_rank.
            q_pe: Raw q_pe (no RoPE yet). Shape: [B, N, R] where R = qk_rope_head_dim.
            positions: Position indices. Shape: [B]
            q_scale: Scale for FP8 quantization (unused, scale is 1.0).

        Returns:
            FP8 quantized tensor with RoPE applied. Shape: [B, N, L+R]
        """
        assert self.rotary_emb is not None, (
            "rotary_emb must be set for fused RoPE+quant"
        )
        B, N, L = ql_nope.shape
        R = q_pe.shape[-1]

        # Output tensor: [B, N, L+R] in FP8
        q_out = torch.empty(
            B,
            N,
            L + R,
            dtype=torch.float8_e4m3fn,
            device=q_pe.device,
        )

        # flashinfer requires cos_sin_cache to be float32
        cos_sin_cache_f32 = self.rotary_emb.cos_sin_cache.float()

        # The flashinfer kernel requires K tensors to have the same batch size
        # as Q tensors. For decode, K is already in cache with RoPE applied,
        # so we pass dummy K tensors and ignore the output.
        # K tensors need shape [B, 1, dim] to match Q's batch size.
        # Use empty instead of zeros since these are dummy tensors - the
        # output is ignored.
        k_rope_dummy = torch.empty(B, 1, R, dtype=q_pe.dtype, device=q_pe.device)
        k_nope_dummy = torch.empty(B, 1, L, dtype=ql_nope.dtype, device=ql_nope.device)
        k_rope_out_dummy = torch.empty(
            B, 1, R, dtype=torch.float8_e4m3fn, device=q_pe.device
        )
        k_nope_out_dummy = torch.empty(
            B, 1, L, dtype=torch.float8_e4m3fn, device=ql_nope.device
        )

        # Call fused kernel
        mla_rope_quantize_fp8(
            q_rope=q_pe,
            k_rope=k_rope_dummy,
            q_nope=ql_nope,
            k_nope=k_nope_dummy,
            cos_sin_cache=cos_sin_cache_f32,
            pos_ids=positions,
            is_neox=False,  # MLA uses GPT-J style RoPE
            quantize_dtype=torch.float8_e4m3fn,
            q_rope_out=q_out[..., L:],  # RoPE portion goes after nope
            q_nope_out=q_out[..., :L],  # nope portion goes first
            k_rope_out=k_rope_out_dummy,  # ignored
            k_nope_out=k_nope_out_dummy,  # ignored
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )

        return q_out

    def _forward_decode(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = torch.cat([q_nope, q_pe], dim=-1)

        # trtllm API requires extra dimension q_len_per_request for MTP
        if attn_metadata.num_decode_tokens % attn_metadata.num_decodes != 0:
            logger.warning_once(
                """FlashInferMLAImpl got a query of uneven length.
                This usually indicates an issue in batch reordering
                or incorrect setup in dummy_run."""
            )
            q = q.unsqueeze(1)
        else:
            q = q.view(attn_metadata.num_decodes, -1, q.shape[-2], q.shape[-1])

        if self.bmm1_scale is None:
            self.bmm1_scale = layer._q_scale_float * layer._k_scale_float * self.scale
        if self.bmm2_scale is None:
            self.bmm2_scale = layer._v_scale_float

        o = trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=attn_metadata.decode.block_table,
            seq_lens=attn_metadata.decode.seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
        )

        # Flatten the output for consistent shape
        o = o.view(-1, o.shape[-2], o.shape[-1])

        # TODO: Return LSE pending support from Flashinfer API:
        # https://github.com/flashinfer-ai/flashinfer/pull/1566
        return o, None
