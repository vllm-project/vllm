import os

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import ClassVar

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

from vllm._custom_ops import (
    mla_absorption_bmm_bf16,
    mla_fused_cache_nope,
    mla_fused_cache_rope,
    mla_rope_quantize_fp8,
)
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import aux_stream, current_stream
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType

logger = init_logger(__name__)

FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024


class FlashInferMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM


class FlashInferMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
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
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # FlashInfer MLA kernel requires qk_nope_head_dim in [64, 128, 192]
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
            if qk_nope_head_dim not in [64, 128, 192]:
                return (
                    "FlashInfer MLA kernel requires qk_nope_head_dim "
                    f"in [64, 128, 192], but got {qk_nope_head_dim}"
                )
        return None

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

        # Enable fused RoPE+quant for FP8 attention.
        # use_fused_rope_quant requires at least one fused path
        # (rope_cache or absorption) to be enabled, otherwise the
        # standalone fused-quant path outputs FP8 K tensors that
        # concat_and_cache_mla cannot accept.
        _is_fp8 = kv_cache_dtype.startswith("fp8")
        _want_rope_cache = os.getenv("VLLM_MLA_FUSED_ROPE_CACHE", "0") == "1"
        _want_absorption = os.getenv("VLLM_MLA_FUSED_ABSORPTION", "0") == "1"
        self.use_fused_rope_cache = _is_fp8 and _want_rope_cache
        self.use_fused_absorption = _is_fp8 and _want_absorption
        self.use_fused_rope_quant = (
            self.use_fused_rope_cache or self.use_fused_absorption
        )

        # Pre-allocate constant scale_b=1.0 for absorption BMM to avoid
        # per-call tensor allocation.
        self._absorption_scale_b = torch.ones(1, dtype=torch.float32, device="cuda")
        # Cached scale_a = q_scale * w_uk_scale (computed on first call).
        self._absorption_scale_a: torch.Tensor | None = None

    def _fused_rope_quant(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        q_scale: float,
        k_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused RoPE + FP8 quantization for Q and K.

        Uses vLLM's in-tree mla_rope_quantize_fp8 kernel (yanked from
        FlashInfer) to apply RoPE and quantize in a single fused kernel.

        Args:
            ql_nope: Projected q_nope. Shape: [B, N, L] where L = kv_lora_rank.
            q_pe: Raw q_pe (no RoPE yet). Shape: [B, N, R] where R = qk_rope_head_dim.
            k_nope: k_c_normed (latent). Shape: [B, L] (2D, no head dim).
            k_pe: Raw k_pe (no RoPE yet). Shape: [B, R] (2D, squeezed by caller).
            positions: Position indices. Shape: [B]
            q_scale: Scale for FP8 quantization of Q (host float).
            k_scale: Scale for FP8 quantization of K (host float).

        Returns:
            tuple of:
            - q_out: FP8 quantized Q with RoPE. Shape: [B, N, L+R]
            - k_nope_out: FP8 quantized k_nope. Shape: [B, L]
            - k_pe_out: FP8 quantized k_pe with RoPE. Shape: [B, R]
        """
        assert self.rotary_emb is not None, (
            "rotary_emb must be set for fused RoPE+quant"
        )
        L = ql_nope.shape[-1]
        attn_dtype = torch.float8_e4m3fn

        # Output tensors - k_nope and k_pe are 2D for MLA
        q_out = q_pe.new_empty(
            q_pe.shape[0], q_pe.shape[1], L + q_pe.shape[2], dtype=attn_dtype
        )
        k_nope_out = k_nope.new_empty(k_nope.shape, dtype=attn_dtype)
        k_pe_out = k_pe.new_empty(k_pe.shape, dtype=attn_dtype)

        # Dynamic kernel: computes cos/sin on-the-fly from inv_freq,
        # eliminating the cos_sin_cache entirely.
        inv_freq = self.rotary_emb.inv_freq

        mla_rope_quantize_fp8(
            q_pe,  # q_rope_in
            k_pe,  # k_rope_in
            ql_nope,  # q_nope_in
            k_nope,  # k_nope_in
            q_out[..., L:],  # q_rope_out (RoPE portion after nope)
            k_pe_out,  # k_rope_out
            q_out[..., :L],  # q_nope_out (nope portion first)
            k_nope_out,  # k_nope_out
            inv_freq,  # inv_freq (32 floats, computes cos/sin via __sincosf)
            positions,  # pos_ids
            q_scale,  # quant_scale_q
            k_scale,  # quant_scale_kv
            True,  # interleave (= not is_neox; MLA uses GPT-J style)
            False,  # enable_pdl
        )

        return q_out, k_nope_out, k_pe_out

    def _fused_rope_quant_cache(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        q_scale: float,
        k_scale: float,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Fused RoPE + FP8 quant + KV cache scatter-write.

        Two independent kernels on separate CUDA streams:
          - main stream:  nope kernel (heavier, ~32us)
          - aux stream:   rope kernel (lighter, ~12us)
        Uses vLLM's aux_stream() singleton so all layers share one
        stream, avoiding excessive stream allocations during CUDA
        graph capture/replay.
        """
        assert self.rotary_emb is not None
        L = ql_nope.shape[-1]
        attn_dtype = torch.float8_e4m3fn

        q_out = q_pe.new_empty(
            q_pe.shape[0],
            q_pe.shape[1],
            L + q_pe.shape[2],
            dtype=attn_dtype,
        )

        inv_freq = self.rotary_emb.inv_freq
        num_kv_heads = 1 if k_pe.dim() == 2 else k_pe.size(1)

        rope_stream = aux_stream()

        # Fork: rope stream waits for main stream's prior work
        rope_stream.wait_stream(current_stream())

        with torch.cuda.stream(rope_stream):
            mla_fused_cache_rope(
                q_pe,
                q_out[..., L:],
                k_pe,
                kv_cache,
                slot_mapping,
                inv_freq,
                positions,
                num_kv_heads,
                L,
                q_scale,
                k_scale,
                True,
            )

        # Nope kernel on main stream (runs in parallel with rope)
        mla_fused_cache_nope(
            ql_nope,
            q_out[..., :L],
            k_nope,
            kv_cache,
            slot_mapping,
            num_kv_heads,
            q_scale,
            k_scale,
        )

        # Join: main stream waits for rope stream
        current_stream().wait_stream(rope_stream)

        return q_out

    def _fused_absorption_rope_cache(
        self,
        mqa_q_nope: torch.Tensor,
        mqa_q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        q_scale: float,
        k_scale: float,
        w_uk_bf16: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Fused CUTLASS BF16×BF16→FP8 absorption BMM + RoPE + KV cache write.

        Uses BF16 MMA with FP8 output via ScaledEpilogue. W_UK is
        pre-dequantized to BF16 during model init (one-time cost).

        Dual-stream execution:
          main:  CUTLASS BMM
          aux:   rope kernel + nope kernel (KV cache writes)  [starts before BMM]

        Args:
            mqa_q_nope: [N_heads, B, P=128] bf16, pre-transposed.
            mqa_q_pe:   [B, N_heads, R=64] bf16.
            k_nope:     [B, L=512] bf16 (k_c_normed).
            k_pe:       [B, R=64] bf16.
            positions:  [B] int64.
            q_scale:    FP8 quant scale for Q (host float).
            k_scale:    FP8 quant scale for K (host float).
            w_uk_bf16:  [N_heads, 512, 128] bf16 (pre-dequantized weight).
            kv_cache:   paged KV cache tensor.
            slot_mapping: [B] int64 slot indices.

        Returns:
            q_out: [B, N_heads, L+R] fp8 — ready for attention.
        """
        assert self.rotary_emb is not None
        N = mqa_q_nope.shape[0]
        B = mqa_q_nope.shape[1]
        L = w_uk_bf16.shape[1]
        R = mqa_q_pe.shape[-1]
        attn_dtype = torch.float8_e4m3fn

        q_out = mqa_q_pe.new_empty(B, N, L + R, dtype=attn_dtype)

        # --- 1. CUTLASS BF16×BF16→FP8 BMM -> q_out[..., :L] ---
        # Both q_nope and W_UK are BF16. No runtime quantization.
        # scale_a = q_scale (constant, cached on first call).
        if self._absorption_scale_a is None:
            self._absorption_scale_a = torch.tensor(
                q_scale, dtype=torch.float32, device="cuda"
            ).reshape(1)

        # --- 2. Dual-stream: BMM (main) / rope+nope (aux) ---
        # Fork aux stream BEFORE BMM: neither RoPE nor nope depend
        # on the BMM output. RoPE needs q_pe/k_pe, nope needs k_nope —
        # all from q_b_proj. Both ~16us combined, hidden behind ~35us BMM.
        inv_freq = self.rotary_emb.inv_freq
        num_kv_heads = 1 if k_pe.dim() == 2 else k_pe.size(1)

        rope_stream = aux_stream()
        rope_stream.wait_stream(current_stream())

        with torch.cuda.stream(rope_stream):
            mla_fused_cache_rope(
                mqa_q_pe,
                q_out[..., L:],
                k_pe,
                kv_cache,
                slot_mapping,
                inv_freq,
                positions,
                num_kv_heads,
                L,
                q_scale,
                k_scale,
                True,
            )

            # Nope kernel for K cache write only.
            # Zero-Q-head tensors [B, 0, L] so only K blocks launch.
            zero_q_in = k_nope.new_empty(B, 0, L)
            zero_q_out = q_out.new_empty(B, 0, L)
            mla_fused_cache_nope(
                zero_q_in,
                zero_q_out,
                k_nope,
                kv_cache,
                slot_mapping,
                num_kv_heads,
                q_scale,
                k_scale,
            )

        mla_absorption_bmm_bf16(
            q_out,
            mqa_q_nope,
            w_uk_bf16,
            self._absorption_scale_a,
            self._absorption_scale_b,
        )

        current_stream().wait_stream(rope_stream)

        return q_out

    def forward_mqa(
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
            self.bmm1_scale = self.scale
            if self.kv_cache_dtype.startswith("fp8"):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float

        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if self.kv_cache_dtype.startswith("fp8"):
                self.bmm2_scale *= layer._k_scale_float

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
