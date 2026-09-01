# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    get_dtype_size,
    kv_cache_dtype_str_to_dtype,
    nvfp4_kv_cache_full_dim,
    nvfp4_split_data_scale,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
    is_quantized_kv_cache,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer

# KV cache dtypes accepted for the QSA main cache. fp8_e5m2 is left out on
# purpose (smaller mantissa, no benefit for K/V); nvfp4_4over6 changes the
# block-scale search in the writer and has not been evaluated on this path.
# The indexer caches are not affected: they are constructed in bf16 (the
# key-state cache packs int64 mRoPE positions into bf16 cells).
_QSA_KV_CACHE_DTYPES: tuple[str, ...] = ("auto", "bfloat16", "fp8", "fp8_e4m3", "nvfp4")

# Storage dtypes of the main cache. Quantized caches are allocated as uint8
# (STR_DTYPE_TO_TORCH_DTYPE); the bytes are reinterpreted right before the
# kernel.
_QSA_KV_STORAGE_DTYPES = (torch.bfloat16, torch.uint8, torch.float8_e4m3fn)

# Block-scale layout written by reshape_and_cache_flash for nvfp4: K block
# scales are stored linearly, V block scales are swizzled for the TRT-LLM
# decode kernels. The read kernel follows the same convention.
_NVFP4_V_SCALE_SWIZZLED = True


def _nvfp4_attention_spec(spec: AttentionSpec) -> AttentionSpec:
    """Apply the nvfp4 page geometry to a QSA main-cache spec.

    K and V get separate head slots (all K heads first, then all V heads, the
    order ``kv_cache.split(num_kv_heads, dim=1)`` expects) and a cell shrinks
    from ``head_size + head_size_v`` bf16 values to
    ``head_size // 2 + head_size // 16`` bytes of fp4 data plus fp8 block
    scales. Same geometry as ``FlashInferBackend.customize_spec``. Idempotent,
    so it does not matter whether the model runner calls ``customize_spec``
    in addition to the owner applying it.
    """
    if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
        return spec
    hs_k = nvfp4_kv_cache_full_dim(spec.head_size)
    hs_v = nvfp4_kv_cache_full_dim(spec.head_size_v)
    assert hs_k == hs_v, "nvfp4 with asymmetric K/V head sizes not yet supported"
    return replace(
        spec,
        num_head_slots=2 * spec.num_kv_heads,
        state_content_bytes=hs_k * get_dtype_size(spec.dtype),
    )


def _nvfp4_cache_views(
    kv_cache: torch.Tensor, num_kv_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a packed nvfp4 cache ``[B, 2 * H, N, 144]`` into strided views.

    Returns ``(k_data, k_scales, v_data, v_scales)``, each ``[B, N, H, X]``
    with X = ``head_size // 2`` data bytes or ``head_size // 16`` block scales.
    The byte arithmetic is ``nvfp4_split_data_scale``, the same helper the
    writer side uses, so both sides see one layout by construction. Scales are
    returned as raw uint8 bits; the kernel decodes them.
    """
    k_side = kv_cache[:, :num_kv_heads].transpose(1, 2)
    v_side = kv_cache[:, num_kv_heads:].transpose(1, 2)
    k_data, k_scales = nvfp4_split_data_scale(k_side)
    v_data, v_scales = nvfp4_split_data_scale(v_side)
    return k_data, k_scales.view(torch.uint8), v_data, v_scales.view(torch.uint8)


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "nvfp4",
    ]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # The model runner passes every attention spec through this hook and
        # the hybrid page alignment on the platform uses it for the per-token
        # page size. Without the override the nvfp4 page would be declared
        # four times too large and the GDN page padding would be wrong.
        return _nvfp4_attention_spec(spec)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # The model runner asks the backend for the cache shape and
        # reinterprets the raw buffer with it; it does not read
        # num_head_slots / state_content_bytes from the spec. For nvfp4 the
        # packed rows are 2 * num_kv_heads slots of
        # head_size // 2 + head_size // 16 bytes.
        if isinstance(cache_dtype_str, str) and cache_dtype_str.startswith("nvfp4"):
            return (
                num_blocks,
                2 * num_kv_heads,
                block_size,
                nvfp4_kv_cache_full_dim(head_size),
            )
        return FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # nvfp4 needs the HND layout: a page is [K_data | K_scale | V_data |
        # V_scale], so the K half must be one contiguous half page. Under NHD
        # the K and V head slots interleave per token, nvfp4_split_data_scale
        # then derives strides across all 2 * num_kv_heads slots and the scale
        # region overlaps the data. That does not crash, it produces wrong
        # numbers. bf16 and fp8 keep the inherited choice.
        cfg = get_current_vllm_config_or_none()
        cache_dtype = getattr(getattr(cfg, "cache_config", None), "cache_dtype", "auto")
        if isinstance(cache_dtype, str) and cache_dtype.startswith("nvfp4"):
            if include_num_layers_dimension:
                return (1, 2, 0, 3, 4)
            return (0, 1, 2, 3)
        return FlashAttentionBackend.get_kv_cache_stride_order(
            include_num_layers_dimension
        )

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        # The inherited check asks whether FlashAttention kernels can read a
        # quantized cache on this device. The QSA main cache is only read by
        # the Triton kernel in ops/qsa.py, which dequantizes in registers, so
        # the FlashAttention capability is irrelevant here.
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_combination(cls, *args, **kwargs) -> str | None:
        # Same reason: the inherited message ("FP8 KV cache requires FA3 on
        # SM90 or FA4 on SM100") describes FlashAttention kernels this path
        # does not use.
        return None

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        # FlashAttentionImpl.__init__ rejects a quantized KV cache whenever
        # flash_attn_supports_kv_cache_dtype() is false for the device (it
        # knows SM90 with FA3/FA4 and SM100 with FA4). QSA never reads its
        # main cache with a FlashAttention kernel, so the dtype is passed
        # around the base constructor and restored afterwards; the inherited
        # write path (reshape_and_cache_flash) reads it later. kv_cache_dtype
        # is the seventh positional argument of FlashAttentionImpl.__init__.
        kv_cache_dtype: str | None = None
        if (
            len(args) > 6
            and isinstance(args[6], str)
            and args[6] not in ("auto", "bfloat16")
        ):
            kv_cache_dtype = args[6]
            args = (*args[:6], "auto", *args[7:])
        elif kwargs.get("kv_cache_dtype", "auto") not in ("auto", "bfloat16"):
            kv_cache_dtype = kwargs["kv_cache_dtype"]
            kwargs = {**kwargs, "kv_cache_dtype": "auto"}
        super().__init__(*args, **kwargs)
        if kv_cache_dtype is not None:
            self.kv_cache_dtype = kv_cache_dtype
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in _QSA_KV_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP8-E4M3 or NVFP4 main KV cache"
            )
        # nvfp4 is a separate branch (page geometry, writer, read kernel), not
        # fp8 with a different dtype.
        self.kv_cache_nvfp4 = self.kv_cache_dtype.startswith("nvfp4")
        self.kv_cache_fp8 = (
            is_quantized_kv_cache(self.kv_cache_dtype) and not self.kv_cache_nvfp4
        )
        # Strided views on the packed nvfp4 cache, built once per cache buffer.
        # as_strided is metadata only, but it would otherwise run in every
        # decode step, and the pointers stay stable across CUDA graph capture.
        self._nvfp4_views: tuple[int, tuple[torch.Tensor, ...]] | None = None
        self.supports_quant_query_input = False

    def _nvfp4_views_for(self, kv_cache: torch.Tensor) -> tuple[torch.Tensor, ...]:
        key = kv_cache.data_ptr()
        cached = self._nvfp4_views
        if cached is None or cached[0] != key:
            cached = (key, _nvfp4_cache_views(kv_cache, self.num_kv_heads))
            self._nvfp4_views = cached
        return cached[1]

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if not self.kv_cache_nvfp4:
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return None
        # The inherited path splits the cache with
        # kv_cache.transpose(1, 2).split(self.head_size, dim=-1), which assumes
        # K and V share one row of 2 * head_size values. nvfp4 stores them in
        # separate head slots of head_size // 2 + head_size // 16 bytes, so
        # the cache is cut by head slot here: (B, 2n, N, 144) -> two views
        # (B, N, n, 144), the 4D shape reshape_and_cache_flash expects for
        # nvfp4.
        from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

        n = self.num_kv_heads
        reshape_and_cache_flash(
            key,
            value,
            kv_cache[:, :n].transpose(1, 2),
            kv_cache[:, n:].transpose(1, 2),
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
        return None

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        k_scale_cache: torch.Tensor | None = None
        v_scale_cache: torch.Tensor | None = None
        if self.kv_cache_nvfp4:
            # Packed rows carry data and block scales for K and V in separate
            # head slots; there is nothing to split at head_size.
            key_cache, k_scale_cache, value_cache, v_scale_cache = (
                self._nvfp4_views_for(kv_cache)
            )
        else:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size, dim=-1
            )
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if self.kv_cache_fp8:
            # Quantized caches are allocated as uint8; reinterpret as fp8 so
            # the kernel decodes floats instead of integers. view() is a pure
            # reinterpretation (1 byte to 1 byte, unit stride on the last dim).
            # nvfp4 stays uint8: the packed nibbles are unpacked in the kernel.
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
        if query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 queries")
        if key_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16, FP8-E4M3 or NVFP4 K/V"
            )

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
            # Per-layer scales as device buffers (1.0 from
            # set_default_quant_scales(register_buffer=True)): no host-to-device
            # copy on the hot path, so CUDA graph capture is unaffected.
            k_scale=getattr(layer, "_k_scale", None),
            v_scale=getattr(layer, "_v_scale", None),
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            v_scale_swizzled=_NVFP4_V_SCALE_SWIZZLED,
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in _QSA_KV_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP8-E4M3 or NVFP4 main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype not in _QSA_KV_STORAGE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16, FP8-E4M3 or NVFP4 cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # The owner builds its own spec, so the nvfp4 geometry is applied here
        # as well as in customize_spec; both are idempotent.
        return _nvfp4_attention_spec(
            FullAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                head_size_v=self.head_dim,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            )
        )

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        else:
            qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


def qwen4_exp_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
    )


def qwen4_exp_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, output, layer_name


direct_register_custom_op(
    op_name="qwen4_exp_qsa_with_output",
    op_func=qwen4_exp_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen4_exp_qsa_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
    "qwen4_exp_qsa_with_output",
]
