# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import ClassVar, cast

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.config.cache import CacheConfig, CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import (
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.hisparse.runtime import (
    build_hisparse_prefill_staging_plan,
    create_hisparse_cache_handle,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    SparseFullAttentionSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer


@dataclass(kw_only=True)
class Qwen4ExpQSAMetadata(FlashAttentionMetadata):
    num_reqs: int
    req_id_per_token: torch.Tensor
    is_cudagraph_capture: bool = False


class Qwen4ExpQSAQKVIndexerLinear(MergedColumnParallelLinear):
    """Pack sharded Q/gate, K, V and replicated indexer Q/K in one GEMM."""

    def __init__(
        self,
        qkv_proj: QKVParallelLinear,
        index_size: int,
        quant_config: QuantizationConfig | None,
    ) -> None:
        self.num_kv_head_replicas = qkv_proj.num_kv_head_replicas
        super().__init__(
            input_size=qkv_proj.input_size,
            output_sizes=[*qkv_proj.output_sizes, index_size * qkv_proj.tp_size],
            bias=False,
            params_dtype=qkv_proj.params_dtype,
            quant_config=quant_config,
            prefix=qkv_proj.prefix,
        )

    def _load_shard(
        self,
        loader: Callable[..., None],
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | tuple[int, ...] | None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == 3:
            shard_rank = 0
        elif loaded_shard_id in (1, 2):
            shard_rank = tp_rank // self.num_kv_head_replicas
        else:
            shard_rank = tp_rank
        self.tp_rank = shard_rank
        if param_tp_rank is not None:
            param.tp_rank = shard_rank
        try:
            loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(super().weight_loader, param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(
            super().weight_loader_v2, param, loaded_weight, loaded_shard_id
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        def remap_shards():
            for name, weight in weights:
                shard_id = getattr(weight, "shard_id", None)
                if isinstance(shard_id, str):
                    weight = weight.detach()
                    weight.shard_id = {"q": 0, "k": 1, "v": 2}[shard_id]
                yield name, weight

        return super().load_weights(remap_shards())


def qsa_kv_cache_dtype(cache_config: CacheConfig, prefix: str) -> CacheDType:
    """The layer's KV cache dtype, honoring ``--kv-cache-dtype-skip-layers``."""
    if str(extract_layer_index(prefix)) in cache_config.kv_cache_dtype_skip_layers:
        return "auto"
    return cache_config.cache_dtype


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.hisparse_enabled = vllm_config.attention_config.hisparse_config is not None
        if self.hisparse_enabled:
            self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
            self.token_to_req_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                dtype=torch.int32,
                device=device,
            )
            self._hisparse_token_to_req_buffer = torch.empty_like(
                self.token_to_req_buffer
            )
            self._hisparse_token_offsets = torch.arange(
                self.token_to_req_buffer.numel(), dtype=torch.int32, device=device
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if not self.hisparse_enabled:
            return metadata
        (
            metadata.num_decode_reqs,
            metadata.num_prefill_reqs,
            metadata.num_decode_tokens,
            metadata.num_prefill_tokens,
        ) = split_decodes_and_prefills(common_attn_metadata)
        token_to_req = common_attn_metadata.token_to_req_indices(
            self.token_to_req_buffer
        )
        num_tokens = token_to_req.shape[0]
        hisparse_token_to_req = self._hisparse_token_to_req_buffer[:num_tokens]
        # MTP reuse can retain selections in FULL padding rows. Keep those rows
        # out of residency state without changing other groups' shared mapping.
        hisparse_token_to_req.copy_(token_to_req)
        hisparse_token_to_req.masked_fill_(
            self._hisparse_token_offsets[:num_tokens]
            >= common_attn_metadata.query_start_loc[-1],
            -1,
        )
        return Qwen4ExpQSAMetadata(
            **vars(metadata),
            num_reqs=common_attn_metadata.num_reqs,
            req_id_per_token=hisparse_token_to_req,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlashAttentionMetadata:
        metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        if isinstance(metadata, Qwen4ExpQSAMetadata):
            metadata.is_cudagraph_capture = True
        return metadata


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # fp8/fp8_e4m3: e4m3 bytes in a uint8 cache, written by reshape_and_cache
    # with the layer's per-tensor scales and dequantized on load inside the QSA
    # Triton kernel. flash-attn never runs over this cache, so its fp8 probe
    # does not apply (see supports_kv_cache_dtype and the impl constructor).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes

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
        # QSA dequantizes the fp8 KV in its own Triton kernel and never runs
        # flash-attn over the quantized cache, so the parent's fp8-KV rejection
        # does not apply and every combination it is handed is accepted here.
        return None

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
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

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        # The parent constructor probes flash-attn for quantized-KV support and
        # raises where it is unavailable (sm120), but QSA dequantizes fp8 inside
        # its own Triton kernel and never runs flash-attn over the cache. Hand
        # the parent "auto" for that probe and restore the real dtype afterwards:
        # the parent only uses it there, and do_kv_cache_update reads the
        # attribute at call time.
        real_kv_cache_dtype = kv_cache_dtype
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            kv_cache_dtype = "auto"
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
            sinks,
        )
        self.kv_cache_dtype = real_kv_cache_dtype
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
            )
        self.supports_quant_query_input = False

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
        use_prefill_config: bool,
        output_gate: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        physical_indices: bool = False,
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

        topk_buffer = (
            topk_indices
            if topk_indices is not None
            else getattr(layer, "topk_indices_buffer", None)
        )
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        selected_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        k_scale = v_scale = None
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            # The cache is allocated as uint8; reinterpret the e4m3 bytes
            # (same itemsize, so shape and strides are preserved).
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
            # Host-side per-tensor dequant scales (Python floats), as used by
            # other host-scale backends; folded into the kernel's scales.
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float
        if query.dtype != torch.bfloat16 or key_cache.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 Q and BF16 or FP8-e4m3 K/V"
            )

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            selected_indices,
            attn_metadata.block_table,
            token_to_req,
            use_prefill_config,
            output[:num_tokens],
            k_scale=k_scale,
            v_scale=v_scale,
            output_gate=output_gate[:num_tokens],
            physical_indices=physical_indices,
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
        if cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
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
        # Decode/verify batches have at most 1 + num_spec query tokens per
        # request; use_prefill_config (max_query_len > this) steers the
        # config table. Shorter batches take the decode profile — harmless,
        # the difference is tile-shape tuning, not correctness.
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.total_num_heads)
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
        self.kv_cache_dtype = qsa_kv_cache_dtype(cache_config, prefix)
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.uint8):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 or FP8-e4m3 (uint8) cache storage"
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
        self.fuse_indexer_projection = vllm_config.lora_config is None
        if self.fuse_indexer_projection:
            self.index_qk_size = self.indexer.index_qk_proj.output_size
            self.qkv_proj = Qwen4ExpQSAQKVIndexerLinear(
                self.qkv_proj,
                self.index_qk_size,
                model.without_modelopt_fp4(quant_config),
            )
            del self.indexer.index_qk_proj

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # PACKED selection buffer: the trailing column holds each row's
        # valid-entry count (written by the expand kernel) — never a token
        # index; the sparse attention kernel reads it as its loop bound.
        # MTP skip_topk steps reuse rows frozen from step 0; the count is
        # a row column, so compaction/reuse keep it paired with the content.
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.packed_output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        self.hisparse_cache = create_hisparse_cache_handle(
            vllm_config,
            self.indexer.output_width,
            is_index_group_leader=True,
            row_width=self.num_kv_heads * 2 * self.head_dim,
            kv_dtype=self.kv_cache_torch_dtype,
        )
        # Fused prepare writes directly to the ordinary main cache; HiSparse
        # must select its resident write target before updating KV.
        self.use_fused_qsa_prepare = (
            self.hisparse_cache is None
            and self.use_fused_qk_norm_rope_gate
            and self.indexer.use_fused_pre_indexer
        )
        if self.hisparse_cache is not None:
            self.register_buffer(
                "physical_topk_indices_buffer",
                torch.empty_like(self.topk_indices_buffer),
                persistent=False,
            )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self.impl.process_weights_after_loading(act_dtype)
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = None
        if self.fuse_indexer_projection:
            mapper = WeightsMapper(
                orig_to_new_stacked={
                    "indexer.index_qk_proj.": ("qkv_proj.", 3),
                }
            )
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        spec = FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
        if self.hisparse_cache is None:
            return spec
        return SparseFullAttentionSpec(
            **vars(spec),
            top_k=self.indexer.output_width,
            total_num_kv_heads=self.total_num_kv_heads,
        )

    def _hisparse_kv_view(self, rows: torch.Tensor) -> torch.Tensor:
        return rows.unflatten(-1, (self.num_kv_heads, 2 * self.head_dim)).transpose(
            1, 2
        )

    def _forward_hisparse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        output_gate: torch.Tensor,
        metadata: Qwen4ExpQSAMetadata,
    ) -> None:
        cache = self.hisparse_cache
        assert cache is not None and cache.view is not None
        assert cache.block_table is not None and cache.slot_mapping is not None
        cache.prepare_group_for_batch(metadata)
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        resident, slots, num_rows = cache.write_target(
            key.shape[0], metadata.slot_mapping.shape[0]
        )
        impl.do_kv_cache_update(
            self,
            key[:num_rows],
            value[:num_rows],
            self._hisparse_kv_view(resident),
            slots,
        )
        # FULL warmup and capture both run with runtime mode NONE, without a
        # connector mirror phase. Replay mirrors real rows at worker finish.
        if not metadata.is_cudagraph_capture:
            cache.finish_kv_update()

        output.zero_()
        num_decode = metadata.num_decode_tokens
        num_tokens = metadata.num_actual_tokens
        if num_decode:
            assert cache.source_block_table is not None
            logical = self.topk_indices_buffer[:num_decode]
            physical = self.physical_topk_indices_buffer[:num_decode]
            physical[:, -1].copy_(logical[:, -1])
            resolved = cache.swap_in(
                metadata.req_id_per_token[:num_decode],
                cache.source_block_table,
                logical[:, :-1],
                block_size=cache.view.block_size,
                num_valid_rows=metadata.query_start_loc[-1:],
            )
            assert isinstance(resolved, torch.Tensor)
            physical[:, :-1].copy_(resolved)
            impl.forward_qsa(
                self,
                query[:num_decode],
                key[:num_decode],
                value[:num_decode],
                self._hisparse_kv_view(cache.runtime.hot.attention_cache),
                replace(metadata, num_actual_tokens=num_decode),
                output[:num_decode],
                token_to_req=metadata.req_id_per_token[:num_decode],
                use_prefill_config=False,
                output_gate=output_gate[:num_decode],
                topk_indices=physical,
                physical_indices=True,
            )
        if num_decode < num_tokens:
            if (
                metadata.is_cudagraph_capture
                and 1 < metadata.max_query_len <= self._max_decode_query_len
            ):
                logical = self.topk_indices_buffer[num_decode:num_tokens]
                prefill_requests = metadata.req_id_per_token[num_decode:num_tokens]
                staged, resolved = cache.runtime.gather_selected_cache(
                    cache, prefill_requests, logical[:, :-1], logical[:, -1]
                )
                physical = self.physical_topk_indices_buffer[num_decode:num_tokens]
                physical[:, :-1].copy_(resolved)
                physical[:, -1].copy_(logical[:, -1])
                impl.forward_qsa(
                    self,
                    query[num_decode:num_tokens],
                    key[num_decode:num_tokens],
                    value[num_decode:num_tokens],
                    self._hisparse_kv_view(staged),
                    replace(metadata, num_actual_tokens=num_tokens - num_decode),
                    output[num_decode:num_tokens],
                    token_to_req=prefill_requests,
                    use_prefill_config=True,
                    output_gate=output_gate[num_decode:num_tokens],
                    topk_indices=physical,
                    physical_indices=True,
                )
                return
            prefill_cache = resident
            prefill_requests = metadata.req_id_per_token[num_decode:num_tokens]
            if not cache.all_context_pages_resident:
                assert cache.source_block_table is not None
                block_size = cache.view.block_size
                first_prefill = metadata.num_decode_reqs
                capacity = metadata.num_prefill_reqs * (
                    (metadata.max_seq_len + block_size - 1) // block_size
                )
                plan = build_hisparse_prefill_staging_plan(
                    cache.source_block_table[first_prefill : metadata.num_reqs],
                    metadata.seq_lens[first_prefill : metadata.num_reqs],
                    block_size,
                    capacity,
                )
                state_indices = cache.runtime.request_state_indices
                assert state_indices is not None
                plan.ensure_gpu_sources(
                    cache.block_table,
                    state_indices[first_prefill : metadata.num_reqs],
                    block_size,
                )
                prefill_cache = cache.runtime.gather_prefill_cache(
                    cache.runtime.host_cache.view(
                        -1, block_size, cache.runtime.row_width
                    ),
                    plan,
                    resident_cache=resident,
                )
                prefill_table = plan.block_table
                prefill_requests = prefill_requests - first_prefill
            else:
                prefill_table = cache.batch_block_table()
            impl.forward_qsa(
                self,
                query[num_decode:num_tokens],
                key[num_decode:num_tokens],
                value[num_decode:num_tokens],
                self._hisparse_kv_view(prefill_cache),
                replace(
                    metadata,
                    num_actual_tokens=num_tokens - num_decode,
                    block_table=prefill_table,
                ),
                output[num_decode:num_tokens],
                token_to_req=prefill_requests,
                use_prefill_config=True,
                output_gate=output_gate[num_decode:num_tokens],
                topk_indices=self.topk_indices_buffer[num_decode:num_tokens],
            )

    @eager_break_during_capture
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        output_gate: torch.Tensor | None,
        qkv: torch.Tensor,
    ) -> None:
        # query/key/value/output_gate are None when the fused prepare runs
        # inside the indexer launch.
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
        selected, main_outputs = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
            attn=self,
            qkv=qkv,
            slot_mapping=main_metadata.slot_mapping,
        )
        if selected.shape != (num_tokens, self.indexer.packed_output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        if self.hisparse_cache is not None:
            assert query is not None and output_gate is not None
            assert key is not None and value is not None
            self._forward_hisparse(
                query,
                key,
                value,
                output,
                output_gate,
                cast(Qwen4ExpQSAMetadata, main_metadata),
            )
            return
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if main_outputs is None:
            assert key is not None and value is not None
            impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
        else:
            query, output_gate = main_outputs
        assert query is not None and output_gate is not None
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
            output_gate=output_gate,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.fuse_indexer_projection:
            qkv, projected_qk = qkv.split(
                [2 * self.q_size + 2 * self.kv_size, self.index_qk_size], dim=-1
            )
        else:
            projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        num_tokens = hidden_states.shape[0]
        if not self.use_fused_qsa_prepare:
            q, k, v, gate = self._project_qkv_gate(qkv, positions)
            assert gate is not None
            query = q.view(num_tokens, self.num_heads, self.head_dim)
            key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            # Norm/RoPE/gate and the K/V cache write happen inside _run_qsa.
            query = key = value = gate = None
        attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
        self._run_qsa(
            projected_qk,
            positions,
            query,
            key,
            value,
            attn_output,
            gate,
            qkv,
        )
        flat_output = attn_output.view(num_tokens, -1)
        output, _ = self.o_proj(flat_output)
        return output


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
]
