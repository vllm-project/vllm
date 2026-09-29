# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm attention backend for token-major segmented Triton attention."""

from typing import TYPE_CHECKING, ClassVar

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend,
    RocmAttentionImpl,
    RocmAttentionMetadata,
    RocmAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import get_num_attention_heads_from_layers
from vllm.v1.attention.ops.segmented_attention import (
    MAX_QUERY_LEN,
    reserve_segmented_attention_workspace,
    segmented_attention,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode

if TYPE_CHECKING:
    from vllm.platforms.interface import DeviceCapability

logger = init_logger(__name__)


class RocmSegmentedAttentionMetadataBuilder(RocmAttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        model_config = vllm_config.model_config
        self.num_heads_q = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_heads_kv = kv_cache_spec.num_kv_heads
        self.headdim = kv_cache_spec.head_size
        self._reserve_workspace(kv_cache_spec, vllm_config)

    def _reserve_workspace(
        self,
        kv_cache_spec: AttentionSpec,
        vllm_config: VllmConfig,
    ) -> None:
        """Reserve only the scratch used by segmented prefill."""
        from vllm.platforms.rocm import on_gfx1x, on_gfx12x

        model_config = vllm_config.model_config
        fp8_kv_supported = (
            kv_cache_spec.kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
            and kv_cache_spec.dtype.itemsize == 1
            and vllm_config.cache_config.cache_dtype in ("fp8", "fp8_e4m3")
            and on_gfx12x()
        )
        native_kv_supported = (
            kv_cache_spec.kv_quant_mode == KVQuantMode.NONE
            and kv_cache_spec.dtype == model_config.dtype
        )
        if (
            on_gfx1x()
            and (native_kv_supported or fp8_kv_supported)
            and self.headdim in (64, 128, 256)
            and self.num_heads_kv > 0
            and self.num_heads_q % self.num_heads_kv == 0
            and 1 <= self.num_heads_q // self.num_heads_kv <= 16
        ):
            reserve_segmented_attention_workspace(
                vllm_config.scheduler_config.max_num_seqs,
                self.num_heads_q,
                self.num_heads_kv,
                self.headdim,
                model_config.max_model_len,
                max_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
                fp8=fp8_kv_supported,
            )


class RocmSegmentedAttentionBackend(RocmAttentionBackend):
    """Explicit token-major ROCm backend optimized for segmented prefill."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "ROCM_SEGMENTED_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RocmSegmentedAttentionImpl"]:
        return RocmSegmentedAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["RocmSegmentedAttentionMetadataBuilder"]:
        return RocmSegmentedAttentionMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_mm_prefix(cls) -> bool:
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
        device_capability: "DeviceCapability",
    ) -> str | None:
        del (
            head_size,
            dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )
        from vllm.platforms.rocm import on_gfx1x, on_gfx12x

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            if not on_gfx12x():
                return "FP8 segmented attention requires gfx12"
        elif not on_gfx1x():
            return "segmented attention requires gfx1x"
        return None


class RocmSegmentedAttentionImpl(RocmAttentionImpl):
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
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
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
            sinks,
        )
        from vllm.platforms.rocm import on_gfx1x, on_gfx12x

        fp8 = kv_cache_dtype in ("fp8", "fp8_e4m3")
        arch_supported = on_gfx12x() if fp8 else on_gfx1x()
        gqa_ratio_supported = (
            num_kv_heads > 0
            and num_heads % num_kv_heads == 0
            and 1 <= self.num_queries_per_kv <= 16
        )
        if not (
            arch_supported
            and attn_type == AttentionType.DECODER
            and head_size in (64, 128, 256)
            and gqa_ratio_supported
            and kv_cache_dtype in ("auto", "float16", "bfloat16", "fp8", "fp8_e4m3")
            and alibi_slopes is None
        ):
            raise ValueError(
                "ROCM_SEGMENTED_ATTN requires decoder attention on gfx1x "
                "(gfx12 for FP8), head size 64, 128 or 256, GQA ratio 1-16, and "
                "does not support ALiBi."
            )
        logger.info_once("Using token-major ROCm segmented Triton attention")
        self._segmented_attention_warmed_up = False
        self._segmented_attention_config: VllmConfig | None = None

    def _warmup_segmented_attention(self, layer, device, dtype, **limits) -> None:
        if (
            envs.VLLM_ROCM_SEGMENTED_ATTN_AUTOTUNE
            and not self._segmented_attention_warmed_up
            and self.alibi_slopes is None
            and not self.logits_soft_cap
        ):
            from vllm.v1.attention.ops.segmented_attention_tuning import (
                warmup_segmented_attention,
            )

            config = self._segmented_attention_config
            assert config is not None
            spec = layer.get_kv_cache_spec(config)
            assert spec is not None
            sliding_window = self.sliding_window[0]
            query_limit = getattr(layer, "segmented_query_limit", None)
            if not isinstance(query_limit, int):
                query_limit = MAX_QUERY_LEN
            causal = getattr(layer, "segmented_causal", True)
            if not isinstance(causal, bool):
                causal = True
            tuning_max_len = config.model_config.max_model_len
            if sliding_window >= 0:
                tuning_max_len = min(tuning_max_len, sliding_window + query_limit)
            warmup_segmented_attention(
                device,
                dtype,
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                spec.block_size,
                self.scale,
                config.scheduler_config.max_num_batched_tokens,
                tuning_max_len,
                config.scheduler_config.max_num_seqs,
                kv_dtype=spec.dtype if spec.dtype != torch.uint8 else self.fp8_dtype,
                sliding_window=sliding_window,
                causal=causal,
                has_sinks=self.sinks is not None,
                physical_max_len=config.model_config.max_model_len,
                max_query_len=query_limit,
                **limits,
            )
            self._segmented_attention_warmed_up = True

    def _split_kv_cache(
        self,
        kv_cache: torch.Tensor,
        num_kv_heads: int | None = None,
        head_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_kv_heads = num_kv_heads or self.num_kv_heads
        head_size = head_size or self.head_size
        return (
            kv_cache[:, 0].unflatten(-1, (num_kv_heads, head_size)),
            kv_cache[:, 1].unflatten(-1, (num_kv_heads, head_size)),
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not supported"
            )
        if attn_metadata is None:
            return output.fill_(0)
        assert not attn_metadata.use_cascade

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            if query.dtype == self.fp8_dtype and layer._q_scale_float != 1.0:
                raise NotImplementedError(
                    "A non-1.0 q_scale with an FP8 query is not supported"
                )

        segmented_attention(
            query=query[:num_actual_tokens],
            key=key[:num_actual_tokens],
            value=value[:num_actual_tokens],
            output=output[:num_actual_tokens],
            kv_cache_dtype=self.kv_cache_dtype,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            query_start_loc=attn_metadata.query_start_loc,
            seq_lens=attn_metadata.seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
            max_query_len=attn_metadata.max_query_len,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
            sm_scale=self.scale,
            sliding_window=self.sliding_window[0],
            output_scale=output_scale,
            sinks=self.sinks,
            causal=attn_metadata.causal,
            softcap=self.logits_soft_cap,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = self._split_kv_cache(
            kv_cache,
            layer.num_kv_heads,  # type: ignore[attr-defined]
            layer.head_size,  # type: ignore[attr-defined]
        )
        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            True,
            is_fp8_kv_cache,
        )
