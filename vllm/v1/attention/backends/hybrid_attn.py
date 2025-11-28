#! SPDX-License-Identifier: Apache-2.0
#! SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mamba1_attn import (
    Mamba1AttentionMetadata,
    Mamba1AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

logger = init_logger(__name__)


@dataclass
class HybridAttentionMetadata:
    triton_metadata: TritonAttentionMetadata
    mamba_metadata: Mamba1AttentionMetadata

    @property
    def num_actual_tokens(self):
        return self.triton_metadata.num_actual_tokens


class HybridAttentionMetadataBuilder(
    AttentionMetadataBuilder[HybridAttentionMetadata]
):
    """Reuse TritonAttentionMetadataBuilder for hybrid attention."""

    _cudagraph_support: AttentionCGSupport = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # Delegate all heavy lifting to the Triton builder.
        self._triton_builder = TritonAttentionMetadataBuilder(
            kv_cache_spec, layer_names, vllm_config, device
        )

        # Construct a MambaSpec for Mamba metadata generation
        mamba_block_size = vllm_config.cache_config.mamba_block_size
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        mamba_spec = MambaSpec(
            shapes=(),  # Dummy
            dtypes=(),  # Dummy
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type="mamba1",
            num_speculative_blocks=0,
        )
        self._mamba_builder = Mamba1AttentionMetadataBuilder(
            mamba_spec, layer_names, vllm_config, device
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> HybridAttentionMetadata:
        triton_meta = self._triton_builder.build_for_cudagraph_capture(
            common_attn_metadata
        )
        mamba_meta = self._mamba_builder.build_for_cudagraph_capture(
            common_attn_metadata
        )
        return HybridAttentionMetadata(triton_meta, mamba_meta)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> HybridAttentionMetadata:
        triton_meta = self._triton_builder.build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        mamba_meta = self._mamba_builder.build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        return HybridAttentionMetadata(triton_meta, mamba_meta)


class HybridAttentionBackend(AttentionBackend):
    """Backend that combines Triton sliding-window attention with an SSM branch.

    KV cache layout and metadata are identical to the Triton attention backend;
    the only difference is that the implementation fuses in an additional SSM
    contribution computed by a ``HybridSSMAdapter`` owned by the layer.
    """

    accept_output_buffer: bool = True
    supported_dtypes: list[torch.dtype] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: list[CacheDType] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Same kernel requirements as TritonAttentionBackend.
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "HYBRID_ATTN"

    @staticmethod
    def get_impl_cls() -> type["HybridAttentionImpl"]:
        return HybridAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Identical KV cache shape as TritonAttentionBackend.
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        # Initial hybrid implementation does not support cascade attention.
        return False

    @staticmethod
    def get_builder_cls() -> type["HybridAttentionMetadataBuilder"]:
        return HybridAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Mirror Triton attention constraints.
        return head_size >= 32

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Defer detailed filtering to the underlying Triton kernels.
        return True


class HybridAttentionImpl(AttentionImpl):
    """Implementation that wraps TritonAttentionImpl and adds an SSM branch."""

    def fused_output_quant_supported(self, *args, **kwargs):
        # Delegate to the Triton implementation.
        return self._triton_impl.fused_output_quant_supported(*args, **kwargs)

    def supports_quant_query_input(self) -> bool:
        return self._triton_impl.supports_quant_query_input()

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
        # Mirror the Triton implementation's constructor so ``Attention``
        # can instantiate this backend without any special handling.
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type not in [AttentionType.DECODER, AttentionType.ENCODER_DECODER]:
            raise NotImplementedError(
                "Encoder self-attention is not implemented for HybridAttentionImpl"
            )
        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )

        # Internal Triton implementation that performs the sliding-window KV
        # caching and attention compute.
        self._triton_impl = TritonAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes.tolist()
            if isinstance(alibi_slopes, torch.Tensor)
            else alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HybridAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass combining Triton attention with an SSM history branch.

        1. Run standard Triton attention to populate the output tensor.
        2. If the layer exposes an ``ssm_adapter`` attribute, call its
           ``forward_history_branch_decode`` method to obtain an SSM
           contribution and fuse it into the first ``num_actual_tokens``
           positions of the output.
        """
        if output is None:
            raise ValueError("Output tensor must be provided.")

        # Step 1: delegate sliding-window attention to Triton.
        # Unwrap triton metadata if it's our hybrid metadata
        triton_metadata = (
            attn_metadata.triton_metadata if attn_metadata is not None else None
        )
        self._triton_impl.forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            triton_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

        # Profiling / warmup runs: no attention metadata, nothing to fuse.
        if attn_metadata is None:
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return output

        # Step 2: invoke the SSM adapter, if present on the layer.
        ssm_adapter = getattr(layer, "ssm_adapter", None)
        if ssm_adapter is None:
            return output

        # The adapter expects a tensor aligned with the flattened token layout.
        # We use the query representation as input, matching the attention
        # interface: [num_tokens, num_heads, head_size].
        query_tokens = query[:num_actual_tokens]
        try:
            ssm_out = ssm_adapter.forward_history_branch_decode(
                query_tokens, attn_metadata=attn_metadata
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "HybridAttentionImpl: SSM adapter %s raised an error: %s",
                type(ssm_adapter).__name__,
                exc,
            )
            return output

        if ssm_out is None:
            return output

        if ssm_out.shape != query_tokens.shape:
            raise ValueError(
                "HybridSSMAdapter must return a tensor with shape matching "
                f"the query slice. Expected {query_tokens.shape}, got "
                f"{ssm_out.shape}."
            )

        # Fuse SSM contribution into the attention output.
        # ``output`` is [num_tokens, num_heads, head_size].
        output[:num_actual_tokens] += ssm_out
        return output


