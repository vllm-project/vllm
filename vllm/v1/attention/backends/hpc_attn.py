# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with HPC custom CUDA kernels."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

_reshape_and_cache_flash = None
_hpc = None


def _get_reshape_and_cache_flash():
    global _reshape_and_cache_flash
    if _reshape_and_cache_flash is None:
        if (
            current_platform.is_cuda()
            or current_platform.is_xpu()
            or current_platform.is_rocm()
        ):
            from vllm._custom_ops import reshape_and_cache_flash as fn
        else:
            raise RuntimeError(
                "HPC attention backend is not supported on this platform."
            )
        _reshape_and_cache_flash = fn
    return _reshape_and_cache_flash


def _get_hpc():
    """Return the cached hpc module, importing it on first call."""
    global _hpc
    if _hpc is None:
        import hpc as _hpc_mod

        _hpc = _hpc_mod
    return _hpc


def _hpc_decode_use_splitk(
    max_decode_seq_len: int,
    num_decode_tokens: int,
    num_heads: int,
    num_kv_heads: int,
) -> bool:
    """Whether to enable split-K in the HPC decode kernel.

        TODO. add auto-tuning for split-K.
    """

    # special case for 32 heads and 4 kv heads (split-K tuning table).
    if num_heads == 32 and num_kv_heads == 4:
        if num_decode_tokens < 6:
            return True
        if 6 <= num_decode_tokens < 12 and max_decode_seq_len <= 1024:
            return False
        if 12 <= num_decode_tokens < 14 and max_decode_seq_len <= 3072:
            return False
        if 14 <= num_decode_tokens < 16 and max_decode_seq_len <= 4096:
            return False
        if 16 <= num_decode_tokens < 24 and max_decode_seq_len <= 8192:
            return False
        if 24 <= num_decode_tokens < 32 and max_decode_seq_len <= 24576:
            return False
        if num_decode_tokens >= 32 and max_decode_seq_len <= 24576:
            return False
        return True
    else:
        if max_decode_seq_len < 1024:
            return False
        else:
            return True

logger = init_logger(__name__)


@dataclass
class HpcAttentionMetadata(AttentionMetadata):
    """Per-batch metadata for the HPC attention backend."""

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_decodes: int = 0
    num_decode_tokens: int = 0
    max_decode_seq_len: int = 0
    cu_seqlens_q_prefill: torch.Tensor | None = None


class HpcAttentionBackend(AttentionBackend):
    """vLLM v1 attention backend wrapping HPC custom CUDA kernels.

    Supports:
    - bfloat16 only
    - block_size (page_size) = 32 or 64
    - Decoder attention only
    - CUDA graph for uniform single-token decode batches
    - No cascade attention, no sliding window, no ALiBi
    """

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "HPC_ATTN"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [32, 64]

    @staticmethod
    def get_impl_cls() -> type["HpcAttentionImpl"]:
        return HpcAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["HpcAttentionMetadataBuilder"]:
        return HpcAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)


class HpcAttentionMetadataBuilder(AttentionMetadataBuilder):
    """Builds HpcAttentionMetadata from CommonAttentionMetadata."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> "HpcAttentionMetadata":
        attn_metadata = self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        attn_metadata.seq_lens.fill_(1)
        attn_metadata.block_table.fill_(0)
        attn_metadata.max_decode_seq_len = (
            1 if attn_metadata.num_decodes > 0 else 0
        )
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> HpcAttentionMetadata:
        if common_prefix_len > 0:
            raise NotImplementedError(
                "HPC attention backend does not support cascade attention "
                "(common_prefix_len > 0)."
            )

        seq_lens = common_attn_metadata.seq_lens.to(torch.int32, non_blocking=True)
        block_table = common_attn_metadata.block_table_tensor.to(
            torch.int32, non_blocking=True
        )

        num_decodes, num_prefills, num_decode_tokens, _num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
            )
        )

        if num_prefills > 0:
            cu_seqlens_q_prefill_cpu = (
                common_attn_metadata.query_start_loc_cpu[num_decodes:]
                - num_decode_tokens
            ).to(torch.int32)
            cu_seqlens_q_prefill = cu_seqlens_q_prefill_cpu.to(
                common_attn_metadata.query_start_loc.device,
                non_blocking=True,
            )
        else:
            cu_seqlens_q_prefill = None

        if num_decodes > 0:
            max_decode_seq_len = int(
                common_attn_metadata.seq_lens_cpu[:num_decodes].max().item()
            )
        else:
            max_decode_seq_len = 0

        return HpcAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            max_decode_seq_len=max_decode_seq_len,
            cu_seqlens_q_prefill=cu_seqlens_q_prefill,
        )


class HpcAttentionImpl(AttentionImpl[HpcAttentionMetadata]):
    """Attention implementation using HPC custom CUDA kernels.

    KV cache write: handled externally via do_kv_cache_update()
        (uses standard reshape_and_cache_flash).
    Attention compute: handled internally via forward()
        (calls hpc.attention_decode_bf16 or
         hpc.attention_with_kvcache_prefill_bf16).
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        block_size: int = 32,
    ) -> None:
        assert kv_cache_dtype in ("auto", "bfloat16"), (
            f"HPC attention backend only supports bfloat16 KV cache, "
            f"got kv_cache_dtype={kv_cache_dtype!r}. "
            f"Use --kv-cache-dtype auto or --kv-cache-dtype bfloat16."
        )
        assert attn_type == AttentionType.DECODER, (
            f"HPC attention backend only supports decoder attention, "
            f"got attn_type={attn_type!r}."
        )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "HPC attention backend does not support ALiBi slopes."
            )
        if sliding_window is not None:
            raise NotImplementedError(
                "HPC attention backend does not support sliding window attention."
            )
        if logits_soft_cap is not None:
            raise NotImplementedError(
                "HPC attention backend does not support logits soft cap."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.block_size = block_size
        self._kv_cache_view_shape = (-1, block_size, self.num_kv_heads, head_size)

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write new K/V tokens into the paged KV cache.

        Uses the standard vLLM reshape_and_cache_flash op.
        NOTE: slot_mapping drives how many tokens are written —
        no need to slice key/value to num_actual_tokens.
        """
        key_cache, value_cache = kv_cache.unbind(0)
        _get_reshape_and_cache_flash()(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: HpcAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run attention using HPC kernels.

        For mixed batches the framework guarantees that decode requests
        (query_len == 1) are placed at the front via batch reordering.
        We split at the decode/prefill boundary and call the optimal kernel
        for each segment, mirroring the FlashInfer backend pattern:

          query[:num_decode_tokens]  → attention_decode_bf16
          query[num_decode_tokens:n] → attention_with_kvcache_prefill_bf16

        Args:
            query: [num_tokens, num_heads, head_size] bfloat16
            key:   [num_tokens, num_kv_heads, head_size] bfloat16 (unused;
                   KV is read from kv_cache populated by do_kv_cache_update)
            value: [num_tokens, num_kv_heads, head_size] bfloat16 (unused)
            kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: HpcAttentionMetadata
            output: [num_tokens, num_heads, head_size] bfloat16
        Returns:
            output tensor, shape [num_tokens, num_heads, head_size]
        """
        hpc = _get_hpc()

        assert output is not None, "HPC backend requires output tensor."

        if attn_metadata is None:
            return output.fill_(0)

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "HPC attention backend does not support fused output quantization."
            )

        n = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes

        key_cache = kv_cache[0].view(self._kv_cache_view_shape)
        value_cache = kv_cache[1].view(self._kv_cache_view_shape)

        if num_decode_tokens > 0:
            splitk = _hpc_decode_use_splitk(
                attn_metadata.max_decode_seq_len,
                num_decode_tokens,
                self.num_heads,
                self.num_kv_heads,
            )
            hpc.attention_decode_bf16(
                q=query[:num_decode_tokens],
                kcache=key_cache,
                vcache=value_cache,
                block_ids=attn_metadata.block_table[:num_decodes],
                num_seq_kvcache=attn_metadata.seq_lens[:num_decodes],
                new_kv_included=True,
                splitk=splitk,
                output=output[:num_decode_tokens],
            )

        if n > num_decode_tokens:
            hpc.attention_with_kvcache_prefill_bf16(
                q=query[num_decode_tokens:n],
                kcache=key_cache,
                vcache=value_cache,
                cu_seqlens_q=attn_metadata.cu_seqlens_q_prefill,
                block_ids=attn_metadata.block_table[num_decodes:],
                seqlens_kvcache=attn_metadata.seq_lens[num_decodes:],
                max_seqlens_q=attn_metadata.max_query_len,
                output=output[num_decode_tokens:n],
            )

        return output
