# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with TensorRT-LLM attention kernels."""

from dataclasses import dataclass
from typing import ClassVar

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.utils import FP4Tensor

from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    can_use_trtllm_attention,
    supports_trtllm_attention,
)
from vllm.utils.torch_utils import is_strictly_contiguous
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    get_kv_cache_layout,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

# Global workspace buffer for TRTLLM attention
_trtllm_workspace_buffer = None


def _get_trtllm_workspace_buffer():
    """Get or create the workspace buffer for TRTLLM attention."""
    global _trtllm_workspace_buffer
    if _trtllm_workspace_buffer is None:
        _trtllm_workspace_buffer = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda"
        )
    return _trtllm_workspace_buffer


@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    K_CACHE_STRIDE: tl.constexpr,
    KV_CACHE_STRIDE: tl.constexpr,
):
    """Triton kernel to dequantize FP8 KV cache for TRTLLM prefill attention.

    TRTLLM prefill attention does not support BF16 Q with FP8 KV cache,
    so we dequantize the relevant pages to BF16 for the prefill operation.
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx
    ).to(tl.int64)
    if orig_page_num <= 0:
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    # Dequantize K
    k_scale_val = tl.load(k_scale_ptr)
    offset = orig_page_num * KV_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    fp8_vals = tl.load(kv_cache_ptr + offset)
    dequantized_vals = fp8_vals.to(tl.float32) * k_scale_val
    mock_cache_offset = (
        batch_idx * block_table_stride + mock_block_table_idx + 1
    ) * KV_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    dequantized_vals = dequantized_vals.to(dequant_dtype)
    tl.store(mock_kv_cache_ptr + mock_cache_offset, dequantized_vals)

    # Dequantize V
    v_scale_val = tl.load(v_scale_ptr)
    offset = (
        orig_page_num * KV_CACHE_STRIDE + K_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    )
    fp8_vals = tl.load(kv_cache_ptr + offset)
    dequantized_vals = fp8_vals.to(tl.float32) * v_scale_val
    mock_cache_offset = (
        (batch_idx * block_table_stride + mock_block_table_idx + 1) * KV_CACHE_STRIDE
        + K_CACHE_STRIDE
        + tl.arange(0, K_CACHE_STRIDE)
    )
    dequantized_vals = dequantized_vals.to(dequant_dtype)
    tl.store(mock_kv_cache_ptr + mock_cache_offset, dequantized_vals)


def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize FP8 KV cache pages for TRTLLM prefill.

    Creates a mock KV cache with dequantized values and a mock block table
    that sequentially indexes the dequantized pages.

    Args:
        kv_cache: The FP8 KV cache tensor
        block_tables_prefill: Block table for prefill requests
        k_scale: Key quantization scale
        v_scale: Value quantization scale
        dequant_dtype: Target dtype for dequantization (bf16 or fp16)

    Returns:
        Tuple of (mock_kv_cache, mock_block_table)
    """
    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    assert s[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)
    k_cache_stride = s[2] * s[3] * s[4]
    kv_cache_stride = k_cache_stride * s[1]
    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    # mock kv cache contains just the pages needed by this prefill
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device=kv_cache.device)
    # we simply sequentially index the pages needed by this prefill
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device=block_tables_prefill.device,
    ).reshape(batch_size, num_of_page_per_token)
    grid = (batch_size, num_of_page_per_token)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        k_cache_stride,
        kv_cache_stride,
    )
    return mock_kv_cache, mock_block_table


class TRTLLMAttentionBackend(AttentionBackend):
    """TensorRT-LLM attention backend for Blackwell (SM100) GPUs."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64]

    @staticmethod
    def get_name() -> str:
        return "TRTLLM_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TRTLLMImpl"]:
        return TRTLLMImpl

    @staticmethod
    def get_builder_cls() -> type["TRTLLMMetadataBuilder"]:
        return TRTLLMMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # TRTLLM requires HND layout
        cache_layout = get_kv_cache_layout()
        if cache_layout == "HND" and include_num_layers_dimension:
            return (1, 2, 4, 0, 3, 5)
        elif cache_layout == "HND":
            return (0, 1, 3, 2, 4)
        else:
            raise ValueError(
                f"TRTLLM attention requires HND cache layout, got {cache_layout}"
            )

    @staticmethod
    def get_fp8_dtype_for_trtllm(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # TRTLLM attention only supports Blackwell (SM100)
        return capability >= DeviceCapability(10, 0)

    @classmethod
    def supports_sink(cls) -> bool:
        """TRTLLM attention supports attention sinks."""
        return supports_trtllm_attention()

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        return "HND"


@dataclass
class TRTLLMPrefillMetadata:
    """Metadata for TRTLLM prefill attention."""

    block_tables: torch.Tensor
    """Block table tensor for prefill requests.
    
    Shape: [num_prefills, max_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """Sequence lengths for prefill requests. Shape: [num_prefills]"""

    cum_seq_lens_q: torch.Tensor
    """Cumulative query sequence lengths. Shape: [num_prefills + 1]"""

    cum_seq_lens_kv: torch.Tensor
    """Cumulative KV sequence lengths. Shape: [num_prefills + 1]"""

    max_q_len: int
    """Maximum query length among prefill requests."""

    max_seq_len: int
    """Maximum sequence length for KV cache."""


@dataclass
class TRTLLMDecodeMetadata:
    """Metadata for TRTLLM decode attention."""

    block_tables: torch.Tensor
    """Block table tensor for decode requests.
    
    Shape: [num_decodes, max_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """Sequence lengths for decode requests. Shape: [num_decodes]"""

    max_seq_len: int
    """Maximum sequence length for KV cache."""


@dataclass
class TRTLLMMetadata:
    """Attention metadata for TRTLLM backend."""

    num_actual_tokens: int
    """Total number of tokens in the batch (excluding padding)."""

    slot_mapping: torch.Tensor
    """Tensor for writing K/V to the cache. Shape: [num_actual_tokens]"""

    q_data_type: torch.dtype
    """Data type for query tensors."""

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    prefill: TRTLLMPrefillMetadata | None
    """Metadata for prefill portion of batch. None if num_prefill_tokens == 0."""

    decode: TRTLLMDecodeMetadata | None
    """Metadata for decode portion of batch. None if num_decode_tokens == 0."""


class TRTLLMMetadataBuilder(AttentionMetadataBuilder[TRTLLMMetadata]):
    """Builds attention metadata for TRTLLM backend."""

    reorder_batch_threshold: int = 1
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.attention_config = vllm_config.attention_config
        self.parallel_config = vllm_config.parallel_config

        self.num_qo_heads = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        self.page_size = self.kv_cache_spec.block_size

        # Validate TRTLLM is supported
        if not can_use_trtllm_attention(self.num_qo_heads, self.num_kv_heads):
            raise ValueError(
                "TRTLLM attention is not supported for this configuration. "
                f"num_qo_heads={self.num_qo_heads}, num_kv_heads={self.num_kv_heads}"
            )

        # TRTLLM attention requires strictly contiguous KV cache tensors.
        # KV transfer (P/D disaggregation) may permute KV cache into
        # non-contiguous views, causing assertion failures.
        if vllm_config.kv_transfer_config is not None:
            raise ValueError(
                "TRTLLM attention is incompatible with KV transfer "
                "(P/D disaggregation). TRTLLM attention requires strictly "
                "contiguous KV cache tensors which are not guaranteed "
                "with KV transfer. Use FlashInfer or Flash Attention backend."
            )

        # TRTLLM does not support DCP
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            dcp_world_size = get_dcp_group().world_size
        except AssertionError:
            dcp_world_size = 1

        if dcp_world_size > 1:
            raise ValueError(
                "TRTLLM attention does not support Decode Context Parallel (DCP). "
                "Please use FlashInfer backend for DCP."
            )

        self.cache_dtype = self.cache_config.cache_dtype
        if self.cache_dtype.startswith("fp8"):
            self.kv_cache_dtype = TRTLLMAttentionBackend.get_fp8_dtype_for_trtllm(
                self.cache_dtype
            )
        else:
            assert self.kv_cache_spec.dtype == self.model_config.dtype
            self.kv_cache_dtype = self.kv_cache_spec.dtype

        # Use FP8 query if KV cache is FP8 and Q quantization is not disabled
        if (
            self.cache_dtype.startswith("fp8")
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        ):
            self.q_data_type = self.kv_cache_dtype
        else:
            self.q_data_type = self.model_config.dtype

        # Initialize reorder batch threshold for speculative decoding
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TRTLLMMetadata:
        """Build TRTLLM attention metadata from common attention metadata."""
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        max_seq_len = common_attn_metadata.max_seq_len
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        qo_indptr = common_attn_metadata.query_start_loc

        # Cascade attention is not supported by TRTLLM
        if common_prefix_len > 0:
            raise ValueError(
                "TRTLLM attention does not support cascade attention. "
                "Please use FlashInfer backend."
            )

        prefill_metadata = None
        decode_metadata = None

        # Build prefill metadata
        if num_prefills > 0:
            prefill_start = num_decodes
            qo_indptr_prefill = qo_indptr[prefill_start:] - qo_indptr[prefill_start]

            # Compute max_q_len for prefill
            qo_indptr_prefill_cpu = qo_indptr_prefill.cpu()
            query_lens_prefill = qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
            max_q_len_prefill = int(query_lens_prefill.max().item())

            prefill_metadata = TRTLLMPrefillMetadata(
                block_tables=block_table_tensor[prefill_start:],
                seq_lens=seq_lens[prefill_start:],
                cum_seq_lens_q=qo_indptr_prefill,
                cum_seq_lens_kv=qo_indptr_prefill,  # Same as Q for prefill
                max_q_len=max_q_len_prefill,
                max_seq_len=max_seq_len,
            )

        # Build decode metadata
        if num_decodes > 0:
            assert num_decode_tokens % num_decodes == 0, (
                "TRTLLM decode requires uniform query lengths per request."
            )
            decode_metadata = TRTLLMDecodeMetadata(
                block_tables=block_table_tensor[:num_decodes],
                seq_lens=seq_lens[:num_decodes],
                max_seq_len=max_seq_len,
            )

        return TRTLLMMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            q_data_type=self.q_data_type,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        # TRTLLM does not support cascade attention
        return False


class TRTLLMImpl(AttentionImpl):
    """TRTLLM attention implementation."""

    can_return_lse_for_decode: bool = False  # TRTLLM doesn't support LSE

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
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        if alibi_slopes is not None:
            raise ValueError("TRTLLM attention does not support ALiBi.")

        if sliding_window is not None:
            self.window_left = sliding_window - 1
        else:
            self.window_left = -1

        if logits_soft_cap is not None:
            raise ValueError("TRTLLM attention does not support logits soft cap.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "TRTLLM attention only supports decoder attention."
            )

        # Attention sinks
        self.sinks: torch.Tensor | None = None
        if sinks is not None:
            if sinks.shape[0] != num_heads:
                raise ValueError(
                    f"Sinks must have shape[0] == num_heads. "
                    f"Expected {num_heads}, got {sinks.shape[0]}."
                )
            self.sinks = sinks

        # Validate TRTLLM support
        if not can_use_trtllm_attention(num_heads, num_kv_heads):
            raise ValueError(
                f"TRTLLM attention not supported for num_heads={num_heads}, "
                f"num_kv_heads={num_kv_heads}"
            )

        vllm_config = get_current_vllm_config()
        self.supports_quant_query_input = (
            not vllm_config.attention_config.disable_flashinfer_q_quantization
        )

        # Scales for BMM operations (computed lazily)
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None
        self.o_sf_scale: float | None = None

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return self.kv_cache_dtype.startswith("fp8") and quant_key in (
            kFp8StaticTensorSym,
            kNvfp4Dynamic,
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # TRTLLM requires attention sinks to be float32
        if self.sinks is not None and self.sinks.dtype != torch.float32:
            self.sinks = self.sinks.to(torch.float32)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TRTLLMMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with TRTLLM attention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: KV cache tensor (HND layout required)
            attn_metadata: TRTLLM attention metadata

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run
            return output.fill_(0)

        assert attn_metadata.q_data_type == query.dtype, (
            f"Query dtype mismatch: expected {attn_metadata.q_data_type}, "
            f"got {query.dtype}"
        )

        # Compute BMM scales lazily
        if self.bmm1_scale is None:
            self.bmm1_scale = layer._q_scale_float * layer._k_scale_float * self.scale
        if self.bmm2_scale is None:
            self.bmm2_scale = layer._v_scale_float

        # Handle fused output quantization
        if output_scale is None:
            assert output_block_scale is None
        else:
            assert attn_metadata.q_data_type == FP8_DTYPE
            if output.dtype == FP8_DTYPE:
                assert output_block_scale is None
            elif output.dtype == FP4_DTYPE:
                assert output_block_scale is not None
            else:
                raise ValueError(f"Unsupported output dtype: {output.dtype}")

            if layer._o_scale_float is None:
                layer._o_scale_float = output_scale.cpu().item()
                if output.dtype == FP8_DTYPE:
                    self.bmm2_scale = self.bmm2_scale / layer._o_scale_float
                elif output.dtype == FP4_DTYPE:
                    self.o_sf_scale = layer._o_scale_float

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Write K/V to cache
        if self.kv_sharing_target_layer_name is None:
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

            if self.kv_cache_dtype.startswith("fp8"):
                torch_dtype = TRTLLMAttentionBackend.get_fp8_dtype_for_trtllm(
                    self.kv_cache_dtype
                )
                kv_cache = kv_cache.view(torch_dtype)

        # Slice to actual tokens
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        # Get permuted KV cache (HND layout)
        stride_order = TRTLLMAttentionBackend.get_kv_cache_stride_order()
        kv_cache_permute = kv_cache.permute(*stride_order)

        workspace_buffer = _get_trtllm_workspace_buffer()

        # Prefill path
        if num_prefill_tokens > 0:
            assert attn_metadata.prefill is not None
            prefill_query = query[num_decode_tokens:]
            prefill_query = prefill_query.contiguous().reshape(prefill_query.shape)

            block_tables_prefill = attn_metadata.prefill.block_tables
            seq_lens_prefill = attn_metadata.prefill.seq_lens

            assert get_kv_cache_layout() == "HND"
            assert is_strictly_contiguous(prefill_query)
            assert is_strictly_contiguous(kv_cache_permute)
            assert is_strictly_contiguous(workspace_buffer)
            assert is_strictly_contiguous(block_tables_prefill)
            assert is_strictly_contiguous(seq_lens_prefill)

            # Handle output quantization
            if output.dtype == FP4_DTYPE:
                assert self.o_sf_scale is not None
                out = FP4Tensor(
                    data=output[num_decode_tokens:],
                    scale=output_block_scale,
                    scale_start_index=num_decode_tokens,
                    original_shape=prefill_query.shape,
                )
            else:
                assert self.o_sf_scale is None
                out = output[num_decode_tokens:]

            # Handle BF16 Q with FP8 KV cache
            if (
                attn_metadata.q_data_type != FP8_DTYPE
                and self.kv_cache_dtype.startswith("fp8")
            ):
                mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
                    kv_cache_permute,
                    block_tables_prefill,
                    layer._k_scale,
                    layer._v_scale,
                    attn_metadata.q_data_type,
                )
            else:
                mock_kv_cache = kv_cache_permute
                mock_block_table = block_tables_prefill

            trtllm_batch_context_with_kv_cache(
                query=prefill_query,
                kv_cache=mock_kv_cache,
                workspace_buffer=workspace_buffer,
                block_tables=mock_block_table,
                seq_lens=seq_lens_prefill,
                max_q_len=attn_metadata.prefill.max_q_len,
                max_kv_len=attn_metadata.prefill.max_seq_len,
                bmm1_scale=self.bmm1_scale,
                bmm2_scale=self.bmm2_scale,
                batch_size=attn_metadata.num_prefills,
                cum_seq_lens_q=attn_metadata.prefill.cum_seq_lens_q,
                cum_seq_lens_kv=attn_metadata.prefill.cum_seq_lens_kv,
                window_left=self.window_left,
                sinks=self.sinks,
                o_sf_scale=self.o_sf_scale,
                out=out,
            )

        # Decode path
        if num_decode_tokens > 0:
            assert attn_metadata.decode is not None
            decode_query = query[:num_decode_tokens]
            decode_query = decode_query.contiguous().reshape(decode_query.shape)

            block_tables_decode = attn_metadata.decode.block_tables
            seq_lens_decode = attn_metadata.decode.seq_lens

            assert get_kv_cache_layout() == "HND"
            assert is_strictly_contiguous(decode_query)
            assert is_strictly_contiguous(kv_cache_permute)
            assert is_strictly_contiguous(workspace_buffer)
            assert is_strictly_contiguous(block_tables_decode)
            assert is_strictly_contiguous(seq_lens_decode)

            # Handle output quantization
            if output.dtype == FP4_DTYPE:
                assert self.o_sf_scale is not None
                out = FP4Tensor(
                    data=output[:num_decode_tokens],
                    scale=output_block_scale,
                    scale_start_index=0,
                    original_shape=decode_query.shape,
                )
            else:
                assert self.o_sf_scale is None
                out = output[:num_decode_tokens]

            # Compute q_len_per_req for speculative decoding
            if num_decode_tokens % attn_metadata.num_decodes != 0:
                q_len_per_req = 1
            else:
                q_len_per_req = num_decode_tokens // attn_metadata.num_decodes

            trtllm_batch_decode_with_kv_cache(
                query=decode_query,
                kv_cache=kv_cache_permute,
                workspace_buffer=workspace_buffer,
                block_tables=block_tables_decode,
                seq_lens=seq_lens_decode,
                max_seq_len=attn_metadata.decode.max_seq_len,
                bmm1_scale=self.bmm1_scale,
                bmm2_scale=self.bmm2_scale,
                window_left=self.window_left,
                sinks=self.sinks,
                o_sf_scale=self.o_sf_scale,
                out=out,
                q_len_per_req=q_len_per_req,
            )

        return output_padded
