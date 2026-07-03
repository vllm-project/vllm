# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPC Attention Backend.

Pure attention (prefill + decode), without RoPE or RMSNorm.
Independent metadata / builder; KV cache layout is NHD:
(num_blocks, 2, block_size, num_kv_heads, head_size).
"""

import importlib.util
from dataclasses import dataclass
from typing import ClassVar

import torch
from typing_extensions import override

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
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    get_per_layer_parameters,
    infer_global_hyperparameters,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

FP8_DTYPE = current_platform.fp8_dtype()


def _get_fp8_dtype_for_kv_cache(kv_cache_dtype: str) -> torch.dtype:
    """Return the torch FP8 dtype for the given kv_cache_dtype string."""
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    elif kv_cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    else:
        raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


def _hpc_decode_use_splitk(
    splitk_seq_len_hint: int,
    num_decode_tokens: int,
    num_heads: int,
    num_kv_heads: int,
) -> bool:
    """Whether to enable split-K in the HPC decode kernel.

    TODO: replace this hand-tuned table with proper auto-tuning.
    """
    if num_decode_tokens < 8:
        return True
    if 8 <= num_decode_tokens < 12 and splitk_seq_len_hint <= 1024:
        return False
    if 12 <= num_decode_tokens < 16 and splitk_seq_len_hint <= 4096:
        return False
    if 16 <= num_decode_tokens < 24 and splitk_seq_len_hint <= 8192:
        return False
    if 24 <= num_decode_tokens and splitk_seq_len_hint <= 24576:
        return False
    return True


@dataclass
class HpcAttnMetadata(AttentionMetadata):
    """Metadata required by the HPC attention kernel."""

    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    max_query_len: int

    slot_mapping: torch.Tensor
    """Slot mapping for KV cache writes. shape = [num_actual_tokens]"""

    seq_lens: torch.Tensor
    """KV cache length per request. shape = [batch_size]"""

    block_table_tensor: torch.Tensor
    """Paged KV-cache block table.
    shape = [batch_size, max_num_blocks_per_seq]"""

    qo_indptr: torch.Tensor | None = None
    """Cumulative query lengths for prefill requests (GPU tensor).
    shape = [num_prefills + 1]. None when num_prefills == 0."""

    splitk_seq_len_hint: int = 0
    """Upper bound on decode KV length used by ``_hpc_decode_use_splitk``."""

    # --- HPC RopeNorm pass-through fields ---
    # Set by HpcRopeNorm._forward_impl(); consumed & reset by
    # HpcAttentionImpl.forward().  Defaults are safe for the standard
    # (non-RopeNorm) path and for profiling runs (attn_metadata=None).
    hpc_kv_written: bool = False
    """True when HpcRopeNorm already wrote KV cache."""
    hpc_prefill_q_scale: torch.Tensor | None = None
    """FP8 per-token-per-head Q scale for prefill (from RopeNorm)."""
    hpc_decode_q_scale: torch.Tensor | None = None
    """FP8 per-token-per-head Q scale for decode (from RopeNorm)."""
    hpc_split_k_flag: torch.Tensor | None = None
    """Split-K flag tensor for FP8 decode (from RopeNorm)."""


class HpcAttnMetadataBuilder(AttentionMetadataBuilder[HpcAttnMetadata]):
    """Build HpcAttnMetadata from CommonAttentionMetadata."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        self.num_qo_heads = self.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.head_dim = kv_cache_spec.head_size
        self.page_size = kv_cache_spec.block_size

        self.cache_dtype = self.cache_config.cache_dtype

        self.global_hyperparameters = infer_global_hyperparameters(
            get_per_layer_parameters(vllm_config, layer_names, HpcAttentionImpl)
        )

    @override  # type: ignore[misc]
    @classmethod
    def get_cudagraph_support(
        cls: type["HpcAttnMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> HpcAttnMetadata:
        attn_metadata = self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        attn_metadata.splitk_seq_len_hint = self.model_config.max_model_len
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> HpcAttnMetadata:
        """Build HpcAttnMetadata from CommonAttentionMetadata."""
        num_actual_tokens = common_attn_metadata.num_actual_tokens

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=False,
            )
        )

        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        max_query_len = common_attn_metadata.max_query_len

        splitk_seq_len_hint = (
            common_attn_metadata.max_seq_len if num_decodes > 0 else 0
        )

        qo_indptr = None
        if num_prefills > 0:
            qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu
            prefill_start = num_decodes
            qo_indptr_prefill_cpu = (
                qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[prefill_start]
            )
            qo_indptr = qo_indptr_prefill_cpu.to(self.device, non_blocking=True)

        return HpcAttnMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            max_query_len=max_query_len,
            splitk_seq_len_hint=splitk_seq_len_hint,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            block_table_tensor=block_table_tensor,
            qo_indptr=qo_indptr,
            hpc_kv_written=True,
            hpc_prefill_q_scale=None,
            hpc_decode_q_scale=None,
            hpc_split_k_flag=None,
        )


class HpcAttentionBackend(AttentionBackend):
    """HPC attention backend (pure attention, no RoPE/Norm).

    KV cache layout: NHD (num_blocks, 2, block_size, num_kv_heads, head_size).
    """

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8_e4m3",
    ]

    # Avoid attention abstracted method call cache insert
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "HPC_ATTN"

    @staticmethod
    def get_impl_cls() -> type["HpcAttentionImpl"]:
        return HpcAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["HpcAttnMetadataBuilder"]:
        return HpcAttnMetadataBuilder

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
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(9, 0)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: "CacheDType | None") -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        return "NHD"


class HpcAttentionImpl(AttentionImpl[HpcAttnMetadata]):
    """HPC pure attention implementation (no RoPE/Norm).

    Constraints:
    - head_dim == 128
    - num_heads // num_kv_heads in {4, 8}
    - kv_cache_dtype in {"auto", "fp8_e4m3"}
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        if importlib.util.find_spec("hpc") is None:
            raise ImportError(
                "HPC attention requires the hpc module to be installed. "
                "Please install it from https://github.com/Tencent/hpc-ops"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("HPC attention only supports decoder attention")
        if alibi_slopes is not None:
            raise NotImplementedError("HPC attention does not support ALiBi")
        if logits_soft_cap is not None:
            raise NotImplementedError("HPC attention does not support logits_soft_cap")

        if head_size != 128:
            raise ValueError(
                f"HPC attention only supports head_dim=128, got {head_size}"
            )

        num_queries_per_kv = num_heads // num_kv_heads
        if num_queries_per_kv not in (4, 8):
            raise ValueError(
                f"HPC attention only supports head_per_group in {{4, 8}}, "
                f"got {num_queries_per_kv} "
                f"(num_heads={num_heads}, num_kv_heads={num_kv_heads})"
            )

        if kv_cache_dtype not in ("auto", "fp8_e4m3"):
            raise ValueError(
                f"HPC attention only supports kv_cache_dtype 'auto' or "
                f"'fp8_e4m3', got '{kv_cache_dtype}'"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = num_queries_per_kv

        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        self.use_fp8 = kv_cache_dtype == "fp8_e4m3"

        self.supports_quant_query_input = False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HpcAttnMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """HPC attention forward (standard vLLM backend interface).

        Two modes:
        1. Standard: upstream handles RoPE/Norm; this backend writes KV + attn.
        2. HpcRopeNorm: fused op already did RoPE/Norm/KV-Write/Q-Quant;
           extra params passed via attn_metadata.hpc_* fields.
        """
        import hpc

        assert output is not None, "Output tensor must be provided."
        assert output_scale is None, "HPC attention does not support fused output quant"
        assert output_block_scale is None

        if attn_metadata is None:
            return output.fill_(0)

        hpc_kv_written = attn_metadata.hpc_kv_written
        hpc_prefill_q_scale = attn_metadata.hpc_prefill_q_scale
        hpc_decode_q_scale = attn_metadata.hpc_decode_q_scale
        hpc_split_k_flag = attn_metadata.hpc_split_k_flag

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_prefill_reqs = attn_metadata.num_prefills
        num_decode_reqs = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        # Write KV cache if not already done by HpcRopeNorm.
        if self.kv_sharing_target_layer_name is None and not hpc_kv_written:
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

        if self.use_fp8:
            torch_dtype = _get_fp8_dtype_for_kv_cache(self.kv_cache_dtype)
            kv_cache = kv_cache.view(torch_dtype)

        if self.use_fp8:
            if not hpc_kv_written:
                raise RuntimeError(
                    "HpcAttentionImpl: FP8 mode requires HpcRopeNorm. "
                    "Ensure hpc_rope_norm is enabled or set "
                    "kv_cache_dtype='auto' for bf16 mode."
                    f" (layer={getattr(layer, 'layer_name', '?')})"
                )
            k_scale = layer._k_scale.reshape(1)
            v_scale = layer._v_scale.reshape(1)

        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        # --- Prefill ---
        if num_prefill_reqs > 0:
            seq_lens_prefill = attn_metadata.seq_lens[num_decode_reqs:]
            cu_seqlens_prefill = attn_metadata.qo_indptr
            max_seqlens = attn_metadata.max_query_len
            block_table_prefill = attn_metadata.block_table_tensor[num_decode_reqs:]

            q_prefill = query[num_decode_tokens:]
            output_prefill = output[num_decode_tokens:]

            if self.use_fp8:
                hpc.attention_with_kvcache_prefill_fp8(
                    q_prefill,
                    kv_cache[:, 0],
                    kv_cache[:, 1],
                    hpc_prefill_q_scale,
                    k_scale,
                    v_scale,
                    cu_seqlens_prefill,
                    block_table_prefill,
                    seq_lens_prefill,
                    max_seqlens,
                    output=output_prefill,
                )
            else:
                hpc.attention_with_kvcache_prefill_bf16(
                    q_prefill,
                    kv_cache[:, 0],
                    kv_cache[:, 1],
                    cu_seqlens_prefill,
                    block_table_prefill,
                    seq_lens_prefill,
                    max_seqlens,
                    output=output_prefill,
                )

        # --- Decode ---
        if num_decode_reqs > 0:
            num_seq_kvcache = attn_metadata.seq_lens[:num_decode_reqs]
            block_table_decode = attn_metadata.block_table_tensor[:num_decode_reqs]

            q_decode = query[:num_decode_tokens]
            output_decode = output[:num_decode_tokens]

            splitk = _hpc_decode_use_splitk(
                attn_metadata.splitk_seq_len_hint,
                num_decode_tokens,
                self.num_heads,
                self.num_kv_heads,
            )

            if self.use_fp8:
                hpc.attention_decode_fp8(
                    q_decode,
                    kv_cache[:, 0],
                    kv_cache[:, 1],
                    block_table_decode,
                    num_seq_kvcache,
                    hpc_decode_q_scale,
                    k_scale,
                    v_scale,
                    new_kv_included=True,
                    splitk=splitk,
                    split_flag=hpc_split_k_flag,
                    output=output_decode,
                )
            else:
                hpc.attention_decode_bf16(
                    q_decode,
                    kv_cache[:, 0],
                    kv_cache[:, 1],
                    block_table_decode,
                    num_seq_kvcache,
                    output=output_decode,
                    new_kv_included=True,
                    splitk=splitk,
                )

        return output_padded
