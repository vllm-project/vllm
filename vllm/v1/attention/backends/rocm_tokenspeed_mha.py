# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TokenSpeed-kernel AMD MHA backend for ROCm gfx950."""

import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_prefills_and_extends,
)
from vllm.v1.attention.ops.rocm_tokenspeed_mha import (
    rocm_tokenspeed_mha_decode,
    rocm_tokenspeed_mha_extend,
    rocm_tokenspeed_mha_prefill,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

_MAX_PREFILL_TOKENS_PER_CALL = 64 * 1024
_MAX_FULL_DECODE_TOKENS_PER_CALL = 2048
_MAX_SLIDING_DECODE_TOKENS_PER_CALL = 8192


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}.")
    return parsed


def _env_extend_kernel_mode() -> str:
    value = os.getenv("VLLM_ROCM_TOKENSPEED_MHA_USE_EXTEND_KERNEL")
    if value is None:
        return "off"
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on", "all"):
        return "all"
    if value in ("full", "full_attention"):
        return "full"
    if value in ("sliding", "sliding_window", "swa"):
        return "sliding"
    if value in ("0", "false", "no", "off"):
        return "off"
    raise ValueError(
        "VLLM_ROCM_TOKENSPEED_MHA_USE_EXTEND_KERNEL must be one of "
        "'off', 'full', 'sliding', or 'all'."
    )


def _env_block_table_expansion_mode() -> str:
    value = os.getenv("VLLM_ROCM_TOKENSPEED_MHA_BLOCK_TABLE_EXPANSION")
    if value is None:
        return "index_select"
    value = value.strip().lower()
    if value in ("index_select", "default"):
        return "index_select"
    if value in ("advanced_index", "advanced", "getitem"):
        return "advanced_index"
    raise ValueError(
        "VLLM_ROCM_TOKENSPEED_MHA_BLOCK_TABLE_EXPANSION must be one of "
        "'index_select' or 'advanced_index'."
    )


@dataclass
class RocmTokenspeedMHAMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_decodes: int
    num_decode_tokens: int
    num_extends: int
    num_extend_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    prefill_query_start_loc: torch.Tensor | None
    prefill_query_start_loc_cpu: torch.Tensor | None
    prefill_max_query_len: int
    prefill_token_to_req: torch.Tensor | None
    prefill_decode_seq_lens: torch.Tensor | None
    prefill_decode_max_seq_len: int
    extend_token_to_req: torch.Tensor | None
    extend_decode_seq_lens: torch.Tensor | None
    extend_query_start_loc: torch.Tensor | None
    extend_max_query_len: int
    extend_max_seq_len: int
    causal: bool | torch.Tensor = True


class RocmTokenspeedMHAMetadataBuilder(
    AttentionMetadataBuilder[RocmTokenspeedMHAMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> RocmTokenspeedMHAMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        common_attn_metadata.query_start_loc.zero_()
        common_attn_metadata.query_start_loc_cpu.zero_()
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmTokenspeedMHAMetadata:
        if common_prefix_len > 0:
            raise NotImplementedError("ROCM_TOKENSPEED_MHA does not support cascade.")

        (
            num_decodes,
            num_extends,
            num_prefills,
            num_decode_tokens,
            num_extend_tokens,
            num_prefill_tokens,
        ) = split_decodes_prefills_and_extends(common_attn_metadata)

        first_prefill = num_decodes + num_extends
        prefill_query_start_loc = None
        prefill_query_start_loc_cpu = None
        prefill_max_query_len = 0
        prefill_token_to_req = None
        prefill_decode_seq_lens = None
        prefill_decode_max_seq_len = 0
        if num_prefills > 0:
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            prefill_start_token = query_start_loc_cpu[first_prefill].item()
            prefill_query_start_loc = (
                common_attn_metadata.query_start_loc[first_prefill:]
                - prefill_start_token
            )
            prefill_query_start_loc_cpu = (
                query_start_loc_cpu[first_prefill:] - prefill_start_token
            )
            prefill_query_lens_cpu = (
                prefill_query_start_loc_cpu[1:] - prefill_query_start_loc_cpu[:-1]
            )
            prefill_max_query_len = prefill_query_lens_cpu.max().item()
            prefill_seq_lens = common_attn_metadata.seq_lens_cpu[first_prefill:]
            token_to_req_cpu = torch.repeat_interleave(
                torch.arange(num_prefills, dtype=torch.int32),
                prefill_query_lens_cpu,
            )
            decode_seq_lens_cpu = torch.empty(num_prefill_tokens, dtype=torch.int32)
            offset = 0
            for req_idx in range(num_prefills):
                query_len = prefill_query_lens_cpu[req_idx].item()
                seq_len = prefill_seq_lens[req_idx].item()
                start_seq_len = seq_len - query_len + 1
                decode_seq_lens_cpu[offset : offset + query_len] = torch.arange(
                    start_seq_len,
                    seq_len + 1,
                    dtype=torch.int32,
                )
                offset += query_len
            prefill_token_to_req = token_to_req_cpu.to(self.device, non_blocking=True)
            prefill_decode_seq_lens = decode_seq_lens_cpu.to(
                self.device, non_blocking=True
            )
            prefill_decode_max_seq_len = prefill_seq_lens.max().item()

        extend_token_to_req = None
        extend_decode_seq_lens = None
        extend_query_start_loc = None
        extend_max_query_len = 0
        extend_max_seq_len = 0
        if num_extends > 0:
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            extend_slice = slice(num_decodes, num_decodes + num_extends)
            extend_query_lens = query_lens_cpu[extend_slice]
            extend_seq_lens = common_attn_metadata.seq_lens_cpu[extend_slice]
            token_to_req_cpu = torch.repeat_interleave(
                torch.arange(num_extends, dtype=torch.int32),
                extend_query_lens,
            )
            decode_seq_lens_cpu = torch.empty(num_extend_tokens, dtype=torch.int32)
            offset = 0
            for req_idx in range(num_extends):
                query_len = extend_query_lens[req_idx].item()
                seq_len = extend_seq_lens[req_idx].item()
                start_seq_len = seq_len - query_len + 1
                decode_seq_lens_cpu[offset : offset + query_len] = torch.arange(
                    start_seq_len,
                    seq_len + 1,
                    dtype=torch.int32,
                )
                offset += query_len
            extend_token_to_req = token_to_req_cpu.to(self.device, non_blocking=True)
            extend_decode_seq_lens = decode_seq_lens_cpu.to(
                self.device, non_blocking=True
            )
            extend_start_token = query_start_loc_cpu[num_decodes].item()
            extend_query_start_loc = (
                common_attn_metadata.query_start_loc[
                    num_decodes : num_decodes + num_extends + 1
                ]
                - extend_start_token
            )
            extend_max_query_len = extend_query_lens.max().item()
            extend_max_seq_len = extend_seq_lens.max().item()

        return RocmTokenspeedMHAMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_extends=num_extends,
            num_extend_tokens=num_extend_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_query_start_loc_cpu=prefill_query_start_loc_cpu,
            prefill_max_query_len=prefill_max_query_len,
            prefill_token_to_req=prefill_token_to_req,
            prefill_decode_seq_lens=prefill_decode_seq_lens,
            prefill_decode_max_seq_len=prefill_decode_max_seq_len,
            extend_token_to_req=extend_token_to_req,
            extend_decode_seq_lens=extend_decode_seq_lens,
            extend_query_start_loc=extend_query_start_loc,
            extend_max_query_len=extend_max_query_len,
            extend_max_seq_len=extend_max_seq_len,
            causal=common_attn_metadata.causal,
        )


class RocmTokenspeedMHABackend(AttentionBackend):
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "ROCM_TOKENSPEED_MHA"

    @staticmethod
    def get_impl_cls() -> type["RocmTokenspeedMHAImpl"]:
        return RocmTokenspeedMHAImpl

    @staticmethod
    def get_builder_cls() -> type["RocmTokenspeedMHAMetadataBuilder"]:
        return RocmTokenspeedMHAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size != 64:
            raise ValueError("ROCM_TOKENSPEED_MHA requires block_size=64.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 64

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        from vllm.platforms import current_platform

        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx950

            return on_gfx950()
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        # ROCM_TOKENSPEED_MHA uses (2, num_blocks, ...) KV cache layout like
        # ROCM_ATTN, which is incompatible with KV connectors that require
        # blocks-first layout.
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

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
        if block_size is not None and block_size != 64:
            return "ROCm TokenSpeed MHA gfx950 kernels require block_size 64"
        if use_mm_prefix:
            return "ROCm TokenSpeed MHA does not support multimodal prefix attention"
        return None


class RocmTokenspeedMHAImpl(AttentionImpl[RocmTokenspeedMHAMetadata]):
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
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.prefill_sliding_window = (
            -1 if sliding_window is None else sliding_window - 1
        )
        self.decode_sliding_window = -1 if sliding_window is None else sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = 0 if logits_soft_cap is None else logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.sinks = sinks
        self.direct_output_prefill = _env_flag_enabled(
            "VLLM_ROCM_TOKENSPEED_MHA_DIRECT_OUTPUT_PREFILL"
        )
        self.direct_output_decode = _env_flag_enabled(
            "VLLM_ROCM_TOKENSPEED_MHA_DIRECT_OUTPUT_DECODE"
        )
        self.direct_output_extend = _env_flag_enabled(
            "VLLM_ROCM_TOKENSPEED_MHA_DIRECT_OUTPUT_EXTEND"
        )
        self.extend_kernel_mode = _env_extend_kernel_mode()
        self.block_table_expansion_mode = _env_block_table_expansion_mode()
        self.combine_mixed_decode = _env_flag_enabled(
            "VLLM_ROCM_TOKENSPEED_MHA_COMBINE_MIXED_DECODE",
            default=False,
        )
        self.full_decode_tokens_per_call = _env_int(
            "VLLM_ROCM_TOKENSPEED_MHA_FULL_DECODE_TOKENS_PER_CALL",
            _MAX_FULL_DECODE_TOKENS_PER_CALL,
        )
        self.sliding_decode_tokens_per_call = _env_int(
            "VLLM_ROCM_TOKENSPEED_MHA_SLIDING_DECODE_TOKENS_PER_CALL",
            _MAX_SLIDING_DECODE_TOKENS_PER_CALL,
        )
        self.extend_min_query_len = _env_int(
            "VLLM_ROCM_TOKENSPEED_MHA_EXTEND_MIN_QUERY_LEN",
            1,
        )
        if alibi_slopes is not None:
            raise NotImplementedError("ROCM_TOKENSPEED_MHA does not support ALiBi.")
        if self.logits_soft_cap != 0:
            raise NotImplementedError(
                "ROCM_TOKENSPEED_MHA does not support logits soft cap."
            )
        if head_size != 64:
            raise ValueError("ROCM_TOKENSPEED_MHA only supports head_size=64.")
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )
        logger.info_once("Using TokenSpeed-kernel MHA attention on ROCm gfx950")

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return kv_cache.unbind(0)

    def _prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        max_query_len: int,
        sliding_window: int,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        call_output = output if self.direct_output_prefill else None
        total_tokens = query.shape[0]
        if total_tokens <= _MAX_PREFILL_TOKENS_PER_CALL:
            result = rocm_tokenspeed_mha_prefill(
                query=query,
                key=key,
                value=value,
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                max_query_len=max_query_len,
                sliding_window=sliding_window,
                sinks=self.sinks,
                output=call_output,
            )
            if call_output is None and output is not None:
                output.copy_(result)
                return output
            return result

        if output is None:
            output = torch.empty_like(query)
        num_seqs = query_start_loc_cpu.numel() - 1
        seq_start = 0
        while seq_start < num_seqs:
            seq_end = seq_start + 1
            token_start = query_start_loc_cpu[seq_start].item()
            token_end = query_start_loc_cpu[seq_end].item()
            while seq_end < num_seqs:
                next_token_end = query_start_loc_cpu[seq_end + 1].item()
                if next_token_end - token_start > _MAX_PREFILL_TOKENS_PER_CALL:
                    break
                seq_end += 1
                token_end = next_token_end

            cu_cpu = query_start_loc_cpu[seq_start : seq_end + 1] - token_start
            cu = query_start_loc[seq_start : seq_end + 1] - token_start
            query_lens = cu_cpu[1:] - cu_cpu[:-1]
            result = rocm_tokenspeed_mha_prefill(
                query=query[token_start:token_end],
                key=key[token_start:token_end],
                value=value[token_start:token_end],
                query_start_loc=cu,
                query_start_loc_cpu=cu_cpu,
                max_query_len=query_lens.max().item(),
                sliding_window=sliding_window,
                sinks=self.sinks,
                output=output[token_start:token_end]
                if self.direct_output_prefill
                else None,
            )
            if not self.direct_output_prefill:
                output[token_start:token_end].copy_(result)
            seq_start = seq_end
        return output

    def _decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        max_query_len: int,
        sliding_window: int,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        call_output = output if self.direct_output_decode else None
        total_tokens = query.shape[0]
        max_tokens_per_call = (
            self.sliding_decode_tokens_per_call
            if sliding_window >= 0
            else self.full_decode_tokens_per_call
        )
        if total_tokens <= max_tokens_per_call:
            result = rocm_tokenspeed_mha_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                max_query_len=max_query_len,
                sliding_window=sliding_window,
                sinks=self.sinks,
                output=call_output,
            )
            if call_output is None and output is not None:
                output.copy_(result)
                return output
            return result

        if output is None:
            output = torch.empty_like(query)
        for start in range(0, total_tokens, max_tokens_per_call):
            end = min(start + max_tokens_per_call, total_tokens)
            result = rocm_tokenspeed_mha_decode(
                query=query[start:end],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table[start:end],
                seq_lens=seq_lens[start:end],
                max_seq_len=max_seq_len,
                max_query_len=max_query_len,
                sliding_window=sliding_window,
                sinks=self.sinks,
                output=output[start:end] if self.direct_output_decode else None,
            )
            if not self.direct_output_decode:
                output[start:end].copy_(result)
        return output

    def _extend(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        query_start_loc: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        max_query_len: int,
        sliding_window: int,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        call_output = output if self.direct_output_extend else None
        result = rocm_tokenspeed_mha_extend(
            query=query,
            query_start_loc=query_start_loc,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            max_query_len=max_query_len,
            sliding_window=sliding_window,
            sinks=self.sinks,
            output=call_output,
        )
        if call_output is None and output is not None:
            output.copy_(result)
            return output
        return result

    def _expand_block_table(
        self, req_block_table: torch.Tensor, token_to_req: torch.Tensor
    ) -> torch.Tensor:
        token_to_req_i64 = token_to_req.long()
        if self.block_table_expansion_mode == "advanced_index":
            return req_block_table[token_to_req_i64]
        return req_block_table.index_select(0, token_to_req_i64)

    @staticmethod
    def _is_pure_prefill(attn_metadata: RocmTokenspeedMHAMetadata) -> bool:
        return attn_metadata.num_prefills > 0 and (
            attn_metadata.num_decodes == 0 and attn_metadata.num_extends == 0
        )

    @staticmethod
    def _is_pure_decode(attn_metadata: RocmTokenspeedMHAMetadata) -> bool:
        return attn_metadata.num_decodes > 0 and (
            attn_metadata.num_prefills == 0 and attn_metadata.num_extends == 0
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmTokenspeedMHAMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "ROCM_TOKENSPEED_MHA does not support fused output quantization."  # noqa: E501
            )
        if attn_metadata is None:
            return output.fill_(0)
        if not attn_metadata.causal:
            raise NotImplementedError(
                "ROCM_TOKENSPEED_MHA only supports causal attention."  # noqa: E501
            )
        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "ROCM_TOKENSPEED_MHA only supports decoder attention."  # noqa: E501
            )
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "ROCM_TOKENSPEED_MHA does not support FP8 KV cache."  # noqa: E501
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        output = output[:num_actual_tokens]
        prefill_sliding_window = self.prefill_sliding_window
        decode_sliding_window = self.decode_sliding_window

        if self._is_pure_prefill(attn_metadata):
            self._prefill(
                query=query,
                key=key[:num_actual_tokens],
                value=value[:num_actual_tokens],
                query_start_loc=attn_metadata.query_start_loc,
                query_start_loc_cpu=attn_metadata.query_start_loc_cpu,
                max_query_len=attn_metadata.max_query_len,
                sliding_window=prefill_sliding_window,
                output=output,
            )
            return output

        if self._is_pure_decode(attn_metadata):
            key_cache, value_cache = self._split_kv_cache(kv_cache)
            self._decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                max_seq_len=attn_metadata.max_seq_len,
                max_query_len=attn_metadata.max_query_len,
                sliding_window=decode_sliding_window,
                output=output,
            )
            return output

        if attn_metadata.num_extends > 0 or attn_metadata.num_prefills > 0:
            key_cache, value_cache = self._split_kv_cache(kv_cache)
            if self.combine_mixed_decode and self.extend_kernel_mode == "off":
                block_table_parts: list[torch.Tensor] = []
                seq_lens_parts: list[torch.Tensor] = []
                if attn_metadata.num_decodes > 0:
                    block_table_parts.append(
                        attn_metadata.block_table[: attn_metadata.num_decodes]
                    )
                    seq_lens_parts.append(
                        attn_metadata.seq_lens[: attn_metadata.num_decodes]
                    )
                if attn_metadata.num_extends > 0:
                    assert attn_metadata.extend_token_to_req is not None
                    assert attn_metadata.extend_decode_seq_lens is not None
                    extend_req_block_table = attn_metadata.block_table[
                        attn_metadata.num_decodes : attn_metadata.num_decodes
                        + attn_metadata.num_extends
                    ]
                    extend_block_table = self._expand_block_table(
                        extend_req_block_table,
                        attn_metadata.extend_token_to_req,
                    )
                    block_table_parts.append(extend_block_table)
                    seq_lens_parts.append(attn_metadata.extend_decode_seq_lens)
                if attn_metadata.num_prefills > 0:
                    assert attn_metadata.prefill_token_to_req is not None
                    assert attn_metadata.prefill_decode_seq_lens is not None
                    prefill_req_block_table = attn_metadata.block_table[
                        attn_metadata.num_decodes + attn_metadata.num_extends :
                    ]
                    prefill_block_table = self._expand_block_table(
                        prefill_req_block_table,
                        attn_metadata.prefill_token_to_req,
                    )
                    block_table_parts.append(prefill_block_table)
                    seq_lens_parts.append(attn_metadata.prefill_decode_seq_lens)

                mixed_block_table = (
                    block_table_parts[0]
                    if len(block_table_parts) == 1
                    else torch.cat(block_table_parts, dim=0)
                )
                mixed_seq_lens = (
                    seq_lens_parts[0]
                    if len(seq_lens_parts) == 1
                    else torch.cat(seq_lens_parts, dim=0)
                )
                self._decode(
                    query=query,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=mixed_block_table,
                    seq_lens=mixed_seq_lens,
                    max_seq_len=attn_metadata.max_seq_len,
                    max_query_len=1,
                    sliding_window=decode_sliding_window,
                    output=output,
                )
                return output

            if attn_metadata.num_decodes > 0:
                decode_tokens = attn_metadata.num_decode_tokens
                self._decode(
                    query=query[:decode_tokens],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=attn_metadata.block_table[: attn_metadata.num_decodes],
                    seq_lens=attn_metadata.seq_lens[: attn_metadata.num_decodes],
                    max_seq_len=attn_metadata.max_seq_len,
                    max_query_len=1,
                    sliding_window=decode_sliding_window,
                    output=output[:decode_tokens],
                )

            if attn_metadata.num_extends > 0:
                extend_start = attn_metadata.num_decode_tokens
                extend_end = extend_start + attn_metadata.num_extend_tokens
                extend_req_block_table = attn_metadata.block_table[
                    attn_metadata.num_decodes : attn_metadata.num_decodes
                    + attn_metadata.num_extends
                ]
                use_extend_kernel = (
                    self.extend_kernel_mode == "all"
                    or (
                        self.extend_kernel_mode == "sliding"
                        and decode_sliding_window >= 0
                    )
                    or (self.extend_kernel_mode == "full" and decode_sliding_window < 0)
                )
                use_extend_kernel = (
                    use_extend_kernel
                    and attn_metadata.extend_max_query_len >= self.extend_min_query_len
                )
                if use_extend_kernel:
                    assert attn_metadata.extend_query_start_loc is not None
                    self._extend(
                        query=query[extend_start:extend_end],
                        key_cache=key_cache,
                        value_cache=value_cache,
                        query_start_loc=attn_metadata.extend_query_start_loc,
                        block_table=extend_req_block_table,
                        seq_lens=attn_metadata.seq_lens[
                            attn_metadata.num_decodes : attn_metadata.num_decodes
                            + attn_metadata.num_extends
                        ],
                        max_seq_len=attn_metadata.extend_max_seq_len,
                        max_query_len=attn_metadata.extend_max_query_len,
                        sliding_window=decode_sliding_window,
                        output=output[extend_start:extend_end],
                    )
                else:
                    assert attn_metadata.extend_token_to_req is not None
                    assert attn_metadata.extend_decode_seq_lens is not None
                    extend_block_table = self._expand_block_table(
                        extend_req_block_table,
                        attn_metadata.extend_token_to_req,
                    )
                    self._decode(
                        query=query[extend_start:extend_end],
                        key_cache=key_cache,
                        value_cache=value_cache,
                        block_table=extend_block_table,
                        seq_lens=attn_metadata.extend_decode_seq_lens,
                        max_seq_len=attn_metadata.extend_max_seq_len,
                        max_query_len=1,
                        sliding_window=decode_sliding_window,
                        output=output[extend_start:extend_end],
                    )

            if attn_metadata.num_prefills == 0:
                return output

            prefill_start = (
                attn_metadata.num_decode_tokens + attn_metadata.num_extend_tokens
            )
            assert attn_metadata.prefill_query_start_loc is not None
            assert attn_metadata.prefill_query_start_loc_cpu is not None
            prefill_req_block_table = attn_metadata.block_table[
                attn_metadata.num_decodes + attn_metadata.num_extends :
            ]
            assert attn_metadata.prefill_token_to_req is not None
            assert attn_metadata.prefill_decode_seq_lens is not None
            prefill_block_table = self._expand_block_table(
                prefill_req_block_table,
                attn_metadata.prefill_token_to_req,
            )
            self._decode(
                query=query[prefill_start:],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=prefill_block_table,
                seq_lens=attn_metadata.prefill_decode_seq_lens,
                max_seq_len=attn_metadata.prefill_decode_max_seq_len,
                max_query_len=1,
                sliding_window=decode_sliding_window,
                output=output[prefill_start:],
            )
            return output

        raise NotImplementedError(
            "Unhandled ROCM_TOKENSPEED_MHA attention batch shape."
        )

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
