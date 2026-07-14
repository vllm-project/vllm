# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
)

logger = init_logger(__name__)


class CPUAttentionBackend(AttentionBackend):
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512]

    @staticmethod
    def get_name() -> str:
        return "CPU_ATTN"

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """CPU attention supports decoder,
        encoder-only and encoder-decoder attention."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["CPUAttentionBackendImpl"]:
        return CPUAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["CPUAttentionMetadataBuilder"]:
        return CPUAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return num_blocks, num_kv_heads, block_size, 2 * head_size

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        return "HND"

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class CPUAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    causal: bool = True
    dynamic_causal: torch.Tensor | None = None

    # can be removed after deprecate sdpa
    use_sdpa_prefill: bool = False
    num_decode_tokens: int = 0
    sdpa_attn_masks: list[torch.Tensor | None] | None = None
    sdpa_start_loc: torch.Tensor | None = None

    encoder_cache: torch.Tensor | None = None


class CPUAttentionMetadataBuilder(AttentionMetadataBuilder[CPUAttentionMetadata]):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config

        parallel_config = vllm_config.parallel_config
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.num_heads = vllm_config.model_config.get_num_attention_heads(
            parallel_config
        )
        self.head_dim = kv_cache_spec.head_size
        self.dtype = vllm_config.model_config.dtype
        self.window_size = getattr(kv_cache_spec, "sliding_window", -1)
        if self.window_size is None:
            self.window_size = -1
        self.block_size = vllm_config.cache_config.block_size
        kv_cache_dtype_str = vllm_config.cache_config.cache_dtype
        self.isa = _get_attn_isa(
            self.dtype,
            self.block_size,
            self.head_dim,
            kv_cache_dtype_str,
        )
        self.is_cross_attention = isinstance(kv_cache_spec, CrossAttentionSpec)
        self.is_encoder_only_attention = isinstance(
            kv_cache_spec, EncoderOnlyAttentionSpec
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CPUAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        is_dynamic_casual = isinstance(common_attn_metadata.causal, torch.Tensor)
        dynamic_casual = None
        if is_dynamic_casual:
            dynamic_casual = common_attn_metadata.causal

        causal = (
            False
            if self.is_cross_attention or is_dynamic_casual
            else common_attn_metadata.causal
        )

        encoder_cache_tensor = None
        if self.is_encoder_only_attention:
            block_nums = (seq_lens + self.block_size - 1) // self.block_size
            start_block_ids = torch.zeros_like(seq_lens)
            torch.cumsum(block_nums[:-1], 0, out=start_block_ids[1:])
            total_block_num: int = block_nums.sum().item()
            max_block_num = block_nums.max().item()
            block_offsets = torch.arange(
                0, max_block_num, dtype=block_table_tensor.dtype
            )
            encoder_block_table = start_block_ids[:, None] + block_offsets[None, :]
            torch.ops._C.compute_slot_mapping_kernel_impl(
                query_start_loc,
                common_attn_metadata.positions,
                encoder_block_table,
                slot_mapping,
                self.block_size,
            )
            encoder_cache_tensor = torch.zeros(
                (
                    total_block_num,
                    self.num_kv_heads,
                    self.block_size,
                    2 * self.head_dim,
                ),
                dtype=self.dtype,
            )
            block_table_tensor = encoder_block_table

        scheduler_metadata = ops.cpu_attn_get_scheduler_metadata(
            num_reqs=num_reqs,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            seq_lens=seq_lens,
            dtype=self.dtype,
            query_start_loc=query_start_loc,
            causal=causal,
            sliding_window_size=self.window_size,
            isa=self.isa,
            enable_kv_split=envs.VLLM_CPU_ATTN_SPLIT_KV,
            dynamic_causal=dynamic_casual,
        )

        attn_metadata = CPUAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            scheduler_metadata=scheduler_metadata,
            causal=causal,
            encoder_cache=encoder_cache_tensor,
            dynamic_causal=dynamic_casual,
        )

        return attn_metadata


class CPUAttentionBackendImpl(AttentionImpl):
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
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        if logits_soft_cap is not None and attn_type in (
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        ):
            logger.warning_once(
                "CPU_ATTN does not support logits softcap for"
                " ENCODER and ENCODER_ONLY, outputs may be slightly off"
            )
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = -1
        else:
            self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.is_fp8_kv_cache = is_quantized_kv_cache(kv_cache_dtype)
        self.attn_type = attn_type

        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        vllm_config = get_current_vllm_config()
        self.isa = _get_attn_isa(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.block_size,
            self.head_size,
            self.kv_cache_dtype,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CPUAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for CPU attention backend.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, num_kv_heads, block_size, 2 * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for CPUAttentionBackendImpl"
            )

        # For warming-up
        if attn_metadata is None:
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens

        # For encoder attention
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            kv_cache = attn_metadata.encoder_cache

        # KV cache size are [num_blocks, num_kv_heads, block_size,
        # 2 * head_size]. Make a view [num_blocks, num_kv_heads,
        # block_size * 2, head_size]. Then slice KV at dim 2
        num_blocks, num_kv_heads, block_size, _ = kv_cache.size()
        kv_cache = kv_cache.view((num_blocks, num_kv_heads, block_size * 2, -1))
        key_cache, value_cache = kv_cache.chunk(2, dim=2)

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            ops.cpu_attn_reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.isa,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                kv_cache_dtype=self.kv_cache_dtype,
            )

        ops.cpu_attention_with_kv_cache(
            query=query[:num_actual_tokens],
            key_cache=key_cache,
            value_cache=value_cache,
            output=output[:num_actual_tokens],  # type: ignore
            query_start_loc=attn_metadata.query_start_loc,
            seq_lens=attn_metadata.seq_lens,
            scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,  # type: ignore
            sliding_window=self.sliding_window,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            s_aux=self.sinks,
            dynamic_causal=attn_metadata.dynamic_causal,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            kv_cache_dtype=self.kv_cache_dtype,
        )

        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        num_blocks, num_kv_heads, block_size, _ = kv_cache.size()
        kv_cache = kv_cache.view((num_blocks, num_kv_heads, block_size * 2, -1))
        key_cache, value_cache = kv_cache.chunk(2, dim=2)
        ops.cpu_attn_reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.isa,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            kv_cache_dtype=self.kv_cache_dtype,
        )


@functools.lru_cache(maxsize=1)
def _riscv_supports_rvv() -> bool:
    """Whether the C++ RVV attention path is usable.

    The kernel in csrc/cpu/cpu_attn_rvv.hpp uses VLEN-agnostic RVVI()
    macros and supports VLEN=128 and VLEN=256.  CMake auto-detects the
    largest zvl<N>b from /proc/cpuinfo and passes it via -mrvv-vector-bits.
    The RVV path is compiled whenever __riscv_v_min_vlen is defined, so
    we check that at least one supported zvl<N>b is advertised.
    """
    # The C++ compile-time check is the ground truth: it knows which
    # VLEN the binary was actually compiled for.  The cpuinfo check
    # below is only a fast-path shortcut.
    try:
        import torch

        if torch.ops._C.cpu_attn_has_isa("rvv"):
            return True
    except Exception:
        pass

    # Fallback: check /proc/cpuinfo for zvl128b/zvl256b.
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
    except OSError:
        return False
    return any(f"zvl{n}b" in cpuinfo for n in (128, 256))


def _get_attn_isa(
    dtype: torch.dtype,
    block_size: int,
    head_size: int | None = None,
    kv_cache_dtype: str | None = None,
) -> str:
    fp8_kv = is_quantized_kv_cache(kv_cache_dtype) if kv_cache_dtype else False
    if head_size is not None and head_size % 32 != 0 and head_size % 16 == 0:
        if fp8_kv:
            raise NotImplementedError(
                "FP8 KV cache requires head_size divisible by 32 on CPU."
            )
        return "vec16"
    supports_amx = torch.cpu._is_amx_tile_supported()
    arch = current_platform.get_cpu_architecture()
    supports_arm = arch == CpuArchEnum.ARM
    supports_vxe = arch == CpuArchEnum.S390X
    supports_riscv = arch == CpuArchEnum.RISCV
    supports_vsx = arch == CpuArchEnum.POWERPC
    supports_avx512 = torch.cpu._is_avx512_supported()
    if fp8_kv and not supports_amx and not supports_avx512:
        raise NotImplementedError(
            "FP8 KV cache on CPU requires x86 with AVX-512 or AMX."
        )
    if supports_amx and dtype in (torch.bfloat16,) and block_size % 32 == 0:
        return "amx"
    elif block_size % 32 == 0:
        if supports_arm:
            # support ARM NEON FMLA and BFMMLA (bf16) for block size 32
            return "neon"
        elif supports_riscv and _riscv_supports_rvv():
            return "rvv"
        elif supports_vxe:
            return "vxe"
        elif supports_vsx:
            return "vsx"
        else:
            return "vec"
    else:
        return "vec16"
