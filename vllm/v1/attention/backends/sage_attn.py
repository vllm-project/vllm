# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SageAttention V1 attention backend (exact/paged prefill + FlashAttn decode).

KV cache layout is ``(2, num_blocks, block_size, H, D)`` so ``kv_cache[0/1]``
are contiguous ``[num_blocks, block_size, H, D]`` slabs (FA paged NHD +
Sage paged kernels).

Hybrid dispatch
---------------
1. Prefill / mixed: paged quant + ``seqlens_paged`` (no gather / permute);
   chunked-causal falls back to exact ``sageattn()``.
2. Pure decode (default): FlashAttention paged varlen — same CG contract as
   FA2 (``seq_lens`` / ``block_table`` updated in-place on replay).
3. Opt-in Sage seqlens decode: ``VLLM_SAGE_USE_SEQLENS=1``.

Builder reports ``UNIFORM_SINGLE_TOKEN_DECODE`` so resolve can use
``FULL_AND_PIECEWISE`` (FULL for pure decode, PIECEWISE for mixed).

Requires the ``sageattention`` package for prefill.
"""

from __future__ import annotations

import copy
import os
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar, Iterator

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
else:  # pragma: no cover
    flash_attn_varlen_func = None  # type: ignore[assignment]

try:
    from sageattention import sageattn

    _SAGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    sageattn = None  # type: ignore[assignment]
    _SAGE_AVAILABLE = False

try:
    from sageattention.core import sageattn_qk_int8_pv_fp8_cuda_seqlens

    _SAGE_SEQLENS_AVAILABLE = True
except ImportError:  # pragma: no cover
    sageattn_qk_int8_pv_fp8_cuda_seqlens = None  # type: ignore[assignment]
    _SAGE_SEQLENS_AVAILABLE = False

try:
    from sageattention.core import sageattn_qk_int8_pv_fp8_cuda_seqlens_paged

    _SAGE_SEQLENS_PAGED_AVAILABLE = True
except ImportError:  # pragma: no cover
    sageattn_qk_int8_pv_fp8_cuda_seqlens_paged = None  # type: ignore[assignment]
    _SAGE_SEQLENS_PAGED_AVAILABLE = False

# Backward-compat: older SageAttention seqlens_paged API has no `kv_cap`.
_SAGE_SEQLENS_PAGED_HAS_KV_CAP = False
if _SAGE_SEQLENS_PAGED_AVAILABLE and sageattn_qk_int8_pv_fp8_cuda_seqlens_paged is not None:
    try:
        _SAGE_SEQLENS_PAGED_HAS_KV_CAP = (
            "kv_cap"
            in inspect.signature(
                sageattn_qk_int8_pv_fp8_cuda_seqlens_paged
            ).parameters
        )
    except Exception:
        _SAGE_SEQLENS_PAGED_HAS_KV_CAP = False


class _SageSharedWorkspace:
    """Process-wide persistent gather / GQA / padded-decode buffers."""

    _gather: dict[tuple, tuple[torch.Tensor, torch.Tensor, int]] = {}
    _gqa: dict[tuple, tuple[torch.Tensor, torch.Tensor, int]] = {}
    # key -> (k_hnd, v_hnd, k_flat, v_flat, token_cap)
    # Padded decode uses a *micro-batch* workspace (default 1) so memory is
    # O(S_cap) not O(B * S_cap) — full B×max_model_len OOMs during FULL CG
    # capture when max_cudagraph_capture_size is large.
    _pad: dict[
        tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
    ] = {}
    _cu_pad: dict[tuple, torch.Tensor] = {}
    _arange: dict[tuple, torch.Tensor] = {}

    @classmethod
    def get_gather(
        cls,
        *,
        min_tokens: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cap = _max_persistent_gather_tokens()
        key = (device, num_kv_heads, head_size, dtype)
        k_buf, v_buf, capacity = cls._gather.get(key, (None, None, 0))
        need = min(max(min_tokens, 1), cap)
        if k_buf is None or capacity < need:
            new_cap = max(need, capacity * 2 if capacity else need)
            new_cap = min(new_cap, cap)
            k_buf = torch.empty(
                (new_cap, num_kv_heads, head_size), dtype=dtype, device=device
            )
            v_buf = torch.empty(
                (new_cap, num_kv_heads, head_size), dtype=dtype, device=device
            )
            cls._gather[key] = (k_buf, v_buf, new_cap)
        return k_buf, v_buf

    @classmethod
    def get_gqa(
        cls,
        *,
        min_tokens: int,
        num_q_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cap = _max_persistent_gather_tokens()
        key = (device, num_q_heads, head_size, dtype)
        k_buf, v_buf, capacity = cls._gqa.get(key, (None, None, 0))
        need = min(max(min_tokens, 1), cap)
        if k_buf is None or capacity < need:
            new_cap = max(need, capacity * 2 if capacity else need)
            new_cap = min(new_cap, cap)
            k_buf = torch.empty(
                (new_cap, num_q_heads, head_size), dtype=dtype, device=device
            )
            v_buf = torch.empty(
                (new_cap, num_q_heads, head_size), dtype=dtype, device=device
            )
            cls._gqa[key] = (k_buf, v_buf, new_cap)
        return k_buf, v_buf

    @classmethod
    def get_padded(
        cls,
        *,
        micro_batch: int,
        kv_cap: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """HND ``[M,H,S,D]`` + NHD flats ``[M*S,H,D]`` for decode micro-batch."""
        key = (device, num_kv_heads, head_size, dtype)
        entry = cls._pad.get(key)
        need_tokens = micro_batch * kv_cap
        if (
            entry is None
            or entry[4] < need_tokens
            or entry[0].shape[0] < micro_batch
            or entry[0].shape[2] < kv_cap
        ):
            b = max(micro_batch, entry[0].shape[0] if entry else micro_batch)
            s = max(kv_cap, entry[0].shape[2] if entry else kv_cap)
            k_hnd = torch.empty(
                (b, num_kv_heads, s, head_size), dtype=dtype, device=device
            )
            v_hnd = torch.empty(
                (b, num_kv_heads, s, head_size), dtype=dtype, device=device
            )
            k_flat = torch.empty(
                (b * s, num_kv_heads, head_size), dtype=dtype, device=device
            )
            v_flat = torch.empty(
                (b * s, num_kv_heads, head_size), dtype=dtype, device=device
            )
            cls._pad[key] = (k_hnd, v_hnd, k_flat, v_flat, b * s)
            entry = cls._pad[key]
        return entry[0], entry[1], entry[2], entry[3]

    @classmethod
    def get_cu_padded(
        cls, *, micro_batch: int, kv_cap: int, device: torch.device
    ) -> torch.Tensor:
        """Fixed ``cu_seqlens`` for padded gather: ``[0, S, 2S, ...]``."""
        key = (device, micro_batch, kv_cap)
        cu = cls._cu_pad.get(key)
        if cu is None:
            cu = torch.arange(
                0,
                (micro_batch + 1) * kv_cap,
                kv_cap,
                dtype=torch.int32,
                device=device,
            )
            cls._cu_pad[key] = cu
        return cu

    @classmethod
    def get_arange(cls, *, kv_cap: int, device: torch.device) -> torch.Tensor:
        key = (device, kv_cap)
        ar = cls._arange.get(key)
        if ar is None:
            ar = torch.arange(kv_cap, device=device, dtype=torch.int32)
            cls._arange[key] = ar
        return ar


def _decode_micro_batch() -> int:
    """Micro-batch for seqlens decode workspace (memory ∝ this, not full B)."""
    env = os.getenv("VLLM_SAGE_DECODE_MICRO_BATCH")
    if env is not None:
        return max(int(env), 1)
    return 1


def _seqlens_decode_enabled() -> bool:
    """Opt-in Sage seqlens/paged decode (default off; FA decode is default).

    Set ``VLLM_SAGE_USE_SEQLENS=1`` to use Sage int8/fp8 seqlens decode
    instead of FlashAttention.
    """
    return os.getenv("VLLM_SAGE_USE_SEQLENS") == "1"


@contextmanager
def _suppress_cuda_set_device() -> Iterator[None]:
    """SageAttention calls ``torch.cuda.set_device`` which breaks CG capture.

    Devices are already matched by the caller; no-op the set during our
    sageattn invocations so PIECEWISE capture/replay stays healthy.
    """
    orig = torch.cuda.set_device

    def _noop_set_device(*_args, **_kwargs) -> None:
        return None

    torch.cuda.set_device = _noop_set_device  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.cuda.set_device = orig  # type: ignore[assignment]


def _max_persistent_gather_tokens() -> int:
    env_cap = os.getenv("VLLM_SAGE_MAX_GATHER_TOKENS")
    if env_cap is not None:
        return max(int(env_cap), 1)
    try:
        cfg = get_current_vllm_config()
        max_cg = cfg.compilation_config.max_cudagraph_capture_size or 0
        max_model = cfg.model_config.max_model_len or 0
        if max_cg > 0:
            return max(max_cg * 16, max_cg, max_model)
        if max_model > 0:
            return max_model
    except AssertionError:
        pass
    return 1 << 20


def _default_kv_cap() -> int:
    try:
        cfg = get_current_vllm_config()
        return max(int(cfg.model_config.max_model_len or 0), 1)
    except AssertionError:
        return 8192


@dataclass
class SageAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool
    cu_seqlens_k: torch.Tensor
    queries_eq_seqs: bool
    pure_decode: bool
    # Host-side values filled in build() — only for eager exact-length path.
    num_reqs: int
    total_kv_tokens: int
    cu_q_offsets: list[int]
    cu_k_offsets: list[int]
    # Fixed padded capacity for decode seqlens path (CG-stable).
    kv_cap: int


class SageAttentionBackend(AttentionBackend):
    """Attention backend backed by SageAttention CUDA (prefill + decode)."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SageAttentionMetadataBuilder"]:
        return SageAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # (2, num_blocks, block_size, H, D): kv_cache[0/1] are contiguous
        # NHD slabs. Do NOT use (num_blocks, block_size, 2, H, D) — slicing
        # [:, :, 0] is non-contiguous and breaks paged Sage kernels.
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # (2, num_blocks, num_layers, block_size, H, D)
            return (0, 1, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 80, 96, 112, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        if not current_platform.is_cuda():
            return False
        return capability.major >= 8

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
        if not _SAGE_AVAILABLE:
            return "sageattention package is not installed"
        if is_quantized_kv_cache(kv_cache_dtype):
            return "quantized KV cache is not supported by SAGE_ATTN yet"
        if has_sink:
            return "attention sinks are not supported by SAGE_ATTN"
        if use_sparse:
            return "sparse attention is not supported by SAGE_ATTN"
        if use_mm_prefix:
            return "mm prefix attention is not supported by SAGE_ATTN"
        return None


class SageAttentionMetadataBuilder(AttentionMetadataBuilder[SageAttentionMetadata]):
    # FULL CG for uniform single-token decode (FlashAttn); prefill stays
    # outside FULL graphs (exact sage uses host offsets).
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self.kv_cap = max(int(vllm_config.model_config.max_model_len or 0), 1)
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self._cu_seqlens_k = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        cap = _max_persistent_gather_tokens()
        dtype = vllm_config.model_config.dtype
        _SageSharedWorkspace.get_gather(
            min_tokens=cap,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            head_size=kv_cache_spec.head_size,
            dtype=dtype,
            device=device,
        )
        # Lazy padded alloc on first decode; only warm arange here.
        # Do NOT prealloc max_cg * max_model_len (can be tens of GB).
        _SageSharedWorkspace.get_arange(kv_cap=self.kv_cap, device=device)
        # Micro-batch decode workspace (O(S_cap), CG-safe).
        _SageSharedWorkspace.get_padded(
            micro_batch=_decode_micro_batch(),
            kv_cap=self.kv_cap,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            head_size=kv_cache_spec.head_size,
            dtype=dtype,
            device=device,
        )
        _SageSharedWorkspace.get_cu_padded(
            micro_batch=_decode_micro_batch(),
            kv_cap=self.kv_cap,
            device=device,
        )

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Default decode uses FlashAttn (CG-safe). Seqlens is opt-in.
        if _HAS_FLASH_ATTN or _SAGE_SEQLENS_AVAILABLE:
            return cls._cudagraph_support
        return AttentionCGSupport.NEVER

    def update_block_table(
        self,
        metadata: SageAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> SageAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SageAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        query_lens = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]

        cu_seqlens_k = self._cu_seqlens_k[: num_reqs + 1]
        cu_seqlens_k[0] = 0
        torch.cumsum(seq_lens[:num_reqs], dim=0, out=cu_seqlens_k[1:])

        queries_eq_seqs = bool(torch.equal(query_lens, seq_lens[:num_reqs]))
        pure_decode = bool((query_lens == 1).all().item())

        causal = common_attn_metadata.causal
        if isinstance(causal, torch.Tensor):
            causal = bool(causal.reshape(-1)[:num_reqs].all().item())

        total_kv_tokens = int(seq_lens[:num_reqs].sum().item())
        cu_q_offsets = query_start_loc[: num_reqs + 1].detach().cpu().tolist()
        cu_k_offsets = cu_seqlens_k[: num_reqs + 1].detach().cpu().tolist()

        return SageAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=causal,
            cu_seqlens_k=cu_seqlens_k,
            queries_eq_seqs=queries_eq_seqs,
            pure_decode=pure_decode,
            num_reqs=num_reqs,
            total_kv_tokens=total_kv_tokens,
            cu_q_offsets=cu_q_offsets,
            cu_k_offsets=cu_k_offsets,
            kv_cap=self.kv_cap,
        )


class SageAttentionImpl(AttentionImpl[SageAttentionMetadata]):
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
        sinks: torch.Tensor | None = None,
    ) -> None:
        if not _SAGE_AVAILABLE:
            raise ImportError(
                "SAGE_ATTN backend requires the sageattention package. "
                "Install it into the same Python env as vLLM."
            )
        if alibi_slopes is not None:
            raise NotImplementedError("SAGE_ATTN does not support ALiBi")
        if sliding_window is not None and sliding_window > 0:
            raise NotImplementedError("SAGE_ATTN does not support sliding window")
        if logits_soft_cap is not None and logits_soft_cap != 0:
            raise NotImplementedError("SAGE_ATTN does not support logits soft cap")
        if sinks is not None:
            raise NotImplementedError("SAGE_ATTN does not support attention sinks")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"SAGE_ATTN only supports decoder attention, got {attn_type}"
            )
        if is_quantized_kv_cache(kv_cache_dtype):
            raise NotImplementedError(
                "SAGE_ATTN does not support quantized KV cache yet"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self._dense_out_buf: torch.Tensor | None = None
        self._decode_q_buf: torch.Tensor | None = None
        self.kv_cap = _default_kv_cap()
        self.fa_version = get_flash_attn_version(head_size=head_size)

        if _seqlens_decode_enabled() and not _SAGE_SEQLENS_AVAILABLE:
            logger.warning_once(
                "VLLM_SAGE_USE_SEQLENS=1 but seqlens kernel is missing; "
                "falling back to FlashAttn decode."
            )
        if not _HAS_FLASH_ATTN and not _SAGE_SEQLENS_AVAILABLE:
            logger.warning_once(
                "Neither FlashAttn nor Sage seqlens is available; "
                "pure decode will use exact sageattn (not FULL-CG safe)."
            )

        decode_backend = (
            "seqlens"
            if (_seqlens_decode_enabled() and _SAGE_SEQLENS_AVAILABLE)
            else ("flash_attn" if _HAS_FLASH_ATTN else "exact_sage")
        )
        logger.info_once(
            "Using SAGE_ATTN backend (exact prefill; decode=%s; "
            "cudagraph=%s seqlens_pkg=%s fa=%s)",
            decode_backend,
            AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE.name,
            _SAGE_SEQLENS_AVAILABLE,
            _HAS_FLASH_ATTN,
        )

    def _should_use_seqlens_decode(self, attn_metadata: SageAttentionMetadata) -> bool:
        """Opt-in Sage seqlens decode (``VLLM_SAGE_USE_SEQLENS=1``)."""
        return (
            _seqlens_decode_enabled()
            and attn_metadata.pure_decode
            and _SAGE_SEQLENS_AVAILABLE
        )

    def _should_use_fa_decode(self, attn_metadata: SageAttentionMetadata) -> bool:
        """Default pure-decode path: FlashAttention paged varlen."""
        return (
            attn_metadata.pure_decode
            and _HAS_FLASH_ATTN
            and not self._should_use_seqlens_decode(attn_metadata)
        )

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # kv_cache: (2, num_blocks, block_size, num_kv_heads, head_size)
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]
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

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for SAGE_ATTN"
            )
        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        q = query[:num_actual_tokens]
        out = output[:num_actual_tokens]

        # Contiguous slabs: (num_blocks, block_size, num_kv_heads, head_size)
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        # Pure decode: FlashAttn (default) or opt-in Sage seqlens.
        if self._should_use_fa_decode(attn_metadata):
            self._forward_decode_fa(q, key_cache, value_cache, attn_metadata, out)
            return output
        if self._should_use_seqlens_decode(attn_metadata):
            o = self._forward_decode_seqlens(
                q, key_cache, value_cache, attn_metadata
            )
            out.copy_(o)
            return output

        # Prefill / mixed: paged quant+attn (no gather / TransposePadPermute).
        # Chunked causal (lq < lk, lq > 1) still needs gather+exact sage fallback.
        if _SAGE_SEQLENS_PAGED_AVAILABLE and _SAGE_SEQLENS_PAGED_HAS_KV_CAP:
            o = self._forward_prefill_paged(
                q, key_cache, value_cache, attn_metadata
            )
            if o is not None:
                out.copy_(o)
                return output

        # Fallback: gather + exact dense sageattn.
        k_varlen, v_varlen = self._gather_kv(
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.cu_seqlens_k,
            attn_metadata.num_reqs,
            alloc_kv_tokens=attn_metadata.total_kv_tokens,
        )

        if self.num_queries_per_kv > 1:
            k_varlen, v_varlen = self._expand_kv_gqa(
                k_varlen, v_varlen, attn_metadata.total_kv_tokens
            )

        o = self._forward_dense_sageattn(q, k_varlen, v_varlen, attn_metadata)
        out.copy_(o)
        return output

    def _forward_decode_fa(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
        out: torch.Tensor,
    ) -> None:
        """Paged FlashAttention varlen decode (CG-safe like FA2 backend)."""
        assert flash_attn_varlen_func is not None
        kwargs: dict = dict(
            q=q,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            block_table=attn_metadata.block_table,
        )
        if self.fa_version is not None:
            kwargs["fa_version"] = self.fa_version
        flash_attn_varlen_func(**kwargs)

    def _ensure_buf(
        self,
        buf: torch.Tensor | None,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            buf is None
            or buf.dtype != dtype
            or buf.device != device
            or buf.ndim != len(shape)
            or any(buf.shape[i] < shape[i] for i in range(len(shape)))
        ):
            return torch.empty(shape, dtype=dtype, device=device)
        return buf

    def _gather_kv(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        num_reqs: int,
        *,
        alloc_kv_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]
        device = key_cache.device
        dtype = key_cache.dtype
        entry_size = num_kv_heads * head_size
        cu_k = cu_seqlens_k[: num_reqs + 1]

        k_buf, v_buf = _SageSharedWorkspace.get_gather(
            min_tokens=max(alloc_kv_tokens, 1),
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            device=device,
        )
        k_flat = k_buf.view(-1, entry_size)
        v_flat = v_buf.view(-1, entry_size)

        ops.cp_gather_cache(
            key_cache,
            k_flat,
            block_table[:num_reqs],
            cu_k,
            num_reqs,
        )
        ops.cp_gather_cache(
            value_cache,
            v_flat,
            block_table[:num_reqs],
            cu_k,
            num_reqs,
        )
        n = max(alloc_kv_tokens, 1)
        return k_buf[:n], v_buf[:n]

    def _expand_kv_gqa(
        self,
        k_varlen: torch.Tensor,
        v_varlen: torch.Tensor,
        alloc_kv_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nrep = self.num_queries_per_kv
        total_tokens = k_varlen.shape[0]
        num_kv_heads = k_varlen.shape[1]
        head_size = k_varlen.shape[2]
        num_q_heads = num_kv_heads * nrep
        device, dtype = k_varlen.device, k_varlen.dtype

        k_buf, v_buf = _SageSharedWorkspace.get_gqa(
            min_tokens=max(alloc_kv_tokens, total_tokens, 1),
            num_q_heads=num_q_heads,
            head_size=head_size,
            dtype=dtype,
            device=device,
        )
        k_out = k_buf[:total_tokens]
        v_out = v_buf[:total_tokens]
        k_out.copy_(
            k_varlen.unsqueeze(2)
            .expand(-1, -1, nrep, -1)
            .reshape(total_tokens, num_q_heads, head_size)
        )
        v_out.copy_(
            v_varlen.unsqueeze(2)
            .expand(-1, -1, nrep, -1)
            .reshape(total_tokens, num_q_heads, head_size)
        )
        return k_out, v_out

    def _forward_decode_seqlens(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
    ) -> torch.Tensor:
        """Seqlens decode (prefer paged quant; fallback to gather+padded path)."""
        # Preferred path: paged cache -> paged quant -> paged seqlens kernel.
        if _SAGE_SEQLENS_PAGED_AVAILABLE:
            assert sageattn_qk_int8_pv_fp8_cuda_seqlens_paged is not None
            num_reqs = attn_metadata.num_reqs
            seq_lens = attn_metadata.seq_lens[:num_reqs].to(torch.int32)
            block_table = attn_metadata.block_table[:num_reqs].to(torch.int32)
            # pure decode => one query token per request
            q_hnd = q[:num_reqs].unsqueeze(2)
            # Paged Sage kernels assume contiguous [num_blocks, page, H, D].
            assert key_cache.is_contiguous() and value_cache.is_contiguous(), (
                "SAGE_ATTN paged path requires contiguous K/V cache slabs; "
                f"got strides key={key_cache.stride()} value={value_cache.stride()}"
            )
            with _suppress_cuda_set_device():
                if _SAGE_SEQLENS_PAGED_HAS_KV_CAP:
                    o_hnd = sageattn_qk_int8_pv_fp8_cuda_seqlens_paged(
                        q_hnd,
                        key_cache,
                        value_cache,
                        seq_lens,
                        block_table,
                        key_cache.shape[1],
                        attn_metadata.kv_cap,
                        sm_scale=self.scale,
                        is_causal=False,
                    )
                else:
                    # Older signature:
                    # (q, k_cache, v_cache, seq_lens, block_table, block_size, sm_scale=None, ...)
                    o_hnd = sageattn_qk_int8_pv_fp8_cuda_seqlens_paged(
                        q_hnd,
                        key_cache,
                        value_cache,
                        seq_lens,
                        block_table,
                        key_cache.shape[1],
                        sm_scale=self.scale,
                    )
            self._dense_out_buf = self._ensure_buf(
                self._dense_out_buf,
                (num_reqs, self.num_heads, self.head_size),
                dtype=q.dtype,
                device=q.device,
            )
            out = self._dense_out_buf[:num_reqs]
            out.copy_(o_hnd.squeeze(2))
            return out
        # Fallback: gather+padded seqlens path.

        """Padded gather + device ``seq_lens`` (CUDA-graph safe, FA2-like).

        Workspace is a **micro-batch** (default 1), not full ``B × S_cap``:
        full-batch padding OOMs when FULL CG capture sizes are large
        (``max_cudagraph_capture_size × max_model_len``).

        For each micro-chunk we gather a fixed ``S_cap`` tokens per request
        into stable buffers, zero-pad with device ``seq_lens``, and call
        ``sageattn_qk_int8_pv_fp8_cuda_seqlens``. Loop trip count is the
        capture-time ``num_reqs`` (frozen under CG); buffer *contents* and
        ``seq_lens`` / ``block_table`` update on replay — same contract as FA2.
        """
        assert sageattn_qk_int8_pv_fp8_cuda_seqlens is not None
        num_reqs = attn_metadata.num_reqs
        kv_cap = attn_metadata.kv_cap
        device = q.device
        dtype = q.dtype
        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]
        entry_size = num_kv_heads * head_size
        micro = _decode_micro_batch()

        k_hnd, v_hnd, k_flat, v_flat = _SageSharedWorkspace.get_padded(
            micro_batch=micro,
            kv_cap=kv_cap,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            device=device,
        )
        cu_pad = _SageSharedWorkspace.get_cu_padded(
            micro_batch=micro, kv_cap=kv_cap, device=device
        )
        ar = _SageSharedWorkspace.get_arange(kv_cap=kv_cap, device=device)
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table

        self._decode_q_buf = self._ensure_buf(
            self._decode_q_buf,
            (micro, self.num_heads, 1, head_size),
            dtype=dtype,
            device=device,
        )
        self._dense_out_buf = self._ensure_buf(
            self._dense_out_buf,
            (num_reqs, self.num_heads, head_size),
            dtype=dtype,
            device=device,
        )
        out = self._dense_out_buf[:num_reqs]

        # num_reqs is fixed per captured graph; chunk over a small workspace.
        for start in range(0, num_reqs, micro):
            end = min(start + micro, num_reqs)
            n = end - start
            k_flat_b = k_flat[: n * kv_cap]
            v_flat_b = v_flat[: n * kv_cap]
            cu = cu_pad if n == micro else cu_pad[: n + 1]

            ops.cp_gather_cache(
                key_cache,
                k_flat_b.view(-1, entry_size),
                block_table[start:end],
                cu,
                n,
            )
            ops.cp_gather_cache(
                value_cache,
                v_flat_b.view(-1, entry_size),
                block_table[start:end],
                cu,
                n,
            )

            k_dst = k_hnd[:n, :, :kv_cap]
            v_dst = v_hnd[:n, :, :kv_cap]
            k_dst.copy_(
                k_flat_b.view(n, kv_cap, num_kv_heads, head_size).permute(0, 2, 1, 3)
            )
            v_dst.copy_(
                v_flat_b.view(n, kv_cap, num_kv_heads, head_size).permute(0, 2, 1, 3)
            )

            sl = seq_lens[start:end]
            pad_mask = ar[None, :] >= sl[:, None]
            k_dst.masked_fill_(pad_mask[:, None, :, None], 0)
            v_dst.masked_fill_(pad_mask[:, None, :, None], 0)

            q_hnd = self._decode_q_buf[:n]
            q_hnd.copy_(q[start:end].unsqueeze(2))

            with _suppress_cuda_set_device():
                o_hnd = sageattn_qk_int8_pv_fp8_cuda_seqlens(
                    q_hnd,
                    k_dst,
                    v_dst,
                    sl.to(torch.int32),
                    sm_scale=self.scale,
                )
            out[start:end].copy_(o_hnd.squeeze(2))

        return out

    def _forward_prefill_paged(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
    ) -> torch.Tensor | None:
        """Prefill via paged quant+attn — skips gather and TransposePadPermute.

        Returns None to fall back to gather+exact sage when a request needs
        chunked causal (lq < lk and lq > 1), which needs Q index offset.
        """
        assert sageattn_qk_int8_pv_fp8_cuda_seqlens_paged is not None
        num_reqs = attn_metadata.num_reqs
        cu_q = attn_metadata.cu_q_offsets
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        page_size = int(key_cache.shape[1])
        kv_cap = int(attn_metadata.kv_cap)

        assert key_cache.is_contiguous() and value_cache.is_contiguous(), (
            "SAGE_ATTN paged prefill requires contiguous K/V cache slabs; "
            f"got strides key={key_cache.stride()} value={value_cache.stride()}"
        )

        # Fast reject: any chunked-causal request → gather fallback.
        for i in range(num_reqs):
            lq = cu_q[i + 1] - cu_q[i]
            lk = int(seq_lens[i].item())
            if lq > 1 and lq != lk and attn_metadata.causal:
                return None

        self._dense_out_buf = self._ensure_buf(
            self._dense_out_buf,
            tuple(q.shape),
            dtype=q.dtype,
            device=q.device,
        )
        out = self._dense_out_buf[: q.shape[0], : q.shape[1], : q.shape[2]]

        for i in range(num_reqs):
            q_s = cu_q[i]
            q_e = cu_q[i + 1]
            lq = q_e - q_s
            if lq == 0:
                continue
            lk = int(seq_lens[i].item())
            # [Lq, H, D] NHD → [1, H, Lq, D] HND
            q_hnd = q[q_s:q_e].unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            sl = seq_lens[i : i + 1].to(torch.int32)
            bt = block_table[i : i + 1].to(torch.int32)
            use_causal = bool(attn_metadata.causal and lq == lk)
            with _suppress_cuda_set_device():
                o_hnd = sageattn_qk_int8_pv_fp8_cuda_seqlens_paged(
                    q_hnd,
                    key_cache,
                    value_cache,
                    sl,
                    bt,
                    page_size,
                    kv_cap,
                    sm_scale=self.scale,
                    is_causal=use_causal,
                )
            # [1, H, Lq, D] → [Lq, H, D]
            out[q_s:q_e].copy_(o_hnd.squeeze(0).permute(1, 0, 2))
        return out

    def _forward_dense_sageattn(
        self,
        q: torch.Tensor,
        k_varlen: torch.Tensor,
        v_varlen: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
    ) -> torch.Tensor:
        """Per-request ``sageattn()`` CUDA — offsets from ``build()`` (eager)."""
        num_reqs = attn_metadata.num_reqs
        cu_q = attn_metadata.cu_q_offsets
        cu_k = attn_metadata.cu_k_offsets
        self._dense_out_buf = self._ensure_buf(
            self._dense_out_buf,
            tuple(q.shape),
            dtype=q.dtype,
            device=q.device,
        )
        out = self._dense_out_buf[: q.shape[0], : q.shape[1], : q.shape[2]]

        # Use NHD ([B,L,H,D]). On current sageattention (sm120), HND is
        # numerically wrong vs SDPA (amp collapse / garbage tokens); NHD matches.
        for i in range(num_reqs):
            q_s = cu_q[i]
            q_e = cu_q[i + 1]
            k_s = cu_k[i]
            k_e = cu_k[i + 1]
            q_i = q[q_s:q_e]
            k_i = k_varlen[k_s:k_e]
            v_i = v_varlen[k_s:k_e]
            lq = q_e - q_s
            lk = k_e - k_s
            if lq == 0:
                continue
            with _suppress_cuda_set_device():
                if lq == lk:
                    o_i = sageattn(
                        q_i.unsqueeze(0),
                        k_i.unsqueeze(0),
                        v_i.unsqueeze(0),
                        tensor_layout="NHD",
                        is_causal=attn_metadata.causal,
                        sm_scale=self.scale,
                    )
                    out[q_s:q_e].copy_(o_i.squeeze(0))
                elif lq == 1 or not attn_metadata.causal:
                    o_i = sageattn(
                        q_i.unsqueeze(0),
                        k_i.unsqueeze(0),
                        v_i.unsqueeze(0),
                        tensor_layout="NHD",
                        is_causal=False,
                        sm_scale=self.scale,
                    )
                    out[q_s:q_e].copy_(o_i.squeeze(0))
                else:
                    q_pad = q.new_zeros(lk, q_i.shape[1], q_i.shape[2])
                    q_pad[lk - lq :] = q_i
                    o_full = sageattn(
                        q_pad.unsqueeze(0),
                        k_i.unsqueeze(0),
                        v_i.unsqueeze(0),
                        tensor_layout="NHD",
                        is_causal=True,
                        sm_scale=self.scale,
                    )
                    out[q_s:q_e].copy_(o_full.squeeze(0)[lk - lq :])
        return out
