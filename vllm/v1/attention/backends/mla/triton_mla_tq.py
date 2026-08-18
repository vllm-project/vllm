# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant TRITON_MLA backend for DeepSeek-style MLA.

Supported presets (via --kv-cache-dtype):
    turboquant_k8v4     — FP8 kv_c (8-bit, no Hadamard rotation)
    turboquant_4bit_nc  — 4-bit MSE kv_c with Hadamard + Lloyd-Max + norm_correction
    turboquant_k3v4_nc  — 3-bit MSE kv_c with Hadamard + Lloyd-Max + norm_correction
    turboquant_3bit_nc  — 3-bit MSE kv_c with Hadamard + Lloyd-Max + norm_correction
                          (in MLA, k/v bits collapse onto kv_c → same as k3v4_nc)

Cache layout per token (uint8, byte-packed):
    [ kv_c_packed (key_packed_size bytes) | k_pe (k_pe_bytes) ]

k_pe bytes:
  default: 2 * qk_rope_head_dim bytes, raw bf16
  optional: qk_rope_head_dim fp8 bytes + 2-byte fp16 scale
            when VLLM_TQ_KPE_FP8=1

kv_c_packed:
  FP8 path:  kv_lora_rank bytes (1 B/elem, e4m3)
  MSE path:  ceil(kv_lora_rank * mse_bits / 8) index bytes
             + 2 B fp16 per-token scale (vec_norm, or eff_scale if norm_correction)

Implementation strategy:
  - do_kv_cache_update: quantize latent kv_c / ctkv and scatter packed bytes
    (MSE 4bit: fused Triton store by default — see triton_turboquant_mla_store.py)
    into the uint8 KV cache.
  - forward_mqa (sparse): prefill uses gather+dedup+flash_mla_sparse_fwd;
    decode uses fused 2-stage sparse.  Selected by ``--kv-cache-dtype`` and
    backend choice — no runtime env toggles required for production ``*_nc``.
  - The legacy FP8 workspace path is kept behind _TQ_FP8_WORKSPACE=1 as a
    regression anchor only.

Reference material:
  - #38479 general TurboQuant infrastructure
  - vllm/v1/attention/backends/turboquant_attn.py
  - vllm/model_executor/layers/quantization/turboquant/{config,centroids}.py
  - vllm/v1/attention/ops/triton_turboquant_mla_decode.py
"""

import math
import os
from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.config.vllm import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata,
)
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAImpl,
    TritonMLAMetadataBuilder,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd
from vllm.v1.attention.ops.tq_mla_defaults import (
    default_kpe_4bit,
    default_kpe_fp8,
    default_store_fwht,
)
from vllm.v1.attention.ops.triton_decode_attention import (
    _decode_softmax_reducev_fwd,
)
from vllm.v1.attention.ops.triton_turboquant_mla_decode import (
    fused_mla_dequant_mse,
    fused_mla_sparse_topk_gather_dequant_mse,
    fused_mla_tq_decode_stage1,
    fused_mla_tq_sparse_decode_stage1,
    sparse_decode_softmax_reducev_fwd,
    tq_mla_sparse_adaptive_enabled,
    tq_mla_sparse_split_count,
)
from vllm.v1.attention.ops.triton_turboquant_mla_store import (
    kpe_mse_index_bytes,
    kpe_packed_bytes,
    tq_mla_fused_kv_cache_store,
    tq_mla_fused_store_enabled,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0
_BF16 = torch.bfloat16

# One bf16 top-k dequant workspace per GPU for legacy sparse gather+flash.
# All MLA layers share it (forward is sequential).  Do NOT pool on per-layer
# ``self._tq_buffers`` — that would retain peak ``B*topk`` on every layer.
_LEGACY_TOPK_DEQUANT_WS: dict[str, torch.Tensor] = {}

# Reusable per-GPU buffers for bucket dedup (top-k sparse prefill).
# Keyed by device string; grows on demand and is reused across layers/steps.
_DEDUP_BUCKET_PRESENT_WS: dict[str, torch.Tensor] = {}
_DEDUP_BUCKET_IDMAP_WS: dict[str, torch.Tensor] = {}


def _in_cuda_graph_capture() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _sparse_topk_dedup_enabled() -> bool:
    # Production default ON for sparse latent prefill. Opt-out: =0.
    if os.environ.get("VLLM_TQ_MLA_SPARSE_TOPK_DEDUP", "1") == "0":
        return False
    # torch.unique / Tensor.any() sync during CUDAGraph capture →
    # cudaErrorStreamCaptureUnsupported (breaks engine startup).
    return not _in_cuda_graph_capture()


def _dedup_bucket_enabled() -> bool:
    """Bucket dedup (presence-bitmap + nonzero) instead of ``torch.unique`` sort.

    ``torch.unique(sorted=True)`` sorts the whole ``(T*K,)`` topk array
    (O(N log N), N up to ~16M for an 8k prefill chunk).  Global slot ids are
    bounded by the KV pool size, so a presence bitmap + ``nonzero`` (+ scatter
    remap) computes the identical sorted-unique result in O(N + pool) with no
    sort.  Set ``VLLM_TQ_DEDUP_BUCKET=0`` to fall back to ``torch.unique``.
    """
    return os.environ.get("VLLM_TQ_DEDUP_BUCKET", "1") == "1"


def _dedup_global_topk(
    global_topk: torch.Tensor,
    num_cache_slots: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Unique global slots + per-(token,topk) workspace row indices.

    Returns ``(unique_slots, remapped_local_topk, num_unique)`` where
    ``remapped_local_topk[token, k]`` indexes into the deduped workspace
    (or -1 when invalid).
    """
    num_tokens, topk = global_topk.shape
    device = global_topk.device
    flat = global_topk.reshape(-1)
    valid = flat >= 0

    if _dedup_bucket_enabled():
        # Bound the bitmap by the KV pool when known (avoids a max() reduction).
        if num_cache_slots is not None and num_cache_slots > 0:
            n_slots = int(num_cache_slots)
        else:
            # Fallback (rare): derive from data.
            if not bool(valid.any()):
                empty = torch.empty(0, dtype=torch.int32, device=device)
                remapped = torch.full(
                    (num_tokens, topk), -1, dtype=torch.int32, device=device
                )
                return empty, remapped, 0
            n_slots = int(flat[valid].max().item()) + 1

        key = str(device)
        need = n_slots + 1  # +1 sentinel for invalid (-1) entries.
        present = _DEDUP_BUCKET_PRESENT_WS.get(key)
        if present is None or present.numel() < need:
            present = torch.empty(need, dtype=torch.bool, device=device)
            _DEDUP_BUCKET_PRESENT_WS[key] = present
        else:
            present = present[:need]
        id_map = _DEDUP_BUCKET_IDMAP_WS.get(key)
        if id_map is None or id_map.numel() < need:
            id_map = torch.empty(need, dtype=torch.int32, device=device)
            _DEDUP_BUCKET_IDMAP_WS[key] = id_map
        else:
            id_map = id_map[:need]

        # Compaction-free path:
        #   - map invalid indices to sentinel slot ``n_slots``
        #   - build presence bitmap in one scatter
        #   - remap by table lookup (sentinel maps to -1)
        sentinel_i32 = torch.full_like(flat, n_slots)
        safe_i32 = torch.where(valid, flat, sentinel_i32)
        safe_long = safe_i32.to(torch.long)

        present.fill_(False)
        present[safe_long] = True
        present[n_slots] = False  # drop invalid sentinel
        unique_slots = (
            present[:-1].nonzero(as_tuple=True)[0].to(torch.int32)
        )  # ascending
        num_unique = int(unique_slots.numel())
        id_map.fill_(-1)
        id_map[unique_slots.to(torch.long)] = torch.arange(
            num_unique, dtype=torch.int32, device=device
        )
        remapped_flat = id_map[safe_long]
        remapped = remapped_flat.view(num_tokens, topk)
        return unique_slots, remapped, num_unique

    if not bool(valid.any()):
        empty = torch.empty(0, dtype=torch.int32, device=device)
        remapped = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=device)
        return empty, remapped, 0

    unique_slots, inverse = torch.unique(flat[valid], sorted=True, return_inverse=True)
    remapped_flat = torch.full((flat.numel(),), -1, dtype=torch.int32, device=device)
    remapped_flat[valid] = inverse.to(torch.int32)
    remapped = remapped_flat.view(num_tokens, topk)
    return unique_slots, remapped, int(unique_slots.numel())


def _get_global_topk_dequant_workspace(
    num_slots: int,
    L: int,
    R: int,
    device: torch.device,
    max_slots: int,
) -> torch.Tensor:
    """Return legacy top-k dequant workspace view (one tensor per GPU).

    Must NOT use WorkspaceManager here: growing that blob replaces the
    underlying storage and invalidates the ``q_concat_buffer`` view allocated
    earlier from the same manager (garbage outputs / UAF).

    Memory cost is ``min(num_slots, max_slots) * (L+R) * 2`` bytes — cap with
    ``--max-num-batched-tokens``.
    """
    need = min(num_slots, max_slots)
    head_dim = L + R
    key = str(device)
    ws = _LEGACY_TOPK_DEQUANT_WS.get(key)
    if ws is None or ws.shape[2] != head_dim or ws.shape[0] < need:
        ws = torch.empty((need, 1, head_dim), dtype=_BF16, device=device)
        _LEGACY_TOPK_DEQUANT_WS[key] = ws
    return ws[:num_slots]


def _enumerate_packed_sizes() -> list[int]:
    """Return all head_size (=packed_bytes per slot) values this backend
    accepts, across (L, R, preset, kpe_fp8) combinations supported by the
    code paths in this module.

    Concrete reasoning:
      - L (kv_lora_rank): power of 2 in {128, 256, 512, 1024}
      - R (qk_rope_head_dim): in {32, 64, 96, 128}
      - kv_c_bytes for preset:
          k8v4: L              (1 B/elem fp8)
          4bit: ceil(L*4/8)+2 = L/2 + 2
          3bit: ceil(L*3/8)+2
      - k_pe_bytes: 2*R (bf16) or R+2 (fp8 + fp16 scale)

    We enumerate the cross product so that any vLLM `get_supported_head_sizes`
    membership check at startup will succeed.
    """
    import math as _math

    sizes: set[int] = set()
    for L in (128, 256, 512, 1024):
        # k8v4 fp8 keys (1 byte/elem):
        kv_c_options = [L]
        # MSE 3-bit and 4-bit keys, +2 fp16 vec_norm:
        for bits in (3, 4):
            kv_c_options.append(_math.ceil(L * bits / 8) + 2)
        for R in (32, 64, 96, 128):
            k_pe_options = [2 * R, R + 2, kpe_mse_index_bytes(R, 4) + 2]
            for k_pe in k_pe_options:
                for kv_c in kv_c_options:
                    sizes.add(kv_c + k_pe)
    return sorted(sizes)


# ----------------------------------------------------------------------
# Bit-packing helpers (pure PyTorch, correctness-first).
# ----------------------------------------------------------------------


def _pack_bits_rows(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack (N, D) int indices into (N, ceil(D*bits/8)) uint8 rows.

    Because the `bits`-wide fields land in disjoint bit ranges within each
    byte, we can use integer addition (scatter_add_) as a stand-in for
    bitwise-OR and convert the int32 accumulator back to uint8 at the end.

    Args:
        idx: (N, D) integer tensor in [0, 2**bits).
        bits: 3 or 4.

    Returns:
        (N, ceil(D*bits/8)) uint8 tensor.
    """
    assert bits in (3, 4), f"pack supports 3/4 bits only, got {bits}"
    N, D = idx.shape
    n_bytes = math.ceil(D * bits / 8)
    device = idx.device

    idx_i = idx.to(torch.int32)
    out = torch.zeros((N, n_bytes), dtype=torch.int32, device=device)

    d = torch.arange(D, device=device)
    bit_off = d * bits
    byte_idx = (bit_off // 8).to(torch.long)  # (D,)
    bit_shift = (bit_off % 8).to(torch.int32)  # (D,)

    # Low part: bits landing in byte_idx.
    low = (idx_i << bit_shift.view(1, D)) & 0xFF
    out.scatter_add_(1, byte_idx.view(1, D).expand(N, D), low)

    # High part: spill bits to byte_idx+1 when bit_shift + bits > 8.
    spans = (bit_shift + bits > 8).view(1, D).expand(N, D)
    spill_shift = (8 - bit_shift).clamp(min=0)
    high = (idx_i >> spill_shift.view(1, D)) & 0xFF
    high = torch.where(spans, high, torch.zeros_like(high))
    high_byte_idx = (byte_idx + 1).clamp(max=n_bytes - 1)
    out.scatter_add_(1, high_byte_idx.view(1, D).expand(N, D), high)
    return out.to(torch.uint8)


def _unpack_bits_rows(packed: torch.Tensor, bits: int, D: int) -> torch.Tensor:
    """Inverse of _pack_bits_rows.

    Args:
        packed: (..., n_bytes) uint8.
        bits: 3 or 4.
        D: expected output width.

    Returns:
        (..., D) int64 tensor of indices in [0, 2**bits).
    """
    assert bits in (3, 4), f"unpack supports 3/4 bits only, got {bits}"
    device = packed.device
    n_bytes = packed.shape[-1]

    d = torch.arange(D, device=device)
    bit_off = d * bits
    byte_idx = (bit_off // 8).to(torch.long)
    bit_shift = (bit_off % 8).to(torch.long)
    mask = (1 << bits) - 1

    raw0 = packed[..., byte_idx].to(torch.int32)
    # byte_idx + 1 may equal n_bytes; pad by clamping and masking the spill.
    safe_next = (byte_idx + 1).clamp(max=n_bytes - 1)
    raw1 = packed[..., safe_next].to(torch.int32)
    raw16 = raw0 | (raw1 << 8)
    out = (raw16 >> bit_shift.view(*([1] * (packed.dim() - 1)), D)) & mask
    return out.to(torch.int64)


class TritonMLATurboQuantMetadataBuilder(TritonMLAMetadataBuilder):
    """MLA metadata builder that advertises CUDA-graph support.

    Parent MLACommonMetadataBuilder defaults to AttentionCGSupport.NEVER.
    TurboQuant's decode path (after Step 2: no torch.unique, no host syncs)
    is CG-safe for decode-only batches, so UNIFORM_BATCH is the correct
    level — matching FlashMLA / FlashAttn MLA.

    `build_for_cudagraph_capture` is inherited from MLACommonMetadataBuilder;
    it asserts decode-only and calls self.build(0, m), which is fine here.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLATurboQuantBackend(TritonMLABackend):
    """TurboQuant-aware MLA backend on top of TritonMLA."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_TURBOQUANT"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # head_size in this backend == packed_bytes per slot (uint8 cache),
        # so the value depends on (kv_lora_rank, qk_rope_head_dim, preset, kpe_fp8).
        # DeepSeek-V2/V3 (L=512, R=64): bf16 k_pe → 576; fp8 k_pe → 514.
        # Other models: see _enumerate_packed_sizes for the full set we accept.
        return _enumerate_packed_sizes()

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_k8v4",
    ) -> tuple[int, ...]:
        # head_size is the packed byte count per token, computed by
        # MLAAttention.get_kv_cache_spec using TurboQuantConfig.
        return (num_blocks, block_size, head_size)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # FP8 path needs SM89+; MSE path only needs SM80+. Gate at SM80.
        return capability.major >= 8

    @staticmethod
    def get_impl_cls() -> type["TritonMLATurboQuantImpl"]:
        return TritonMLATurboQuantImpl

    @staticmethod
    def get_builder_cls() -> type[TritonMLAMetadataBuilder]:
        return TritonMLATurboQuantMetadataBuilder


class TritonMLATurboQuantImpl(TritonMLAImpl):
    """TritonMLA impl with a TurboQuant byte-packed KV cache.

    Supports FP8 keys (k8v4) and MSE Lloyd-Max keys (4bit_nc / k3v4_nc /
    3bit_nc). Value compression is not applicable in MLA: V is recovered
    from kv_c via W_UV, so there is no independent V slot to quantize.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # P3-1: kv_lora_rank can now be any power of 2 (Sylvester Hadamard
        # requirement). qk_rope_head_dim can be any positive int (R is just a
        # bf16 byte slice on the cache, no Hadamard, only BLOCK_R=next_pow2(R)
        # masking inside the kernel).
        L = self.kv_lora_rank
        assert L > 0 and (L & (L - 1)) == 0, (
            f"TritonMLATurboQuant requires kv_lora_rank to be a power of 2 "
            f"(Sylvester Hadamard); got {L}"
        )
        assert self.qk_rope_head_dim > 0, (
            f"qk_rope_head_dim must be positive; got {self.qk_rope_head_dim}"
        )
        # Build TQ config from the kv_cache_dtype string. self.kv_cache_dtype
        # is set by the base impl (turboquant_{k8v4,4bit_nc,k3v4_nc,3bit_nc}).
        self.tq_config: TurboQuantConfig = TurboQuantConfig.from_cache_dtype(
            self.kv_cache_dtype, head_dim=self.kv_lora_rank
        )
        # Per-slot byte layout (parameterized on L = kv_lora_rank):
        #   FP8  : L bytes (1 B/elem, no norm)
        #   MSE  : ceil(L * bits / 8) + 2 bytes (indices + fp16 vec_norm)
        # Concrete: L=512 → FP8 512B, 4bit 258B, 3bit 194B.
        self._kv_c_bytes = self.tq_config.key_packed_size
        # P3-2 / P6-4: k_pe FP8 (e4m3) compression (default on).
        # bf16 layout: 2*R bytes when VLLM_TQ_KPE_FP8=0.
        # fp8 layout:  R bytes fp8 + 2 bytes per-token fp16 scale.
        # 4bit: ceil(R*4/8) index bytes + 2-byte fp16 scale
        # (VLLM_TQ_KPE_4BIT=1).
        # R=64 → 128B → 66B (~48% smaller k_pe, ~16% smaller 4bit_nc slot).
        # k_pe layout follows --kv-cache-dtype (NC presets → 4bit + FWHT store).
        self._kpe_4bit = default_kpe_4bit(self.kv_cache_dtype, self.tq_config)
        self._store_fwht = default_store_fwht(self.kv_cache_dtype, self.tq_config)
        self._kpe_fp8 = default_kpe_fp8(
            self.kv_cache_dtype,
            self.tq_config,
            kpe_4bit=self._kpe_4bit,
        )
        self._k_pe_bytes = kpe_packed_bytes(
            self.qk_rope_head_dim,
            kpe_4bit=self._kpe_4bit,
            kpe_fp8=self._kpe_fp8,
        )
        self._packed_bytes = self._kv_c_bytes + self._k_pe_bytes
        # For MSE path: number of "index bytes" before the 2-byte vec_norm.
        self._mse_index_bytes = (
            math.ceil(self.kv_lora_rank * self.tq_config.key_mse_bits / 8)
            if not self.tq_config.key_fp8
            else 0
        )

        # Override: TritonMLA sets supports_quant_query_input=False when
        # is_quantized_kv_cache is True; we want the same.
        self.supports_quant_query_input = False

        # P0-1: cache layer._k_scale as a Python float on first use so we
        # never call `.item()` inside CUDA Graph capture (host-device sync
        # is forbidden during capture). First read happens during eager
        # warmup, before capture, so this is safe. Assumes scale is fixed
        # after weight load (true unless calculate_kv_scales=True with
        # runtime recompute, which already breaks graph capture anyway).
        self._cached_k_scale: float | None = None

        # Fixed NUM_KV_SPLITS for dense decode (CUDA Graph requires constant
        # grid dims). Do not derive from attn_metadata.max_seq_len — capture
        # uses max_model_len and would bake 128 splits at 40k max_model_len.
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = min(
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph,
            self._sm_count * 2,
        )

    def _get_dense_decode_num_kv_splits(self) -> int:
        if envs.VLLM_BATCH_INVARIANT:
            return 1
        return self.max_num_kv_splits

    # ---------------- impl-level TQ buffers ----------------
    # Hadamard + Lloyd-Max centroids depend only on (kv_lora_rank, device)
    # and bit-width, not on any layer-specific state. Caching on `self`
    # avoids threading `layer` through do_kv_cache_update (which has no
    # layer argument in the unified dispatch path).
    def _ensure_buffers(self, device: torch.device) -> None:
        key = str(device)
        if not hasattr(self, "_tq_buffers"):
            self._tq_buffers: dict[str, dict] = {}
        if key in self._tq_buffers:
            return
        D = self.kv_lora_rank  # power of 2 (e.g. 512 for DeepSeek-V2/V3)
        H = _build_hadamard(D, str(device)).to(torch.float32)
        buf: dict = {"Pi": H, "PiT": H, "Pi_bf16": H.to(_BF16)}
        if not self.tq_config.key_fp8:
            cents = get_centroids(D, self.tq_config.key_mse_bits).to(
                device=device, dtype=torch.float32
            )
            c_sorted, _ = cents.sort()
            buf["centroids"] = cents
            buf["centroids_bf16"] = cents.to(_BF16)
            buf["midpoints"] = (c_sorted[:-1] + c_sorted[1:]) / 2
        if self._kpe_4bit:
            R = self.qk_rope_head_dim
            kpe_cents = get_centroids(R, 4).to(device=device, dtype=torch.float32)
            kpe_sorted, _ = kpe_cents.sort()
            buf["kpe_centroids"] = kpe_cents
            buf["kpe_centroids_bf16"] = kpe_cents.to(_BF16)
            buf["kpe_midpoints"] = (kpe_sorted[:-1] + kpe_sorted[1:]) / 2
            buf["kpe_mse_bytes"] = kpe_mse_index_bytes(R, 4)
        # P0-3: permanent v_shape_holder. `_decode_softmax_reducev_fwd` only
        # reads `v_buffer.shape[-1]` (Lv); the content is never accessed. We
        # used to `torch.empty((1, L), ...)` on every forward — wasted alloc
        # since the dtype/shape never change. Stash one and reuse forever.
        buf["v_shape_holder"] = torch.empty((1, D), device=device, dtype=_BF16)
        self._tq_buffers[key] = buf

    def _get_buffers(self, device: torch.device) -> dict:
        self._ensure_buffers(device)
        return self._tq_buffers[str(device)]

    def _kpe_decode_kwargs(self, buf: dict) -> dict:
        return {
            "kpe_fp8": self._kpe_fp8,
            "kpe_4bit": self._kpe_4bit,
            "kpe_centroids_bf16": buf.get("kpe_centroids_bf16"),
        }

    # P0-2: pool the three per-forward decode scratch tensors
    # (attn_logits, o_unrot, lse) through vLLM's WorkspaceManager so we
    # don't pay a `torch.empty` (cudaMalloc / allocator hit) on every
    # attention layer of every step. WorkspaceManager hands back views
    # into a single growing buffer that is reused across layers and
    # safely included in CUDA Graph captures. Attention layers run
    # sequentially and each layer fully consumes attn_out/lse before
    # the next layer's forward runs, so view aliasing is safe.
    def _get_decode_workspaces(
        self,
        B: int,
        H_q: int,
        num_kv_splits: int,
        L: int,
        out_dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shapes_and_dtypes = (
            ((B, H_q, num_kv_splits, L + 1), torch.float32),  # attn_logits
            ((B, H_q, L), out_dtype),  # o_unrot
            ((B, H_q), out_dtype),  # lse
        )
        if is_workspace_manager_initialized():
            attn_logits, o_unrot, lse = current_workspace_manager().get_simultaneous(
                *shapes_and_dtypes
            )
            return attn_logits, o_unrot, lse
        # Fallback for environments without an initialized WorkspaceManager
        # (e.g. unit tests). Behaves like the original code.
        return (
            torch.empty(shapes_and_dtypes[0][0], dtype=torch.float32, device=device),
            torch.empty(shapes_and_dtypes[1][0], dtype=out_dtype, device=device),
            torch.empty(shapes_and_dtypes[2][0], dtype=out_dtype, device=device),
        )

    def _reserve_decode_workspace_peak(
        self,
        max_b: int,
        max_h: int,
        num_kv_splits: int,
        device: torch.device,
        out_dtype: torch.dtype = _BF16,
    ) -> None:
        """Grow WorkspaceManager to peak decode/sparse scratch before lock.

        CUDA-graph warmup often exercises smaller batch descriptors than
        ``max_num_batched_tokens``.  Prefill on the fused sparse path calls
        ``_get_decode_workspaces(B, ...)`` with the live batch size; after
        ``lock_workspace()`` that growth fails (e.g. 1536 MB reserved vs
        1555 MB required for a slightly larger prefill chunk).
        """
        if not is_workspace_manager_initialized():
            return
        self._get_decode_workspaces(
            max_b,
            max_h,
            num_kv_splits,
            self.kv_lora_rank,
            out_dtype,
            device,
        )

    # P0-1: cache layer._k_scale as a Python float. Calling `.item()` inside
    # a CUDA Graph capture triggers a host-device sync and fails with
    # "operation not permitted when stream is capturing". The very first
    # forward runs in eager mode during warmup (before capture), so the
    # one-time `.item()` here is safe.
    def _get_layer_k_scale(self, layer: AttentionLayer) -> float:
        if self._cached_k_scale is not None:
            return self._cached_k_scale
        k_scale_t = getattr(layer, "_k_scale", None)
        if k_scale_t is None:
            self._cached_k_scale = 1.0
        else:
            self._cached_k_scale = float(k_scale_t.item())
        return self._cached_k_scale

    def _maybe_fold_pi_into_layer(self, layer) -> None:
        """K2: fold Pi into layer.W_UK_T and layer.W_UV once, eliminating two
        per-forward bf16 GEMMs (q-side rotation, V un-rotation).

        Mathematical identity (Pi is self-inverse symmetric Hadamard):
          - q_rot = q[..., :L] @ Pi  →  fold W_UK_T_new = W_UK_T @ Pi
          - o_unrot @ Pi @ W_UV = o_unrot @ W_UV_new  with W_UV_new = Pi @ W_UV
        Done in fp32 then cast back to weight dtype for precision; net effect
        is *more* accurate than the runtime bf16 rotations.
        """
        if getattr(layer, "_tq_pi_folded", False):
            return
        if not (hasattr(layer, "W_UK_T") and hasattr(layer, "W_UV")):
            layer._tq_pi_folded = False
            return  # Backend supports only the absorbed-MLA decode path.
        Pi = self._get_buffers(layer.W_UK_T.device)["Pi_bf16"]
        Pi_f32 = Pi.to(torch.float32)
        # W_UK_T: (N, P, L). Fold along last dim.
        wuk = layer.W_UK_T
        wuv = layer.W_UV
        with torch.no_grad():
            # Preserve Parameter identity: v0.27.1 registers both tensors as
            # parameters and rejects replacing them with plain tensors.
            wuk.copy_((wuk.to(torch.float32) @ Pi_f32).to(wuk.dtype))
            # W_UV: (N, L, V). Fold along middle (L): W_UV_new = Pi @ W_UV.
            wuv.copy_((Pi_f32 @ wuv.to(torch.float32)).to(wuv.dtype))
        layer._tq_pi_folded = True

    def fold_pi_at_load(self, layer) -> None:
        """Fold Pi_L into W_UK_T / W_UV once at weight load (before any bmm).

        Unified-576 uses the same W_UK / W_UV fold; q_pe still gets Pi_R inside
        the sparse kernel (RoPE prevents folding Pi_R into W_QR).
        """
        if self.tq_config.key_fp8:
            return
        self._ensure_buffers(layer.W_UK_T.device)
        self._maybe_fold_pi_into_layer(layer)

    # ---------------- store ----------------
    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,  # (N, kv_lora_rank)
        k_pe: torch.Tensor,  # (N, 1, qk_rope_head_dim) or (N, qk_rope_head_dim)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, packed_bytes) uint8
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,  # float scalar tensor
    ) -> None:
        if kv_cache.numel() == 0:
            return
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = kv_cache.device
        buf = self._get_buffers(device)

        k_pe_ = k_pe.squeeze(1) if k_pe.dim() == 3 else k_pe
        kv_c_ = kv_c_normed[:N]
        k_pe_ = k_pe_[:N]

        # CUDA TQ4 store (opt-in via VLLM_TQ_MLA_CUDA_STORE=1). Replaces the Triton
        # bf16-direct fused store with the FlashMLA warp-shuffle FWHT-512 kernel.
        # Same 292B layout + numerics (gated bit-level: codes 100%, round-trip ==
        # Triton). Only for the FWHT + 4bit-kpe + norm_correction (512/64) preset.
        if (
            os.environ.get("VLLM_TQ_MLA_CUDA_STORE") == "1"
            and not self.tq_config.key_fp8
            and self._store_fwht
            and self._kpe_4bit
            and self.tq_config.key_mse_bits == 4
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
        ):
            import flash_mla.cuda as _fm

            if not getattr(self, "_tq4_cuda_store_logged", False):
                print("[TQ4-CUDA] store branch ACTIVE", flush=True)
                self._tq4_cuda_store_logged = True
            _fm.tq4_store_fwd(
                kv_c_.contiguous(),
                k_pe_.contiguous(),
                kv_cache,
                slot_mapping[:N].to(torch.int32).contiguous(),
                buf["centroids"],
                buf["kpe_centroids"],
                buf["midpoints"],
                buf["kpe_midpoints"],
                bool(self.tq_config.norm_correction),
            )
            return

        if (
            not self.tq_config.key_fp8
            and tq_mla_fused_store_enabled()
            and self.tq_config.key_mse_bits == 4
        ):
            tq_mla_fused_kv_cache_store(
                kv_c_,
                k_pe_,
                kv_cache,
                slot_mapping,
                pi_t=buf["PiT"],
                midpoints=buf["midpoints"],
                centroids_fp32=buf["centroids"],
                mse_bits=self.tq_config.key_mse_bits,
                mse_bytes=self._mse_index_bytes,
                kv_c_bytes=self._kv_c_bytes,
                packed_bytes=self._packed_bytes,
                kpe_fp8=self._kpe_fp8,
                kpe_4bit=self._kpe_4bit,
                use_fwht=self._store_fwht,
                kpe_midpoints=buf.get("kpe_midpoints"),
                kpe_centroids_fp32=buf.get("kpe_centroids"),
                kpe_mse_bytes=buf.get("kpe_mse_bytes", 0),
                norm_correction=bool(self.tq_config.norm_correction),
            )
            return

        if self.tq_config.key_fp8:
            kv_c_packed = self._quantize_kv_c_fp8(kv_c_, k_scale)
        else:
            kv_c_packed = self._quantize_kv_c_mse(kv_c_, buf)

        if self._kpe_fp8:
            # P3-2: per-token fp8 e4m3 + fp16 scale.
            # scale = max_abs / FP8_MAX, with floor to avoid div-by-zero.
            kpe_f32 = k_pe_.to(torch.float32)
            max_abs = kpe_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = max_abs / _FP8_MAX  # (N, 1)
            inv_scale = torch.where(scale > 0, 1.0 / scale, torch.ones_like(scale))
            x = (kpe_f32 * inv_scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
            kpe_data = x.view(torch.uint8).view(N, self.qk_rope_head_dim)  # R bytes
            scale_fp16 = scale.squeeze(-1).to(torch.float16)  # (N,)
            scale_bytes = scale_fp16.view(torch.uint8).view(N, 2)
            k_pe_bytes = torch.cat([kpe_data, scale_bytes], dim=-1)  # (N, R+2)
        else:
            # bf16 k_pe as raw bytes.
            k_pe_bf16 = k_pe_.to(_BF16).contiguous()
            k_pe_bytes = k_pe_bf16.view(torch.uint8).view(N, -1)

        combined = torch.cat([kv_c_packed, k_pe_bytes], dim=-1)  # (N, packed_bytes)
        assert combined.shape[-1] == self._packed_bytes, (
            f"combined bytes {combined.shape[-1]} != expected {self._packed_bytes}"
        )

        cache_flat = kv_cache.view(-1, self._packed_bytes)
        slot = slot_mapping.flatten().to(torch.int64)
        # Replace `if not bool(valid.all()): combined = combined[valid]` with
        # a mask-based variant that has no host sync and fixed shapes:
        #   * clamp invalid slots (-1) to 0
        #   * for invalid rows, gather current slot-0 bytes and write them
        #     back unchanged (torch.where); for valid rows, write `combined`.
        # This avoids corrupting slot 0 with zeros and keeps every op CUDA-graph
        # capturable. In the common all-valid case the where still picks
        # `combined` row-wise, so correctness is preserved.
        valid = slot >= 0
        slot = slot.clamp(min=0)
        current = cache_flat.index_select(0, slot)  # (N, packed_bytes) uint8
        new = torch.where(valid.view(-1, 1), combined, current)
        cache_flat.index_copy_(0, slot, new)

    def _quantize_kv_c_fp8(
        self, kv_c: torch.Tensor, k_scale: torch.Tensor
    ) -> torch.Tensor:
        """FP8 path: kv_c / k_scale → fp8_e4m3fn → reinterpret as bytes.

        Output shape: (N, kv_lora_rank) uint8.
        """
        scale = k_scale.to(torch.float32)
        inv_scale = torch.where(scale > 0, 1.0 / scale, torch.ones_like(scale))
        x = kv_c.to(torch.float32) * inv_scale
        x = x.clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
        return x.view(torch.uint8)  # (N, kv_lora_rank)

    def _quantize_kv_c_mse(self, kv_c: torch.Tensor, buf: dict) -> torch.Tensor:
        """MSE path: Hadamard rotation + Lloyd-Max bucketize + pack + vec_norm.

        Output shape: (N, key_packed_size) uint8 where
            key_packed_size = mse_index_bytes + 2 (fp16 vec_norm)
        """
        N, D = kv_c.shape
        bits = self.tq_config.key_mse_bits
        kv_c_f = kv_c.to(torch.float32)
        norms = kv_c_f.norm(dim=1, keepdim=True)  # (N, 1)
        safe = norms.clamp(min=1e-8)
        x_hat = kv_c_f / safe
        # Post-rotation: y = x_hat @ PiT. Pi is symmetric (Hadamard), so PiT == Pi.
        pi_t = buf["PiT"].to(torch.float32)
        y = x_hat @ pi_t  # (N, D)
        # Bucketize to midpoints -> integer index in [0, 2**bits).
        idx = torch.bucketize(y.contiguous(), buf["midpoints"])  # (N, D) int64
        idx = idx.clamp(max=(1 << bits) - 1)
        packed_idx = _pack_bits_rows(idx, bits=bits)  # (N, mse_index_bytes)
        vec_norm_f32 = norms.squeeze(1)  # (N,)
        if self.tq_config.norm_correction:
            # P6-3: fold norm_correction into store so decode skips 512-dim sum/sqrt.
            # eff_scale = vec_norm / ||centroid(idx)||_2 (same 2B slot as vec_norm).
            y_hat_raw = buf["centroids_bf16"][idx.to(torch.int64)]
            c_norm = y_hat_raw.to(torch.float32).norm(dim=-1).clamp(min=1e-8)
            scale_f32 = vec_norm_f32 / c_norm
        else:
            scale_f32 = vec_norm_f32
        scale_fp16 = scale_f32.to(torch.float16)
        scale_bytes = scale_fp16.view(torch.uint8).view(N, 2)
        return torch.cat([packed_idx, scale_bytes], dim=-1)  # (N, key_packed_size)

    # ---------------- prefix-caching prefill ----------------
    # P3-3b: upstream's `ops.gather_and_maybe_dequant_cache` (C++) only knows
    # {auto, fp8*} dtypes and raises a TORCH_CHECK on `turboquant_*`. Override the
    # context-gather step so chunked_prefill + enable_prefix_caching works.
    #
    # Strategy: reproduce the C++ kernel's (token_id → block_id, slot_id)
    # mapping in Python, gather packed rows from the uint8 cache, then reuse
    # our fused dequant kernel (same one forward_mqa uses).
    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        k_scale: torch.Tensor,
    ):
        from vllm.model_executor.layers.attention.mla_attention import (
            merge_attn_states,
        )
        from vllm.platforms import current_platform

        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.chunked_context is not None

        use_fp8_prefill = prefill_metadata.q_data_type == current_platform.fp8_dtype()
        if use_fp8_prefill:
            q = q.to(prefill_metadata.q_data_type)

        device = kv_c_and_k_pe_cache.device
        buf = self._get_buffers(device)
        L = self.kv_lora_rank
        R = self.qk_rope_head_dim
        num_blocks, block_size, packed_bytes = kv_c_and_k_pe_cache.shape
        assert packed_bytes == self._packed_bytes

        cache_flat = kv_c_and_k_pe_cache.view(num_blocks * block_size, packed_bytes)

        output = None
        output_lse = None
        chunked_context = prefill_metadata.chunked_context
        assert chunked_context is not None
        assert prefill_metadata.prefill_backend is not None
        workspace = chunked_context.workspace

        for chunk in chunked_context.chunks:
            toks = chunk.num_context_tokens
            if toks <= 0:
                continue

            # Replicate C++ kernel address math:
            #   batch_id      = token_to_seq[token_id]
            #   batch_start   = cu_seq_lens[batch_id]
            #   batch_offset  = token_id - batch_start + seq_starts[batch_id]
            #   block_id      = block_table[batch_id, batch_offset // block_size]
            #   slot_id       = batch_offset % block_size
            num_tokens = toks
            token_to_seq = chunk.token_to_seq[:num_tokens].to(torch.int64)
            cu_seq_lens = chunk.cu_seq_lens.to(torch.int64)
            seq_starts_t = chunk.starts
            block_table = prefill_metadata.block_table[chunk.request_slice].to(
                torch.int64
            )

            # All shape-dependent gather math stays on-device (CG-compatible).
            batch_ids = token_to_seq  # (num_tokens,)
            batch_starts = cu_seq_lens.index_select(0, batch_ids)
            tok_ids = torch.arange(num_tokens, device=device, dtype=torch.int64)
            batch_offsets = tok_ids - batch_starts
            if seq_starts_t is not None:
                seq_starts_i = seq_starts_t.to(torch.int64).index_select(0, batch_ids)
                batch_offsets = batch_offsets + seq_starts_i
            block_table_ids = batch_offsets // block_size
            slot_ids = batch_offsets % block_size
            bt_stride = block_table.stride(0)
            bt_flat_idx = batch_ids * bt_stride + block_table_ids
            block_ids = block_table.view(-1).index_select(0, bt_flat_idx)
            row_ids = block_ids * block_size + slot_ids  # (num_tokens,)

            # Gather packed bytes → (num_tokens, packed_bytes) uint8.
            gathered = cache_flat.index_select(0, row_ids).contiguous()

            # Shape to what fused_mla_dequant_mse expects:
            #   cache_view: (nb=toks, bs=1, packed_bytes)
            #   out_view:   (nb=toks, bs=1, L+R) bf16
            cache_view = gathered.view(num_tokens, 1, packed_bytes)
            out_view = workspace[:num_tokens].view(num_tokens, 1, L + R)

            if self.tq_config.key_fp8:
                # FP8 kv_c: reinterpret bytes as fp8_e4m3 → bf16 * k_scale.
                kv_c_fp8 = (
                    cache_view[..., : self._kv_c_bytes].contiguous().view(_FP8_DTYPE)
                )
                scale_f32 = k_scale.to(torch.float32)
                out_view[..., :L] = (kv_c_fp8.to(torch.float32) * scale_f32).to(_BF16)
                self._write_kpe_into_workspace(cache_view, out_view, L, R)
            else:
                # MSE path: fused kernel writes (y*vec_norm) + k_pe, then
                # we apply inverse Hadamard as the final bf16 GEMM.
                if (
                    os.environ.get("VLLM_TQ_MLA_CUDA_DEQUANT") == "1"
                    and self._kpe_4bit
                    and L == 512
                    and R == 64
                ):
                    import flash_mla.cuda as _fm

                    if not getattr(self, "_tq4_cuda_deq_dense_logged", False):
                        print(
                            "[TQ4-CUDA] prefill dense dequant branch ACTIVE", flush=True
                        )
                        self._tq4_cuda_deq_dense_logged = True
                    _fm.tq4_dequant_fwd(
                        cache_view.contiguous(),
                        buf["centroids_bf16"],
                        buf["kpe_centroids_bf16"],
                        out_view,
                    )
                else:
                    fused_mla_dequant_mse(
                        cache_view,
                        buf["centroids_bf16"],
                        out_view,
                        L=L,
                        R=R,
                        mse_bits=self.tq_config.key_mse_bits,
                        mse_bytes=self._mse_index_bytes,
                        kv_c_bytes=self._kv_c_bytes,
                        norm_correction=bool(self.tq_config.norm_correction),
                        **self._kpe_decode_kwargs(buf),
                    )
                kvc = out_view[..., :L].view(num_tokens, L)
                out_view[..., :L] = (kvc @ buf["Pi_bf16"]).view(num_tokens, 1, L)

            # Extract kv_c_normed / k_pe from workspace (shape matches base).
            kv_c_normed = workspace[:toks][..., :L]
            _kv_b_proj_w_dtype = (
                self.kv_b_proj.weight.dtype
                if hasattr(self.kv_b_proj, "weight")
                else self.kv_b_proj.params_dtype
            )
            if (
                use_fp8_prefill or _kv_b_proj_w_dtype != current_platform.fp8_dtype()
            ) and _kv_b_proj_w_dtype != torch.uint8:
                kv_c_normed = kv_c_normed.to(_kv_b_proj_w_dtype)

            k_pe = workspace[:toks][..., L:].unsqueeze(1)
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            if use_fp8_prefill:
                kv_nope = kv_nope.to(prefill_metadata.q_data_type)
                k_pe = k_pe.to(prefill_metadata.q_data_type)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = (
                prefill_metadata.prefill_backend.run_prefill_context_chunk(
                    chunk=chunk,
                    q=q[chunk.token_slice],
                    k=k,
                    v=v,
                )
            )
            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    # ---------------- decode ----------------
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        # (num_blocks, block_size, packed_bytes) uint8
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        # FP8 path: K3 fused FP8 decode kernel (no bf16 staging). FP8 has
        # no Hadamard rotation so the K2 weight fold must NOT run on this
        # path. Set _TQ_FP8_WORKSPACE=1 to fall back to the legacy staging
        # flow (kept for regression comparison).
        if self.tq_config.key_fp8:
            if os.environ.get("_TQ_FP8_WORKSPACE") == "1":
                return self._forward_mqa_fp8_workspace(
                    q, kv_c_and_k_pe_cache, attn_metadata, layer
                )
            return self._forward_mqa_fp8_fused(
                q, kv_c_and_k_pe_cache, attn_metadata, layer
            )

        # K2: Pi was folded into W_UK_T / W_UV at load time (fold_pi_at_load).

        # ----- MSE (3/4-bit) fused path -----
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        assert isinstance(q, torch.Tensor)

        device = kv_c_and_k_pe_cache.device
        buf = self._get_buffers(device)
        L = self.kv_lora_rank
        R = self.qk_rope_head_dim
        B = q.shape[0]
        H_q = q.shape[1]

        # K2-a: q is already in rotated space because Pi has been folded into
        # layer.W_UK_T (the projection that produced q[..., :L]). Skip runtime
        # rotation.
        q_rot = q

        # 2) Allocate stage1 output (logits + LSE) and final output buffers.
        decode_md = attn_metadata.decode
        num_kv_splits = self._get_dense_decode_num_kv_splits()

        # P0-2: pool scratch buffers via WorkspaceManager instead of
        # allocating fresh tensors every forward.
        attn_logits, o_unrot, lse = self._get_decode_workspaces(
            B,
            H_q,
            num_kv_splits,
            L,
            q.dtype,
            device,
        )

        # 3) Fused stage1 directly on packed uint8 cache.
        fused_mla_tq_decode_stage1(
            q_rot,
            kv_c_and_k_pe_cache,
            buf["centroids_bf16"],
            attn_logits,
            decode_md.block_table,
            decode_md.seq_lens,
            sm_scale=self.scale,
            page_size=kv_c_and_k_pe_cache.shape[1],
            L=L,
            R=R,
            mse_bits=self.tq_config.key_mse_bits,
            mse_bytes=self._mse_index_bytes,
            kv_c_bytes=self._kv_c_bytes,
            norm_correction=bool(self.tq_config.norm_correction),
            **self._kpe_decode_kwargs(buf),
            num_kv_splits=num_kv_splits,
            logit_cap=0.0,
        )

        # 4) Stage2 reduce. v_buffer is only read for its last-dim size (Lv);
        #    pass the permanent placeholder cached in `buf` (P0-3).
        v_shape_holder = buf["v_shape_holder"]
        _decode_softmax_reducev_fwd(
            attn_logits,
            q_rot,
            o_unrot,
            lse,
            v_shape_holder,
            decode_md.seq_lens,
            num_kv_splits,
        )

        # K2-b: leave o in rotated kv_c space; Pi has been folded into
        # layer.W_UV so the downstream BMM (`x @ W_UV`) un-rotates implicitly.
        o = o_unrot.view(B, H_q, L)

        return o, lse

    def _forward_mqa_fp8_fused(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """K3: FP8 fused decode — `fused_mla_tq_decode_stage1(key_fp8=True)`
        reads the fp8 cache directly inside the inner KV loop, eliminating
        the bf16 workspace materialization that `_forward_mqa_fp8_workspace`
        used to do. The layer's k_scale is passed as a kernel constexpr.
        """
        assert attn_metadata.decode is not None
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        assert isinstance(q, torch.Tensor)

        device = kv_c_and_k_pe_cache.device
        L = self.kv_lora_rank
        R = self.qk_rope_head_dim
        B = q.shape[0]
        H_q = q.shape[1]

        decode_md = attn_metadata.decode
        num_kv_splits = self._get_dense_decode_num_kv_splits()

        # P0-2: pool scratch buffers via WorkspaceManager.
        attn_logits, o_unrot, lse = self._get_decode_workspaces(
            B,
            H_q,
            num_kv_splits,
            L,
            q.dtype,
            device,
        )

        # Empty centroid placeholder — KEY_FP8 branch never reads it; Triton
        # still requires a bf16 1-D tensor argument.
        buf = self._get_buffers(device)
        centroids_unused = buf.get(
            "_fp8_centroid_placeholder",
            torch.empty(0, device=device, dtype=_BF16),
        )
        buf["_fp8_centroid_placeholder"] = centroids_unused

        # P0-1: cached float, no `.item()` on the hot path → CUDA Graph safe.
        k_scale = self._get_layer_k_scale(layer)

        fused_mla_tq_decode_stage1(
            q,
            kv_c_and_k_pe_cache,
            centroids_unused,
            attn_logits,
            decode_md.block_table,
            decode_md.seq_lens,
            sm_scale=self.scale,
            page_size=kv_c_and_k_pe_cache.shape[1],
            L=L,
            R=R,
            mse_bits=0,
            mse_bytes=0,
            kv_c_bytes=L,
            norm_correction=False,
            **self._kpe_decode_kwargs(buf),
            key_fp8=True,
            k_scale=k_scale,
            num_kv_splits=num_kv_splits,
            logit_cap=0.0,
        )

        # P0-3: reuse permanent v_shape_holder (only shape is read).
        v_shape_holder = buf["v_shape_holder"]
        _decode_softmax_reducev_fwd(
            attn_logits,
            q,
            o_unrot,
            lse,
            v_shape_holder,
            decode_md.seq_lens,
            num_kv_splits,
        )
        o = o_unrot.view(B, H_q, L)
        return o, lse

    def _forward_mqa_fp8_workspace(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Legacy FP8 path — dequant fp8 → bf16 active-block workspace, then
        call upstream `decode_attention_fwd_grouped`. Kept reachable via the
        env `_TQ_FP8_WORKSPACE=1` for regression comparison against
        `_forward_mqa_fp8_fused`.
        """
        device = kv_c_and_k_pe_cache.device
        L = self.kv_lora_rank
        R = self.qk_rope_head_dim
        num_blocks, block_size, _ = kv_c_and_k_pe_cache.shape

        active_block_ids = self._collect_active_blocks(attn_metadata, num_blocks)
        if active_block_ids is None or active_block_ids.numel() == 0:
            q_for_count = q[0] if isinstance(q, tuple) else q
            if q_for_count.shape[0] == 0:
                empty_ws = torch.empty(
                    (num_blocks, block_size, L + R), dtype=_BF16, device=device
                )
                return super().forward_mqa(q, empty_ws, attn_metadata, layer)
            raise RuntimeError(
                "TRITON_MLA_TURBOQUANT (FP8) requires active block metadata."
            )

        n_active = int(active_block_ids.shape[0])
        sub_ws = torch.empty((n_active, block_size, L + R), dtype=_BF16, device=device)
        sub_cache = kv_c_and_k_pe_cache.index_select(0, active_block_ids)

        self._dequant_kv_c_fp8(sub_cache, layer, sub_ws)
        self._write_kpe_into_workspace(sub_cache, sub_ws, L, R)

        workspace = torch.empty(
            (num_blocks, block_size, L + R), dtype=_BF16, device=device
        )
        workspace.index_copy_(0, active_block_ids, sub_ws)
        return super().forward_mqa(q, workspace, attn_metadata, layer)

    @staticmethod
    def _collect_active_blocks(
        attn_metadata: MLACommonMetadata, num_blocks: int
    ) -> torch.Tensor | None:
        """Return block ids referenced by the current batch.

        Deliberately does NOT call torch.unique — that op has a
        data-dependent output shape which blocks CUDA-graph capture and
        forces a CPU sync. Duplicate ids in the return value are fine:
        dequanting the same block twice into the workspace is idempotent.

        Pulls from whichever sub-metadata is populated (decode / prefill /
        chunked_prefill). Returns None when no block_table is available.
        """
        ids: list[torch.Tensor] = []
        for attr in ("decode", "prefill", "chunked_prefill"):
            sub = getattr(attn_metadata, attr, None)
            if sub is None:
                continue
            bt = getattr(sub, "block_table", None)
            if bt is None or not isinstance(bt, torch.Tensor) or bt.numel() == 0:
                continue
            ids.append(bt.flatten())
        # Fallback: top-level block_table if backend exposes one.
        bt_top = getattr(attn_metadata, "block_table", None)
        if isinstance(bt_top, torch.Tensor) and bt_top.numel() > 0:
            ids.append(bt_top.flatten())
        if not ids:
            return None
        flat = torch.cat(ids).to(torch.int64)
        # Clamp out-of-range ids to 0 rather than filtering (fixed shape).
        # Out-of-range is a fault condition that shouldn't happen in
        # practice; clamping keeps the op CG-safe while turning any
        # stray id into a harmless re-dequant of block 0.
        flat = flat.clamp(min=0, max=num_blocks - 1)
        if flat.numel() == 0:
            return None
        return flat

    def _dequant_kv_c_fp8(self, cache, layer, workspace) -> None:
        L = self.kv_lora_rank
        kv_c_fp8 = cache[..., : self._kv_c_bytes].contiguous().view(_FP8_DTYPE)
        scale = layer._k_scale.to(torch.float32)
        workspace[..., :L] = (kv_c_fp8.to(torch.float32) * scale).to(_BF16)

    def _dequant_kv_c_mse(self, cache, buf: dict, workspace) -> None:
        """Fused Triton dequant: unpack + centroid gather + (optional) norm
        correction + vec_norm multiply, plus inline k_pe bf16 copy.

        Writes workspace[..., :L] = (y_normed * vec_norm), un-rotated.
        Caller must follow with workspace[..., :L] @= Pi (bf16 GEMM) to
        finish the inverse Hadamard. Also writes workspace[..., L:] = k_pe.

        Set _TQ_USE_TORCH_DEQUANT=1 to fall back to the pure-PyTorch path
        (used for numerical equivalence regression).
        """
        if os.environ.get("_TQ_USE_TORCH_DEQUANT") == "1":
            self._dequant_kv_c_mse_torch_ref(cache, buf, workspace)
            return

        L = self.kv_lora_rank
        R = self.qk_rope_head_dim
        bits = self.tq_config.key_mse_bits

        # Kernel writes (y_normed * vec_norm) into workspace[..., :L] and
        # k_pe into workspace[..., L:]. Caller (e.g. _compute_prefill_context)
        # is responsible for the inverse Hadamard `@ Pi`.
        fused_mla_dequant_mse(
            cache,
            buf["centroids_bf16"],
            workspace,
            L=L,
            R=R,
            mse_bits=bits,
            mse_bytes=self._mse_index_bytes,
            kv_c_bytes=self._kv_c_bytes,
            norm_correction=bool(self.tq_config.norm_correction),
            **self._kpe_decode_kwargs(buf),
        )

    def _dequant_kv_c_mse_torch_ref(self, cache, buf: dict, workspace) -> None:
        """Pure-PyTorch reference path. Pre-Step5 implementation kept here
        for numerical regression: set _TQ_USE_TORCH_DEQUANT=1 to use it.
        Also writes workspace[..., L:] = k_pe so the contract matches the
        fused kernel.
        """
        L = self.kv_lora_rank
        bits = self.tq_config.key_mse_bits
        idx_bytes = cache[..., : self._mse_index_bytes].contiguous()
        norms_bytes = cache[
            ..., self._mse_index_bytes : self._mse_index_bytes + 2
        ].contiguous()

        # Unpack indices. int32 is enough for counts up to ~2B and halves
        # the temporary footprint vs int64.
        idx = _unpack_bits_rows(idx_bytes, bits=bits, D=L).to(torch.int32)

        y_hat = buf["centroids_bf16"][idx.to(torch.int64)]  # (nb, bs, L) bf16
        token_scale = norms_bytes.view(torch.float16).to(torch.float32).unsqueeze(-1)
        kv_c_recovered = (y_hat.to(torch.float32) * token_scale).to(_BF16)

        workspace[..., :L] = kv_c_recovered
        # Match fused kernel contract: also write k_pe.
        self._write_kpe_into_workspace(cache, workspace, L, self.qk_rope_head_dim)

    # ------------------------------------------------------------------
    # k_pe writer (handles both bf16 and fp8 layouts).
    # P3-2: when VLLM_TQ_KPE_FP8=1, k_pe is stored as R fp8 e4m3 bytes
    # followed by 2 bytes per-token fp16 scale; otherwise raw bf16 (2*R bytes).
    # ------------------------------------------------------------------
    def _write_kpe_into_workspace(self, cache, workspace, L, R) -> None:
        if self._kpe_fp8:
            # cache layout: [..., kv_c_bytes : kv_c_bytes + R] = fp8 data
            #               [kv_c_bytes + R : kv_c_bytes + R + 2] = fp16 scale
            kpe_fp8 = (
                cache[..., self._kv_c_bytes : self._kv_c_bytes + R]
                .contiguous()
                .view(_FP8_DTYPE)
            )
            # (..., 2) bytes → (..., 1) fp16 → (..., 1) bf16 broadcast scalar
            scale_fp16 = (
                cache[..., self._kv_c_bytes + R : self._kv_c_bytes + R + 2]
                .contiguous()
                .view(torch.float16)
            )
            scale_bf = scale_fp16.to(_BF16)  # (..., 1)
            workspace[..., L:] = kpe_fp8.to(_BF16) * scale_bf
        else:
            workspace[..., L:] = cache[..., self._kv_c_bytes :].contiguous().view(_BF16)


# =============================================================================
# Sparse DSA path: Indexer topk + TurboQuant packed MLA cache.
# Sparse DSA path: indexer top-k + TurboQuant packed MLA cache.
# Prefill: gather+dedup+flash_mla_sparse_fwd; decode: fused 2-stage sparse.
# =============================================================================


@dataclass
class TritonMLATurboQuantSparseMetadata(MLACommonMetadata):
    """MLACommonMetadata plus sparse top-k indexing fields."""

    req_id_per_token: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    block_size: int = 64
    topk_tokens: int = 2048
    prefill_max_seq_len: int = 0


class TritonMLATurboQuantSparseMetadataBuilder(TritonMLATurboQuantMetadataBuilder):
    """Build MLA prefill/decode metadata and sparse top-k index fields."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.metadata_cls = TritonMLATurboQuantSparseMetadata
        # DCP+MTP fix: the base helper resets reorder_batch_threshold back to 1
        # when decode_context_parallel_size>1 unless supports_dcp_with_varlen is
        # set. Without this, MTP spec-decode (uniform query_len = 1+num_spec) is
        # forced to threshold=1 under DCP and full-cudagraph decode capture asserts
        # `max_query_len <= reorder_batch_threshold`. The sparse TQ decode handles
        # the flattened uniform multi-token queries, so keep the spec threshold.
        self._init_reorder_batch_threshold(
            1, supports_spec_as_decode=True, supports_dcp_with_varlen=True
        )
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonMLATurboQuantSparseMetadata:
        saved_cls = self.metadata_cls
        self.metadata_cls = MLACommonMetadata
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        self.metadata_cls = saved_cls

        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )

        sparse_fields = {
            f.name: getattr(base, f.name) for f in fields(MLACommonMetadata)
        }
        return TritonMLATurboQuantSparseMetadata(
            **sparse_fields,
            req_id_per_token=self.req_id_per_token_buffer[:num_tokens],
            block_table=common_attn_metadata.block_table_tensor,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            prefill_max_seq_len=base.max_seq_len if base.num_prefills else 0,
        )


class TritonMLATurboQuantSparseBackend(TritonMLATurboQuantBackend):
    """TurboQuant MLA backend with DSA (sparse top-k) support."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_TURBOQUANT_SPARSE"

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # flash_mla_sparse_fwd requires Hopper/Blackwell (same as FlashMLA sparse).
        return capability.major in (9, 10)

    @staticmethod
    def get_impl_cls() -> type["TritonMLATurboQuantSparseImpl"]:
        return TritonMLATurboQuantSparseImpl

    @staticmethod
    def get_builder_cls() -> type[TritonMLAMetadataBuilder]:
        return TritonMLATurboQuantSparseMetadataBuilder


class TritonMLATurboQuantSparseImpl(
    SparseMLACommonImpl[TritonMLATurboQuantSparseMetadata],
    TritonMLATurboQuantImpl,
):
    """Sparse MLA over a TurboQuant byte-packed cache.

    Prefill and decode both run ``forward_mqa``.  Prefill uses gather+dedup+
    ``flash_mla_sparse_fwd``; decode uses fused 2-stage inline dequant+attention.
    Production ``turboquant_*_nc`` paths are enabled by ``--kv-cache-dtype`` —
    no env toggles required.
    """

    supports_hybrid_prefill: ClassVar[bool] = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: object | None = None,
        **mla_args,
    ) -> None:
        mla_args.pop("topk_indices_buffer", None)
        TritonMLATurboQuantImpl.__init__(
            self,
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            **mla_args,
        )
        self.topk_indices_buffer = (
            indexer.topk_indices_buffer  # type: ignore[attr-defined]
            if indexer is not None
            else topk_indices_buffer
        )
        self.masked_mha_available = False
        assert self.topk_indices_buffer is not None
        self.softmax_scale = float(scale)
        self.prefill_padding = (
            128 if current_platform.is_device_capability_family(100) else 64
        )
        from vllm.config import get_current_vllm_config
        from vllm.distributed import get_dcp_group

        vllm_config = get_current_vllm_config()
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        from vllm.v1.worker.workspace import current_workspace_manager

        # DCP: MLAAttention all-gathers Q over heads then merges per-rank
        # partials via LSE (cp_lse_ag_out_rs / dcp_a2a_lse_reduce). Indexer
        # stays FP8 under DCP; this wires the TQ *main* KV sparse path.
        _pc = vllm_config.parallel_config
        self.dcp_world_size = _pc.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.dcp_interleave_size = _pc.cp_kv_cache_interleave_size

        topk = self.topk_indices_buffer.shape[1]
        # Decode always uses fused 2-stage; _TQ_SPARSE_WORKSPACE only switches
        # prefill to gather+flash+dedup.  Reserve peak 2-stage scratch either way.
        # Under DCP, Q arrives with num_heads * dcp_world_size after all-gather.
        max_h = (
            self.prefill_padding
            if os.environ.get("VLLM_TQ_SPARSE_Q_HEAD_PAD", "0") == "1"
            else num_heads
        )
        if self.dcp_world_size > 1:
            max_h = max_h * self.dcp_world_size
        reserve_b = max_tokens
        if tq_mla_sparse_adaptive_enabled():
            # Adaptive caps B*splits at ~sm_count*8 (see tq_mla_sparse_split_count),
            # so peak f32 scratch is max(sm_count*8, mnbt) * H * 1 * (L+1) instead
            # of mnbt * H * 64.  Reserve with splits=1 so KV profiling matches tqa.
            target = self._sm_count * 8
            reserve_b = max(target, max_tokens)
            num_kv_splits = 1
        else:
            num_kv_splits = tq_mla_sparse_split_count(topk, self._sm_count * 2)
            if envs.VLLM_BATCH_INVARIANT:
                num_kv_splits = 1
            num_kv_splits = max(num_kv_splits, self._get_dense_decode_num_kv_splits())
        # Grow the shared WorkspaceManager blob to decode peak BEFORE taking
        # q_concat_buffer views. Otherwise a later blob grow can reallocate the
        # underlying storage and leave cached q views dangling.
        self._reserve_decode_workspace_peak(
            reserve_b,
            max_h,
            num_kv_splits,
            self.topk_indices_buffer.device,
        )
        q_concat_shape = (max_tokens, num_heads, head_size)
        (self.q_concat_buffer,) = current_workspace_manager().get_simultaneous(
            (q_concat_shape, torch.bfloat16),
        )
        # Legacy prefill pool: do not preallocate at model load time. Growing
        # the workspace here steals memory from KV profiling and sampler warmup.
        # It grows via _get_global_topk_dequant_workspace on
        # first forward, before lock_workspace() after cudagraph capture.
        self._max_decode_tokens = max_tokens
        self._sparse_b_seqlen: torch.Tensor | None = None
        self._sparse_topk_val: int | None = None

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        return TritonMLATurboQuantImpl.do_kv_cache_update(
            self,
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    def _get_topk_dequant_workspace(
        self,
        num_slots: int,
        L: int,
        R: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Device-global bf16 workspace for legacy ``_gather_dequant_topk_workspace``.

        Shared across all attention layers on this GPU; safe because layer
        forwards are sequential and each layer fully consumes the view first.
        """
        assert self.topk_indices_buffer is not None
        return _get_global_topk_dequant_workspace(
            num_slots,
            L,
            R,
            device,
            self.topk_indices_buffer.numel(),
        )

    def _gather_dequant_topk_workspace(
        self,
        packed_cache: torch.Tensor,
        global_topk: torch.Tensor,
        layer: AttentionLayer,
        *,
        apply_pi_on_k: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Dequant top-k packed slots into bf16 workspace.

        Returns ``(dequant_kv, remapped_local_topk)``.  When P3 dedup is off,
        ``remapped_local_topk`` is None and workspace has shape ``(T*K, 1, L+R)``.
        When dedup is on, workspace has ``(U, 1, L+R)`` and remapped indices
        point into those ``U`` unique global slots.
        """
        num_tokens, topk = global_topk.shape
        device = packed_cache.device
        buf = self._get_buffers(device)
        L = self.kv_lora_rank
        R = self.qk_rope_head_dim
        packed_bytes = self._packed_bytes
        page_size = packed_cache.shape[1]

        remapped_local: torch.Tensor | None = None
        gather_topk = global_topk
        num_out_slots = num_tokens * topk
        if _sparse_topk_dedup_enabled():
            num_cache_slots = packed_cache.shape[0] * packed_cache.shape[1]
            unique_slots, remapped_local, num_unique = _dedup_global_topk(
                global_topk, num_cache_slots=num_cache_slots
            )
            if num_unique > 0:
                gather_topk = unique_slots.unsqueeze(1)
                num_out_slots = num_unique

        out_view = self._get_topk_dequant_workspace(num_out_slots, L, R, device)

        use_fused_topk_gather = not self.tq_config.key_fp8

        def _pytorch_gather_cache_view(topk_for_gather: torch.Tensor) -> torch.Tensor:
            cache_flat = packed_cache.view(-1, packed_bytes)
            valid = topk_for_gather >= 0
            safe = torch.where(
                valid, topk_for_gather, torch.zeros_like(topk_for_gather)
            )
            g_rows, g_cols = topk_for_gather.shape
            gathered = cache_flat[safe.reshape(-1)].view(g_rows, g_cols, packed_bytes)
            return gathered.view(g_rows * g_cols, 1, packed_bytes)

        if self.tq_config.key_fp8:
            k_scale = self._get_layer_k_scale(layer)
            cache_view = _pytorch_gather_cache_view(gather_topk)
            kv_c_fp8 = cache_view[..., : self._kv_c_bytes].contiguous().view(_FP8_DTYPE)
            out_view[..., :L] = (kv_c_fp8.to(torch.float32) * k_scale).to(_BF16)
            self._write_kpe_into_workspace(cache_view, out_view, L, R)
        elif use_fused_topk_gather:
            # CUDA TQ4 sparse gather-dequant (opt-in VLLM_TQ_MLA_CUDA_DEQUANT=1).
            # dedup stays in Python (_dedup_global_topk); the CUDA kernel only
            # replaces the Triton dequant pass over the deduped unique slots.
            if (
                os.environ.get("VLLM_TQ_MLA_CUDA_DEQUANT") == "1"
                and self._kpe_4bit
                and L == 512
                and R == 64
            ):
                import flash_mla.cuda as _fm

                if not getattr(self, "_tq4_cuda_deq_sparse_logged", False):
                    print("[TQ4-CUDA] sparse gather-dequant branch ACTIVE", flush=True)
                    self._tq4_cuda_deq_sparse_logged = True
                _fm.tq4_sparse_gather_dequant_fwd(
                    packed_cache,
                    gather_topk.reshape(-1).to(torch.int32).contiguous(),
                    buf["centroids_bf16"],
                    buf["kpe_centroids_bf16"],
                    out_view,
                )
            else:
                fused_mla_sparse_topk_gather_dequant_mse(
                    packed_cache,
                    gather_topk,
                    buf["centroids_bf16"],
                    out_view,
                    page_size=page_size,
                    L=L,
                    R=R,
                    mse_bits=self.tq_config.key_mse_bits,
                    mse_bytes=self._mse_index_bytes,
                    kv_c_bytes=self._kv_c_bytes,
                    norm_correction=bool(self.tq_config.norm_correction),
                    **self._kpe_decode_kwargs(buf),
                )
            if apply_pi_on_k:
                kvc = out_view[..., :L].reshape(-1, L)
                out_view[..., :L] = (kvc @ buf["Pi_bf16"]).view_as(out_view[..., :L])
        else:
            cache_view = _pytorch_gather_cache_view(gather_topk)
            fused_mla_dequant_mse(
                cache_view,
                buf["centroids_bf16"],
                out_view,
                L=L,
                R=R,
                mse_bits=self.tq_config.key_mse_bits,
                mse_bytes=self._mse_index_bytes,
                kv_c_bytes=self._kv_c_bytes,
                norm_correction=bool(self.tq_config.norm_correction),
                **self._kpe_decode_kwargs(buf),
            )
            if apply_pi_on_k:
                kvc = out_view[..., :L].reshape(-1, L)
                out_view[..., :L] = (kvc @ buf["Pi_bf16"]).view_as(out_view[..., :L])

        return out_view, remapped_local

    def _sparse_flash_mla(
        self,
        q: torch.Tensor,
        dequant_kv: torch.Tensor,
        global_topk: torch.Tensor,
        *,
        remapped_local_topk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens, topk = global_topk.shape
        device = q.device
        # Under DCP, Q was all-gathered over heads before forward_mqa, so
        # self.num_heads (per-rank) is wrong; use the head count in q.
        actual_num_heads = q.shape[1]

        if remapped_local_topk is not None:
            local_topk = remapped_local_topk
        else:
            local_topk = (
                torch.arange(topk, device=device, dtype=torch.int32)
                .unsqueeze(0)
                .expand(num_tokens, topk)
                + torch.arange(num_tokens, device=device, dtype=torch.int32).unsqueeze(
                    1
                )
                * topk
            )
            local_topk = torch.where(
                global_topk >= 0, local_topk, torch.full_like(local_topk, -1)
            )

        if actual_num_heads % self.prefill_padding != 0:
            padded_num_heads = (
                (actual_num_heads + self.prefill_padding - 1) // self.prefill_padding
            ) * self.prefill_padding
            q_padded = q.new_empty((q.shape[0], padded_num_heads, q.shape[2]))
            q_padded[:, :actual_num_heads, :] = q
            q = q_padded

        # flash_mla_sparse_fwd -> (output, max_logits, lse); LSE needed for DCP.
        output, _max_logits, lse = flash_mla_sparse_fwd(
            q,
            dequant_kv,
            local_topk.view(num_tokens, 1, topk),
            self.softmax_scale,
            d_v=self.kv_lora_rank,
        )
        return output[:, :actual_num_heads, :], lse[:, :actual_num_heads]

    def _get_sparse_b_seqlen(
        self, batch: int, topk: int, device: torch.device
    ) -> torch.Tensor:
        if (
            self._sparse_b_seqlen is None
            or self._sparse_topk_val != topk
            or self._sparse_b_seqlen.device != device
            or self._sparse_b_seqlen.shape[0] < batch
        ):
            self._sparse_topk_val = topk
            self._sparse_b_seqlen = torch.full(
                (self._max_decode_tokens,),
                topk,
                dtype=torch.int32,
                device=device,
            )
        return self._sparse_b_seqlen[:batch]

    def _forward_mqa_sparse_fused(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        global_topk: torch.Tensor,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused sparse decode: inline dequant over top-k global slots."""
        # q is commonly a view into q_concat_buffer, and decode stage1 writes
        # attn_logits from the same workspace blob. Clone to break potential
        # q/attn_logits aliasing within a single fused pass.
        q = q.clone()
        device = kv_c_and_k_pe_cache.device
        buf = self._get_buffers(device)
        L = self.kv_lora_rank
        R = self.qk_rope_head_dim
        B = q.shape[0]
        H_q = q.shape[1]
        topk = global_topk.shape[1]

        # K2-a (same as dense decode): q latent is already Pi-rotated because
        # fold_pi_at_load folded Pi into layer.W_UK_T at weight load.
        q_work = q
        H_run = H_q

        # FlashMLA native-TQ4 decode (opt-in via VLLM_TQ_MLA_FLASHMLA_DECODE=1).
        # The CUDA producer reads the 292B cache and writes pre-Hadamard K; q_pe
        # is rotated host-side here (nope is already Pi-rotated via the W_UK_T
        # fold). Output is o_unrot (Hadamard space) + lse — identical contract to
        # the Triton stage1+stage2 path below (Pi is folded into W_UV downstream).
        if (
            os.environ.get("VLLM_TQ_MLA_FLASHMLA_DECODE") == "1"
            and not self.tq_config.key_fp8
        ):
            import flash_mla.cuda as _fm

            from vllm.v1.attention.ops.triton_turboquant_mla_decode import (
                _rotate_qpe_for_kpe_4bit,
            )

            if not getattr(self, "_tq4_flashmla_logged", False):
                print("[FlashMLA-TQ4] sparse decode branch ACTIVE", flush=True)
                self._tq4_flashmla_logged = True
            q_rot = _rotate_qpe_for_kpe_4bit(q_work, L, R, self._kpe_4bit)
            # DCP compaction: pass per-row topk_length so the kernel scans only the
            # valid prefix. global_topk is already compacted (valid slots first,
            # -1 tail), so (>=0).sum recovers the count as a pure-dataflow reduction.
            _tlen = None
            if getattr(self, "_dcp_compact_active", False):
                _tlen = (global_topk >= 0).sum(dim=1).to(torch.int32)
                if not getattr(self, "_dcp_compact_logged", False):
                    print("[TQ4-CUDA] DCP topk compaction branch ACTIVE", flush=True)
                    self._dcp_compact_logged = True
            o, lse, _, _ = _fm.sparse_decode_tq4_fwd(
                q_rot.unsqueeze(1),  # [B, 1, H, 576]
                kv_c_and_k_pe_cache.unsqueeze(2),  # [nb, page, 1, 292] uint8
                global_topk.unsqueeze(1),  # [B, 1, topk]
                _tlen,
                None,
                None,
                None,
                None,
                None,
                None,
                L,
                self.softmax_scale,
                buf["centroids_bf16"],
                buf["kpe_centroids_bf16"],
            )
            o = o.view(B, H_run, L)
            lse = lse.reshape(B, H_run)
            return o, lse

        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            num_kv_splits = tq_mla_sparse_split_count(topk, self._sm_count * 2, batch=B)

        attn_logits, o_unrot, lse = self._get_decode_workspaces(
            B,
            H_run,
            num_kv_splits,
            L,
            q.dtype,
            device,
        )
        b_seqlen = self._get_sparse_b_seqlen(B, topk, device)

        if self.tq_config.key_fp8:
            k_scale = self._get_layer_k_scale(layer)
            fused_mla_tq_sparse_decode_stage1(
                q_work,
                kv_c_and_k_pe_cache,
                buf["centroids_bf16"],
                attn_logits,
                global_topk,
                b_seqlen,
                sm_scale=self.softmax_scale,
                page_size=kv_c_and_k_pe_cache.shape[1],
                topk=topk,
                L=L,
                R=R,
                mse_bits=self.tq_config.key_mse_bits,
                mse_bytes=self._mse_index_bytes,
                kv_c_bytes=self._kv_c_bytes,
                norm_correction=bool(self.tq_config.norm_correction),
                **self._kpe_decode_kwargs(buf),
                key_fp8=True,
                k_scale=k_scale,
                num_kv_splits=num_kv_splits,
            )
        else:
            fused_mla_tq_sparse_decode_stage1(
                q_work,
                kv_c_and_k_pe_cache,
                buf["centroids_bf16"],
                attn_logits,
                global_topk,
                b_seqlen,
                sm_scale=self.softmax_scale,
                page_size=kv_c_and_k_pe_cache.shape[1],
                topk=topk,
                L=L,
                R=R,
                mse_bits=self.tq_config.key_mse_bits,
                mse_bytes=self._mse_index_bytes,
                kv_c_bytes=self._kv_c_bytes,
                norm_correction=bool(self.tq_config.norm_correction),
                **self._kpe_decode_kwargs(buf),
                num_kv_splits=num_kv_splits,
            )

        sparse_decode_softmax_reducev_fwd(
            attn_logits,
            q_work,
            o_unrot,
            lse,
            buf["v_shape_holder"],
            b_seqlen,
            num_kv_splits,
        )

        # K2-b: leave o in rotated kv_c space; Pi is folded into layer.W_UV.
        # Return LSE so DCP can merge per-rank partials (invalid topk slots are
        # already masked to -inf inside the fused sparse kernel).
        o = o_unrot.view(B, H_run, L)
        return o, lse

    def _forward_mqa_gather_flash(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        global_topk: torch.Tensor,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dequant_kv, remapped_local = self._gather_dequant_topk_workspace(
            kv_c_and_k_pe_cache,
            global_topk,
            layer,
            apply_pi_on_k=False,
        )
        return self._sparse_flash_mla(
            q,
            dequant_kv,
            global_topk,
            remapped_local_topk=remapped_local,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sparse MLA: prefill may use gather+flash; decode uses fused 2-stage.

        Decode (2-stage) and prefill (gather+flash) have different scratch
        lifetimes, so each pass materializes its own query slice immediately
        before use via ``_materialize_q`` instead of concatenating the whole
        batch up front. The decode 2-stage scratch (``attn_logits`` from
        ``_get_decode_workspaces``) is pooled in the same WorkspaceManager blob
        as ``q_concat_buffer`` and may alias it; running decode first and then
        re-materializing the prefill query into that same buffer (a fresh write)
        means prefill never reads decode-corrupted data, with no extra copy.
        """
        assert isinstance(attn_metadata, TritonMLATurboQuantSparseMetadata)
        if isinstance(q, tuple):
            q_nope_full, q_pe_full = q
            num_actual_toks = q_nope_full.shape[0]
        else:
            q_nope_full = q_pe_full = None
            num_actual_toks = q.shape[0]

        def _materialize_q(start: int, end: int) -> torch.Tensor:
            # Concat the [start:end) query into the shared q_concat_buffer
            # (zero-copy reuse) right before its consumer reads it, or slice a
            # pre-concatenated tensor input. Writing the buffer immediately
            # before each consumer guarantees a prior pass's WorkspaceManager
            # scratch cannot corrupt this pass's query.
            if q_nope_full is not None:
                out = self.q_concat_buffer[: end - start]
                ops.concat_mla_q(q_nope_full[start:end], q_pe_full[start:end], out)
                return out
            return q[start:end]

        assert self.topk_indices_buffer is not None
        assert attn_metadata.req_id_per_token is not None
        assert attn_metadata.block_table is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # Under DCP: filter topk to tokens this rank holds and remap to local
        # cache slots (others -> -1). Matches FLASHMLA_SPARSE / sparse_utils.
        #
        # DCP dead-slot compaction (opt-in VLLM_TQ_MLA_DCP_COMPACT_TOPK=1): on a
        # PURE-DECODE step with FlashMLA decode + dcp>1, compact the scattered
        # valid slots to a contiguous prefix (folded into the convert pass) so the
        # decode kernel scans only ceil(valid/64) blocks via per-row topk_length,
        # instead of the full topk. Falls back to the scattered convert otherwise
        # (prefill/mixed steps are never compacted -> their correctness is intact).
        self._dcp_compact_active = (
            os.environ.get("VLLM_TQ_MLA_DCP_COMPACT_TOPK") == "1"
            and os.environ.get("VLLM_TQ_MLA_FLASHMLA_DECODE") == "1"
            and self.dcp_world_size > 1
            and not self.tq_config.key_fp8
            and num_actual_toks == attn_metadata.num_decode_tokens
        )
        if self._dcp_compact_active:
            global_topk = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=self.dcp_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            )
        else:
            global_topk = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                dcp_world_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                dcp_interleave_size=self.dcp_interleave_size,
            )
        assert isinstance(global_topk, torch.Tensor)

        # Production sparse prefill: gather+flash (phase1).
        # Set _TQ_SPARSE_WORKSPACE=0 to opt out.
        use_gather_flash_prefill = os.environ.get("_TQ_SPARSE_WORKSPACE", "1") != "0"
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = num_actual_toks - num_decode_tokens
        # Non-DCP callers ignore LSE; DCP combine in MLAAttention needs it.
        want_lse = self.dcp_world_size > 1

        def _dcp_finalize(
            out: torch.Tensor, lse: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # A query whose entire top-k missed this rank's shard has e_sum==0
            # in the fused reduce-v -> NaN out + (-inf) lse. correct_attn_out
            # weights this rank by exp(lse - lse_max) == 0, but NaN * 0 stays
            # NaN, so zero the non-finite rows explicitly. lse == -inf already
            # makes them contribute nothing. Upcast lse to fp32 to match the
            # cross-rank combine (correct_attn_out / dcp_a2a_lse_reduce).
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            return out, lse.to(torch.float32)

        if not use_gather_flash_prefill:
            attn_out, lse = self._forward_mqa_sparse_fused(
                _materialize_q(0, num_actual_toks),
                kv_c_and_k_pe_cache,
                global_topk,
                layer,
            )
            if not want_lse:
                return attn_out, None
            return _dcp_finalize(attn_out, lse)

        if num_prefill_tokens == 0:
            attn_out, lse = self._forward_mqa_sparse_fused(
                _materialize_q(0, num_actual_toks),
                kv_c_and_k_pe_cache,
                global_topk,
                layer,
            )
            if not want_lse:
                return attn_out, None
            return _dcp_finalize(attn_out, lse)

        if num_decode_tokens == 0:
            attn_out, lse = self._forward_mqa_gather_flash(
                _materialize_q(0, num_actual_toks),
                kv_c_and_k_pe_cache,
                global_topk,
                layer,
            )
            if not want_lse:
                return attn_out, None
            return _dcp_finalize(attn_out, lse)

        # Mixed batch: decode (2-stage) runs first and may write
        # WorkspaceManager scratch that aliases q_concat_buffer; prefill
        # (gather+flash) then re-materializes its own query into that same
        # buffer (a fresh write) so it never reads decode-corrupted data.
        q_decode = _materialize_q(0, num_decode_tokens)
        # Under DCP, q may already be all-gathered over heads.
        out_heads = q_decode.shape[1]
        attn_out = q_decode.new_empty(
            (num_actual_toks, out_heads, self.kv_lora_rank),
            dtype=q_decode.dtype,
            device=q_decode.device,
        )
        decode_out, decode_lse = self._forward_mqa_sparse_fused(
            q_decode,
            kv_c_and_k_pe_cache,
            global_topk[:num_decode_tokens],
            layer,
        )
        attn_out[:num_decode_tokens] = decode_out
        prefill_out, prefill_lse = self._forward_mqa_gather_flash(
            _materialize_q(num_decode_tokens, num_actual_toks),
            kv_c_and_k_pe_cache,
            global_topk[num_decode_tokens:],
            layer,
        )
        attn_out[num_decode_tokens:] = prefill_out
        if not want_lse:
            return attn_out, None
        lse = decode_lse.new_empty((num_actual_toks, out_heads), dtype=torch.float32)
        lse[:num_decode_tokens] = decode_lse.to(torch.float32)
        lse[num_decode_tokens:] = prefill_lse.to(torch.float32)
        return _dcp_finalize(attn_out, lse)
