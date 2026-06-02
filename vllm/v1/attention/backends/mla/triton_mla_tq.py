# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant TRITON_MLA backend for DeepSeek-style MLA.

Supported presets (via --kv-cache-dtype):
    turboquant_k8v4     — FP8 kv_c (8-bit, no Hadamard rotation), SM89+ required
    turboquant_4bit_nc  — 4-bit MSE kv_c with Hadamard + Lloyd-Max + norm_correction
    turboquant_k3v4_nc  — 3-bit MSE kv_c with Hadamard + Lloyd-Max + norm_correction
    turboquant_3bit_nc  — 3-bit MSE kv_c with Hadamard + Lloyd-Max + norm_correction
                          (in MLA, k/v bits collapse onto kv_c → same as k3v4_nc)

Cache layout per token (uint8, byte-packed):
    [ kv_c_packed (key_packed_size bytes) | k_pe (k_pe_bytes) ]

k_pe bytes (controlled by TurboQuantConfig.k_pe_fp8):
  default: 2 * qk_rope_head_dim bytes, raw bf16
  optional: qk_rope_head_dim fp8 bytes + 2-byte fp16 scale

kv_c_packed:
  FP8 path:  kv_lora_rank bytes (1 B/elem, e4m3)
  MSE path:  ceil(kv_lora_rank * mse_bits / 8) index bytes
             + 2 B fp16 vec_norm

Implementation strategy:
  - do_kv_cache_update: quantize latent kv_c / ctkv and scatter packed bytes
    into the uint8 KV cache.
  - forward_mqa:
      * k8v4 uses the fused KEY_FP8 branch in fused_mla_tq_decode_stage1.
      * MSE presets use the fused bit-unpack + centroid-gather + vec_norm
        branch in fused_mla_tq_decode_stage1.

Reference material:
  - #38479 general TurboQuant infrastructure
  - vllm/v1/attention/backends/turboquant_attn.py
  - vllm/model_executor/layers/quantization/turboquant/{config,centroids}.py
  - vllm/v1/attention/ops/triton_turboquant_mla_decode.py
"""

import math
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAImpl,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_decode_attention import (
    _decode_softmax_reducev_fwd,
)
from vllm.v1.attention.ops.triton_turboquant_mla_decode import (
    fused_mla_dequant_mse,
    fused_mla_tq_decode_stage1,
)
from vllm.v1.attention.ops.triton_turboquant_store import (
    _mla_fused_store_fp8,
    _mla_fused_store_mse,
)

logger = init_logger(__name__)

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0
_BF16 = torch.bfloat16


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
            for k_pe in (2 * R, R + 2):  # bf16 vs fp8 layout
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


class TritonMLATurboQuantMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
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

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
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
        # MSE path needs SM80+; FP8 path (k8v4) needs SM89+.
        # Gate at SM80 here — the FP8 SM89+ requirement is enforced at
        # runtime in TritonMLATurboQuantImpl.__init__ so that A100 users
        # can still use MSE presets (4bit_nc, k3v4_nc, 3bit_nc).
        return capability.major >= 8

    @staticmethod
    def get_impl_cls() -> type["TritonMLATurboQuantImpl"]:
        return TritonMLATurboQuantImpl

    @staticmethod
    def get_builder_cls() -> type[TritonMLATurboQuantMetadataBuilder]:
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
        # k_pe_fp8 flag is read once here and stored in the config; the rest
        # of the code reads from tq_config.k_pe_fp8 rather than re-checking
        # the env var. This matches mla_attention.get_kv_cache_spec which
        # also bakes the flag into the config at cache-spec time.
        k_pe_fp8 = envs.VLLM_TQ_KPE_FP8
        self.tq_config: TurboQuantConfig = TurboQuantConfig.from_cache_dtype(
            self.kv_cache_dtype,
            head_dim=self.kv_lora_rank,
            rope_head_dim=self.qk_rope_head_dim,
            k_pe_fp8=k_pe_fp8,
        )
        # FP8 preset (k8v4) requires SM89+ for native fp8e4m3 support.
        # MSE presets (4bit_nc, k3v4_nc, 3bit_nc) only need SM80+.
        if self.tq_config.key_fp8:
            from vllm.platforms import current_platform

            cap = current_platform.get_device_capability()
            if cap is not None and (
                cap.major < 8 or (cap.major == 8 and cap.minor < 9)
            ):
                raise RuntimeError(
                    f"TurboQuant FP8 preset '{self.kv_cache_dtype}' requires "
                    f"SM89+ (Hopper+), but current GPU is SM{cap.major}{cap.minor}. "
                    f"Use an MSE preset (turboquant_4bit_nc, turboquant_k3v4_nc, "
                    f"turboquant_3bit_nc) instead, which work on SM80+."
                )
        # Per-slot byte layout (parameterized on L = kv_lora_rank):
        #   FP8  : L bytes (1 B/elem, no norm)
        #   MSE  : ceil(L * bits / 8) + 2 bytes (indices + fp16 vec_norm)
        # Concrete: L=512 → FP8 512B, 4bit 258B, 3bit 194B.
        self._kv_c_bytes = self.tq_config.key_packed_size
        # k_pe layout: single source of truth from TurboQuantConfig.
        # The config was created in mla_attention.get_kv_cache_spec with the
        # k_pe_fp8 flag already baked in, so we read it from the config
        # rather than re-reading the env var.
        self._kpe_fp8 = self.tq_config.k_pe_fp8
        self._k_pe_bytes = self.tq_config.k_pe_bytes
        self._packed_bytes = self.tq_config.mla_packed_bytes

        # For MSE path: number of "index bytes" before the 2-byte vec_norm.
        self._mse_index_bytes = (
            math.ceil(self.kv_lora_rank * self.tq_config.key_mse_bits / 8)
            if not self.tq_config.key_fp8
            else 0
        )

        # Override: TritonMLA sets supports_quant_query_input=False when
        # is_quantized_kv_cache is True; we want the same.
        self.supports_quant_query_input = False

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
        H = _build_hadamard(D, str(device))  # symmetric → Pi == PiT
        buf: dict = {"Pi": H, "PiT": H, "Pi_bf16": H.to(_BF16)}
        if not self.tq_config.key_fp8:
            cents = get_centroids(D, self.tq_config.key_mse_bits).to(
                device=device, dtype=torch.float32
            )
            c_sorted, _ = cents.sort()
            buf["centroids"] = cents
            buf["centroids_bf16"] = cents.to(_BF16)
            buf["midpoints"] = (c_sorted[:-1] + c_sorted[1:]) / 2
        self._tq_buffers[key] = buf

    def _get_buffers(self, device: torch.device) -> dict:
        self._ensure_buffers(device)
        return self._tq_buffers[str(device)]

    def _get_decode_buffers(
        self,
        B: int,
        H_q: int,
        num_kv_splits: int,
        L: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cached decode temporaries. Reuses tensors when shapes match,
        reducing CUDA allocator pressure in high-throughput decode."""
        key = (B, H_q, num_kv_splits, L, dtype)
        if not hasattr(self, "_decode_buf_cache"):
            self._decode_buf_cache: dict = {}
        if key in self._decode_buf_cache:
            return self._decode_buf_cache[key]
        attn_logits = torch.empty(
            (B, H_q, num_kv_splits, L + 1),
            dtype=torch.float32,
            device=device,
        )
        o_unrot = torch.empty(B, H_q, L, dtype=dtype, device=device)
        lse = torch.empty(B, H_q, dtype=dtype, device=device)
        v_shape_holder = torch.empty((1, L), device=device, dtype=dtype)
        bufs = (attn_logits, o_unrot, lse, v_shape_holder)
        self._decode_buf_cache[key] = bufs
        return bufs

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
            return  # Backend supports only the absorbed-MLA decode path.
        Pi = self._get_buffers(layer.W_UK_T.device)["Pi_bf16"]
        Pi_f32 = Pi.to(torch.float32)
        # W_UK_T: (N, P, L). Fold along last dim.
        wuk = layer.W_UK_T
        wuk_new = (wuk.to(torch.float32) @ Pi_f32).to(wuk.dtype)
        layer.W_UK_T = wuk_new
        # W_UV: (N, L, V). Fold along middle (L) dim: W_UV_new = Pi @ W_UV.
        wuv = layer.W_UV
        wuv_new = (Pi_f32 @ wuv.to(torch.float32)).to(wuv.dtype)
        layer.W_UV = wuv_new
        layer._tq_pi_folded = True

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
        torch.cuda.nvtx.range_push("tq_store")
        if kv_cache.numel() == 0:
            torch.cuda.nvtx.range_pop()
            return
        N = slot_mapping.shape[0]
        if N <= 0:
            torch.cuda.nvtx.range_pop()
            return

        k_pe_ = k_pe.squeeze(1) if k_pe.dim() == 3 else k_pe
        kv_c_ = kv_c_normed[:N]
        k_pe_ = k_pe_[:N]

        # Cache k_scale as fp32 to avoid .to() call (CUDA graph safe)
        if not hasattr(self, "_tq_k_scale_f32"):
            self._tq_k_scale_f32 = k_scale.to(torch.float32).contiguous()

        if self.tq_config.key_fp8:
            # Fused FP8 store: quantize kv_c + k_pe and scatter in one kernel
            _mla_fused_store_fp8(
                kv_c_,
                k_pe_,
                kv_cache,
                slot_mapping,
                self._tq_k_scale_f32,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self._kpe_fp8,
            )
        else:
            # Fused MSE store: normalize + rotate + bucketize + pack + scatter
            device = kv_cache.device
            buf = self._get_buffers(device)
            _mla_fused_store_mse(
                kv_c_,
                k_pe_,
                kv_cache,
                slot_mapping,
                buf["PiT"],
                buf["midpoints"],
                self.tq_config.key_mse_bits,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self._kpe_fp8,
            )

        torch.cuda.nvtx.range_pop()

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
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            if toks <= 0:
                continue

            # Replicate C++ kernel address math:
            #   batch_id      = token_to_seq[token_id]
            #   batch_start   = cu_seq_lens[batch_id]
            #   batch_offset  = token_id - batch_start + seq_starts[batch_id]
            #   block_id      = block_table[batch_id, batch_offset // block_size]
            #   slot_id       = batch_offset % block_size
            num_tokens = int(prefill_metadata.chunked_context.chunk_total_token[i])
            token_to_seq = prefill_metadata.chunked_context.token_to_seq[i][
                :num_tokens
            ].to(torch.int64)
            cu_seq_lens = prefill_metadata.chunked_context.cu_seq_lens[i].to(
                torch.int64
            )
            seq_starts_t = prefill_metadata.chunked_context.starts[i]
            block_table = prefill_metadata.block_table.to(torch.int64)

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
                    kpe_fp8=self._kpe_fp8,
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
                    chunk_idx=i,
                    q=q,
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
        torch.cuda.nvtx.range_push("tq_decode")
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        # FP8 path: K3 fused FP8 decode kernel (no bf16 staging). FP8 has
        # no Hadamard rotation so the K2 weight fold must NOT run on this
        # path.
        if self.tq_config.key_fp8:
            result = self._forward_mqa_fp8_fused(
                q, kv_c_and_k_pe_cache, attn_metadata, layer
            )
            torch.cuda.nvtx.range_pop()
            return result

        # K2: one-shot fold of Pi into layer.W_UK_T and layer.W_UV (MSE only).
        self._maybe_fold_pi_into_layer(layer)

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
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            ideal_splits = max(1, attn_metadata.max_seq_len // 512)
            ideal_splits = triton.next_power_of_2(ideal_splits)
            num_kv_splits = min(ideal_splits, self._sm_count * 2)

        attn_logits, o_unrot, lse, v_shape_holder = self._get_decode_buffers(
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
            kpe_fp8=self._kpe_fp8,
            num_kv_splits=num_kv_splits,
            logit_cap=0.0,
        )

        # 4) Stage2 reduce. v_buffer is only read for its last-dim size (Lv);
        #    pass a minimal placeholder with last_dim=L (1*L bf16 elements).
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

        torch.cuda.nvtx.range_pop()
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
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            ideal_splits = max(1, attn_metadata.max_seq_len // 512)
            ideal_splits = triton.next_power_of_2(ideal_splits)
            num_kv_splits = min(ideal_splits, self._sm_count * 2)

        attn_logits, o_unrot, lse, v_shape_holder = self._get_decode_buffers(
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

        # CUDA-graph-safe k_scale access: cache the fp32 copy on the layer
        # to avoid a .to() call (which is a CUDA op) during graph capture.
        if not hasattr(layer, "_tq_k_scale_f32"):
            layer._tq_k_scale_f32 = (
                layer._k_scale.to(torch.float32).contiguous()
                if hasattr(layer, "_k_scale")
                else torch.tensor(1.0, dtype=torch.float32, device=device)
            )
        k_scale = layer._tq_k_scale_f32

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
            kpe_fp8=self._kpe_fp8,
            key_fp8=True,
            k_scale=k_scale,
            num_kv_splits=num_kv_splits,
            logit_cap=0.0,
        )

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
