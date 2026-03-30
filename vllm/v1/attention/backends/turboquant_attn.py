# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for compressed KV cache.

Separate backend for TurboQuant (ICLR 2026) KV cache quantization.
Stores K/V in packed uint8 with outlier-aware layout, decodes to bf16
before running standard Triton attention kernels.
"""

import math
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------


def _pack_3bit_vectorized(
    indices: torch.Tensor,  # [N, head_size] uint8
    head_size: int,
    packed_bytes: int,
) -> torch.Tensor:
    """Pack 3-bit indices into bytes: 10 values per 30 bits (4 bytes)."""
    N = indices.shape[0]
    device = indices.device

    padded = ((head_size + 9) // 10) * 10
    if head_size < padded:
        indices = torch.nn.functional.pad(indices, (0, padded - head_size), value=0)

    num_groups = padded // 10
    grouped = indices.reshape(N, num_groups, 10).to(torch.int32)

    packed_u32 = torch.zeros(N, num_groups, dtype=torch.int32, device=device)
    for shift_idx in range(10):
        packed_u32 |= (grouped[:, :, shift_idx] & 0x7) << (shift_idx * 3)

    packed_u8 = packed_u32.view(torch.uint8).reshape(N, -1)
    return packed_u8[:, :packed_bytes]


def _unpack_3bit_vectorized(
    packed: torch.Tensor,  # [N, packed_bytes] uint8
    head_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Unpack 3-bit indices from bytes: 10 values per 4 bytes."""
    N = packed.shape[0]
    num_groups = (head_size + 9) // 10

    needed_bytes = num_groups * 4
    if packed.shape[1] < needed_bytes:
        packed = torch.nn.functional.pad(
            packed, (0, needed_bytes - packed.shape[1]), value=0
        )

    packed_u32 = packed[:, :needed_bytes].reshape(N, num_groups, 4)
    words = packed_u32.to(torch.int32)
    words = (
        words[:, :, 0]
        | (words[:, :, 1] << 8)
        | (words[:, :, 2] << 16)
        | (words[:, :, 3] << 24)
    )

    indices_list = []
    for shift_idx in range(10):
        indices_list.append((words >> (shift_idx * 3)) & 0x7)
    indices = torch.stack(indices_list, dim=-1).reshape(N, -1).to(torch.uint8)
    return indices[:, :head_size]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend for TurboQuant compressed KV cache."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["turboquant"]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TritonAttentionMetadataBuilder]:
        # Reuse Triton metadata builder — same metadata format
        return TritonAttentionMetadataBuilder

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
        # head_size here is actually slot_bytes (set by get_kv_cache_spec)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major >= 8

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "turboquant"


# ---------------------------------------------------------------------------
# Impl
# ---------------------------------------------------------------------------


class TurboQuantAttentionImpl(AttentionImpl):
    """TurboQuant attention: decode compressed KV → bf16 → Triton attention."""

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
        **kwargs,
    ) -> None:
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
        self.logits_soft_cap = logits_soft_cap or 0
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(1)

        # Decode compressed uint8 blocks → bf16.
        # Trim block_table to only needed blocks (avoids decoding padding).
        block_table = attn_metadata.block_table
        if hasattr(layer, "_tq_k_state"):
            block_size = key_cache.shape[1]
            max_blocks_needed = (
                attn_metadata.max_seq_len + block_size - 1
            ) // block_size
            trimmed_bt = block_table[:, :max_blocks_needed]
            key_cache, value_cache, block_table = self._decode_turboquant_cache(
                key_cache, value_cache, layer, trimmed_bt
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
        )

        return output

    @torch.compiler.disable
    def _decode_turboquant_cache(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer: torch.nn.Module,
        block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode only referenced blocks from packed uint8 to bf16.

        Returns compact bf16 caches and remapped block_table.
        Uses fused Triton kernel for 4-bit (single kernel: unpack → codebook
        → inv Hadamard → norm scale → write bf16 with outlier interleaving).
        """
        k_bits = int(layer._tq_k_state.config.bit_width)
        v_bits = int(layer._tq_v_state.config.bit_width)

        flat_bt = block_table.reshape(-1)
        num_entries = flat_bt.shape[0]

        # Cache the remapped block_table (same size every call in CUDA graph)
        cache_key = (num_entries, block_table.shape[0], block_table.shape[1])
        if not hasattr(self, "_bt_cache") or self._bt_cache[0] != cache_key:
            self._bt_cache = (
                cache_key,
                torch.arange(
                    num_entries,
                    device=block_table.device,
                    dtype=block_table.dtype,
                ).reshape(block_table.shape),
            )
        new_block_table = self._bt_cache[1]

        if layer._tq_k_state.config.lite_mode:
            return self._decode_lite(
                key_cache,
                value_cache,
                layer,
                flat_bt,
                num_entries,
                new_block_table,
            )

        # When K and V use the same bit-width, use the fast fused path
        # for 4-bit. With asymmetric bits, fall back to unfused which
        # handles per-state bit_width correctly.
        if k_bits == 4 and v_bits == 4:
            return self._decode_fused_4bit(
                key_cache,
                value_cache,
                layer,
                flat_bt,
                num_entries,
                new_block_table,
            )

        return self._decode_unfused(
            key_cache,
            value_cache,
            layer,
            flat_bt,
            num_entries,
            new_block_table,
        )

    def _decode_fused_4bit(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer: torch.nn.Module,
        flat_bt: torch.Tensor,
        num_entries: int,
        new_block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused 4-bit decode. Uses CUDA WPH kernel if available."""
        import os

        use_cuda_wph = os.environ.get("TQ_CUDA_WPH", "0") in ("1", "true")
        if use_cuda_wph:
            try:
                from vllm.v1.attention.ops.cuda_turboquant_decode import (
                    cuda_wph_decode_from_slots,
                )

                return self._decode_cuda_wph(
                    key_cache,
                    value_cache,
                    layer,
                    flat_bt,
                    num_entries,
                    new_block_table,
                    cuda_wph_decode_from_slots,
                )
            except Exception:
                pass  # Fall through to Triton

        from vllm.v1.attention.ops.triton_fused_turboquant import (
            fused_paged_decode,
        )

        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        head_size = k_state.head_size
        normal_size = k_state.normal_size
        n_outliers = head_size - normal_size
        packed_bytes = math.ceil(normal_size * 4 / 8)

        decoded_caches = []
        for cache, state in [(key_cache, k_state), (value_cache, v_state)]:
            decoded = fused_paged_decode(
                cache=cache,
                flat_bt=flat_bt,
                sign_flips=state.sign_flips,
                codebook=state.codebook,
                normal_idx=state.normal_idx,
                outlier_idx=state.outlier_idx,
                head_size=head_size,
                normal_size=normal_size,
                n_outliers=n_outliers,
                packed_bytes=packed_bytes,
            )
            decoded_caches.append(decoded)

        return decoded_caches[0], decoded_caches[1], new_block_table

    def _decode_cuda_wph(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer: torch.nn.Module,
        flat_bt: torch.Tensor,
        num_entries: int,
        new_block_table: torch.Tensor,
        decode_fn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CUDA warp-per-head decode (no shared memory races)."""
        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        head_size = k_state.head_size
        normal_size = k_state.normal_size
        n_outliers = head_size - normal_size
        packed_bytes = math.ceil(normal_size * 4 / 8)

        decoded_caches = []
        for cache, state in [(key_cache, k_state), (value_cache, v_state)]:
            _, block_size, num_kv_heads, slot_bytes = cache.shape
            used = cache[flat_bt]
            N = num_entries * block_size * num_kv_heads
            flat = used.reshape(N, slot_bytes)

            decoded_flat = decode_fn(
                flat_slots=flat,
                sign_flips=state.sign_flips,
                codebook=state.codebook,
                normal_idx=state.normal_idx,
                outlier_idx=state.outlier_idx,
                head_size=head_size,
                normal_size=normal_size,
                n_outliers=n_outliers,
                packed_bytes=packed_bytes,
            )
            decoded_caches.append(
                decoded_flat.reshape(num_entries, block_size, num_kv_heads, head_size)
            )

        return decoded_caches[0], decoded_caches[1], new_block_table

    def _decode_unfused(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer: torch.nn.Module,
        flat_bt: torch.Tensor,
        num_entries: int,
        new_block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Non-fused decode for 2-bit/3-bit/8-bit."""
        from vllm.v1.attention.ops.triton_hadamard_turboquant import (
            hadamard_turboquant_decode,
        )

        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        head_size = k_state.head_size
        normal_size = k_state.normal_size
        n_outliers = head_size - normal_size
        outlier_byte_count = n_outliers * 2

        decoded_caches = []
        for cache, state in [(key_cache, k_state), (value_cache, v_state)]:
            bits = int(state.config.bit_width)
            packed_bytes = math.ceil(normal_size * bits / 8)

            _, block_size, num_kv_heads, slot_bytes = cache.shape
            used = cache[flat_bt]
            N = num_entries * block_size * num_kv_heads
            flat = used.reshape(N, slot_bytes)

            pos = 0
            outlier_vals = None
            if n_outliers > 0:
                outlier_vals = (
                    flat[:, pos : pos + outlier_byte_count]
                    .clone()
                    .view(torch.bfloat16)
                    .reshape(N, n_outliers)
                )
                pos += outlier_byte_count
            flat_packed = flat[:, pos : pos + packed_bytes]
            pos += packed_bytes

            norms = flat[:, pos : pos + 2].clone().view(torch.float16).reshape(N)

            if bits == 4:
                low = flat_packed & 0x0F
                high = (flat_packed >> 4) & 0x0F
                indices = torch.stack([low, high], dim=-1).reshape(N, -1)[
                    :, :normal_size
                ]
            elif bits == 2:
                b0 = flat_packed & 0x03
                b1 = (flat_packed >> 2) & 0x03
                b2 = (flat_packed >> 4) & 0x03
                b3 = (flat_packed >> 6) & 0x03
                indices = torch.stack([b0, b1, b2, b3], dim=-1).reshape(N, -1)[
                    :, :normal_size
                ]
            elif bits == 3:
                indices = _unpack_3bit_vectorized(
                    flat_packed, normal_size, cache.device
                )
                indices = indices[:N, :normal_size]
            else:
                indices = flat_packed[:, :normal_size]

            indices_3d = indices.reshape(N, 1, normal_size)
            norms_2d = norms.reshape(N, 1)
            normal_decoded = hadamard_turboquant_decode(
                indices_3d,
                norms_2d,
                state.sign_flips,
                state.codebook,
                output_dtype=torch.bfloat16,
            ).reshape(N, normal_size)

            full = torch.empty(N, head_size, dtype=torch.bfloat16, device=cache.device)
            if state.normal_idx is not None and outlier_vals is not None:
                full[:, state.normal_idx] = normal_decoded
                full[:, state.outlier_idx] = outlier_vals
            else:
                full = normal_decoded

            decoded = full.reshape(num_entries, block_size, num_kv_heads, head_size)
            decoded_caches.append(decoded)

        return decoded_caches[0], decoded_caches[1], new_block_table

    def _decode_lite(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer: torch.nn.Module,
        flat_bt: torch.Tensor,
        num_entries: int,
        new_block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lite decode: unpack, codebook lookup, scale by norm. No Hadamard."""
        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        head_size = k_state.head_size
        normal_size = k_state.normal_size
        n_outliers = head_size - normal_size
        outlier_byte_count = n_outliers * 2

        decoded_caches = []
        for cache, state in [(key_cache, k_state), (value_cache, v_state)]:
            bits = int(state.config.bit_width)
            packed_bytes = math.ceil(normal_size * bits / 8)

            _, block_size, num_kv_heads, slot_bytes = cache.shape
            used = cache[flat_bt]
            N = num_entries * block_size * num_kv_heads
            flat = used.reshape(N, slot_bytes)

            # Parse slot layout: [outlier_bytes | packed | norm(2B)]
            pos = 0
            outlier_vals = None
            if n_outliers > 0:
                outlier_vals = (
                    flat[:, pos : pos + outlier_byte_count]
                    .clone()
                    .view(torch.bfloat16)
                    .reshape(N, n_outliers)
                )
                pos += outlier_byte_count
            flat_packed = flat[:, pos : pos + packed_bytes]
            pos += packed_bytes
            norms = flat[:, pos : pos + 2].clone().view(torch.float16).reshape(N)

            # Unpack indices
            if bits == 4:
                low = flat_packed & 0x0F
                high = (flat_packed >> 4) & 0x0F
                indices = torch.stack([low, high], dim=-1).reshape(N, -1)[
                    :, :normal_size
                ]
            elif bits == 2:
                b0 = flat_packed & 0x03
                b1 = (flat_packed >> 2) & 0x03
                b2 = (flat_packed >> 4) & 0x03
                b3 = (flat_packed >> 6) & 0x03
                indices = torch.stack([b0, b1, b2, b3], dim=-1).reshape(N, -1)[
                    :, :normal_size
                ]
            elif bits == 3:
                indices = _unpack_3bit_vectorized(
                    flat_packed, normal_size, cache.device
                )
                indices = indices[:N, :normal_size]
            else:
                indices = flat_packed[:, :normal_size]

            # Codebook lookup + scale by norm (no inverse Hadamard)
            reconstructed = state.codebook[indices.long()]
            normal_decoded = (reconstructed * norms.unsqueeze(-1).float()).to(
                torch.bfloat16
            )

            # Reassemble outlier + normal channels
            full = torch.empty(N, head_size, dtype=torch.bfloat16, device=cache.device)
            if state.normal_idx is not None and outlier_vals is not None:
                full[:, state.normal_idx] = normal_decoded
                full[:, state.outlier_idx] = outlier_vals
            else:
                full = normal_decoded

            decoded = full.reshape(num_entries, block_size, num_kv_heads, head_size)
            decoded_caches.append(decoded)

        return decoded_caches[0], decoded_caches[1], new_block_table

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if not hasattr(layer, "_tq_k_state"):
            return

        # Calibrate outlier channels on first REAL batch (not profiling).
        # During profiling/warmup, attn_metadata is None and K/V values
        # come from dummy inputs — calibrating on those produces wrong
        # outlier channels. We detect profiling via the forward context.
        if getattr(layer, "_tq_needs_calibration", False):
            from vllm.forward_context import get_forward_context

            ctx = get_forward_context()
            is_profile = ctx.attn_metadata is None
            if not is_profile:
                num_actual = slot_mapping.shape[0]
                k_flat = key[:num_actual].reshape(-1, key.shape[-1])
                v_flat = value[:num_actual].reshape(-1, value.shape[-1])
                layer._tq_k_state.calibrate_outliers(k_flat)
                layer._tq_v_state.calibrate_outliers(v_flat)
                layer._tq_needs_calibration = False

        self._encode_turboquant_cache(key, value, kv_cache, slot_mapping, layer)

    @torch.compiler.disable
    def _encode_turboquant_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer: torch.nn.Module,
    ) -> None:
        """Encode K/V with outlier-aware layout into paged uint8 cache."""
        # Clamp slot_mapping: padding tokens (-1) go to slot 0.
        # This is safe because calibration is skipped during profiling
        # (checked in do_kv_cache_update), and real requests have no
        # -1 entries. Using clamp avoids GPU→CPU sync that would break
        # CUDA graph capture.
        num_actual = slot_mapping.shape[0]
        clamped_slots = slot_mapping.clamp(min=0)

        k_bits = int(layer._tq_k_state.config.bit_width)
        v_bits = int(layer._tq_v_state.config.bit_width)
        block_size = kv_cache.shape[2]
        block_indices = clamped_slots // block_size
        block_offsets = clamped_slots % block_size

        if layer._tq_k_state.config.lite_mode:
            self._encode_lite(
                key[:num_actual],
                value[:num_actual],
                kv_cache,
                block_indices,
                block_offsets,
                layer,
            )
        elif k_bits == 4 and v_bits == 4:
            self._encode_fused_4bit(
                key[:num_actual],
                value[:num_actual],
                kv_cache,
                block_indices,
                block_offsets,
                layer,
            )
        else:
            self._encode_unfused(
                key[:num_actual],
                value[:num_actual],
                kv_cache,
                block_indices,
                block_offsets,
                layer,
            )

    def _encode_fused_4bit(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        block_indices: torch.Tensor,
        block_offsets: torch.Tensor,
        layer: torch.nn.Module,
    ) -> None:
        """Fused encode path for 4-bit: single Triton kernel per K/V."""
        from vllm.v1.attention.ops.triton_fused_turboquant import (
            fused_hadamard_encode_and_store,
        )

        for kv_idx, (tensor, state) in enumerate(
            [
                (key, layer._tq_k_state),
                (value, layer._tq_v_state),
            ]
        ):
            if state.outlier_idx is not None:
                normal_x = tensor[..., state.normal_idx].contiguous()
                outlier_x = tensor[..., state.outlier_idx]
            else:
                normal_x = tensor
                outlier_x = None

            cache = kv_cache[:, kv_idx]
            fused_hadamard_encode_and_store(
                normal_x=normal_x,
                outlier_x=outlier_x,
                sign_flips=state.sign_flips,
                boundaries=state.boundaries,
                cache=cache,
                block_indices=block_indices,
                block_offsets=block_offsets,
                bit_width=4,
            )

    def _encode_unfused(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        block_indices: torch.Tensor,
        block_offsets: torch.Tensor,
        layer: torch.nn.Module,
    ) -> None:
        """Non-fused encode path for 2-bit/3-bit/8-bit."""
        from vllm.v1.attention.ops.triton_hadamard_turboquant import (
            hadamard_turboquant_encode,
        )

        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        head_size = k_state.head_size
        normal_size = k_state.normal_size
        n_outliers = head_size - normal_size
        slot_bytes = kv_cache.shape[-1]
        num_actual = key.shape[0]

        for kv_idx, (tensor, state) in enumerate(
            [
                (key, k_state),
                (value, v_state),
            ]
        ):
            bits = int(state.config.bit_width)
            packed_bytes = math.ceil(normal_size * bits / 8)

            if state.outlier_idx is not None:
                normal_x = tensor[..., state.normal_idx].contiguous()
                outlier_x = tensor[..., state.outlier_idx]
            else:
                normal_x = tensor
                outlier_x = None

            indices, norms = hadamard_turboquant_encode(
                normal_x, state.sign_flips, state.codebook, state.boundaries
            )

            flat_indices = indices.reshape(-1, normal_size)
            N = flat_indices.shape[0]
            # Pad for 4-bit/2-bit interleaving (3-bit packs internally)
            if bits in (4, 2):
                align = {4: 2, 2: 4}[bits]
                if normal_size % align != 0:
                    pad = align - (normal_size % align)
                    flat_indices = torch.nn.functional.pad(
                        flat_indices, (0, pad), value=0
                    )
            if bits == 4:
                packed = flat_indices[:, 0::2] | (flat_indices[:, 1::2] << 4)
            elif bits == 2:
                packed = (
                    flat_indices[:, 0::4]
                    | (flat_indices[:, 1::4] << 2)
                    | (flat_indices[:, 2::4] << 4)
                    | (flat_indices[:, 3::4] << 6)
                )
            elif bits == 3:
                packed = _pack_3bit_vectorized(flat_indices, normal_size, packed_bytes)
            else:
                packed = flat_indices[:, :packed_bytes]
            packed = packed[:, :packed_bytes]

            parts = []
            if outlier_x is not None:
                ob = (
                    outlier_x.reshape(N, n_outliers)
                    .to(torch.bfloat16)
                    .view(torch.uint8)
                    .reshape(N, n_outliers * 2)
                )
                parts.append(ob)
            parts.append(packed)
            norm_bytes_data = (
                norms.reshape(N).to(torch.float16).view(torch.uint8).reshape(N, 2)
            )
            parts.append(norm_bytes_data)
            slot_data = torch.cat(parts, dim=-1)

            # Pad to slot_bytes when asymmetric V uses fewer bits than K,
            # leaving unused trailing bytes in the slot.
            actual_bytes = slot_data.shape[-1]
            if actual_bytes < slot_bytes:
                slot_data = torch.nn.functional.pad(
                    slot_data, (0, slot_bytes - actual_bytes), value=0
                )

            num_kv_heads = tensor.shape[1]
            slot_3d = slot_data.reshape(num_actual, num_kv_heads, slot_bytes)
            cache = kv_cache[:, kv_idx]
            cache[block_indices, block_offsets] = slot_3d

    def _encode_lite(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        block_indices: torch.Tensor,
        block_offsets: torch.Tensor,
        layer: torch.nn.Module,
    ) -> None:
        """Lite encode path: no rotation, pure scalar quantization.

        Layout per slot (same as standard): [outlier_bytes | packed | norm(2B)]
        No Hadamard, no sign flips — just normalize, quantize, pack, write.
        """
        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        head_size = k_state.head_size
        normal_size = k_state.normal_size
        n_outliers = head_size - normal_size
        slot_bytes = kv_cache.shape[-1]
        num_actual = key.shape[0]

        for kv_idx, (tensor, state) in enumerate(
            [
                (key, k_state),
                (value, v_state),
            ]
        ):
            bits = int(state.config.bit_width)
            packed_bytes = math.ceil(normal_size * bits / 8)

            # Split outlier / normal channels
            if state.outlier_idx is not None:
                normal_x = tensor[..., state.normal_idx].contiguous()
                outlier_x = tensor[..., state.outlier_idx]
            else:
                normal_x = tensor
                outlier_x = None

            # Flatten to (N, normal_size)
            flat = normal_x.reshape(-1, normal_size).float()
            N = flat.shape[0]

            # Compute norms and normalize
            norms = torch.norm(flat, dim=-1, keepdim=True)
            flat_normed = flat / (norms + 1e-16)

            # Scalar quantize (no rotation)
            indices = torch.bucketize(flat_normed.contiguous(), state.boundaries).to(
                torch.uint8
            )

            # Pack indices
            flat_indices = indices.reshape(N, normal_size)
            align = {4: 2, 2: 4, 3: 10}.get(bits, 1)
            if normal_size % align != 0:
                pad = align - (normal_size % align)
                flat_indices = torch.nn.functional.pad(flat_indices, (0, pad), value=0)
            if bits == 4:
                packed = flat_indices[:, 0::2] | (flat_indices[:, 1::2] << 4)
            elif bits == 2:
                packed = (
                    flat_indices[:, 0::4]
                    | (flat_indices[:, 1::4] << 2)
                    | (flat_indices[:, 2::4] << 4)
                    | (flat_indices[:, 3::4] << 6)
                )
            elif bits == 3:
                packed = _pack_3bit_vectorized(flat_indices, normal_size, packed_bytes)
            else:
                packed = flat_indices[:, :packed_bytes]
            packed = packed[:, :packed_bytes]

            # Assemble slot data: [outlier_bytes | packed | norm_bytes]
            parts: list[torch.Tensor] = []
            if outlier_x is not None:
                ob = (
                    outlier_x.reshape(N, n_outliers)
                    .to(torch.bfloat16)
                    .view(torch.uint8)
                    .reshape(N, n_outliers * 2)
                )
                parts.append(ob)
            parts.append(packed)
            norm_bytes_data = (
                norms.reshape(N).to(torch.float16).view(torch.uint8).reshape(N, 2)
            )
            parts.append(norm_bytes_data)
            slot_data = torch.cat(parts, dim=-1)

            # Pad to slot_bytes when asymmetric V uses fewer bits than K
            actual_bytes = slot_data.shape[-1]
            if actual_bytes < slot_bytes:
                slot_data = torch.nn.functional.pad(
                    slot_data, (0, slot_bytes - actual_bytes), value=0
                )

            num_kv_heads = tensor.shape[1]
            slot_3d = slot_data.reshape(num_actual, num_kv_heads, slot_bytes)
            cache = kv_cache[:, kv_idx]
            cache[block_indices, block_offsets] = slot_3d
