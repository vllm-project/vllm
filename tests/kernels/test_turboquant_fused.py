# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused TurboQuant encode/decode Triton kernels.

Verifies that the fused kernels produce identical results to the
non-fused (separate encode + Python pack/unpack) path.
"""

import math

import pytest
import torch

from vllm.v1.attention.ops.triton_hadamard_turboquant import (
    hadamard_turboquant_decode,
    hadamard_turboquant_encode,
)

# Skip if no CUDA
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# Default test parameters
OUTLIER_FRACTION = 0.15
BIT_WIDTH = 4
BLOCK_SIZE = 16
NUM_BLOCKS = 32


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _setup_turboquant_state(device, head_size=128):
    """Create TurboQuant parameters for testing."""
    n_outliers = max(1, int(head_size * OUTLIER_FRACTION))
    normal_size = head_size - n_outliers
    padded_d = _next_power_of_2(normal_size)

    # Sign flips
    gen = torch.Generator(device="cpu").manual_seed(42)
    sign_flips = torch.where(
        torch.rand(padded_d, generator=gen) > 0.5,
        torch.ones(padded_d),
        -torch.ones(padded_d),
    ).to(device=device, dtype=torch.float32)

    # 4-bit codebook (16 centroids, Lloyd-Max for Gaussian)
    num_centroids = 2**BIT_WIDTH
    codebook = torch.linspace(-1.5, 1.5, num_centroids, device=device)
    boundaries = (codebook[:-1] + codebook[1:]) / 2

    # Outlier/normal indices
    outlier_idx = torch.arange(n_outliers, dtype=torch.long, device=device)
    normal_idx = torch.arange(n_outliers, head_size, dtype=torch.long, device=device)

    return {
        "sign_flips": sign_flips,
        "codebook": codebook,
        "boundaries": boundaries,
        "outlier_idx": outlier_idx,
        "normal_idx": normal_idx,
        "normal_size": normal_size,
        "n_outliers": n_outliers,
        "padded_d": padded_d,
        "head_size": head_size,
    }


def _python_pack_4bit(indices, normal_size):
    """Reference Python 4-bit packing."""
    flat = indices.reshape(-1, normal_size)
    if normal_size % 2 != 0:
        flat = torch.nn.functional.pad(flat, (0, 1), value=0)
    packed = flat[:, 0::2] | (flat[:, 1::2] << 4)
    packed_bytes = math.ceil(normal_size * 4 / 8)
    return packed[:, :packed_bytes]


def _python_unpack_4bit(packed, normal_size):
    """Reference Python 4-bit unpacking."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    indices = torch.stack([low, high], dim=-1).reshape(packed.shape[0], -1)
    return indices[:, :normal_size]


class TestFusedEncode:
    """Test fused encode kernel against non-fused reference."""

    @pytest.mark.parametrize("num_tokens", [1, 4, 16, 64])
    @pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
    @pytest.mark.parametrize("head_size", [128])
    def test_fused_encode_matches_reference(self, num_tokens, num_kv_heads, head_size):
        device = torch.device("cuda")
        state = _setup_turboquant_state(device, head_size)

        # Create input
        x = torch.randn(
            num_tokens,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )
        normal_x = x[..., state["normal_idx"]].contiguous()
        outlier_x = x[..., state["outlier_idx"]]

        # --- Reference path: separate encode + Python pack ---
        ref_indices, ref_norms = hadamard_turboquant_encode(
            normal_x,
            state["sign_flips"],
            state["codebook"],
            state["boundaries"],
        )
        ref_packed = _python_pack_4bit(
            ref_indices.reshape(-1, state["normal_size"]),
            state["normal_size"],
        )
        packed_bytes = math.ceil(state["normal_size"] * BIT_WIDTH / 8)
        n_outliers = state["n_outliers"]
        outlier_bytes_count = n_outliers * 2
        slot_bytes = outlier_bytes_count + packed_bytes + 2

        # Build reference slot
        N = num_tokens * num_kv_heads
        ref_parts = []
        ob = (
            outlier_x.reshape(N, n_outliers)
            .to(torch.bfloat16)
            .view(torch.uint8)
            .reshape(N, outlier_bytes_count)
        )
        ref_parts.append(ob)
        ref_parts.append(ref_packed)
        norm_bytes = (
            ref_norms.reshape(N).to(torch.float16).view(torch.uint8).reshape(N, 2)
        )
        ref_parts.append(norm_bytes)
        ref_slot = torch.cat(ref_parts, dim=-1)

        # --- Fused path ---
        from vllm.v1.attention.ops.triton_fused_turboquant import (
            fused_hadamard_encode_and_store,
        )

        cache = torch.zeros(
            NUM_BLOCKS,
            BLOCK_SIZE,
            num_kv_heads,
            slot_bytes,
            dtype=torch.uint8,
            device=device,
        )
        # Simple slot mapping: token i → slot i
        slot_mapping = torch.arange(num_tokens, device=device)
        block_indices = slot_mapping // BLOCK_SIZE
        block_offsets = slot_mapping % BLOCK_SIZE

        fused_hadamard_encode_and_store(
            normal_x=normal_x,
            outlier_x=outlier_x,
            sign_flips=state["sign_flips"],
            boundaries=state["boundaries"],
            cache=cache,
            block_indices=block_indices,
            block_offsets=block_offsets,
            bit_width=BIT_WIDTH,
        )

        # Read back from cache
        fused_slots = []
        for t in range(num_tokens):
            bi = block_indices[t].item()
            bo = block_offsets[t].item()
            for h in range(num_kv_heads):
                fused_slots.append(cache[bi, bo, h])
        fused_slot = torch.stack(fused_slots)

        # Compare: outlier bytes and norm bytes must match exactly.
        outlier_match = torch.equal(
            ref_slot[:, :outlier_bytes_count],
            fused_slot[:, :outlier_bytes_count],
        )
        norm_match = torch.equal(
            ref_slot[:, -2:],
            fused_slot[:, -2:],
        )
        assert outlier_match, "Outlier bytes mismatch!"
        assert norm_match, "Norm bytes mismatch!"
        # Packed index bytes: allow small differences (boundary effects)
        packed_ref = ref_slot[:, outlier_bytes_count:-2]
        packed_fused = fused_slot[:, outlier_bytes_count:-2]
        diff_count = (packed_ref != packed_fused).sum().item()
        total = packed_ref.numel()
        diff_pct = diff_count / total * 100
        assert diff_pct < 10.0, (
            f"Too many packed byte differences: {diff_count}/{total} ({diff_pct:.1f}%)"
        )


class TestFusedDecode:
    """Test fused decode kernel against non-fused reference."""

    @pytest.mark.parametrize("num_tokens", [1, 4, 16])
    @pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
    def test_fused_decode_matches_reference(self, num_tokens, num_kv_heads):
        device = torch.device("cuda")
        head_size = 128
        state = _setup_turboquant_state(device, head_size)
        normal_size = state["normal_size"]
        n_outliers = state["n_outliers"]
        packed_bytes = math.ceil(normal_size * BIT_WIDTH / 8)
        outlier_bytes_count = n_outliers * 2

        # Create random input and encode it
        x = torch.randn(
            num_tokens,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )
        normal_x = x[..., state["normal_idx"]].contiguous()
        outlier_x = x[..., state["outlier_idx"]]

        indices, norms = hadamard_turboquant_encode(
            normal_x,
            state["sign_flips"],
            state["codebook"],
            state["boundaries"],
        )

        # Build slot data
        N = num_tokens * num_kv_heads
        packed = _python_pack_4bit(indices.reshape(-1, normal_size), normal_size)
        parts = []
        ob = (
            outlier_x.reshape(N, n_outliers)
            .to(torch.bfloat16)
            .view(torch.uint8)
            .reshape(N, outlier_bytes_count)
        )
        parts.append(ob)
        parts.append(packed)
        norm_bytes = norms.reshape(N).to(torch.float16).view(torch.uint8).reshape(N, 2)
        parts.append(norm_bytes)
        flat_slots = torch.cat(parts, dim=-1)

        # --- Reference decode: Python unpack + Triton decode ---
        unpacked = _python_unpack_4bit(packed, normal_size)
        indices_3d = unpacked.reshape(N, 1, normal_size)
        norms_2d = norms.reshape(N, 1)
        ref_normal = hadamard_turboquant_decode(
            indices_3d,
            norms_2d,
            state["sign_flips"],
            state["codebook"],
            output_dtype=torch.bfloat16,
        ).reshape(N, normal_size)

        ref_full = torch.empty(N, head_size, dtype=torch.bfloat16, device=device)
        ref_full[:, state["normal_idx"]] = ref_normal
        outlier_vals = outlier_x.reshape(N, n_outliers).to(torch.bfloat16)
        ref_full[:, state["outlier_idx"]] = outlier_vals

        # --- Fused decode ---
        from vllm.v1.attention.ops.triton_fused_turboquant import (
            fused_hadamard_decode_from_slots,
        )

        fused_full = fused_hadamard_decode_from_slots(
            flat_slots=flat_slots,
            sign_flips=state["sign_flips"],
            codebook=state["codebook"],
            normal_idx=state["normal_idx"],
            outlier_idx=state["outlier_idx"],
            head_size=head_size,
            normal_size=normal_size,
            n_outliers=n_outliers,
            packed_bytes=packed_bytes,
        )

        # Compare
        normal_close = torch.allclose(
            fused_full[:, state["normal_idx"]],
            ref_full[:, state["normal_idx"]],
            atol=1e-3,
        )
        outlier_close = torch.allclose(
            fused_full[:, state["outlier_idx"]],
            ref_full[:, state["outlier_idx"]],
            atol=0,
        )

        n_idx = state["normal_idx"]
        o_idx = state["outlier_idx"]
        assert normal_close, (
            f"Normal channel mismatch! max diff: "
            f"{(fused_full[:, n_idx] - ref_full[:, n_idx]).abs().max()}"
        )
        assert outlier_close, (
            f"Outlier channel mismatch! max diff: "
            f"{(fused_full[:, o_idx] - ref_full[:, o_idx]).abs().max()}"
        )


class TestRoundTrip:
    """Test fused round-trip matches non-fused round-trip quality."""

    @pytest.mark.parametrize("num_tokens", [1, 4, 16])
    @pytest.mark.parametrize("head_size", [128, 96])
    def test_roundtrip_matches_reference(self, num_tokens, head_size):
        """Fused encode→decode should have similar quality to non-fused."""
        device = torch.device("cuda")
        num_kv_heads = 4
        state = _setup_turboquant_state(device, head_size)
        normal_size = state["normal_size"]
        n_outliers = state["n_outliers"]
        packed_bytes = math.ceil(normal_size * BIT_WIDTH / 8)
        outlier_bytes_count = n_outliers * 2
        slot_bytes = outlier_bytes_count + packed_bytes + 2

        x = torch.randn(
            num_tokens,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )
        normal_x = x[..., state["normal_idx"]].contiguous()
        outlier_x = x[..., state["outlier_idx"]]
        N = num_tokens * num_kv_heads

        # --- Reference round-trip (non-fused) ---
        ref_indices, ref_norms = hadamard_turboquant_encode(
            normal_x,
            state["sign_flips"],
            state["codebook"],
            state["boundaries"],
        )
        ref_normal = hadamard_turboquant_decode(
            ref_indices,
            ref_norms,
            state["sign_flips"],
            state["codebook"],
            output_dtype=torch.bfloat16,
        ).reshape(N, normal_size)
        ref_full = torch.empty(N, head_size, dtype=torch.bfloat16, device=device)
        ref_full[:, state["normal_idx"]] = ref_normal
        ref_full[:, state["outlier_idx"]] = outlier_x.reshape(N, n_outliers).to(
            torch.bfloat16
        )

        # --- Fused round-trip ---
        from vllm.v1.attention.ops.triton_fused_turboquant import (
            fused_hadamard_decode_from_slots,
            fused_hadamard_encode_and_store,
        )

        cache = torch.zeros(
            NUM_BLOCKS,
            BLOCK_SIZE,
            num_kv_heads,
            slot_bytes,
            dtype=torch.uint8,
            device=device,
        )
        slot_mapping = torch.arange(num_tokens, device=device)
        block_indices = slot_mapping // BLOCK_SIZE
        block_offsets = slot_mapping % BLOCK_SIZE

        fused_hadamard_encode_and_store(
            normal_x=normal_x,
            outlier_x=outlier_x,
            sign_flips=state["sign_flips"],
            boundaries=state["boundaries"],
            cache=cache,
            block_indices=block_indices,
            block_offsets=block_offsets,
            bit_width=BIT_WIDTH,
        )

        flat_slots = []
        for t in range(num_tokens):
            bi = block_indices[t].item()
            bo = block_offsets[t].item()
            for h in range(num_kv_heads):
                flat_slots.append(cache[bi, bo, h])
        flat_slots = torch.stack(flat_slots)

        fused_full = fused_hadamard_decode_from_slots(
            flat_slots=flat_slots,
            sign_flips=state["sign_flips"],
            codebook=state["codebook"],
            normal_idx=state["normal_idx"],
            outlier_idx=state["outlier_idx"],
            head_size=head_size,
            normal_size=normal_size,
            n_outliers=n_outliers,
            packed_bytes=packed_bytes,
        )

        # Compare: fused quality should be within 5% of reference quality
        original = x.reshape(N, head_size).float()
        ref_cos = (
            torch.nn.functional.cosine_similarity(original, ref_full.float(), dim=1)
            .mean()
            .item()
        )
        fused_cos = (
            torch.nn.functional.cosine_similarity(original, fused_full.float(), dim=1)
            .mean()
            .item()
        )

        assert abs(ref_cos - fused_cos) < 0.05, (
            f"Fused quality differs too much from reference: "
            f"ref={ref_cos:.4f}, fused={fused_cos:.4f}"
        )


class TestFusedPagedDecode:
    """Test fused_paged_decode (the production decode path)."""

    @pytest.mark.parametrize("num_tokens", [4, 16])
    def test_paged_decode_matches_slot_decode(self, num_tokens):
        """fused_paged_decode should produce same output as slot-based decode."""
        device = torch.device("cuda")
        num_kv_heads = 4
        head_size = 128
        state = _setup_turboquant_state(device, head_size)
        normal_size = state["normal_size"]
        n_outliers = state["n_outliers"]
        packed_bytes = math.ceil(normal_size * BIT_WIDTH / 8)
        outlier_bytes_count = n_outliers * 2
        slot_bytes = outlier_bytes_count + packed_bytes + 2

        from vllm.v1.attention.ops.triton_fused_turboquant import (
            fused_hadamard_decode_from_slots,
            fused_hadamard_encode_and_store,
            fused_paged_decode,
        )

        x = torch.randn(
            num_tokens,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )
        normal_x = x[..., state["normal_idx"]].contiguous()
        outlier_x = x[..., state["outlier_idx"]]

        cache = torch.zeros(
            NUM_BLOCKS,
            BLOCK_SIZE,
            num_kv_heads,
            slot_bytes,
            dtype=torch.uint8,
            device=device,
        )
        slot_mapping = torch.arange(num_tokens, device=device)
        block_indices = slot_mapping // BLOCK_SIZE
        block_offsets = slot_mapping % BLOCK_SIZE

        fused_hadamard_encode_and_store(
            normal_x=normal_x,
            outlier_x=outlier_x,
            sign_flips=state["sign_flips"],
            boundaries=state["boundaries"],
            cache=cache,
            block_indices=block_indices,
            block_offsets=block_offsets,
            bit_width=BIT_WIDTH,
        )

        # Path 1: slot-based decode
        flat_slots = []
        for t in range(num_tokens):
            bi = block_indices[t].item()
            bo = block_offsets[t].item()
            for h in range(num_kv_heads):
                flat_slots.append(cache[bi, bo, h])
        flat_slots = torch.stack(flat_slots)

        slot_decoded = fused_hadamard_decode_from_slots(
            flat_slots=flat_slots,
            sign_flips=state["sign_flips"],
            codebook=state["codebook"],
            normal_idx=state["normal_idx"],
            outlier_idx=state["outlier_idx"],
            head_size=head_size,
            normal_size=normal_size,
            n_outliers=n_outliers,
            packed_bytes=packed_bytes,
        )

        # Path 2: fused paged decode
        # Determine which blocks are used
        max_block = (num_tokens - 1) // BLOCK_SIZE + 1
        flat_bt = torch.arange(max_block, device=device, dtype=torch.long)

        paged_decoded = fused_paged_decode(
            cache=cache,
            flat_bt=flat_bt,
            sign_flips=state["sign_flips"],
            codebook=state["codebook"],
            normal_idx=state["normal_idx"],
            outlier_idx=state["outlier_idx"],
            head_size=head_size,
            normal_size=normal_size,
            n_outliers=n_outliers,
            packed_bytes=packed_bytes,
        )

        # Extract the tokens we wrote from paged output
        # paged shape: [max_block, BLOCK_SIZE, num_kv_heads, head_size]
        paged_flat = paged_decoded.reshape(-1, num_kv_heads, head_size)
        paged_tokens = paged_flat[:num_tokens].reshape(
            num_tokens * num_kv_heads, head_size
        )

        assert torch.allclose(slot_decoded, paged_tokens, atol=1e-3), (
            f"Paged decode mismatch! max diff: "
            f"{(slot_decoded - paged_tokens).abs().max()}"
        )


class TestNoOutliers:
    """Test fused path when outlier_fraction=0 (no outlier channels)."""

    def test_encode_decode_no_outliers(self):
        device = torch.device("cuda")
        num_tokens = 4
        num_kv_heads = 4
        head_size = 128
        padded_d = _next_power_of_2(head_size)

        gen = torch.Generator(device="cpu").manual_seed(42)
        sign_flips = torch.where(
            torch.rand(padded_d, generator=gen) > 0.5,
            torch.ones(padded_d),
            -torch.ones(padded_d),
        ).to(device=device, dtype=torch.float32)

        num_centroids = 2**BIT_WIDTH
        codebook = torch.linspace(-1.5, 1.5, num_centroids, device=device)
        boundaries = (codebook[:-1] + codebook[1:]) / 2

        x = torch.randn(
            num_tokens,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )

        # Encode with no outliers
        indices, norms = hadamard_turboquant_encode(x, sign_flips, codebook, boundaries)

        # Decode
        decoded = hadamard_turboquant_decode(
            indices, norms, sign_flips, codebook, output_dtype=torch.bfloat16
        )

        N = num_tokens * num_kv_heads
        original = x.reshape(N, head_size).float()
        reconstructed = decoded.reshape(N, head_size).float()
        cos_sim = (
            torch.nn.functional.cosine_similarity(original, reconstructed, dim=1)
            .mean()
            .item()
        )

        assert cos_sim > 0.98, f"No-outlier cos_sim={cos_sim:.4f} too low"
