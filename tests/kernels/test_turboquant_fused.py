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

# Test parameters
NUM_TOKENS = 4
NUM_KV_HEADS = 4
HEAD_SIZE = 128
OUTLIER_FRACTION = 0.15
BIT_WIDTH = 4
BLOCK_SIZE = 16
NUM_BLOCKS = 32


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _setup_turboquant_state(device):
    """Create TurboQuant parameters for testing."""
    n_outliers = max(1, int(HEAD_SIZE * OUTLIER_FRACTION))
    normal_size = HEAD_SIZE - n_outliers
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
    normal_idx = torch.arange(n_outliers, HEAD_SIZE, dtype=torch.long, device=device)

    return {
        "sign_flips": sign_flips,
        "codebook": codebook,
        "boundaries": boundaries,
        "outlier_idx": outlier_idx,
        "normal_idx": normal_idx,
        "normal_size": normal_size,
        "n_outliers": n_outliers,
        "padded_d": padded_d,
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

    def test_fused_encode_matches_reference(self):
        device = torch.device("cuda")
        state = _setup_turboquant_state(device)

        # Create input
        x = torch.randn(
            NUM_TOKENS,
            NUM_KV_HEADS,
            HEAD_SIZE,
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
        N = NUM_TOKENS * NUM_KV_HEADS
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
            NUM_KV_HEADS,
            slot_bytes,
            dtype=torch.uint8,
            device=device,
        )
        # Simple slot mapping: token i → slot i
        slot_mapping = torch.arange(NUM_TOKENS, device=device)
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
        for t in range(NUM_TOKENS):
            bi = block_indices[t].item()
            bo = block_offsets[t].item()
            for h in range(NUM_KV_HEADS):
                fused_slots.append(cache[bi, bo, h])
        fused_slot = torch.stack(fused_slots)

        # Compare
        assert torch.equal(ref_slot, fused_slot), (
            f"Fused encode mismatch! "
            f"Differing rows: "
            f"{(ref_slot != fused_slot).any(dim=1).nonzero().flatten()[:5].tolist()}"
        )


class TestFusedDecode:
    """Test fused decode kernel against non-fused reference."""

    def test_fused_decode_matches_reference(self):
        device = torch.device("cuda")
        state = _setup_turboquant_state(device)
        normal_size = state["normal_size"]
        n_outliers = state["n_outliers"]
        packed_bytes = math.ceil(normal_size * BIT_WIDTH / 8)
        outlier_bytes_count = n_outliers * 2

        # Create random input and encode it
        x = torch.randn(
            NUM_TOKENS,
            NUM_KV_HEADS,
            HEAD_SIZE,
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
        N = NUM_TOKENS * NUM_KV_HEADS
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

        ref_full = torch.empty(N, HEAD_SIZE, dtype=torch.bfloat16, device=device)
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
            head_size=HEAD_SIZE,
            normal_size=normal_size,
            n_outliers=n_outliers,
            packed_bytes=packed_bytes,
        )

        # Compare
        # Normal channels should match exactly (same quantized values)
        normal_close = torch.allclose(
            fused_full[:, state["normal_idx"]],
            ref_full[:, state["normal_idx"]],
            atol=1e-3,
        )
        # Outlier channels should match exactly (just byte copy)
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
    """Test fused encode → fused decode round-trip quality."""

    def test_roundtrip_cosine_similarity(self):
        device = torch.device("cuda")
        state = _setup_turboquant_state(device)
        normal_size = state["normal_size"]
        n_outliers = state["n_outliers"]
        packed_bytes = math.ceil(normal_size * BIT_WIDTH / 8)
        outlier_bytes_count = n_outliers * 2
        slot_bytes = outlier_bytes_count + packed_bytes + 2

        from vllm.v1.attention.ops.triton_fused_turboquant import (
            fused_hadamard_decode_from_slots,
            fused_hadamard_encode_and_store,
        )

        x = torch.randn(
            NUM_TOKENS,
            NUM_KV_HEADS,
            HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        normal_x = x[..., state["normal_idx"]].contiguous()
        outlier_x = x[..., state["outlier_idx"]]

        # Fused encode
        cache = torch.zeros(
            NUM_BLOCKS,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            slot_bytes,
            dtype=torch.uint8,
            device=device,
        )
        slot_mapping = torch.arange(NUM_TOKENS, device=device)
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

        # Extract slots and fused decode
        flat_slots = []
        for t in range(NUM_TOKENS):
            bi = block_indices[t].item()
            bo = block_offsets[t].item()
            for h in range(NUM_KV_HEADS):
                flat_slots.append(cache[bi, bo, h])
        flat_slots = torch.stack(flat_slots)

        decoded = fused_hadamard_decode_from_slots(
            flat_slots=flat_slots,
            sign_flips=state["sign_flips"],
            codebook=state["codebook"],
            normal_idx=state["normal_idx"],
            outlier_idx=state["outlier_idx"],
            head_size=HEAD_SIZE,
            normal_size=normal_size,
            n_outliers=n_outliers,
            packed_bytes=packed_bytes,
        )

        # Check cosine similarity
        N = NUM_TOKENS * NUM_KV_HEADS
        original = x.reshape(N, HEAD_SIZE).float()
        reconstructed = decoded.float()
        cos_sim = torch.nn.functional.cosine_similarity(original, reconstructed, dim=1)
        mean_cos = cos_sim.mean().item()

        assert mean_cos > 0.95, f"Round-trip cosine similarity too low: {mean_cos:.4f}"
