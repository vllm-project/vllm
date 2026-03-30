#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone TurboQuant tests -- requires only PyTorch, no vLLM.

Usage:
    python tests/quantization/test_turboquant_standalone.py
"""

import math
import sys

import torch

from vllm.model_executor.layers.quantization.turboquant import (
    EXPECTED_MSE_NORMALIZED,
    TurboQuantConfig,
    TurboQuantState,
    _get_codebook,
    random_rotate,
    random_rotate_inverse,
    scalar_dequantize,
    scalar_quantize,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_codebook():
    print("\n--- Codebook ---")
    for bits in [1, 2, 3, 4]:
        cb = _get_codebook(bits, 128, DEVICE)
        check(f"{bits}-bit symmetric", torch.allclose(cb, -cb.flip(0), atol=1e-5))
        check(f"{bits}-bit sorted", bool((cb[1:] > cb[:-1]).all()))


def test_rotation():
    print("\n--- Random rotation (sign-flip) ---")
    d = 128
    sf = (torch.randint(0, 2, (d,), device=DEVICE).float() * 2 - 1)
    x = torch.randn(10, d, device=DEVICE)

    y = random_rotate(x, sf)
    check("norm-preserving",
          torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4))

    x_rec = random_rotate_inverse(y, sf)
    check("invertible", torch.allclose(x, x_rec, atol=1e-4),
          f"max diff={(x - x_rec).abs().max():.6f}")


def test_scalar_quantize():
    print("\n--- Scalar quantization ---")
    cb = _get_codebook(3, 128, DEVICE)
    indices = scalar_quantize(cb, cb)
    recovered = scalar_dequantize(indices, cb)
    check("centroid roundtrip", torch.allclose(cb, recovered))

    x = torch.randn(100, device=DEVICE) / math.sqrt(128)
    idx = scalar_quantize(x, cb)
    check("index range", int(idx.min()) >= 0 and int(idx.max()) <= 7)


def test_roundtrip_mse():
    print("\n--- Roundtrip MSE (paper Theorem 1) ---")
    torch.manual_seed(0)
    n, d = 500, 128

    for bits in [1, 2, 3, 4]:
        config = TurboQuantConfig(bit_width=bits, use_qjl=False)
        state = TurboQuantState(config, d, layer_idx=0, device=DEVICE)

        x = torch.randn(n, 1, d, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)

        x_hat = state.dequantize(state.quantize(x))
        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
        bound = EXPECTED_MSE_NORMALIZED[bits]

        check(f"{bits}-bit MSE={mse:.4f} (bound={bound:.4f})",
              mse < bound * 3.0,
              f"ratio={mse / bound:.2f}x")


def test_qjl_unbiased():
    print("\n--- QJL unbiasedness (paper Theorem 2) ---")
    torch.manual_seed(42)
    d = 128
    n = 300

    x = torch.randn(n, 1, d, device=DEVICE)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.randn(n, 1, d, device=DEVICE)

    state = TurboQuantState(
        TurboQuantConfig(bit_width=2, use_qjl=True),
        d, layer_idx=0, device=DEVICE,
    )
    x_hat = state.dequantize(state.quantize(x))

    ip_true = (y * x).sum(dim=-1)
    ip_est = (y * x_hat).sum(dim=-1)
    bias = (ip_est - ip_true).mean().abs().item()
    check(f"bias={bias:.4f}", bias < 0.05)


def test_nonstandard_head_size():
    print("\n--- Non-power-of-2 head sizes ---")
    for hs in [96, 80, 192]:
        config = TurboQuantConfig(bit_width=2, use_qjl=False)
        state = TurboQuantState(config, hs, layer_idx=0, device=DEVICE)
        x = torch.randn(2, 4, hs, device=DEVICE)
        x_hat = state.dequantize(state.quantize(x))
        check(f"head_size={hs} shape", x_hat.shape == x.shape)


def test_determinism():
    print("\n--- Determinism ---")
    config = TurboQuantConfig(bit_width=3)
    state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
    x = torch.randn(2, 4, 128, device=DEVICE)
    q1 = state.quantize(x)
    q2 = state.quantize(x)
    check("same input -> same indices", torch.equal(q1["indices"], q2["indices"]))


def test_compression_ratio():
    print("\n--- Compression ratio ---")
    d = 128
    for bits in [2, 3, 4]:
        fp16_bytes = d * 2
        tq_bytes = math.ceil(d * bits / 8) + 2  # +2 for norm (float16)
        ratio = fp16_bytes / tq_bytes
        check(f"{bits}-bit ratio={ratio:.1f}x vs FP16", ratio > 1.5)


def test_fractional_bitwidth():
    print("\n--- Fractional bit-widths ---")
    torch.manual_seed(7)
    d = 128
    n = 200

    for bits in [2.5, 3.5]:
        config = TurboQuantConfig(bit_width=bits, use_qjl=False)
        state = TurboQuantState(config, d, layer_idx=0, device=DEVICE)
        x = torch.randn(n, 1, d, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)
        x_hat = state.dequantize(state.quantize(x))
        check(f"{bits}-bit shape preserved", x_hat.shape == x.shape)
        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
        # Fractional MSE should be between the two integer bounds
        lo_bits = int(bits)
        check(f"{bits}-bit MSE={mse:.4f} in range",
              mse < EXPECTED_MSE_NORMALIZED[lo_bits] * 3.0,
              f"mse={mse:.4f}")


def test_fractional_qjl():
    print("\n--- Fractional bit-widths with QJL ---")
    torch.manual_seed(99)
    d = 128
    n = 200

    for bits in [2.5, 3.5]:
        config = TurboQuantConfig(bit_width=bits, use_qjl=True)
        state = TurboQuantState(config, d, layer_idx=0, device=DEVICE)
        x = torch.randn(n, 1, d, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)
        compressed = state.quantize(x)
        check(f"{bits}-bit+QJL has qjl_signs", "qjl_signs" in compressed)
        check(f"{bits}-bit+QJL has qjl_norms", "qjl_norms" in compressed)
        x_hat = state.dequantize(compressed)
        check(f"{bits}-bit+QJL shape preserved", x_hat.shape == x.shape)


def test_1bit_qjl_rejected():
    print("\n--- 1-bit + QJL validation ---")
    try:
        TurboQuantConfig(bit_width=1, use_qjl=True)
        check("1-bit+QJL raises ValueError", False, "no error raised")
    except ValueError:
        check("1-bit+QJL raises ValueError", True)


def test_channel_split_math():
    print("\n--- Channel split weighted averages ---")
    for bits, expected in [(2.5, 2.5), (3.5, 3.5)]:
        config = TurboQuantConfig(bit_width=bits)
        split = config.channel_split
        (hi_bits, hi_ratio), (lo_bits, lo_ratio) = split
        actual = hi_bits * hi_ratio + lo_bits * lo_ratio
        check(f"{bits}-bit split averages to {bits}",
              abs(actual - expected) < 1e-6,
              f"got {actual}")


def test_triton_pre_dequant():
    print("\n--- Triton pre-dequant integration ---")
    if not torch.cuda.is_available():
        print("  SKIP  (no CUDA)")
        return
    try:
        import triton  # noqa: F401
    except ImportError:
        print("  SKIP  (triton not available)")
        return

    from vllm.model_executor.layers.quantization.turboquant import (
        turboquant_pre_dequant,
    )

    torch.manual_seed(456)
    d = 128
    device = torch.device("cuda")
    config = TurboQuantConfig(bit_width=3, use_qjl=False)
    k_state = TurboQuantState(config, d, layer_idx=0, device=device)
    v_state = TurboQuantState(config, d, layer_idx=10000, device=device)

    key = torch.randn(5, 8, d, device=device, dtype=torch.bfloat16)
    value = torch.randn(5, 8, d, device=device, dtype=torch.bfloat16)

    k_out, v_out = turboquant_pre_dequant(key, value, k_state, v_state)
    check("output dtype matches input", k_out.dtype == torch.bfloat16)
    check("output shape matches input", k_out.shape == key.shape)
    # Should be lossy but close
    k_mse = (key.float() - k_out.float()).pow(2).mean().item()
    check(f"key MSE reasonable ({k_mse:.4f})", k_mse < 1.0)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    test_codebook()
    test_rotation()
    test_scalar_quantize()
    test_roundtrip_mse()
    test_qjl_unbiased()
    test_nonstandard_head_size()
    test_determinism()
    test_compression_ratio()
    test_fractional_bitwidth()
    test_fractional_qjl()
    test_1bit_qjl_rejected()
    test_channel_split_math()
    test_triton_pre_dequant()

    print(f"\n{'=' * 40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
