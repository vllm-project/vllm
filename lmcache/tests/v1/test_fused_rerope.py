# SPDX-License-Identifier: Apache-2.0
"""Fused-K/V CacheBlend re-RoPE: rotate the K half of a packed
``[.., num_kv_heads, 2*head_size]`` buffer two ways and compare.

Option 1 (reference): gather K contiguous -> rotary_embedding_k_fused -> scatter.
Option 2: rotary_embedding_k_fused_strided rotates K in place. Option 2 must
match Option 1's K exactly and leave V byte-identical.
Run for the timing table: ``python tests/v1/test_fused_rerope.py``.
"""

# Standard
import time

# Third Party
import torch

try:
    # Third Party
    import pytest

    _skipif = pytest.mark.skipif
except Exception:  # allow running as a plain script (bench) without pytest

    def _skipif(*_a, **_k):
        return lambda f: f


# First Party
try:
    # First Party
    import lmcache.c_ops as lmc_ops

    _HAS_STRIDED = hasattr(lmc_ops, "rotary_embedding_k_fused_strided")
except Exception:  # CPU-only / no built c_ops (e.g. the unit-test runner)
    lmc_ops = None
    _HAS_STRIDED = False

# These tests need CUDA and the built c_ops carrying the strided kernel; without
# both they skip cleanly (keeps CPU-only pytest collection from erroring).
_REQ = torch.cuda.is_available() and _HAS_STRIDED


def _make_inputs(n_tokens, n_heads, head_size, max_pos, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    # Packed per head: [K(head_size) | V(head_size)] -> [T, H, 2, head_size].
    packed = torch.randn(
        n_tokens, n_heads, 2, head_size, dtype=dtype, device=device, generator=g
    )
    old_pos = torch.randint(0, max_pos, (n_tokens,), dtype=torch.int64, device=device)
    new_pos = torch.randint(0, max_pos, (n_tokens,), dtype=torch.int64, device=device)
    # Real rotary cos_sin_cache: [max_pos, rot_dim] with rot_dim == head_size,
    # cos in [:head_size//2], sin in [head_size//2:] (matches the kernel). Unit
    # magnitude -> repeated in-place rotation in the bench stays bounded.
    half = head_size // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    ang = torch.outer(torch.arange(max_pos, device=device).float(), inv_freq)
    cos_sin = torch.cat([ang.cos(), ang.sin()], dim=1).to(dtype)
    return packed, old_pos, new_pos, cos_sin


def _option1_copy(packed, old_pos, new_pos, cos_sin, head_size, is_neox=True):
    """Gather K contiguous -> existing kernel -> scatter back (in place)."""
    t, h = packed.shape[0], packed.shape[1]
    k = packed[:, :, 0, :].reshape(t, h, head_size).contiguous()
    lmc_ops.rotary_embedding_k_fused(old_pos, new_pos, k, head_size, cos_sin, is_neox)
    packed[:, :, 0, :] = k.reshape(t, h, head_size)


def _option2_strided(packed, old_pos, new_pos, cos_sin, head_size, is_neox=True):
    """Strided kernel: rotate the K half in place (head_stride = 2*head_size)."""
    t, h = packed.shape[0], packed.shape[1]
    view = packed.reshape(t, h, 2 * head_size)  # contiguous view
    lmc_ops.rotary_embedding_k_fused_strided(
        old_pos, new_pos, view, head_size, 2 * head_size, cos_sin, is_neox
    )


@_skipif(not _REQ, reason="requires CUDA + built c_ops")
def test_option2_matches_option1_and_preserves_v():
    dev, dtype = "cuda", torch.bfloat16
    packed, old_pos, new_pos, cos_sin = _make_inputs(
        n_tokens=512, n_heads=8, head_size=64, max_pos=4096, dtype=dtype, device=dev
    )
    p0 = packed.clone()  # original (for V-untouched check)
    p1, p2 = packed.clone(), packed.clone()

    _option1_copy(p1, old_pos, new_pos, cos_sin, 64)
    _option2_strided(p2, old_pos, new_pos, cos_sin, 64)
    torch.cuda.synchronize()

    # K rotated identically by both paths.
    assert torch.equal(p1[:, :, 0, :], p2[:, :, 0, :]), "option2 K != option1 K"
    # V left byte-identical by both paths.
    assert torch.equal(p2[:, :, 1, :], p0[:, :, 1, :]), "option2 modified V"
    assert torch.equal(p1[:, :, 1, :], p0[:, :, 1, :]), "option1 modified V"
    # And re-RoPE actually changed K (sanity: not a no-op).
    assert not torch.equal(p2[:, :, 0, :], p0[:, :, 0, :]), "K unchanged (no-op?)"


def _bench(fn, packed, *args, iters=200):
    # In place, no per-iter clone, so the timing isolates the op (not a full
    # KV clone). Bounded rotary cos/sin keeps repeated rotation numerically safe.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(packed, *args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


@_skipif(not _REQ, reason="requires CUDA + built c_ops")
def test_bench_option1_vs_option2(capsys):
    # Timing report; run with `pytest -s` to see the table.
    with capsys.disabled():
        main()


def main():
    if not _REQ:
        print("CUDA unavailable; skipping fused re-RoPE bench")
        return
    dev, dtype = "cuda", torch.bfloat16
    print(
        f"{'config (T x H x D)':>22} | {'opt1 copy':>10} | "
        f"{'opt2 strided':>12} | speedup"
    )
    for t, h, d in [(512, 8, 64), (2048, 8, 128), (4096, 16, 128), (8192, 8, 128)]:
        packed, old_pos, new_pos, cos_sin = _make_inputs(t, h, d, 16384, dtype, dev)
        # warmup (compile/caches)
        _option1_copy(packed.clone(), old_pos, new_pos, cos_sin, d)
        _option2_strided(packed.clone(), old_pos, new_pos, cos_sin, d)
        ms1 = _bench(_option1_copy, packed, old_pos, new_pos, cos_sin, d)
        ms2 = _bench(_option2_strided, packed, old_pos, new_pos, cos_sin, d)
        print(
            f"{t:>6} x {h:>2} x {d:>3}       | {ms1:>8.4f}ms | {ms2:>10.4f}ms | "
            f"{ms1 / ms2:>5.2f}x"
        )


if __name__ == "__main__":
    main()
