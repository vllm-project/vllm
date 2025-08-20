import pytest
import torch

from vllm._rope_fastpath import rope_torch_baseline, rope_complex_fast

@pytest.mark.parametrize("hd", [64, 128])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rope_fastpath_matches_baseline(hd, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Use float32 for stable numeric comparison on both CPU and CUDA.
    dtype = torch.float32
    B, H, T = 1, 4, 16

    q = torch.randn(B, H, T, hd, device=device, dtype=dtype)
    k = torch.randn(B, H, T, hd, device=device, dtype=dtype)
    cos = torch.randn(T, hd // 2, device=device, dtype=dtype)
    sin = torch.randn(T, hd // 2, device=device, dtype=dtype)

    qb_ref, kb_ref = rope_torch_baseline(q, k, cos, sin)
    qb_fast, kb_fast = rope_complex_fast(q, k, cos, sin)

    assert torch.allclose(qb_ref, qb_fast, rtol=1e-5, atol=1e-5)
    assert torch.allclose(kb_ref, kb_fast, rtol=1e-5, atol=1e-5)

def test_cpu_fp16_bf16_fallback():
    import torch
    from vllm._rope_fastpath import rope_complex_fast, rope_torch_baseline

    for dtype in (torch.float16, torch.bfloat16):
        q = torch.randn(2, 4, 8, 64, dtype=dtype, device="cpu")
        k = torch.randn(2, 4, 8, 64, dtype=dtype, device="cpu")
        cos = torch.randn(8, 32, dtype=torch.float32, device="cpu")  # [T, hd/2]
        sin = torch.randn(8, 32, dtype=torch.float32, device="cpu")

        out_fast_q, out_fast_k = rope_complex_fast(q, k, cos, sin)
        out_base_q, out_base_k = rope_torch_baseline(q, k, cos, sin)

        # Compare in fp32 for stability.
        assert torch.allclose(out_fast_q.float(), out_base_q.float(), rtol=1e-5, atol=1e-5)
        assert torch.allclose(out_fast_k.float(), out_base_k.float(), rtol=1e-5, atol=1e-5)
