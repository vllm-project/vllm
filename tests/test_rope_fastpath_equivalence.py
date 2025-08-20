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
