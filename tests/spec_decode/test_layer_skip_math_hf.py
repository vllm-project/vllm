import os
import pytest
import torch
from .conftest import hf_postnorm_at_k, TINY_LLAMA_HF

pytestmark = pytest.mark.hf
pytestmark = pytest.mark.skipif(os.getenv("VLLM_TEST_ALLOW_HF") != "1", reason="HF tests disabled")


def test_hf_postnorm_at_k_is_deterministic():
    input_ids = torch.randint(5, 50, (1, 8), dtype=torch.long)
    out1 = hf_postnorm_at_k(TINY_LLAMA_HF, input_ids, k=1, dtype=torch.float32, device="cpu")
    out2 = hf_postnorm_at_k(TINY_LLAMA_HF, input_ids, k=1, dtype=torch.float32, device="cpu")
    assert torch.allclose(out1, out2, rtol=1e-6, atol=1e-7)


def test_hf_postnorm_changes_with_k():
    input_ids = torch.randint(5, 50, (1, 8), dtype=torch.long)
    early = hf_postnorm_at_k(TINY_LLAMA_HF, input_ids, k=0)  # After first layer
    late = hf_postnorm_at_k(TINY_LLAMA_HF, input_ids, k=1)   # After second layer
    assert early.shape == late.shape
    assert not torch.allclose(early, late, rtol=1e-3, atol=1e-4)