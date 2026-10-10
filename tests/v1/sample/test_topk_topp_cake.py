# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cake_sampling
from vllm.v1.sample.ops.topk_topp_cake import (
    apply_top_k_top_p_cake,
    cake_eligible,
    cake_sample,
)
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    apply_top_k_top_p_pytorch,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.has_device_capability(90)
    or not has_flashinfer_cake_sampling(),
    reason="FlashInfer Cake sampling needs FlashInfer >= 0.7.1 on SM90+",
)

DEVICE = "cuda"


def _params(batch_size, top_k, top_p):
    k = torch.full((batch_size,), top_k, device=DEVICE, dtype=torch.int32)
    p = None if top_p is None else torch.full((batch_size,), top_p, device=DEVICE)
    return k, p


@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("vocab_size", [32000, 151936])
@pytest.mark.parametrize("top_k", [1, 20, 1024])
@pytest.mark.parametrize("top_p", [None, 0.9])
def test_mask_matches_reference(batch_size, vocab_size, top_k, top_p):
    torch.manual_seed(0)
    logits = torch.randn(batch_size, vocab_size, device=DEVICE) * 3
    k, p = _params(batch_size, top_k, top_p)

    expected = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    actual = apply_top_k_top_p_cake(logits.clone(), k, p, top_k)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mask_per_row_params():
    torch.manual_seed(1)
    batch_size, vocab_size = 16, 50000
    logits = torch.randn(batch_size, vocab_size, device=DEVICE) * 2
    k = torch.randint(1, 300, (batch_size,), device=DEVICE, dtype=torch.int32)
    p = torch.rand(batch_size, device=DEVICE) * 0.8 + 0.1

    expected = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    actual = apply_top_k_top_p_cake(logits.clone(), k, p, int(k.max()))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mask_ties_keep_lowest_index():
    logits = torch.zeros(2, 1000, device=DEVICE)
    logits[:, 10:20] = 5.0
    k, _ = _params(2, 4, None)

    kept = apply_top_k_top_p_cake(logits.clone(), k, None, 4) > float("-inf")

    assert kept.sum(dim=1).tolist() == [4, 4]
    assert kept[:, 10:14].all()


@pytest.mark.parametrize("top_p", [None, 0.8])
def test_sample_within_kept_set(top_p):
    torch.manual_seed(2)
    batch_size, vocab_size, top_k = 32, 32000, 50
    logits = torch.randn(batch_size, vocab_size, device=DEVICE) * 2
    k, p = _params(batch_size, top_k, top_p)
    kept = apply_top_k_top_p_pytorch(logits.clone(), k, p) > float("-inf")

    for _ in range(20):
        tokens = cake_sample(logits, k, p, top_k).long()
        assert kept.gather(1, tokens[:, None]).all()


def test_sample_preserves_nonuniform_probabilities():
    """Catch biased draws that a support-only check misses."""
    torch.manual_seed(4)
    batch_size = 128
    logits = torch.full((batch_size, 32000), -torch.inf, device=DEVICE)
    probabilities = torch.tensor([0.4, 0.3, 0.2, 0.1], device=DEVICE)
    logits[:, :4] = probabilities.log()
    k, p = _params(batch_size, 4, 1.0)
    draws = torch.cat([cake_sample(logits, k, p, 4).clone() for _ in range(128)])
    observed = draws.long().bincount(minlength=32000).float() / draws.numel()
    assert observed[4:].sum() == 0
    torch.testing.assert_close(observed[:4], probabilities, atol=0.03, rtol=0)


@pytest.mark.parametrize("batch_size", [8, 128])
def test_dispatch_matches_reference(batch_size):
    """Small batches take Cake, large ones Triton; both match the reference."""
    torch.manual_seed(3)
    logits = torch.randn(batch_size, 32000, device=DEVICE) * 3
    k, p = _params(batch_size, 20, 0.9)

    expected = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    actual = apply_top_k_top_p(logits.clone(), k, p, k_max=20)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_eligibility(monkeypatch):
    logits = torch.zeros(4, 32000, device=DEVICE)
    assert cake_eligible(logits, 20)
    assert cake_eligible(logits, 1024)
    assert not cake_eligible(logits, None)
    assert not cake_eligible(logits, 1025)
    # A request without top-k carries k == vocab_size.
    assert not cake_eligible(logits, 32000)

    monkeypatch.setenv("VLLM_USE_FLASHINFER_CAKE_SAMPLER", "0")
    assert not cake_eligible(logits, 20)
