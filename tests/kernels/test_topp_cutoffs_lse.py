# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""topp_cutoffs_lse against an fp64 reference over x / T.

The op keeps every token tied with the cutoff, so the reference sorts the row
and keeps every value at or above the first one whose cumulative probability
reaches top_p.
"""

import math

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

if not (
    current_platform.is_cuda()
    and current_platform.has_device_capability(90)
    and ops.has_topp_cutoffs_lse()
):
    pytest.skip("topp_cutoffs_lse requires SM90+", allow_module_level=True)

DEVICE = "cuda"


def _params(num_rows: int, temperature: float, top_p: float):
    idx = torch.arange(num_rows, dtype=torch.int32, device=DEVICE)
    temp = torch.full((num_rows,), temperature, device=DEVICE)
    p = torch.full((num_rows,), top_p, device=DEVICE)
    return idx, temp, p


def _reference(
    logits: torch.Tensor, temp: torch.Tensor, top_p: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 ties-kept cutoff and kept log-sum-exp over x / T (fp32, IEEE)."""
    x = logits.float()
    scaled = x / temp[:, None]
    order = x.argsort(dim=-1, descending=True)
    sorted_x = x.gather(1, order)
    probs = torch.softmax(scaled.gather(1, order).double(), dim=-1)
    reach = (probs.cumsum(-1) < top_p.double()[:, None]).sum(-1)
    cut = sorted_x.gather(1, reach.clamp(max=x.shape[1] - 1)[:, None])[:, 0]
    cut = torch.where(top_p >= 1, float("-inf"), cut)
    kept = torch.where(x >= cut[:, None], scaled.double(), float("-inf"))
    return cut, kept.logsumexp(-1)


def _logits(num_rows: int, vocab_size: int) -> torch.Tensor:
    """bf16 rows with a few dominant tokens over a Gaussian body; bf16 rounding
    makes ties at every value."""
    x = torch.randn(num_rows, vocab_size, device=DEVICE) * 2
    spikes = torch.randint(0, vocab_size, (num_rows, 4), device=DEVICE)
    x.scatter_(1, spikes, 12 + torch.rand(num_rows, 4, device=DEVICE) * 6)
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("vocab_size", [32000, 151936, 262144])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
@pytest.mark.parametrize("top_p", [0.5, 0.95])
def test_matches_fp64_reference(vocab_size: int, temperature: float, top_p: float):
    torch.manual_seed(0)
    logits = _logits(64, vocab_size)
    idx, temp, p = _params(64, temperature, top_p)

    thresh, lse = ops.topp_cutoffs_lse(logits, idx, temp, p)
    ref_thresh, ref_lse = _reference(logits, temp, p)

    torch.testing.assert_close(thresh, ref_thresh, rtol=0, atol=0)
    torch.testing.assert_close(lse, ref_lse, rtol=0, atol=1e-6)
    # The max token's log-probability x~max - lse is never positive.
    scaled_max = (logits.float() / temp[:, None]).max(-1).values.double()
    assert (lse >= scaled_max).all()


def test_fp64_path_matches_integer_path():
    torch.manual_seed(1)
    logits = _logits(32, 151936)
    idx, temp, p = _params(32, 0.7, 0.9)

    thresh, lse = ops.topp_cutoffs_lse(logits, idx, temp, p)
    thresh64, lse64 = ops.topp_cutoffs_lse(logits, idx, temp, p, force_fp64=True)

    torch.testing.assert_close(thresh64, thresh, rtol=0, atol=0)
    torch.testing.assert_close(lse64, lse, rtol=0, atol=1e-6)


def test_row_kinds():
    vocab_size = 20000
    logits = _logits(5, vocab_size)
    logits[3] = float("-inf")  # no finite logit
    logits[4, 7] = float("inf")
    idx = torch.arange(5, dtype=torch.int32, device=DEVICE)
    # Greedy, top_p = 1, top_p < 1, then the two special rows.
    temp = torch.tensor([0.0, 0.8, 0.8, 1.0, 1.0], device=DEVICE)
    p = torch.tensor([0.9, 1.0, 0.9, 0.9, 0.9], device=DEVICE)

    thresh, lse = ops.topp_cutoffs_lse(logits, idx, temp, p)
    ref_thresh, ref_lse = _reference(logits[1:3], temp[1:3], p[1:3])

    assert thresh[0] == float("-inf") and torch.isnan(lse[0])
    torch.testing.assert_close(thresh[1:3], ref_thresh, rtol=0, atol=0)
    torch.testing.assert_close(lse[1:3], ref_lse, rtol=0, atol=1e-6)
    assert thresh[3] == float("-inf") and lse[3] == float("-inf")
    assert thresh[4] == float("inf") and lse[4] == float("inf")


def test_runner_up_far_below_max():
    """The kept mass beside the max token is resolved down to 2^-40 of it, so
    the max token's probability stays below 1 for gaps up to 40 ln 2."""
    gaps = [4, 8, 12, 16, 20, 24, 27]
    logits = torch.full((len(gaps), 151936), -40.0, device=DEVICE)
    for i, g in enumerate(gaps):
        logits[i, 1000] = 25.0
        logits[i, 30000] = 25.0 - g
    logits = logits.to(torch.bfloat16)
    idx, temp, p = _params(len(gaps), 1.0, 1.0)

    _, lse = ops.topp_cutoffs_lse(logits, idx, temp, p)

    expected = torch.tensor(
        [math.log1p(math.exp(-g)) for g in gaps], dtype=torch.float64, device=DEVICE
    )
    # Each runner-up's fixed-point mass is floored to an integer of 2^-40 units.
    torch.testing.assert_close(lse - 25.0, expected, rtol=0.05, atol=0)


def test_strided_rows_match_contiguous():
    """A row stride of V + 3 also leaves odd rows off 16-byte alignment, which
    takes the scalar-load path."""
    torch.manual_seed(2)
    padded = _logits(16, 151936 + 3)
    logits = padded[:, :151936]
    idx, temp, p = _params(16, 0.9, 0.95)

    thresh, lse = ops.topp_cutoffs_lse(logits, idx, temp, p)
    ref_thresh, ref_lse = ops.topp_cutoffs_lse(logits.contiguous(), idx, temp, p)

    torch.testing.assert_close(thresh, ref_thresh, rtol=0, atol=0)
    torch.testing.assert_close(lse, ref_lse, rtol=0, atol=0)
