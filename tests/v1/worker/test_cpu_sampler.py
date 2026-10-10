# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU candidate sampler must pick the token the full-vocab path picks:
top-k/top-p masking over the whole vocab, then a Gumbel-max with the same
noise."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_model

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p  # noqa: E402
from vllm.v1.worker.cpu.kernels.gumbel import _gumbel_argmax  # noqa: E402
from vllm.v1.worker.cpu.sampler import _sample_candidates  # noqa: E402

NUM_TOKENS = 64
VOCAB_SIZE = 32000


def _inputs(scale: float):
    generator = torch.Generator().manual_seed(0)
    # fp32 logits, so ties at the top-k/top-p boundary do not occur.
    logits = torch.randn(NUM_TOKENS, VOCAB_SIZE, generator=generator) * scale
    temp = torch.full((NUM_TOKENS,), 0.8)
    temp[::4] = 0.0
    seed = torch.randint(0, 2**62, (NUM_TOKENS,), generator=generator)
    pos = torch.randint(0, 4096, (NUM_TOKENS,), generator=generator)
    return logits, temp, seed, pos


@pytest.mark.parametrize("top_k", [None, 1, 50])
@pytest.mark.parametrize("top_p", [None, 0.5, 0.95])
@pytest.mark.parametrize("use_fp64", [False, True])
def test_matches_full_vocab_sampling(top_k, top_p, use_fp64):
    if top_k is None and top_p is None:
        pytest.skip("the candidate path only handles top-k/top-p")
    # Peaked logits keep every row's top-p nucleus within the candidates.
    logits, temp, seed, pos = _inputs(scale=8.0)
    k = None if top_k is None else torch.full((NUM_TOKENS,), top_k)
    p = None if top_p is None else torch.full((NUM_TOKENS,), top_p)

    sampled, covered = _sample_candidates(logits, k, p, temp, seed, pos, use_fp64)
    masked = apply_top_k_top_p(logits.clone(), k, p)
    _, expected = _gumbel_argmax(masked, temp, seed, pos, False, use_fp64)

    assert covered.all()
    torch.testing.assert_close(sampled, expected)


def test_flat_rows_fall_back():
    # A flat distribution's 0.95 nucleus is far wider than the candidates.
    logits, temp, seed, pos = _inputs(scale=0.1)
    top_p = torch.full((NUM_TOKENS,), 0.95)
    _, covered = _sample_candidates(logits, None, top_p, temp, seed, pos, False)
    assert not covered.any()
