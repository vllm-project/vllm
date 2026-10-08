# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.has_device_capability(89)):
    pytest.skip(
        "The FP8 screen needs CUDA with compute capability 8.9+",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.screened_head import ScreenedGreedyHead
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 4096
HIDDEN_SIZE = 512
MAX_TOKENS = ScreenedGreedyHead.MAX_TOKENS


def _make_sampler() -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        vllm_config=SimpleNamespace(reasoning_config=None),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
    )


def _near_tie_inputs(weight: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Hidden states whose two best rows are pulled close to a tie."""
    x = torch.randn(num_tokens, weight.shape[1], device=DEVICE)
    logits = F.linear(x.to(weight.dtype), weight).float()
    top2 = logits.topk(2, dim=-1)
    diff = weight[top2.indices[:, 0]].float() - weight[top2.indices[:, 1]].float()
    gap = top2.values[:, 0] - top2.values[:, 1]
    shrink = torch.rand(num_tokens, 1, device=DEVICE)
    return (x - shrink * (gap / diff.norm(dim=-1).square())[:, None] * diff).to(
        weight.dtype
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_greedy_token_matches_full_head(dtype: torch.dtype, num_tokens: int):
    """The screen keeps few rows, and the kept rows hold the full head's
    argmax with the full head's logits, also when the top two nearly tie."""
    torch.manual_seed(0)
    # Padded rows beyond the vocabulary must not be scored.
    weight = (torch.randn(VOCAB_SIZE + 64, HIDDEN_SIZE, device=DEVICE) * 0.02).to(dtype)
    head = ScreenedGreedyHead(weight, VOCAB_SIZE, _make_sampler(), MAX_TOKENS)
    x = _near_tie_inputs(weight[:VOCAB_SIZE], num_tokens)

    logits = head(x)
    ref = F.linear(x, weight[:VOCAB_SIZE])

    assert logits.shape == ref.shape and logits.dtype == ref.dtype
    torch.testing.assert_close(logits.argmax(dim=-1), ref.argmax(dim=-1))
    kept = logits.isfinite()
    torch.testing.assert_close(logits[kept], ref[kept])
    assert kept.sum() < kept.numel() / 4


def test_refresh_tracks_weight_updates():
    """After the lm_head is reloaded in place, refresh() re-screens against
    the new weights."""
    torch.manual_seed(0)
    weight = (torch.randn(VOCAB_SIZE, HIDDEN_SIZE, device=DEVICE) * 0.02).bfloat16()
    head = ScreenedGreedyHead(weight, VOCAB_SIZE, _make_sampler(), MAX_TOKENS)
    weight.copy_(weight.flip(0))
    head.refresh()
    x = _near_tie_inputs(weight, 16)

    torch.testing.assert_close(
        head(x).argmax(dim=-1), F.linear(x, weight).argmax(dim=-1)
    )


@pytest.mark.parametrize(
    ("sampling_params", "num_tokens", "expected"),
    [
        pytest.param(SamplingParams(temperature=0.0), 8, True, id="greedy"),
        pytest.param(SamplingParams(temperature=0.7), 8, False, id="random"),
        pytest.param(
            SamplingParams(temperature=0.0, logprobs=1), 8, False, id="logprobs"
        ),
        pytest.param(
            SamplingParams(temperature=0.0, logit_bias={1: 1.0}),
            8,
            False,
            id="logit-bias",
        ),
        pytest.param(
            SamplingParams(temperature=0.0),
            MAX_TOKENS + 1,
            False,
            id="large-batch",
        ),
    ],
)
def test_only_argmax_batches_use_the_screen(
    sampling_params: SamplingParams, num_tokens: int, expected: bool
):
    sampler = _make_sampler()
    sampler.add_request(0, SamplingParams(temperature=0.0))
    sampler.add_request(1, sampling_params)
    weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, device=DEVICE).bfloat16()
    head = ScreenedGreedyHead(weight, VOCAB_SIZE, sampler, MAX_TOKENS)

    assert head.is_eligible(np.array([0, 1]), num_tokens) == expected
