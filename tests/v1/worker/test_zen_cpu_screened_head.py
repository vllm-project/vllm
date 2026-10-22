# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.cpu import buffer_utils as cpu_buffer_utils
from vllm.v1.worker.cpu.screened_lm_head import (
    _OPS,
    ZenScreenedLMHead,
    _bound,
    _quantize_rows,
)
from vllm.v1.worker.gpu import buffer_utils as gpu_buffer_utils
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

VOCAB_SIZE = 16384
HIDDEN_SIZE = 512
GAMMA = HIDDEN_SIZE * 2.0**-24 / (1 - HIDDEN_SIZE * 2.0**-24)
MAX_TOKENS = ZenScreenedLMHead.MAX_TOKENS

requires_zentorch = pytest.mark.skipif(
    not (current_platform.is_zen_cpu() and has_zentorch_op(_OPS)),
    reason="The INT8 screen needs an AMD Zen CPU with zentorch",
)


def _weight(rows: int = VOCAB_SIZE) -> torch.Tensor:
    return (torch.randn(rows, HIDDEN_SIZE) * 0.02).bfloat16()


def _near_tie_inputs(weight: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Hidden states whose two best rows are pulled close to a tie."""
    x = torch.randn(num_tokens, weight.shape[1])
    logits = F.linear(x.to(weight.dtype), weight).float()
    top2 = logits.topk(2, dim=-1)
    diff = weight[top2.indices[:, 0]].float() - weight[top2.indices[:, 1]].float()
    gap = top2.values[:, 0] - top2.values[:, 1]
    shrink = torch.rand(num_tokens, 1)
    return (x - shrink * (gap / diff.norm(dim=-1).square())[:, None] * diff).to(
        weight.dtype
    )


def _inputs(kind: str, weight: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if kind == "near-tie":
        return _near_tie_inputs(weight, num_tokens)
    if kind == "one-hot":
        return torch.eye(num_tokens, HIDDEN_SIZE).bfloat16()
    if kind == "outlier":
        # Every small entry rounds up from just above half an INT8 step.
        x = torch.full((num_tokens, HIDDEN_SIZE), 97 / 16384)
        x[:, 0] = 1.5
        return x.bfloat16()
    return torch.randn(num_tokens, HIDDEN_SIZE).bfloat16()


def _emulated_screen(
    x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """zentorch_dynamic_qlinear in torch: per-token INT8 activations, exact
    integer accumulation, BF16 output."""
    xf = x.float()
    t = (xf.abs().amax(1, keepdim=True).clamp(min=1e-10) / 127).clamp(min=1e-10)
    acc = torch.round(xf / t).double() @ q.double().T
    return (acc * t.bfloat16().double() * scale.double()).bfloat16().float()


@pytest.fixture
def sampler(monkeypatch: pytest.MonkeyPatch) -> Sampler:
    # The CPU worker makes this swap process-wide in vllm/v1/worker/cpu/shm.py.
    monkeypatch.setattr(gpu_buffer_utils, "UvaBuffer", cpu_buffer_utils.UvaBuffer)
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
    )
    return Sampler(
        vllm_config=SimpleNamespace(reasoning_config=None),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
    )


class _FixedScreen:
    """A screen that serves every greedy batch with fixed logits."""

    def __init__(self, logits: torch.Tensor | None):
        self.logits = logits

    def required_top_k(self, idx_mapping_np: np.ndarray, num_tokens: int) -> int:
        return 1

    def __call__(self, x: torch.Tensor, top_k: int) -> torch.Tensor | None:
        return self.logits


@pytest.mark.parametrize(
    ("num_draft_tokens", "grammar_output", "screen_logits", "screened"),
    [
        pytest.param(0, None, torch.zeros(2, 8), True, id="screened"),
        pytest.param(0, None, None, False, id="declined"),
        pytest.param(2, None, torch.zeros(2, 8), False, id="draft-tokens"),
        pytest.param(0, object(), torch.zeros(2, 8), False, id="grammar"),
    ],
)
def test_runner_screens_only_plain_batches(
    num_draft_tokens: int,
    grammar_output: object,
    screen_logits: torch.Tensor | None,
    screened: bool,
):
    """The runner takes the screen's logits only for batches without drafts or
    grammar; otherwise it gets None and computes the full lm_head."""
    runner = SimpleNamespace(screened_head=_FixedScreen(screen_logits))
    batch = SimpleNamespace(
        num_draft_tokens=num_draft_tokens, idx_mapping_np=np.arange(2)
    )

    logits = GPUModelRunner._screened_logits(
        runner, torch.zeros(2, HIDDEN_SIZE), batch, grammar_output
    )

    assert logits is (screen_logits if screened else None)


@pytest.mark.parametrize("kind", ["random", "near-tie", "one-hot", "outlier"])
def test_bound_covers_full_head(kind: str):
    """The bound covers |BF16 head - screen| on every row, and keeps few rows
    on ordinary inputs."""
    torch.manual_seed(0)
    weight = _weight()
    x = _inputs(kind, weight, 8)
    q, scale, stats = _quantize_rows(weight)
    s = _emulated_screen(x, q, scale)

    beta, finite = _bound(x, s, stats, GAMMA)

    assert finite
    assert ((F.linear(x, weight).float() - s).abs() <= beta).all()
    if kind in ("random", "near-tie"):
        keep = (s + beta) >= (s - beta).amax(1, keepdim=True)
        assert keep.any(0).sum() <= VOCAB_SIZE // 16


def test_zero_rows_and_non_finite_inputs():
    weight = _weight()
    weight[3] = 0
    q, scale, stats = _quantize_rows(weight)
    assert scale[3] == 1 and (q[3] == 0).all() and (stats[:, 3] == 0).all()

    x = torch.randn(2, HIDDEN_SIZE).bfloat16()
    x[1, 0] = float("inf")
    _, finite = _bound(x, torch.zeros(2, VOCAB_SIZE), stats, GAMMA)
    assert not finite


@pytest.mark.parametrize(
    "weight",
    [
        torch.zeros(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.float16),
        # UnquantizedLinearMethod on CPU replaces the weight with an empty one.
        torch.empty(0, dtype=torch.bfloat16),
    ],
    ids=["fp16", "removed"],
)
def test_rejects_unsupported_weights(weight: torch.Tensor, sampler: Sampler):
    with pytest.raises(ValueError, match="BF16 lm_head"):
        ZenScreenedLMHead(weight, VOCAB_SIZE, sampler, MAX_TOKENS)


@requires_zentorch
# Random weights have flat logits, so 64 tokens already keep most of the cap.
@pytest.mark.parametrize("num_tokens", [1, 7, 32])
def test_greedy_matches_full_head(num_tokens: int, sampler: Sampler):
    """The kept rows hold the full head's argmax with the full head's logits,
    also when the best two nearly tie."""
    torch.manual_seed(0)
    # Padded rows beyond the vocabulary must not be scored.
    weight = _weight(VOCAB_SIZE + 64)
    head = ZenScreenedLMHead(weight, VOCAB_SIZE, sampler, MAX_TOKENS)
    x = _near_tie_inputs(weight[:VOCAB_SIZE], num_tokens)

    logits = head(x)
    ref = F.linear(x, weight[:VOCAB_SIZE])

    assert logits is not None
    assert logits.shape == ref.shape and logits.dtype == ref.dtype
    kept = logits.isfinite()
    assert kept.gather(1, ref.argmax(dim=-1, keepdim=True)).all()
    torch.testing.assert_close(logits[kept], ref[kept])


@requires_zentorch
def test_greedy_argmax_matches_full_head(sampler: Sampler):
    """Tokens whose best two logits are clearly apart get the full head's
    argmax."""
    torch.manual_seed(0)
    weight = _weight()
    head = ZenScreenedLMHead(weight, VOCAB_SIZE, sampler, MAX_TOKENS)
    x = torch.randn(32, HIDDEN_SIZE).bfloat16()

    logits = head(x)
    ref = F.linear(x, weight)

    assert logits is not None
    top2 = ref.float().topk(2, dim=-1).values
    # Ties within an output ulp or two depend on summation order.
    eps = torch.finfo(ref.dtype).eps
    clear = top2[:, 0] - top2[:, 1] > 2 * eps * top2[:, 0].abs()
    assert clear.sum() >= 8
    torch.testing.assert_close(logits.argmax(-1)[clear], ref.argmax(-1)[clear])


@requires_zentorch
def test_refresh_tracks_weight_updates(sampler: Sampler):
    """After the lm_head is reloaded in place, refresh() re-screens against
    the new weights, also once ZenDNN has cached the old INT8 copy."""
    torch.manual_seed(0)
    weight = _weight()
    head = ZenScreenedLMHead(weight, VOCAB_SIZE, sampler, MAX_TOKENS)
    assert head(_near_tie_inputs(weight, 16)) is not None
    weight.copy_(weight.flip(0))
    head.refresh()
    x = _near_tie_inputs(weight, 16)

    logits = head(x)

    assert logits is not None
    ref = F.linear(x, weight)
    assert logits.isfinite().gather(1, ref.argmax(dim=-1, keepdim=True)).all()


@requires_zentorch
def test_declines_when_most_rows_survive(sampler: Sampler):
    """A batch the screen cannot narrow is left to the full lm_head."""
    weight = _weight(1).expand(VOCAB_SIZE, -1).contiguous()
    head = ZenScreenedLMHead(weight, VOCAB_SIZE, sampler, MAX_TOKENS)

    assert head(torch.randn(2, HIDDEN_SIZE).bfloat16()) is None


@requires_zentorch
@pytest.mark.parametrize(
    ("sampling_params", "expected"),
    [
        pytest.param(SamplingParams(temperature=0.0), 1, id="greedy"),
        pytest.param(SamplingParams(temperature=1.0, top_k=1), None, id="sampled"),
        pytest.param(SamplingParams(temperature=0.0, logprobs=1), None, id="logprobs"),
    ],
)
def test_required_top_k_is_greedy_only(
    sampling_params: SamplingParams, expected: int | None, sampler: Sampler
):
    sampler.add_request(0, SamplingParams(temperature=0.0))
    sampler.add_request(1, sampling_params)
    head = ZenScreenedLMHead(_weight(), VOCAB_SIZE, sampler, MAX_TOKENS)

    assert head.required_top_k(np.array([0, 1]), 8) == expected
