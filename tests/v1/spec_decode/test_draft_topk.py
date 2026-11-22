# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `speculative_config.draft_top_k`: the fused draft sampler
(`sample_draft_topk_fused`) and sparse verification (`rejection_sample_topk`)."""

import pytest
import torch

from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.dflash2.config import DFlash2SpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.draft_topk_cache import (
    SparseDraftCache,
    sample_draft_topk_fused,
)
from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.dspine.config import DSpineSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.eagle.config import (
    Eagle3SpeculatorConfig,
    EagleSpeculatorConfig,
)
from vllm.v1.worker.gpu.spec_decode.lilicorr.config import LiLiCorrSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.config import (
    MultiModuleMTPSpeculatorConfig,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
    rejection_sample_topk,
)
from vllm.v1.worker.gpu.spec_decode.standalone_ar.config import (
    StandaloneARSpeculatorConfig,
)
from vllm.v1.worker.gpu.spec_decode.xpress.config import XPressSpeculatorConfig

if not torch.cuda.is_available():
    pytest.skip("CUDA required for draft_top_k tests", allow_module_level=True)

DEVICE = "cuda"


def _build_topk_inputs(
    target_logits_1d: torch.Tensor,
    draft_logits_1d: torch.Tensor,
    top_k: int,
    num_steps: int,
    temperature: float,
    num_trials: int,
    draft_sampled: torch.Tensor | None = None,
) -> dict:
    """rejection_sample_topk kwargs for a fixed target and draft distribution.
    target_logits_1d already has temperature applied, draft_logits_1d does not.
    Draft tokens are drawn i.i.d. from q' unless draft_sampled is given."""
    num_logits = num_trials * (num_steps + 1)
    vals, idx = torch.topk(draft_logits_1d.float() / max(temperature, 1e-5), top_k)
    if draft_sampled is None:
        local = torch.multinomial(
            torch.softmax(vals, 0).expand(num_trials, -1), num_steps, replacement=True
        )
        draft_sampled = idx[local]
    draft_2d = torch.zeros(num_trials, num_steps + 1, dtype=torch.int64, device=DEVICE)
    draft_2d[:, 1:] = draft_sampled

    cache = SparseDraftCache(num_trials, num_steps, top_k, device=DEVICE)
    cache.ids[:num_trials] = idx
    cache.logq[:num_trials] = torch.log_softmax(vals, 0)
    i32 = dict(dtype=torch.int32, device=DEVICE)
    return dict(
        target_logits=target_logits_1d.expand(num_logits, -1).contiguous(),
        draft_topk_cache=cache,
        draft_sampled=draft_2d.reshape(-1),
        cu_num_logits=torch.arange(num_trials + 1, **i32) * (num_steps + 1),
        pos=torch.arange(num_logits, **i32),
        idx_mapping=torch.arange(num_trials, **i32),
        expanded_idx_mapping=torch.arange(num_trials, **i32).repeat_interleave(
            num_steps + 1
        ),
        expanded_local_pos=torch.arange(num_steps + 1, **i32).repeat(num_trials),
        temperature=torch.full((num_trials,), temperature, device=DEVICE),
        seed=torch.arange(num_trials, dtype=torch.int64, device=DEVICE),
        num_speculative_steps=num_steps,
    )


def _assert_distribution_match(sampled, target_probs, label, min_expected=5.0):
    """Chi-squared goodness of fit with a ~10-sigma threshold, as in
    test_rejection_sampler_utils.py; low-expectation tokens share one bin."""
    n = sampled.shape[0]
    observed = torch.zeros_like(target_probs).scatter_add_(
        0, sampled, torch.ones(n, device=DEVICE)
    )
    expected = target_probs * n
    big = expected >= min_expected
    obs, exp = observed[big], expected[big]
    if expected[~big].sum() >= min_expected:
        obs = torch.cat([obs, observed[~big].sum().view(1)])
        exp = torch.cat([exp, expected[~big].sum().view(1)])
    df = obs.numel() - 1
    if df < 1:  # too few reached samples at this position to test
        return
    chi2 = ((obs - exp) ** 2 / exp).sum().item()
    assert chi2 < df + 10 * (2 * df) ** 0.5, f"{label}: chi2={chi2:.1f}, df={df}"


def _assert_unbiased(inputs, target_logits_1d, num_steps, label):
    sampled, num_sampled = rejection_sample_topk(**inputs, use_fp64=False)
    target_probs = torch.softmax(target_logits_1d, 0)
    for pos in range(num_steps + 1):
        reached = num_sampled >= pos + 1
        _assert_distribution_match(
            sampled[reached, pos], target_probs, f"{label} pos={pos}"
        )


@pytest.mark.parametrize("num_steps", [1, 3])
@pytest.mark.parametrize("top_k", [16, 64])
def test_rejection_sample_topk_is_unbiased(num_steps, top_k):
    torch.manual_seed(42)
    vocab = 4096
    target, draft = torch.randn(2, vocab, device=DEVICE)
    inputs = _build_topk_inputs(target, draft, top_k, num_steps, 1.0, 10 * vocab)
    _assert_unbiased(inputs, target, num_steps, f"top_k={top_k}")


def test_rejection_sample_topk_is_unbiased_with_real_draft_kernel():
    """Drafts come from sample_draft_topk_fused on the (seed, pos) stream that
    verification reads, so correlated draft and resample noise would bias the
    output. Opposite rankings make most trials reach the resample."""
    torch.manual_seed(42)
    vocab, top_k, num_trials = 64, 16, 200_000
    target = torch.randn(vocab, device=DEVICE)
    vals, idx = torch.topk(-target, top_k)
    i64 = dict(dtype=torch.int64, device=DEVICE)
    cache = SparseDraftCache(num_trials, 1, top_k, device=DEVICE)
    drafted = sample_draft_topk_fused(
        cache,
        idx.expand(num_trials, -1).contiguous(),
        vals.expand(num_trials, -1).contiguous(),
        torch.arange(num_trials, **i64),
        torch.ones(num_trials, device=DEVICE),
        torch.arange(num_trials, **i64),
        torch.full((num_trials,), 1000, **i64),
        torch.zeros(num_trials, **i64),
    )
    inputs = _build_topk_inputs(
        target, -target, top_k, 1, 1.0, num_trials, draft_sampled=drafted.unsqueeze(1)
    )
    # Verification positions continue the draft stream: 999 is the bonus row.
    inputs["pos"] = 999 + torch.arange(2, dtype=torch.int32, device=DEVICE).repeat(
        num_trials
    )
    _assert_unbiased(inputs, target, 1, "real draft kernel")


def test_rejection_sample_topk_greedy():
    torch.manual_seed(0)
    target, draft = torch.randn(2, 256, device=DEVICE)
    inputs = _build_topk_inputs(target, draft, 16, 2, 0.0, 4096)
    sampled, _ = rejection_sample_topk(**inputs, use_fp64=False)
    assert (sampled[:, 0] == target.argmax()).all()


@pytest.mark.parametrize("temperature", [0.7, 1.0])
@pytest.mark.parametrize("top_k", [16, 64])
def test_rejection_sample_topk_matches_dense(temperature, top_k):
    """With the same top-K proposal in a dense cache (-inf elsewhere), the dense
    path uses the same accept and resample draws, so decisions may differ only
    on floating-point near-ties."""
    torch.manual_seed(7)
    vocab, num_steps, num_trials = 4096, 3, 8192
    target = torch.randn(vocab, device=DEVICE) / temperature
    draft = target * temperature + 0.8 * torch.randn(vocab, device=DEVICE)
    inputs = _build_topk_inputs(
        target, draft, top_k, num_steps, temperature, num_trials
    )
    ours_s, ours_n = rejection_sample_topk(**inputs, use_fp64=False)

    dense = DraftLogitsCache(
        num_trials,
        num_steps,
        vocab,
        dtype=torch.float32,
        fill=float("-inf"),
        device=DEVICE,
    )
    _, idx = torch.topk(draft, top_k)
    row = torch.full((vocab,), float("-inf"), device=DEVICE)
    row[idx] = draft[idx]
    dense.values[:] = row
    ref_s, ref_n = rejection_sample(
        inputs["target_logits"],
        dense,
        inputs["draft_sampled"],
        inputs["cu_num_logits"],
        inputs["pos"],
        inputs["idx_mapping"],
        inputs["expanded_idx_mapping"],
        inputs["expanded_local_pos"],
        inputs["temperature"],
        inputs["seed"],
        num_steps,
    )
    valid = torch.arange(num_steps + 1, device=DEVICE) < ours_n[:, None]
    agree = (ours_n == ref_n) & ((ours_s == ref_s) | ~valid).all(dim=1)
    assert agree.float().mean().item() > 0.999


def _draft_inputs(num_tokens, vocab, top_k, temperature, draft_step=0):
    logits = torch.randn(num_tokens, vocab, device=DEVICE)
    vals, ids = torch.topk(logits, top_k, dim=-1)
    i64 = dict(dtype=torch.int64, device=DEVICE)
    return dict(
        ids=ids,
        logits=vals,
        idx_mapping=torch.arange(num_tokens, **i64),
        temperature=torch.full((num_tokens,), temperature, device=DEVICE),
        seeds=torch.arange(num_tokens, **i64),
        positions=torch.full((num_tokens,), 1000, **i64),
        draft_step=torch.full((num_tokens,), draft_step, **i64),
    )


def test_sample_draft_topk_fused_distribution_and_cache():
    torch.manual_seed(0)
    top_k, temperature, n = 32, 0.8, 20_000
    vals = torch.randn(top_k, device=DEVICE)
    ids = torch.randperm(512, device=DEVICE)[:top_k]
    i64 = dict(dtype=torch.int64, device=DEVICE)
    cache = SparseDraftCache(n, 1, top_k, device=DEVICE)
    sampled = sample_draft_topk_fused(
        cache,
        ids.expand(n, -1),
        vals.expand(n, -1),
        torch.arange(n, **i64),
        torch.full((n,), temperature, device=DEVICE),
        torch.arange(n, **i64),
        torch.full((n,), 1000, **i64),
        torch.zeros(n, **i64),
    )
    expected_logq = torch.log_softmax(vals / temperature, 0)
    torch.testing.assert_close(cache.logq[:n, 0], expected_logq.expand(n, -1))
    assert torch.equal(cache.ids[:n, 0], ids.expand(n, -1))
    freq = (sampled[:, None] == ids).float().mean(0)
    p = expected_logq.exp()
    assert ((freq - p).abs() < 5 * (p * (1 - p) / n).sqrt() + 1e-3).all()


def test_sample_draft_topk_fused_mixes_step_into_seed():
    """A repeated (seed, pos) at another draft step must draw new noise."""
    torch.manual_seed(2)
    step0 = _draft_inputs(4096, 300, 24, 0.9, draft_step=0)
    step3 = {**step0, "draft_step": torch.full_like(step0["draft_step"], 3)}
    a = sample_draft_topk_fused(SparseDraftCache(4096, 4, 24, device=DEVICE), **step0)
    b = sample_draft_topk_fused(SparseDraftCache(4096, 4, 24, device=DEVICE), **step3)
    assert (a != b).float().mean() > 0.2


def test_sample_draft_topk_fused_is_invariant_to_logit_shift():
    """get_top_k_tokens returns unnormalized logits; q' is renormalized over the
    K entries, so a per-row shift must not change the samples or log q'."""
    torch.manual_seed(4)
    x = _draft_inputs(4096, 300, 24, 0.8, draft_step=1)
    shifted = {**x, "logits": x["logits"] + 20 * torch.randn(4096, 1, device=DEVICE)}
    cache_a = SparseDraftCache(4096, 2, 24, device=DEVICE)
    cache_b = SparseDraftCache(4096, 2, 24, device=DEVICE)
    a = sample_draft_topk_fused(cache_a, **x)
    b = sample_draft_topk_fused(cache_b, **shifted)
    torch.testing.assert_close(
        cache_a.logq[:4096, 1], cache_b.logq[:4096, 1], rtol=1e-4, atol=1e-4
    )
    assert (a == b).float().mean().item() > 0.999


def test_sample_draft_topk_fused_is_cudagraph_safe():
    torch.manual_seed(3)
    x = _draft_inputs(256, 128, 16, 0.8, draft_step=2)
    cache = SparseDraftCache(256, 4, 16, device=DEVICE)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            sample_draft_topk_fused(cache, **x)
    torch.cuda.current_stream().wait_stream(side)
    expected = sample_draft_topk_fused(cache, **x)
    expected_ids = cache.ids[:256, 2].clone()
    expected_logq = cache.logq[:256, 2].clone()

    out = torch.empty_like(expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(sample_draft_topk_fused(cache, **x))
    cache.ids.fill_(-1)
    cache.logq.fill_(float("nan"))
    graph.replay()
    assert torch.equal(out, expected)
    assert torch.equal(cache.ids[:256, 2], expected_ids)
    torch.testing.assert_close(cache.logq[:256, 2], expected_logq)


def test_sample_draft_topk_fused_reads_broadcast_draft_step():
    """Autoregressive drafters pass one scalar step for the batch, expanded with
    stride 0; every row must use it."""
    torch.manual_seed(5)
    x = _draft_inputs(1024, 300, 24, 0.8)
    expanded = {**x, "draft_step": torch.tensor(2, device=DEVICE).expand(1024)}
    materialized = {**x, "draft_step": torch.full((1024,), 2, device=DEVICE)}
    cache_a = SparseDraftCache(1024, 4, 24, device=DEVICE)
    cache_b = SparseDraftCache(1024, 4, 24, device=DEVICE)
    a = sample_draft_topk_fused(cache_a, **expanded)
    b = sample_draft_topk_fused(cache_b, **materialized)
    assert torch.equal(a, b)
    assert torch.equal(cache_a.ids, cache_b.ids)
    assert torch.equal(cache_a.logq, cache_b.logq)


def test_draft_top_k_is_rejected_for_methods_without_support():
    with pytest.raises(ValueError, match="draft_top_k is not supported"):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=3,
            draft_sample_method="probabilistic",
            draft_top_k=64,
        )


@pytest.mark.parametrize(
    "config_cls,supported",
    [
        (DFlashSpeculatorConfig, True),
        (EagleSpeculatorConfig, True),
        (Eagle3SpeculatorConfig, True),
        (MTPSpeculatorConfig, True),
        (StandaloneARSpeculatorConfig, True),
        # These sample drafts outside sample_draft, or use per-step heads.
        (DFlash2SpeculatorConfig, False),
        (DSpineSpeculatorConfig, False),
        (LiLiCorrSpeculatorConfig, False),
        (DSparkSpeculatorConfig, False),
        (XPressSpeculatorConfig, False),
        (MultiModuleMTPSpeculatorConfig, False),
    ],
)
def test_draft_top_k_supported_speculators(config_cls, supported):
    assert ("draft_top_k" in config_cls.supported_options) == supported
