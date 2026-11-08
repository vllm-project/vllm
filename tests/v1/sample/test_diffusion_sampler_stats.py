# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused row statistics of the DiffusionGemma sampler against the PyTorch
reference: argmax, entropy and softmax must match. A zero temperature must
sample the argmax and a positive one must sample from the distribution."""

import math

import pytest
import torch

from vllm.model_executor.models.diffusion_gemma_sampler import (
    sample_row_stats,
    sample_row_stats_reference,
    stash_converged_logprobs,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logprob import (
    LogprobTokenIdsState,
    compute_topk_scores,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused kernel needs a GPU"
)


@pytest.mark.parametrize("vocab", [5000, 9001])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stats_match_reference(vocab: int, dtype: torch.dtype):
    torch.manual_seed(0)
    num_decode, cl = 9, 4
    logits = (torch.randn(num_decode * cl, vocab, device="cuda") * 3).to(dtype)
    temps = torch.tensor([0.0, 0.7, 2.0, 1.0, 0.0, 0.3, 1.5, 1.0, 0.0], device="cuda")
    argmax, sample, entropy, probs = sample_row_stats(
        logits, temps, cl, seed=1234, probs_dtype=torch.bfloat16
    )
    ref_argmax, _, ref_entropy, ref_probs = sample_row_stats_reference(
        logits, temps, cl, probs_dtype=torch.bfloat16
    )
    assert torch.equal(argmax, ref_argmax)
    torch.testing.assert_close(entropy, ref_entropy, atol=2e-3, rtol=1e-3)
    assert probs is not None
    assert ref_probs is not None
    torch.testing.assert_close(probs.float(), ref_probs.float(), atol=5e-3, rtol=2e-2)
    assert probs.dtype == torch.bfloat16
    assert probs.shape == logits.shape
    # A padded (all-zero) row is uniform: max entropy, argmax 0.
    zero = torch.zeros(1, vocab, device="cuda")
    a0, _, e0, _ = sample_row_stats(zero, temps[:1] + 1.0, 1, 1, None)
    assert a0.item() == 0
    assert abs(e0.item() - torch.log(torch.tensor(float(vocab))).item()) < 1e-3


def test_zero_temperature_is_greedy_and_probs_optional():
    torch.manual_seed(0)
    logits = torch.randn(12, 4097, device="cuda")
    temps = torch.zeros(12, device="cuda")
    argmax, sample, entropy, probs = sample_row_stats(logits, temps, 1, 7, None)
    assert torch.equal(sample, argmax)
    assert probs is None
    assert torch.equal(argmax, logits.argmax(dim=-1))
    # Greedy entropy is that of the reference's clamped temperature.
    _, _, ref_entropy, _ = sample_row_stats_reference(logits, temps, 1, None)
    torch.testing.assert_close(entropy, ref_entropy, atol=2e-3, rtol=1e-3)


def test_positive_temperature_samples_from_the_distribution():
    torch.manual_seed(0)
    rows, vocab = 2000, 4099
    # A peaked row samples its argmax almost always and a flat row rarely.
    peaked = torch.zeros(rows, vocab, device="cuda")
    peaked[:, 17] = 16.0  # p(17) = e^16 / (e^16 + 4098) > 0.999
    flat = torch.zeros(rows, vocab, device="cuda")
    temps = torch.ones(rows, device="cuda")
    _, s_peaked, _, _ = sample_row_stats(peaked, temps, 1, 99, None)
    _, s_flat, _, _ = sample_row_stats(flat, temps, 1, 99, None)
    assert (s_peaked == 17).float().mean().item() > 0.99
    assert s_flat.unique().numel() > rows // 2
    # The same seed repeats the draws and another seed changes them.
    _, s_again, _, _ = sample_row_stats(flat, temps, 1, 99, None)
    _, s_other, _, _ = sample_row_stats(flat, temps, 1, 100, None)
    assert torch.equal(s_flat, s_again)
    assert not torch.equal(s_flat, s_other)


def test_empty_batch():
    logits = torch.empty(0, 128, device="cuda")
    temps = torch.empty(0, device="cuda")
    argmax, sample, entropy, probs = sample_row_stats(
        logits, temps, 1, 1, torch.bfloat16
    )
    assert argmax.numel() == sample.numel() == entropy.numel() == 0
    assert probs is not None and probs.shape == (0, 128)


def test_masked_logits_keep_a_finite_entropy():
    """top_k/top_p leave -inf logits. Masked columns gave NaN entropy, and a
    row with one live column has zero entropy."""
    logits = torch.full((4, 5000), float("-inf"), device="cuda")
    logits[:, 3] = 0.0
    logits[2, 9] = 0.0  # two live columns: entropy log 2
    temps = torch.tensor([1.0, 0.0, 1.0, 0.7], device="cuda")
    argmax, sample, entropy, probs = sample_row_stats(
        logits, temps, 1, 5, torch.float32
    )
    assert torch.isfinite(entropy).all()
    assert entropy[0].item() < 1e-6 and entropy[1].item() < 1e-6
    assert abs(entropy[2].item() - math.log(2.0)) < 1e-4
    assert torch.equal(argmax[[0, 1, 3]], torch.tensor([3, 3, 3], device="cuda"))
    assert probs is not None
    assert probs[0, 3].item() == 1.0 and probs[0].sum().item() == 1.0
    _, _, ref_entropy, _ = sample_row_stats_reference(logits, temps, 1, None)
    assert torch.isfinite(ref_entropy).all()


def test_leading_masked_block_keeps_a_finite_entropy():
    """top_k leaves most of the vocabulary -inf, so whole leading blocks can be
    masked before the first live logit. The running max was -inf through
    them, and -inf - -inf poisoned the row with NaN."""
    logits = torch.full((3, 9001), float("-inf"), device="cuda")
    logits[:, 8500] = 0.0  # the only live column, past the first block
    logits[1, 8600] = 0.0  # two live columns: entropy log 2
    logits[2, 4] = -1.0  # a live column in the first block as well
    temps = torch.tensor([1.0, 1.0, 0.7], device="cuda")
    argmax, sample, entropy, probs = sample_row_stats(
        logits, temps, 1, 5, torch.float32
    )
    _, _, ref_entropy, ref_probs = sample_row_stats_reference(
        logits, temps, 1, torch.float32
    )
    assert torch.isfinite(entropy).all() and torch.isfinite(probs).all()
    assert entropy[0].item() < 1e-6
    assert abs(entropy[1].item() - math.log(2.0)) < 1e-4
    torch.testing.assert_close(entropy, ref_entropy, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(probs, ref_probs, atol=1e-5, rtol=1e-4)
    assert argmax.tolist() == [8500, 8500, 8500]
    assert sample[0].item() == 8500


def _by_value_then_id(ids: torch.Tensor, lps: torch.Tensor) -> torch.Tensor:
    """Ids ordered by logprob, ties by id: torch.topk orders tied top-k ids
    arbitrarily, the stash kernels by id."""
    by_id = ids.argsort(dim=-1, stable=True)
    lps = lps.gather(-1, by_id)
    return ids.gather(-1, by_id).gather(
        -1, lps.argsort(dim=-1, descending=True, stable=True)
    )


@pytest.mark.parametrize("num_logprobs", [0, 5, 128])
@pytest.mark.parametrize("with_ids", [False, True])
@pytest.mark.parametrize("logits_mode", [False, True])
def test_stashed_logprobs_match_compute_topk_scores(
    num_logprobs: int, with_ids: bool, logits_mode: bool
):
    """The stash kernels report what compute_topk_scores reports for the
    scaled rows, up to the order of tied top-k ids, and touch only the
    converged slots' real rows. bf16 logits tie often, so the argmax, rank
    and top-k tie conventions are exercised."""
    torch.manual_seed(0)
    # A row spans several kernel chunks of 16 * 1024 logits.
    n, W, vocab, max_reqs, cl = 5, 4, 40001, 8, 6
    tile = (torch.randn(n * W, vocab, device="cuda") * 3).to(torch.bfloat16)
    # A max tied across chunks: the argmax is the first, the rank counts both.
    tile[1, [7, 39000]] = 20.0
    # A row top_k left 3 logits of: the rest of its top-k ties at -inf.
    tile[3 * W] = float("-inf")
    tile[3 * W, [5, 17000, 40000]] = 1.0
    local = torch.tensor([0, 2, 3], device="cuda")  # requests that want logprobs
    slots = torch.tensor([6, 1, 4], device="cuda")
    keep = torch.tensor([True, False, True], device="cuda")
    temps = torch.tensor([0.7, 1.0, 1.3], device="cuda")
    # The last request's canvas is truncated: its last row is padding.
    valid = torch.tensor([W, W, W - 1], device="cuda")

    ids_state = LogprobTokenIdsState(max_reqs, torch.device("cuda"))
    if with_ids:
        ids_state.add_request(4, SamplingParams(logprob_token_ids=[3, 5, 8000]))
        ids_state.apply_staged_writes()
    max_ids = 3 if with_ids else 0

    scaled = tile.view(n, W, vocab)[local].float() / temps[:, None, None]
    scaled = scaled.view(-1, vocab)
    ref = compute_topk_scores(
        scaled,
        num_logprobs,
        scaled.argmax(dim=-1),
        logprob_token_ids_state=ids_state if with_ids else None,
        expanded_idx_mapping=(
            slots.to(torch.int32).repeat_interleave(W) if with_ids else None
        ),
        max_per_req_token_ids=max_ids,
        logits_mode=logits_mode,
    )

    stash_w = 1 + max(num_logprobs, max_ids) + 2
    stash_ids = torch.full((max_reqs, cl, stash_w), 77, device="cuda")
    stash_lps = torch.ones(max_reqs, cl, stash_w, device="cuda")
    stash_ranks = torch.full((max_reqs, cl), 99, device="cuda")
    stash_converged_logprobs(
        tile,
        W,
        local,
        slots,
        keep,
        valid,
        temps,
        num_logprobs,
        ids_state.num_token_ids.gpu,
        ids_state.token_ids.gpu if with_ids else None,
        stash_ids,
        stash_lps,
        stash_ranks,
        logits_mode,
    )

    width = ref.logprobs.shape[1]
    for j, slot in enumerate(slots.tolist()):
        v = int(valid[j])
        rows = slice(j * W, j * W + v)
        if not keep[j]:
            assert (stash_ids[slot] == 77).all() and (stash_ranks[slot] == 99).all()
            continue
        ids, lps = stash_ids[slot, :v, :width], stash_lps[slot, :v, :width]
        # Both sides divide by the temperature with IEEE fp32 division, so
        # ids and ranks match exactly; the logsumexp is a float reduction.
        torch.testing.assert_close(lps, ref.logprobs[rows], rtol=1.3e-6, atol=1e-5)
        assert torch.equal(stash_ranks[slot, :v], ref.selected_token_ranks[rows])
        assert torch.equal(ids[:, 0], ref.logprob_token_ids[rows, 0])
        assert torch.equal(
            _by_value_then_id(ids[:, 1:], lps[:, 1:]),
            _by_value_then_id(ref.logprob_token_ids[rows, 1:], ref.logprobs[rows, 1:]),
        )
        # Padded as compute_topk_scores pads a mixed batch.
        assert not stash_ids[slot, :v, width:].any()
        assert torch.isneginf(stash_lps[slot, :v, width:]).all()
        # Padding rows and positions past this tile's width are left alone.
        assert (stash_ranks[slot, v:] == 99).all()
