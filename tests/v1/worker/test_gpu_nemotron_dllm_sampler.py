# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MaskedDiffusionSampler (Nemotron Labs Diffusion) reports logprobs on the
commit step without host syncs, except for top-k wider than its stash."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.utils.gpu_sync_debug as gsd
from vllm.sampling_params import SamplingParams
from vllm.utils.gpu_sync_debug import with_gpu_sync_check
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu.model_states import nemotron_diffusion as nd
from vllm.v1.worker.gpu.sample.states import SamplingStates

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEVICE = "cuda"
CANVAS_LEN = 8


def _make_sampler(params, vocab, mask, max_logprobs):
    num_reqs = len(params)
    states = nd.NemotronLabsDiffusionRequestStates(
        max_num_reqs=num_reqs,
        canvas_length=CANVAS_LEN,
        mask_token_id=mask,
        max_denoising_steps=2,
        device=torch.device(DEVICE),
    )
    base = SimpleNamespace(
        sampling_states=SamplingStates(num_reqs, vocab),
        req_states=SimpleNamespace(
            draft_tokens=torch.zeros(
                num_reqs, CANVAS_LEN, dtype=torch.int64, device=DEVICE
            )
        ),
    )
    sampler = nd.MaskedDiffusionSampler(
        sampler=base,
        vocab_size=vocab,
        diffusion_states=states,
        selection_policy="leftmost",
        max_logprobs=max_logprobs,
    )
    # Slots start denoising an all-mask canvas.
    for i, p in enumerate(params):
        sampler.add_request(i, p)
    sampler.apply_staged_writes()
    return sampler


def _make_batch(num_reqs):
    cu = np.arange(num_reqs + 1, dtype=np.int64) * CANVAS_LEN
    qsl_np = cu.astype(np.int32)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_draft_tokens=CANVAS_LEN,
        idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int64, device=DEVICE),
        cu_num_logits_np=cu,
        query_start_loc_np=qsl_np,
        query_start_loc=torch.from_numpy(qsl_np).to(DEVICE),
    )


@pytest.fixture
def sync_check_error(monkeypatch):
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", False)
    gsd.enable_gpu_sync_check()
    # Test logprobs independently of Inductor; model tests cover compilation.
    with torch.compiler.set_stance("force_eager"):
        yield


def _denoise_and_commit(sampler, num_reqs, vocab, mask):
    """Run two denoising steps, then the commit step.

    Leftmost reveal over two steps unmasks the first half of each canvas on
    the first step and the rest on the second, where the canvas converges.
    Return the commit step's output and both steps' log-softmax.
    """
    set_random_seed(0)
    step = with_gpu_sync_check(sampler)
    batch = _make_batch(num_reqs)
    log_softmax = []
    for _ in range(2):
        logits = torch.randn(num_reqs * CANVAS_LEN, vocab, device=DEVICE)
        logits[:, mask] = float("-inf")
        assert int(step(logits, batch).num_sampled.sum()) == 0
        log_softmax.append(logits.view(num_reqs, CANVAS_LEN, vocab).log_softmax(-1))
    out = step(torch.randn_like(logits), batch)
    assert out.num_sampled.tolist() == [CANVAS_LEN] * num_reqs
    return out, [lsm.cpu() for lsm in log_softmax]


@pytest.mark.parametrize(
    "num_logprobs",
    [[None, 0, 2, 5], [2, 5, 3, 5]],
    ids=["mixed", "all_topk"],
)
def test_mixed_logprobs_reach_the_commit_step_without_a_host_sync(
    sync_check_error, num_logprobs
):
    """A batch mixing no logprobs, logprobs=0 and top-k widths reports each
    request's own logprobs on its commit step. logprobs=0 reports each
    token's logprob on the step it was revealed; top-k comes from the
    converging step. The host never learns which slots converged or commit,
    so no step syncs with the GPU. A batch where every request asks for
    top-k takes the path that skips the logits copy."""
    vocab, mask = 64, 63
    params = [SamplingParams(temperature=0.0, logprobs=k) for k in num_logprobs]
    sampler = _make_sampler(params, vocab, mask, max_logprobs=20)
    out, (first, converged) = _denoise_and_commit(sampler, len(params), vocab, mask)

    half = CANVAS_LEN // 2
    lists = out.logprobs_tensors.tolists()
    for i, k in enumerate(num_logprobs):
        revealed = torch.cat((first[i, :half], converged[i, half:]))
        tokens = out.sampled_token_ids[i].to(torch.int64).cpu()
        assert torch.equal(tokens, revealed.argmax(-1))
        if k is None:
            continue
        lp = lists.slice_request(i, CANVAS_LEN)
        assert lp.logprob_token_ids[:, 0].tolist() == tokens.tolist()
        if k == 0:
            # Greedy reveals the argmax, so its rank is 1.
            assert (lp.sampled_token_ranks == 1).all()
            torch.testing.assert_close(
                torch.from_numpy(lp.logprobs[:, 0]), revealed.max(-1).values
            )
            continue
        selected = converged[i].gather(-1, tokens[:, None])
        ranks = (converged[i] >= selected).sum(-1)
        assert lp.sampled_token_ranks.tolist() == ranks.tolist()
        torch.testing.assert_close(
            torch.from_numpy(lp.logprobs[:, 0]), selected.squeeze(-1)
        )
        top = converged[i].topk(k)
        assert (lp.logprob_token_ids[:, 1 : k + 1] == top.indices.numpy()).all()
        torch.testing.assert_close(
            torch.from_numpy(lp.logprobs[:, 1 : k + 1]), top.values
        )


def test_full_vocab_logprobs_come_on_the_commit_step(sync_check_error):
    """logprobs=-1 under --max-logprobs -1 asks for every vocab id, wider
    than the stash, next to logprobs=0 and top-2 requests in the same batch.
    Each gets its own number of columns on the commit step, through syncs
    the checker is told about."""
    vocab, mask = 1024, 1023
    params = [
        SamplingParams(temperature=0.0, logprobs=-1),
        SamplingParams(temperature=0.0, logprobs=0),
        SamplingParams(temperature=0.0, logprobs=2),
    ]
    sampler = _make_sampler(params, vocab, mask, max_logprobs=-1)
    out, (_, converged) = _denoise_and_commit(sampler, len(params), vocab, mask)

    lists = out.logprobs_tensors.tolists()
    for i, num_cols in enumerate((1 + vocab, 1, 3)):
        lp = lists.slice_request(i, CANVAS_LEN)
        tokens = out.sampled_token_ids[i].tolist()
        assert lp.logprob_token_ids[:, 0].tolist() == tokens
        # The mask token is never sampled, so it sorts last at -inf.
        torch.testing.assert_close(
            torch.from_numpy(lp.logprobs[:, 1:num_cols]),
            converged[i].sort(-1, descending=True).values[:, : num_cols - 1],
        )
        assert np.isneginf(lp.logprobs[:, num_cols:]).all()
