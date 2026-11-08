# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Teacher-forced logprob parity of --monokernel on and off on test_model's
mini MiniMax M3 (random weights, TP4, no checkpoint), its lm_head scaled for
logits of about 4.3 nats (mean top-1 margin 0.83); ``mono_logprob_parity``
describes the method. Both arms run eager, the same 16 requests in one batch.

Mutants of mono's own view of the last sparse layer's weights on rank 0,
measured as the 6th largest KL (past ``OUTLIERS``) over ``KL``: experts 0 and
1's w2 swapped (``expert_swap``, the test's; 24 positions over the bounds)
19.6x; expert 5's w13 scales an E8M0 step up 50x; expert 7's router bias +1
22.8x. One K block of every expert's w13 scales an E8M0 step up is 2.2x, the
sensitivity limit."""

import dataclasses

import pytest
import torch

from tests.models import mono_logprob_parity as P
from tests.models.minimax_m3.amd_specialized_decode import test_model as M

pytestmark = M.pytestmark
model_dir, _env = M.model_dir, M._env

LM_HEAD_SCALE = 3
REQUESTS, STEPS = 16, 16
# Null (off vs off: same engine, reversed submission order, relaunched, FULL
# graphs; on eager vs on FULL graphs, bit-exact; request seeds 0 and 1): KL
# max 1.13e-3, |dlogprob| max 0.06 but at MoE routing near-ties, KL 0.03-0.26
# at 1-5 positions a pair. Running the batch in two halves instead moves 6% of
# the positions to KL 0.01-0.5 (native's ops are not batch invariant): both
# arms take the same steps.
KL, DLP = 2e-3, 0.15
# the near-ties seen over 10 pairs at this seed, at most 3 a pair: (10, 3),
# (0, 10), (7, 10) and (4, 7)
OUTLIERS = 5
ENTROPY = (3.0, 6.0)
MIN_MARGIN = 0.3


@torch.no_grad()
def _prepare(worker):
    M._randomize(worker)
    worker.model_runner.get_model().lm_head.weight.mul_(LM_HEAD_SCALE)


def _expert_swap(w):
    w2 = w.w2.view(torch.uint8).clone()
    w2[[0, 1]] = w2[[1, 0]]
    return dataclasses.replace(w, w2=w2.view(w.w2.dtype))


def _expert_scale(w):
    s13 = w.s13.clone()
    s13[5] += 1
    return dataclasses.replace(w, s13=s13)


def _e8m0_block(w):
    s13 = w.s13.clone()
    s13[:, :, 0] += 1
    return dataclasses.replace(w, s13=s13)


def _router_bias(w):
    bias = w.router_bias.clone()
    bias[7] += 1
    return dataclasses.replace(w, router_bias=bias)


MUTANTS = dict(
    expert_swap=_expert_swap,
    expert_scale=_expert_scale,
    e8m0_block=_e8m0_block,
    router_bias=_router_bias,
)


def _mutate(worker, mutant):
    """Rebinds rank 0's mono view of the last sparse layer's weights (the
    layer's own tensors, also native's, stay) to a corrupted copy. Eager
    only: a FULL graph keeps the pointers it captured."""
    from vllm.distributed import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() != 0:
        return
    executor = M._executor(worker)
    lid = max(executor._weights)
    executor._weights[lid] = MUTANTS[mutant](executor._weights[lid])
    executor._kernels[lid] = executor._layer_kernel(lid)


def _requests():
    return P.forced_requests(M.VOCAB, REQUESTS, STEPS, seed=0)


def _forced(vllm_runner, model_dir, mono, mutant=None):
    prompts, traces = _requests()
    with M._engine(vllm_runner, model_dir, mono=mono) as runner:
        llm = runner.llm
        llm.collective_rpc(_prepare)
        if mutant is not None:
            llm.collective_rpc(_mutate, args=(mutant,))
        if not mono:
            return P.run_forced(llm, prompts, traces)
        start = llm.collective_rpc(M._count_launches)
        run = P.run_forced(llm, prompts, traces)
        _assert_engaged(llm, start)
        return run


def _assert_engaged(llm, start):
    """Since ``start``, alike on every rank: the prefill ran stock and each of
    the ``STEPS - 1`` batched decode steps one mono launch a sparse layer, so
    a silent fallback cannot pass as parity."""
    moved = {
        tuple(a - b for a, b in zip(after, before))
        for after, before in zip(llm.collective_rpc(M._engagement), start)
    }
    decodes, sparse = STEPS - 1, len(M.SPARSE)
    expected = (STEPS, decodes, 1, *[decodes] * sparse, sparse * decodes)
    assert moved == {expected}, (moved, expected)


@pytest.fixture(scope="module")
def native(vllm_runner, model_dir):
    return _forced(vllm_runner, model_dir, mono=False)


def test_logprobs_match_native(vllm_runner, model_dir, native):
    """Every forced decode position's full-vocabulary logprobs with
    --monokernel eager within the off-vs-off noise of native's."""
    m = P.compare(native, _forced(vllm_runner, model_dir, mono=True))
    print(P.summary(m))
    P.assert_informative(m, ENTROPY, MIN_MARGIN)
    P.assert_parity(m, KL, DLP, OUTLIERS)


def test_mutant_fails_parity(vllm_runner, model_dir, native):
    """The bounds are not vacuous: a corrupted copy of mono's expert weights
    (``expert_swap``) fails them by over 10x."""
    m = P.compare(
        native, _forced(vllm_runner, model_dir, mono=True, mutant="expert_swap")
    )
    print(P.summary(m))
    assert P.worst(m, "kl", OUTLIERS) >= 10 * KL, P.summary(m)
