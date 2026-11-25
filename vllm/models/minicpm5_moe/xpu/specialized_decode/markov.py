# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-vocabulary greedy Markov sampling for FP16 DSpark."""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _markov_partition(
    base,
    w1,
    w2,
    weight_norms,
    seeds,
    anchor,
    output,
    values,
    indices,
    vocab: tl.constexpr,
    step: tl.constexpr,
    rows: tl.constexpr,
    prefer_nan: tl.constexpr,
):
    tile = tl.program_id(0)
    tokens = tile * rows + tl.arange(0, rows)
    lanes = tl.arange(0, 256)
    if step == 0:
        previous = tl.load(anchor).to(tl.int32)
    else:
        previous = tl.load(output + step - 1).to(tl.int32)
    embed = tl.load(w1 + previous * 256 + lanes).to(tl.float32)
    embed_norm = tl.sqrt(tl.sum(embed * embed, axis=0))
    seed = tl.load(seeds + step).to(tl.int32)
    seed_weight = tl.load(w2 + seed * 256 + lanes).to(tl.float32)
    seed_bias = tl.sum(seed_weight * embed, axis=0)
    # The relative slack covers FP32 reduction and norm error for rank 256.
    seed_error = tl.load(weight_norms + seed) * embed_norm * 0.0001
    seed_lower = (seed_bias - seed_error).to(tl.float16).to(tl.float32)
    seed_logit = tl.load(base + step * vocab + seed).to(tl.float32)
    seed_logit = tl.where(seed_logit == seed_logit, seed_logit, -float("inf"))
    lower = (seed_logit + seed_lower).to(tl.float16).to(tl.float32)

    logits = tl.load(base + step * vocab + tokens, tokens < vocab, other=-float("inf"))
    nan = logits != logits
    logits = tl.where(nan, -float("inf"), logits).to(tl.float32)
    norms = tl.load(weight_norms + tokens, tokens < vocab, other=0)
    bound = (norms * embed_norm * 1.0001).to(tl.float16).to(tl.float32)
    upper = (logits + bound).to(tl.float16).to(tl.float32)
    needed = (tokens < vocab) & (upper >= lower)
    value = -float("inf")
    index = vocab
    if tl.sum(needed.to(tl.int32), axis=0) != 0:
        weights = tl.load(
            w2 + tokens[:, None] * 256 + lanes[None, :],
            needed[:, None],
            other=0,
        ).to(tl.float32)
        bias = tl.sum(weights * embed[None, :], axis=1).to(tl.float16).to(tl.float32)
        scores = (logits + bias).to(tl.float16).to(tl.float32)
        value = tl.max(tl.where(needed, scores, -float("inf")), axis=0)
        index = tl.min(tl.where(needed & (scores == value), tokens, vocab), axis=0)
    if prefer_nan:
        first_nan = tl.min(tl.where(nan, tokens, vocab), axis=0)
        value = tl.where(first_nan < vocab, float("nan"), value)
        index = tl.where(first_nan < vocab, first_nan, index)
    tl.store(values + tile, value)
    tl.store(indices + tile, index)


@triton.jit
def _reduce_markov(
    values,
    indices,
    output,
    step: tl.constexpr,
    groups: tl.constexpr,
    width: tl.constexpr,
    vocab: tl.constexpr,
    prefer_nan: tl.constexpr,
):
    positions = tl.arange(0, width)
    scores = tl.load(values + positions, positions < groups, other=-float("inf"))
    ids = tl.load(indices + positions, positions < groups, other=vocab)
    if prefer_nan:
        nan = scores != scores
        first_nan = tl.min(tl.where(nan, ids, vocab), axis=0)
        scores = tl.where(nan, -float("inf"), scores)
    best = tl.max(scores, axis=0)
    winner = tl.min(tl.where(scores == best, ids, vocab), axis=0)
    if prefer_nan:
        winner = tl.where(first_nan < vocab, first_nan, winner)
    tl.store(output + step, winner)


def markov_greedy(base, w1, w2, norms, anchor, output, *, prefer_nan=False):
    steps, vocab = base.shape
    seeds = base.argmax(-1)
    rows = 32
    groups = triton.cdiv(vocab, rows)
    values = torch.empty(groups, device=base.device, dtype=torch.float32)
    indices = torch.empty(groups, device=base.device, dtype=torch.int32)
    for step in range(steps):
        _markov_partition[(groups,)](
            base,
            w1,
            w2,
            norms,
            seeds,
            anchor,
            output,
            values,
            indices,
            vocab,
            step,
            rows,
            prefer_nan,
            num_warps=4,
        )
        _reduce_markov[(1,)](
            values,
            indices,
            output,
            step,
            groups,
            triton.next_power_of_2(groups),
            vocab,
            prefer_nan,
            num_warps=8,
        )


class PrunedMarkovSampler:
    """Use a score bound to skip projection rows in greedy, single-request graphs."""

    def __init__(self, w1, w2, norms, *, prefer_nan):
        self.w1 = w1
        self.w2 = w2
        self.norms = norms
        self.prefer_nan = prefer_nan

    @classmethod
    def from_model(cls, model, *, prefer_nan=False):
        head = model.model.markov_head
        w1, w2 = head.markov_w1.weight, head.markov_w2.weight
        processor = model.logits_processor
        if (
            model.draft_id_to_target_id is not None
            or getattr(model, "has_recurrent_markov_head", False)
            or hasattr(model, "apply_output_logit_transform")
            or processor.logits_as_input
            or processor.scale != 1
            or processor.soft_cap is not None
            or processor.head_dtype not in (None, torch.float16)
            or (model.lm_head.tp_size > 1 and not processor.use_all_gather)
            or head.markov_w2.tp_size != 1
            or not isinstance(head.markov_w2.quant_method, UnquantizedEmbeddingMethod)
            or processor.org_vocab_size != w2.shape[0]
            or w1.shape != w2.shape
            or w2.shape != (130560, 256)
            or w1.device != w2.device
            or not w2.is_xpu
            or w1.dtype != torch.float16
            or w2.dtype != torch.float16
            or not w1.is_contiguous()
            or not w2.is_contiguous()
        ):
            return None
        norms = w2.detach().float().square().sum(-1).sqrt()
        embed_max = w1.detach().float().square().sum(-1).sqrt().amax()
        if not bool(norms.amax() * embed_max < 65000):
            return None
        logger.info("Enabled FP16 DSpark Markov score bounds.")
        return cls(w1, w2, norms, prefer_nan=prefer_nan)

    def __call__(self, speculator, num_reqs, head_hidden):
        if (
            num_reqs != 1
            or speculator.num_speculative_steps != 7
            or speculator.draft_logits is not None
            or speculator.enable_adaptive_verification
            or speculator.use_confidence_head
            or speculator.draft_watermarker is not None
        ):
            return False
        model = speculator.model
        sample_hidden = head_hidden[speculator.sample_indices[:7]]
        base = model.compute_draft_logits(sample_hidden)
        if base.dtype != torch.float16 or not base.is_contiguous():
            return False
        anchor = speculator.input_buffers.input_ids[speculator._anchor_idx[:1]]
        output = speculator.draft_tokens[0, :7]
        markov_greedy(
            base,
            self.w1,
            self.w2,
            self.norms,
            anchor,
            output,
            prefer_nan=self.prefer_nan,
        )
        return True
