# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's Markov loop must give the draft ids and cached draft logits of
adding the Markov bias before sampling, whether or not the Gumbel kernel adds
the bias itself. The LM head, Markov embedding and Markov bias run the real
layers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.qwen3_dspark import (
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
)
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

DEVICE = "cuda"
HIDDEN = 128
MARKOV_RANK = 256
NUM_STEPS = 3
MAX_NUM_REQS = 64
DRAFT_VOCAB = 8000
DTYPE = torch.bfloat16

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


class _LMHead(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.quant_method = UnquantizedEmbeddingMethod()


class _DraftModel(nn.Module):
    markov_embed = Qwen3DSparkForCausalLM.markov_embed
    markov_bias = Qwen3DSparkForCausalLM.markov_bias
    map_draft_to_target = Qwen3DSparkForCausalLM.map_draft_to_target

    def __init__(self, lm_head, markov_head, d2t):
        super().__init__()
        self.lm_head = lm_head
        self.model = SimpleNamespace(markov_head=markov_head)
        self.logits_processor = LogitsProcessor(DRAFT_VOCAB)
        self.draft_id_to_target_id = d2t

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)


def _make_weights(reduced_vocab: bool):
    g = torch.Generator(device=DEVICE).manual_seed(0)
    lm_head = _LMHead(
        torch.randn(DRAFT_VOCAB, HIDDEN, generator=g, device=DEVICE, dtype=DTYPE)
    )
    target_vocab = DRAFT_VOCAB + 500 if reduced_vocab else DRAFT_VOCAB
    markov_head = DSparkMarkovHead(
        target_vocab, DRAFT_VOCAB, MARKOV_RANK, prefix="markov_head"
    ).to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        markov_head.markov_w1.weight.normal_(generator=g)
        markov_head.markov_w2.weight.normal_(generator=g)
    d2t = None
    if reduced_vocab:
        # Draft id i maps to target id i + d2t[i] (sorted, unique targets).
        target_ids = torch.randperm(target_vocab, generator=g, device=DEVICE)
        target_ids = target_ids[:DRAFT_VOCAB].sort().values
        d2t = target_ids - torch.arange(DRAFT_VOCAB, device=DEVICE)
    return lm_head, markov_head, d2t, target_vocab


def _make_speculator(
    lm_head, markov_head, d2t, target_vocab, probabilistic, inputs
) -> DSparkSpeculator:
    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.model = _DraftModel(lm_head, markov_head, d2t)
    spec.num_speculative_steps = NUM_STEPS
    spec.max_num_reqs = MAX_NUM_REQS
    spec.device = torch.device(DEVICE)
    spec.dtype = DTYPE
    spec.use_fp64_gumbel = False
    spec.draft_watermarker = None
    spec.acceptance_estimator = None
    spec.use_confidence_head = False
    spec._draft_topk = None
    spec.draft_logits = (
        DraftLogitsCache(
            MAX_NUM_REQS,
            NUM_STEPS,
            target_vocab,
            dtype=DTYPE,
            fill=0.0,
            device=DEVICE,
            sharded=False,
        )
        if probabilistic
        else None
    )
    spec._d2t_scatter_index = None
    spec._draft_scatter_buf = None
    if probabilistic and d2t is not None:
        spec._d2t_scatter_index = torch.arange(DRAFT_VOCAB, device=DEVICE) + d2t
        spec._draft_scatter_buf = torch.full(
            (MAX_NUM_REQS, target_vocab), float("-inf"), dtype=DTYPE, device=DEVICE
        )
    spec._step_cols = torch.arange(NUM_STEPS, dtype=torch.int32, device=DEVICE)
    spec._anchor_idx = (
        torch.arange(MAX_NUM_REQS, dtype=torch.int64, device=DEVICE) * NUM_STEPS
    )
    spec.draft_tokens = torch.zeros(
        MAX_NUM_REQS, NUM_STEPS, dtype=torch.int64, device=DEVICE
    )
    spec.temperature = inputs["temperature"].clone()
    spec.seeds = inputs["seeds"].clone()
    spec.sample_indices = inputs["sample_indices"].clone()
    spec.sample_pos = inputs["sample_pos"].clone()
    spec.sample_idx_mapping = inputs["sample_idx_mapping"].clone()
    spec.input_buffers = SimpleNamespace(input_ids=inputs["input_ids"].clone())
    return spec


def _make_inputs(rng: np.random.Generator, slots: np.ndarray, target_vocab: int):
    """Buffers as prepare_dflash_inputs leaves them for `slots` (the batch)."""
    num_reqs = len(slots)
    max_rows = MAX_NUM_REQS * NUM_STEPS
    sample_idx_mapping = np.full(max_rows, -1, dtype=np.int32)
    sample_idx_mapping[: num_reqs * NUM_STEPS] = np.repeat(slots, NUM_STEPS)
    sample_indices = np.zeros(max_rows, dtype=np.int64)
    sample_indices[: num_reqs * NUM_STEPS] = np.arange(num_reqs * NUM_STEPS)
    sample_pos = np.zeros(max_rows, dtype=np.int64)
    last_pos = rng.integers(10, 5000, size=num_reqs)
    sample_pos[: num_reqs * NUM_STEPS] = (
        last_pos[:, None] + 2 + np.arange(NUM_STEPS)
    ).ravel()
    temperature = np.zeros(MAX_NUM_REQS, dtype=np.float32)
    # A mix of greedy and seeded random requests.
    temperature[slots] = rng.choice([0.0, 0.6, 1.0, 1.3], size=num_reqs)
    seeds = rng.integers(0, 2**62, size=MAX_NUM_REQS, dtype=np.int64)
    input_ids = rng.integers(0, target_vocab, size=max_rows, dtype=np.int32)

    def dev(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE)

    return {
        "sample_idx_mapping": dev(sample_idx_mapping),
        "sample_indices": dev(sample_indices),
        "sample_pos": dev(sample_pos),
        "temperature": dev(temperature),
        "seeds": dev(seeds),
        "input_ids": dev(input_ids),
    }


def _sample_sequential_unfused(spec, num_reqs: int, head_hidden: torch.Tensor):
    """The Markov loop with the bias added outside the Gumbel kernel."""
    num_sample = num_reqs * NUM_STEPS
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base_logits = spec.model.compute_draft_logits(sample_hidden)
    base_logits = base_logits.view(num_reqs, NUM_STEPS, -1)
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, NUM_STEPS)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, NUM_STEPS)
    prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    for i in range(NUM_STEPS):
        bias = spec.model.markov_bias(spec.model.markov_embed(prev))
        prev = spec._sample_logits(
            base_logits[:, i] + bias, idx_map[:, i], sample_pos[:, i], i
        )
        spec.draft_tokens[:num_reqs, i] = prev


# (slots, padded batch size); padded requests have no slot.
_BATCHES = [
    ([5, 9, 13], 4),
    ([0, 7, 2, 33, 41, 12, 63, 18, 22, 3, 50, 44, 29], 16),
    (None, 40),  # 37 random slots
]


@pytest.mark.parametrize("probabilistic", [True, False])
@pytest.mark.parametrize("reduced_vocab", [False, True])
@pytest.mark.parametrize("batch", range(len(_BATCHES)))
def test_markov_sampling_matches_unfused(
    default_vllm_config, probabilistic: bool, reduced_vocab: bool, batch: int
):
    """In every mode, the Markov loop gives the draft ids and cached draft
    logits of adding the Markov bias first. Probabilistic full-vocab drafting
    adds it inside the Gumbel kernel."""
    rng = np.random.default_rng(batch)
    slots, num_reqs_padded = _BATCHES[batch]
    if slots is None:
        slots = rng.choice(MAX_NUM_REQS, size=37, replace=False)
    lm_head, markov_head, d2t, target_vocab = _make_weights(reduced_vocab)
    inputs = _make_inputs(rng, np.asarray(slots, dtype=np.int64), target_vocab)
    head_hidden = torch.randn(
        num_reqs_padded * NUM_STEPS, HIDDEN, device=DEVICE, dtype=DTYPE
    )
    specs = [
        _make_speculator(lm_head, markov_head, d2t, target_vocab, probabilistic, inputs)
        for _ in range(2)
    ]
    specs[0]._sample_sequential(num_reqs_padded, head_hidden)
    _sample_sequential_unfused(specs[1], num_reqs_padded, head_hidden)
    assert torch.equal(specs[0].draft_tokens, specs[1].draft_tokens)
    if probabilistic:
        assert specs[0].draft_logits is not None and specs[1].draft_logits is not None
        draft_logits = specs[0].draft_logits.values
        assert draft_logits.any()
        assert torch.equal(draft_logits, specs[1].draft_logits.values)
