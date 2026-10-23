# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-sharded DSpark draft sampling must reproduce the replicated path bit
for bit: the same draft ids on every rank and, on each request's owner rank,
the same cached draft-logits rows the rejection sampler reads next step.

Every TP rank is simulated in-process: each rank stages its vocab shard, the
all-to-all and the draft-id all-gather are emulated from the staged buffers.
The LM-head, Markov embedding and Markov bias run the real layers (and the
platform's GEMM dispatch).
"""

from types import SimpleNamespace
from unittest import mock

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
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.sample import batch_shard
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as dspark_speculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

DEVICE = "cuda"
TP_SIZE = 4
HIDDEN = 128
# M3.1's rank: the Markov GEMM's rows then depend on its row count, so a GEMM
# over only the owned rows fails these tests.
MARKOV_RANK = 256
NUM_STEPS = 3
MAX_NUM_REQS = 64
SHARD_WIDTH = 2048
# The draft vocab is smaller than TP_SIZE * SHARD_WIDTH: the last shard ends
# in vocab padding, as with a padded LM head.
DRAFT_VOCAB = 8000
DTYPE = torch.bfloat16

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


class _LMHeadShard(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.quant_method = UnquantizedEmbeddingMethod()
        self.tp_size = TP_SIZE
        self.num_embeddings_per_partition = weight.shape[0]


class _DraftModel(nn.Module):
    """One rank's view of a vocab-parallel Qwen3 DSpark draft head."""

    compute_draft_logits_local = Qwen3DSparkForCausalLM.compute_draft_logits_local
    markov_embed = Qwen3DSparkForCausalLM.markov_embed
    markov_bias = Qwen3DSparkForCausalLM.markov_bias
    map_draft_to_target = Qwen3DSparkForCausalLM.map_draft_to_target

    def __init__(self, shards, rank, markov_head, d2t):
        super().__init__()
        self.shards = shards
        self.lm_head = shards[rank]
        self.model = SimpleNamespace(markov_head=markov_head)
        self.logits_processor = LogitsProcessor(DRAFT_VOCAB)
        self.draft_id_to_target_id = d2t

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The replicated path: every shard (each rank's GEMM), all-gathered
        # along the vocab, then the vocab padding dropped.
        shards = [
            self.logits_processor(shard, hidden_states, skip_gather=True)
            for shard in self.shards
        ]
        return torch.cat(shards, dim=-1)[..., :DRAFT_VOCAB]


def _make_weights(reduced_vocab: bool):
    g = torch.Generator(device=DEVICE).manual_seed(0)
    lm_head = torch.randn(
        TP_SIZE * SHARD_WIDTH, HIDDEN, generator=g, device=DEVICE, dtype=DTYPE
    )
    shards = [_LMHeadShard(w) for w in lm_head.split(SHARD_WIDTH)]
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
    return shards, markov_head, d2t, target_vocab


def _make_speculator(
    shards, markov_head, d2t, target_vocab, rank, probabilistic, inputs
) -> DSparkSpeculator:
    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.model = _DraftModel(shards, rank, markov_head, d2t)
    spec.num_speculative_steps = NUM_STEPS
    spec.max_num_reqs = MAX_NUM_REQS
    spec.device = torch.device(DEVICE)
    spec.dtype = DTYPE
    spec.use_fp64_gumbel = False
    spec.draft_watermarker = None
    spec.acceptance_estimator = None
    spec.use_confidence_head = False
    spec._draft_topk = None
    spec._draft_sharder = None
    spec.draft_logits_num_shards = 1
    spec.draft_logits_shard_idx = 0
    spec.draft_logits = (
        torch.zeros(MAX_NUM_REQS, NUM_STEPS, target_vocab, dtype=DTYPE, device=DEVICE)
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
    spec._req_rows = torch.arange(MAX_NUM_REQS, dtype=torch.int64, device=DEVICE)
    spec.draft_tokens = torch.zeros(
        MAX_NUM_REQS, NUM_STEPS, dtype=torch.int64, device=DEVICE
    )
    # Per-step inputs, shared by value: the same prepared buffers on all ranks.
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


def _fake_group(rank: int):
    return SimpleNamespace(rank_in_group=rank, world_size=TP_SIZE, device_group=None)


def _run_sharded_tails(specs, slots: np.ndarray, num_reqs_padded: int) -> None:
    """Run _sample_after_forward on every simulated rank, emulating the
    all-to-all from the staged send buffers and the all-gather in two passes
    (gather the owners' ids, then rerun with the gathered result)."""
    input_batch = SimpleNamespace(idx_mapping_np=slots, num_reqs=len(slots))
    sends = [spec._draft_sharder.send for spec in specs]

    def run(rank, gathered):
        def fake_all_to_all(
            output, input, output_split_sizes, input_split_sizes, group
        ):
            bounds = np.concatenate([[0], np.cumsum(input_split_sizes)])
            output.copy_(torch.cat([s[bounds[rank] : bounds[rank + 1]] for s in sends]))

        def fake_all_gather(tensor, dim):
            local_ids.append(tensor.clone())
            return gathered if gathered is not None else tensor.repeat(TP_SIZE, 1)

        local_ids: list[torch.Tensor] = []
        with (
            mock.patch.object(
                batch_shard, "get_tp_group", return_value=_fake_group(rank)
            ),
            mock.patch.object(
                batch_shard.torch.distributed, "all_to_all_single", fake_all_to_all
            ),
            mock.patch.object(
                batch_shard, "tensor_model_parallel_all_gather", fake_all_gather
            ),
        ):
            specs[rank]._sample_after_forward(input_batch, num_reqs_padded)
        return local_ids[0]

    local = [run(rank, None) for rank in range(TP_SIZE)]
    gathered = torch.cat(local)
    for rank in range(TP_SIZE):
        run(rank, gathered)


def _enable_sharding(spec, rank: int) -> None:
    with (
        mock.patch.object(batch_shard, "get_tp_group", return_value=_fake_group(rank)),
        mock.patch.object(
            dspark_speculator, "get_tp_group", return_value=_fake_group(rank)
        ),
    ):
        spec.enable_batch_sharded_sampling()
    assert spec._draft_sharder is not None
    if spec.draft_logits is not None:
        # As _init_draft_logits sizes it: this rank's requests only, one row
        # per owned slot.
        spec.draft_logits_num_shards = TP_SIZE
        spec.draft_logits_shard_idx = rank
        spec.draft_logits = spec.draft_logits.new_zeros(
            cdiv(MAX_NUM_REQS, TP_SIZE), *spec.draft_logits.shape[1:]
        )


def _check_against_replicated(ref, specs, slots: np.ndarray) -> None:
    num_reqs = len(slots)
    for rank, spec in enumerate(specs):
        assert torch.equal(spec.draft_tokens[:num_reqs], ref.draft_tokens[:num_reqs])
        if ref.draft_logits is None:
            continue
        owned_rows = set()
        for slot in slots:
            assert ref.draft_logits[slot].any()
            if slot % TP_SIZE == rank:
                row = slot // TP_SIZE
                owned_rows.add(row)
                assert torch.equal(spec.draft_logits[row], ref.draft_logits[slot])
        # Only the owner (the verifying rank) caches a request's rows.
        for row in range(spec.draft_logits.shape[0]):
            if row not in owned_rows:
                assert not spec.draft_logits[row].any()


# (slots, padded batch size). The Markov bias GEMM keeps the padded batch's
# row count; 4 rows pick the skinny GEMM on ROCm, more rows the library GEMM.
_BATCHES = [
    ([5, 9, 13], 4),  # rank 1 owns every request, the others none
    ([0, 7, 2, 33, 41, 12, 63, 18, 22, 3, 50, 44, 29], 16),
    (None, 40),  # 37 random slots
]


@pytest.mark.parametrize("probabilistic", [True, False])
@pytest.mark.parametrize("reduced_vocab", [False, True])
@pytest.mark.parametrize("batch", range(len(_BATCHES)))
def test_sharded_draft_sampling_matches_replicated(
    default_vllm_config, probabilistic: bool, reduced_vocab: bool, batch: int
):
    rng = np.random.default_rng(batch)
    slots, num_reqs_padded = _BATCHES[batch]
    if slots is None:
        slots = rng.choice(MAX_NUM_REQS, size=37, replace=False)
    slots = np.asarray(slots, dtype=np.int64)
    shards, markov_head, d2t, target_vocab = _make_weights(reduced_vocab)
    inputs = _make_inputs(rng, slots, target_vocab)
    head_hidden = torch.randn(
        num_reqs_padded * NUM_STEPS, HIDDEN, device=DEVICE, dtype=DTYPE
    )

    def make(rank):
        spec = _make_speculator(
            shards, markov_head, d2t, target_vocab, rank, probabilistic, inputs
        )
        spec._run_model = lambda *args, **kwargs: head_hidden
        return spec

    ref = make(0)
    ref._generate_draft(num_reqs_padded, num_reqs_padded * NUM_STEPS, None, None, None)

    specs = [make(rank) for rank in range(TP_SIZE)]
    for rank, spec in enumerate(specs):
        _enable_sharding(spec, rank)
        spec._generate_draft(
            num_reqs_padded, num_reqs_padded * NUM_STEPS, None, None, None
        )
    _run_sharded_tails(specs, slots, num_reqs_padded)
    _check_against_replicated(ref, specs, slots)


def test_sharded_draft_stage_replays_in_cuda_graph(default_vllm_config):
    """The staged half of the draft step is captured once and replayed for a
    different batch of the same padded size, like the drafter's FULL graph."""
    num_reqs_padded = 16
    rng = np.random.default_rng(3)
    shards, markov_head, d2t, target_vocab = _make_weights(reduced_vocab=False)
    capture_slots = np.arange(num_reqs_padded, dtype=np.int64)
    capture_inputs = _make_inputs(rng, capture_slots, target_vocab)
    head_hidden = torch.zeros(
        num_reqs_padded * NUM_STEPS, HIDDEN, device=DEVICE, dtype=DTYPE
    )

    specs = []
    graphs = []
    for rank in range(TP_SIZE):
        spec = _make_speculator(
            shards, markov_head, d2t, target_vocab, rank, True, capture_inputs
        )
        spec._run_model = lambda *args, **kwargs: head_hidden
        _enable_sharding(spec, rank)
        args = (num_reqs_padded, num_reqs_padded * NUM_STEPS, None, None, None)
        spec._generate_draft(*args)  # warm up outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            spec._generate_draft(*args)
        specs.append(spec)
        graphs.append(graph)

    # A smaller batch with skewed ownership replays the same graphs.
    slots = np.array([4, 8, 12, 1, 31, 60, 2, 16, 20], dtype=np.int64)
    inputs = _make_inputs(rng, slots, target_vocab)
    head_hidden.copy_(torch.randn_like(head_hidden))
    for spec, graph in zip(specs, graphs):
        for name in ("sample_idx_mapping", "sample_indices", "sample_pos"):
            getattr(spec, name).copy_(inputs[name])
        spec.temperature.copy_(inputs["temperature"])
        spec.seeds.copy_(inputs["seeds"])
        spec.input_buffers.input_ids.copy_(inputs["input_ids"])
        graph.replay()
    _run_sharded_tails(specs, slots, num_reqs_padded)

    ref = _make_speculator(shards, markov_head, d2t, target_vocab, 0, True, inputs)
    ref._run_model = lambda *args, **kwargs: head_hidden
    ref._generate_draft(num_reqs_padded, num_reqs_padded * NUM_STEPS, None, None, None)
    _check_against_replicated(ref, specs, slots)


def _sample_sequential_unfused(spec, num_reqs: int, head_hidden: torch.Tensor):
    """The replicated Markov loop with the bias added outside the Gumbel kernel."""
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


@pytest.mark.parametrize("probabilistic", [True, False])
@pytest.mark.parametrize("reduced_vocab", [False, True])
@pytest.mark.parametrize("batch", range(len(_BATCHES)))
def test_replicated_markov_sampling_matches_unfused(
    default_vllm_config, probabilistic: bool, reduced_vocab: bool, batch: int
):
    """In every mode, the replicated loop gives the draft ids and cached draft
    logits of adding the Markov bias first. Probabilistic full-vocab drafting
    adds it inside the Gumbel kernel."""
    rng = np.random.default_rng(batch)
    slots, num_reqs_padded = _BATCHES[batch]
    if slots is None:
        slots = rng.choice(MAX_NUM_REQS, size=37, replace=False)
    shards, markov_head, d2t, target_vocab = _make_weights(reduced_vocab)
    inputs = _make_inputs(rng, np.asarray(slots, dtype=np.int64), target_vocab)
    head_hidden = torch.randn(
        num_reqs_padded * NUM_STEPS, HIDDEN, device=DEVICE, dtype=DTYPE
    )
    specs = [
        _make_speculator(
            shards, markov_head, d2t, target_vocab, 0, probabilistic, inputs
        )
        for _ in range(2)
    ]
    specs[0]._sample_sequential(num_reqs_padded, head_hidden)
    _sample_sequential_unfused(specs[1], num_reqs_padded, head_hidden)
    assert torch.equal(specs[0].draft_tokens, specs[1].draft_tokens)
    if probabilistic:
        draft_logits = specs[0].draft_logits
        assert draft_logits is not None and draft_logits.any()
        assert torch.equal(draft_logits, specs[1].draft_logits)
