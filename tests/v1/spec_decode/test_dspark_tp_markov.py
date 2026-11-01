# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The vocab-parallel DSpark Markov loop draws the same tokens and caches the
same draft logits as the replicated loop, bitwise, and every TP rank falls back
to the replicated loop alike."""

import queue

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from tests.utils import multi_gpu_test
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.dspark import tp_markov
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.tp_markov import TPMarkovSampler

N_SPEC = 7
MARKOV_RANK = 64
MAX_REQS = 128


class _MarkovDraftModel(nn.Module):
    """A DSpark draft model's Markov head, wired as in Qwen3 DSpark. The base
    draft logits are fed in as the hidden states."""

    def __init__(self, vocab: int):
        super().__init__()
        self.model = nn.Module()
        self.model.markov_head = DSparkMarkovHead(
            vocab, vocab, MARKOV_RANK, prefix="markov_head"
        )
        self.logits_processor = LogitsProcessor(vocab)
        # Rank-identical weights.
        g = torch.Generator(device="cuda").manual_seed(0)
        head = self.model.markov_head
        for w, scale in ((head.markov_w1.weight, 1.0), (head.markov_w2.weight, 0.25)):
            w.data.copy_(torch.randn(w.shape, generator=g) * scale)
        head.markov_w2.quant_method.process_weights_after_loading(head.markov_w2)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)


def _speculator(vocab: int, device: torch.device) -> DSparkSpeculator:
    # A DSparkSpeculator is built from a DSpark checkpoint; set the state that
    # its sequential sampling reads, as test_dspark_topk.py does.
    spc = DSparkSpeculator.__new__(DSparkSpeculator)
    spc.model = _MarkovDraftModel(vocab)
    spc.num_speculative_steps = N_SPEC
    spc.max_num_reqs = MAX_REQS
    num_sample = MAX_REQS * N_SPEC
    spc.input_buffers = InputBuffers(MAX_REQS, num_sample, device)
    spc._anchor_idx = torch.arange(MAX_REQS) * N_SPEC
    spc.sample_indices = torch.arange(num_sample)
    spc.sample_idx_mapping = torch.full((num_sample,), -1)
    spc.sample_pos = torch.zeros(num_sample, dtype=torch.int64)
    spc._step_cols = torch.arange(N_SPEC, dtype=torch.int32)
    spc.draft_tokens = torch.zeros(MAX_REQS, N_SPEC, dtype=torch.int64)
    spc.temperature = torch.ones(MAX_REQS, dtype=torch.float32)
    spc.seeds = torch.zeros(MAX_REQS, dtype=torch.int64)
    spc.draft_logits = DraftLogitsCache(
        MAX_REQS, N_SPEC, vocab, dtype=torch.bfloat16, fill=0.0, device=device
    )
    spc._draft_topk = None
    spc._d2t_scatter_index = None
    spc.draft_watermarker = None
    spc.acceptance_estimator = None
    spc.use_confidence_head = False
    spc.use_fp64_gumbel = False
    spc._tp_markov_sampler = None
    spc._draft_vocab_shard = None
    return spc


def _set_inputs(spc, num_reqs: int, vocab: int, seed: int, hidden: torch.Tensor):
    """Write a batch into the speculator's buffers and `hidden` (the base
    logits): scrambled request rows with a padding, a greedy and a
    temperature-0.6 request."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = num_reqs * N_SPEC
    hidden[:n] = torch.randn(n, vocab, generator=g) * 3
    rows = torch.randperm(MAX_REQS, generator=g)[:num_reqs]
    if num_reqs >= 3:
        rows[-1] = -1
    spc.sample_idx_mapping[:n] = rows.repeat_interleave(N_SPEC)
    start = torch.randint(1, 1 << 20, (num_reqs, 1), generator=g)
    spc.sample_pos[:n] = (start + torch.arange(N_SPEC)).flatten()
    spc.input_buffers.input_ids[spc._anchor_idx[:num_reqs]] = torch.randint(
        0, vocab, (num_reqs,), generator=g, dtype=torch.int32
    )
    spc.seeds.copy_(torch.randint(0, 1 << 40, (MAX_REQS,), generator=g))
    spc.temperature.fill_(1.0)
    if num_reqs > 1:
        spc.temperature[rows[1].clamp(min=0)] = 0.0
    spc.temperature[rows[0]] = 0.6


def _check_unsupported_reason(spc):
    assert TPMarkovSampler.unsupported_reason(spc) is None
    processor = spc.model.logits_processor
    w2_layer = spc.model.model.markov_head.markov_w2

    def cache(**kwargs):
        return DraftLogitsCache(MAX_REQS, N_SPEC, 8, fill=0.0, device="cuda", **kwargs)

    cases = [
        (spc, "draft_logits", None, "greedy"),
        (spc, "draft_logits", cache(dtype=torch.bfloat16, sharded=True), "sharded"),
        (spc, "draft_logits", cache(dtype=torch.float32), "not bf16"),
        (spc, "_draft_topk", 8, "top-k"),
        (spc, "_d2t_scatter_index", torch.zeros(1), "reduced"),
        (spc, "use_confidence_head", True, "adaptive verification"),
        (spc, "use_fp64_gumbel", True, "fp64"),
        (processor, "scale", 0.5, "scales or caps"),
        (processor, "soft_cap", 30.0, "scales or caps"),
        (processor, "head_dtype", torch.float32, "head dtype"),
        (w2_layer, "use_skinny_gemm", True, "skinny GEMM"),
    ]
    for obj, attr, value, expected in cases:
        old = getattr(obj, attr, None)
        setattr(obj, attr, value)
        reason = TPMarkovSampler.unsupported_reason(spc)
        setattr(obj, attr, old)
        assert reason is not None and expected in reason, (attr, reason)
    assert TPMarkovSampler.unsupported_reason(spc) is None


def _worker(local_rank: int, world_size: int, port: int, vocab: int, q):
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    with set_current_vllm_config(config):
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        torch.set_default_device(device)
        torch.set_default_dtype(torch.bfloat16)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(port),
            }
        )
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        cpu_group = get_tp_group().cpu_group
        spc = _speculator(vocab, device)
        _check_unsupported_reason(spc)

        # Symmetric memory fails on one rank only: every rank falls back.
        with pytest.MonkeyPatch.context() as m:
            if local_rank == 1:
                import torch.distributed._symmetric_memory as symm_mem

                def fail(*args, **kwargs):
                    raise RuntimeError("no symmetric memory")

                m.setattr(symm_mem, "empty", fail)
            spc._try_enable_tp_markov_sampling()
            assert spc._tp_markov_sampler is None

        sampler, reason = TPMarkovSampler.create(spc)
        if sampler is None:
            # The projections are inexact on this GPU, or there is no
            # symmetric memory: the replicated loop runs, by design.
            if local_rank == 0:
                q.put(f"vocab-parallel sampling falls back: {reason}")
            return

        hidden = torch.empty(MAX_REQS * N_SPEC, vocab)

        def draft(num_reqs, active, graph=None, late=False):
            """The drafted tokens and the whole draft-logits cache, with the
            vocab-parallel loop if `active`, else the replicated loop."""
            spc._tp_markov_sampler = sampler if active else None
            cache = spc.draft_logits
            assert cache is not None
            cache.values.fill_(-1.0)
            spc.draft_tokens.fill_(-1)
            torch.accelerator.synchronize()
            torch.distributed.barrier(group=cpu_group)
            if late and local_rank != 0:
                torch.cuda._sleep(1 << 29)  # rank 0 reads before this rank writes
            if graph is None:
                spc._sample_sequential(num_reqs, hidden)
            else:
                graph.replay()
            torch.accelerator.synchronize()
            return spc.draft_tokens[:num_reqs].clone(), cache.values.clone()

        def check(num_reqs, seed, graph=None, late=False):
            _set_inputs(spc, num_reqs, vocab, seed, hidden)
            tokens, cache = draft(num_reqs, active=False)
            tp_tokens, tp_cache = draft(num_reqs, True, graph, late)
            assert torch.equal(tp_tokens, tokens)
            assert torch.equal(tp_cache, cache)

        for push_max_reqs in (4, 0):  # push and recompute completion
            sampler.push_max_reqs = push_max_reqs
            for num_reqs in (1, 2, 4, 5, 33, MAX_REQS):
                check(num_reqs, seed=num_reqs)

        # A late peer is waited for at any epoch, including where a 14-bit
        # tag would wrap.
        sampler.push_max_reqs = 4
        for epoch in (0, (1 << 14) - 1, (1 << 22) - 2):
            for num_reqs in (4, 8):
                sampler.epoch.fill_(epoch)
                check(num_reqs, seed=epoch + num_reqs, late=True)

        # CUDA graphs replay with new inputs in the captured buffers.
        for num_reqs in (3, 8):
            _set_inputs(spc, num_reqs, vocab, 0, hidden)
            spc._tp_markov_sampler = sampler
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                spc._sample_sequential(num_reqs, hidden)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                spc._sample_sequential(num_reqs, hidden)
            for seed in range(1, 4):
                check(num_reqs, seed, graph)
            del graph
        sampler.check_exchange()

        # A peer that never writes: the exchange times out and is reported.
        torch.distributed.barrier(group=cpu_group)
        if local_rank == 0:
            with pytest.MonkeyPatch.context() as m:
                m.setattr(tp_markov, "_SPIN_LIMIT", 1 << 10)
                _set_inputs(spc, 1, vocab, 0, hidden)
                spc._tp_markov_sampler = sampler
                spc._sample_sequential(1, hidden)
                torch.accelerator.synchronize()
            with pytest.raises(RuntimeError, match="timed out"):
                sampler.check_exchange()
        torch.distributed.barrier(group=cpu_group)

        # Owned-columns mode (vocab-sharded verification): each rank samples
        # from its owned columns' logits only and caches only those; the cache
        # completion then restores the replicated loop's cache.
        from vllm.v1.worker.gpu.spec_decode.vocab_shard import vocab_shard_range

        ranges = [vocab_shard_range(vocab, world_size, r) for r in range(world_size)]
        c0, c1 = ranges[local_rank]
        owned, reason = TPMarkovSampler.create(spc, ranges)
        assert owned is not None, reason
        values = spc.draft_logits.values
        for num_reqs in (1, 5, MAX_REQS):
            _set_inputs(spc, num_reqs, vocab, num_reqs + 7, hidden)
            tokens, cache = draft(num_reqs, active=False)
            values.fill_(-1.0)
            n = num_reqs * N_SPEC
            owned.sample(
                num_reqs,
                hidden[:n, c0:c1].contiguous().view(num_reqs, N_SPEC, c1 - c0),
                spc.sample_idx_mapping[:n].view(num_reqs, N_SPEC),
                spc.sample_pos[:n].view(num_reqs, N_SPEC),
            )
            torch.accelerator.synchronize()
            assert torch.equal(spc.draft_tokens[:num_reqs], tokens)
            assert torch.equal(values[:, :, c0:c1], cache[:, :, c0:c1])
            rows = spc.sample_idx_mapping[:n:N_SPEC]
            rows = rows[rows >= 0]
            owned.complete_cache(values, rows)
            assert torch.equal(values[rows], cache[rows])
        owned.check_exchange()


def _run(world_size: int, vocab: int):
    q = mp.get_context("spawn").Queue()
    mp.spawn(_worker, args=(world_size, get_open_port(), vocab, q), nprocs=world_size)
    try:
        val = q.get(timeout=1)
    except queue.Empty:
        val = None
    finally:
        cleanup_dist_env_and_memory()
    if val is not None:
        pytest.skip(val)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("vocab", [200064, 151936])
def test_tp_markov_matches_replicated_loop_tp2(vocab):
    _run(2, vocab)


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("vocab", [200064, 151936])
def test_tp_markov_matches_replicated_loop_tp4(vocab):
    _run(4, vocab)
