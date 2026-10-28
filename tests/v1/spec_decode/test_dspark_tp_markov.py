# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The vocab-parallel DSpark Markov loop draws the same tokens and caches the
same draft logits as the replicated loop, bitwise."""

import queue
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

N_SPEC = 7
RANK = 64
MAX_REQS = 32


def _fake_speculator(vocab: int, device: torch.device) -> SimpleNamespace:
    # Rank-identical weights and state.
    g = torch.Generator(device=device).manual_seed(0)
    w1 = torch.randn(vocab, RANK, device=device, generator=g).bfloat16()
    w2 = (torch.randn(vocab, RANK, device=device, generator=g) / 4).bfloat16()
    model = SimpleNamespace(
        model=SimpleNamespace(
            markov_head=SimpleNamespace(markov_w2=SimpleNamespace(weight=w2))
        ),
        logits_processor=SimpleNamespace(org_vocab_size=vocab),
        markov_embed=lambda token_ids: F.embedding(token_ids, w1),
    )
    return SimpleNamespace(
        model=model,
        max_num_reqs=MAX_REQS,
        num_speculative_steps=N_SPEC,
        draft_logits=SimpleNamespace(
            values=torch.zeros(
                MAX_REQS, N_SPEC, vocab, dtype=torch.bfloat16, device=device
            ),
            row_mapping=None,
        ),
        draft_tokens=torch.zeros(MAX_REQS, N_SPEC, dtype=torch.int64, device=device),
        input_buffers=SimpleNamespace(
            input_ids=torch.zeros(MAX_REQS * N_SPEC, dtype=torch.int32, device=device)
        ),
        _anchor_idx=torch.arange(MAX_REQS, device=device) * N_SPEC,
        seeds=torch.randint(0, 1 << 40, (MAX_REQS,), device=device, generator=g),
        temperature=torch.ones(MAX_REQS, device=device),
        _step_cols=torch.arange(N_SPEC, dtype=torch.int32, device=device),
    )


def _replicated_loop(spc, num_reqs, base_logits, idx_map, sample_pos) -> None:
    """DSparkSpeculator._sample_sequential's loop (probabilistic drafting)."""
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    w2 = spc.model.model.markov_head.markov_w2.weight
    prev = spc.input_buffers.input_ids[spc._anchor_idx[:num_reqs]]
    for i in range(N_SPEC):
        logits = base_logits[:, i] + F.linear(spc.model.markov_embed(prev), w2)
        prev = gumbel_sample(
            logits,
            idx_map[:, i],
            spc.temperature,
            spc.seeds,
            sample_pos[:, i] - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=spc.draft_logits.values,
            logits_cache_col=spc._step_cols[i],
        )
        spc.draft_tokens[:num_reqs, i] = prev


def _inputs(spc, num_reqs, vocab, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    base_logits = (
        torch.randn(num_reqs, N_SPEC, vocab, device=device, generator=g) * 3
    ).bfloat16()
    # Request state rows in a scrambled order, one padding request.
    rows = torch.randperm(MAX_REQS, device=device, generator=g)[:num_reqs]
    if num_reqs >= 3:
        rows[-1] = -1
    idx_map = rows.unsqueeze(1).expand(num_reqs, N_SPEC).contiguous()
    sample_pos = (
        torch.randint(1, 1 << 20, (num_reqs, 1), device=device, generator=g)
        + torch.arange(N_SPEC, device=device)
    ).contiguous()
    spc.input_buffers.input_ids[spc._anchor_idx[:num_reqs]] = torch.randint(
        0, vocab, (num_reqs,), device=device, generator=g, dtype=torch.int32
    )
    temperature = torch.ones(MAX_REQS, device=device)
    temperature[rows[1 % num_reqs].clamp(min=0)] = 0.0 if num_reqs > 1 else 1.0
    temperature[rows[0]] = 0.6
    spc.temperature.copy_(temperature)
    return base_logits, idx_map, sample_pos


def _worker(local_rank, world_size, port, vocab, q):
    from vllm.v1.worker.gpu.spec_decode.dspark.tp_markov import TPMarkovSampler

    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    with set_current_vllm_config(config):
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
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
        spc = _fake_speculator(vocab, device)
        # The projections of these vocabularies are exact on the tested GPUs;
        # where they are not, the speculator keeps the replicated loop.
        assert TPMarkovSampler.inexact_projection_reason(spc) is None
        try:
            sampler = TPMarkovSampler(spc)
        except RuntimeError as e:  # no symmetric memory
            q.put(f"symmetric memory is unavailable: {e}")
            return

        def check(num_reqs, seed, push_max_reqs, graph):
            sampler.push_max_reqs = push_max_reqs
            base_logits, idx_map, sample_pos = _inputs(
                spc, num_reqs, vocab, seed, device
            )
            spc.draft_logits.values.fill_(-1.0)
            _replicated_loop(spc, num_reqs, base_logits, idx_map, sample_pos)
            ref_tokens = spc.draft_tokens[:num_reqs].clone()
            ref_cache = spc.draft_logits.values.clone()
            spc.draft_logits.values.fill_(-1.0)
            spc.draft_tokens.fill_(-1)
            if graph is None:
                sampler.sample(spc, num_reqs, base_logits, idx_map, sample_pos)
            else:
                graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(spc.draft_tokens[:num_reqs], ref_tokens)
            assert torch.equal(spc.draft_logits.values, ref_cache)

        for num_reqs in (1, 2, 4, 5, 16, MAX_REQS):
            for push_max_reqs in (4, 0):
                for seed in range(2):
                    check(num_reqs, seed, push_max_reqs, None)

        # CUDA graphs replay with new inputs in the captured buffers.
        for num_reqs, push_max_reqs in ((3, 4), (8, 4)):
            sampler.push_max_reqs = push_max_reqs
            base_logits, idx_map, sample_pos = _inputs(spc, num_reqs, vocab, 0, device)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                sampler.sample(spc, num_reqs, base_logits, idx_map, sample_pos)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                sampler.sample(spc, num_reqs, base_logits, idx_map, sample_pos)
            for seed in range(1, 4):
                new = _inputs(spc, num_reqs, vocab, seed, device)
                for buf, value in zip((base_logits, idx_map, sample_pos), new):
                    buf.copy_(value)
                check(num_reqs, seed, push_max_reqs, graph)
            del graph
        assert sampler.errors.item() == 0


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("vocab", [200064, 151936])
def test_tp_markov_matches_replicated_loop(world_size, vocab):
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
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
