# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 DSpark sharded Markov sampling (ROCm, TP8) vs vLLM's loop.

Reference: ``DSparkSpeculator._sample_sequential``'s loop with a real
``DSparkMarkovHead`` (163840 x 256, random weights) and ``gumbel_sample``.
Both launch modes, 1 / 4 rows, temperatures 1.0 / 0.6 / 0.0, padding rows:
draft tokens match, every rank's draft logits cache matches up to the bias's
last bf16 bit, and a CUDA graph replays to the eager result.
"""

import types
from typing import Any

import pytest
import ray
import torch

from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)
from vllm.platforms import current_platform

V, R, NSPEC = 163840, 256, 7
MAX_REQS = 16


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


class _LogitsProcessor:
    """The draft's LogitsProcessor; ``skip_gather`` gives this rank's shard."""

    def __init__(self, lp, local):
        self.lp, self.local, self.scale, self.soft_cap = lp, local, 1.0, None

    def __call__(self, head, hidden, skip_gather=False):
        return self.local if skip_gather else self.lp(head, hidden)


def _speculator(head, lp, base, base_local, device):
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = types.SimpleNamespace(markov_head=head)
            self.logits_processor = _LogitsProcessor(lp, base_local)
            self.lm_head = None

        def compute_draft_logits(self, hidden):
            return base[: hidden.shape[0]]

        def map_draft_to_target(self, ids):
            return ids

        def markov_embed(self, ids):
            return self.model.markov_head.embed(ids)

        def markov_bias(self, e):
            return self.model.markov_head.bias(e, lp)

    class Spec:
        _sample_sequential = DSparkSpeculator._sample_sequential
        _sample_logits = DSparkSpeculator._sample_logits
        _maybe_predict_acceptance = DraftModelSpeculator._maybe_predict_acceptance

    s: Any = Spec()
    s.model = Model()
    s.num_speculative_steps = NSPEC
    s._draft_topk = None
    s.use_confidence_head = False
    s.acceptance_estimator = None
    s.draft_watermarker = None
    s._d2t_scatter_index = None
    s.use_fp64_gumbel = False
    s.sample_indices = torch.arange(MAX_REQS * NSPEC, device=device)
    s.sample_idx_mapping = torch.zeros(
        MAX_REQS * NSPEC, dtype=torch.int32, device=device
    )
    s.sample_pos = torch.zeros(MAX_REQS * NSPEC, dtype=torch.int64, device=device)
    s.input_buffers = types.SimpleNamespace(
        input_ids=torch.zeros(MAX_REQS * NSPEC, dtype=torch.int32, device=device)
    )
    s._anchor_idx = torch.arange(MAX_REQS, device=device) * NSPEC
    s._step_cols = torch.arange(NSPEC, dtype=torch.int32, device=device)
    s.temperature = torch.ones(MAX_REQS, dtype=torch.float32, device=device)
    s.seeds = torch.zeros(MAX_REQS, dtype=torch.int64, device=device)
    s.draft_logits = DraftLogitsCache(
        MAX_REQS, NSPEC, V, dtype=torch.bfloat16, fill=0.0, device=device
    )
    s.draft_tokens = torch.zeros(MAX_REQS, NSPEC, dtype=torch.int64, device=device)
    return s


def _set_inputs(specs, rows, g, temp, pad_row):
    anchor = torch.randint(0, V, (rows,), generator=g)
    seed = torch.randint(0, 2**62, (rows,), generator=g)
    p0 = torch.randint(10, 900000, (rows,), generator=g)
    pos = (p0[:, None] + 1 + torch.arange(NSPEC)[None]).flatten()
    idx = torch.arange(rows).repeat_interleave(NSPEC)
    if pad_row and rows > 1:
        idx[-NSPEC:] = -1  # a CUDA graph padding row
    for s in specs:
        dev = s.seeds.device
        s.input_buffers.input_ids.zero_()
        s.input_buffers.input_ids[s._anchor_idx[:rows]] = anchor.to(dev, torch.int32)
        s.seeds[:rows] = seed.to(dev)
        s.sample_pos[: rows * NSPEC] = pos.to(dev)
        s.sample_idx_mapping[: rows * NSPEC] = idx.to(dev, torch.int32)
        s.temperature.fill_(temp)
        s.draft_tokens.fill_(-7)
        s.draft_logits.values.fill_(0)


@ray.remote(num_gpus=1, max_calls=1)
def _worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
    from vllm.models.kimi_k3.amd.ops import dspark_markov as dm

    torch.manual_seed(0)  # the same weights on every rank
    with set_current_vllm_config(VllmConfig()), torch.device(device):
        head = DSparkMarkovHead(V, V, R, prefix="model.markov_head")
        lp = LogitsProcessor(V, scale=1.0)
    head = head.to(device, torch.bfloat16)
    head.markov_w1.weight.data.normal_(0, 0.05)
    head.markov_w2.weight.data.normal_(0, 0.05)
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight

    gb = torch.Generator().manual_seed(7)
    base = (torch.randn(MAX_REQS * NSPEC, V, generator=gb) * 2.5).to(
        device, torch.bfloat16
    )
    vl = V // tp_size
    hidden = torch.zeros(MAX_REQS * NSPEC, 16, dtype=torch.bfloat16, device=device)
    cpu_group = get_tp_group().cpu_group
    g = torch.Generator().manual_seed(1234)
    for rows in (1, 4):
        base_local = base[: rows * NSPEC, rank * vl : (rank + 1) * vl].contiguous()
        ref = _speculator(head, lp, base, base_local, device)
        for mode in ("persistent", "round"):
            sp = _speculator(head, lp, base, base_local, device)
            sampler = object.__new__(dm.ShardedMarkovSampler)
            sampler.sharded = dm.ShardedMarkov(
                w1, w2, NSPEC, cpu_group, rank, tp_size, device
            )
            if mode not in sampler.sharded.geom:
                continue
            sampler.sharded.mode = lambda rows, m=mode: m  # type: ignore[method-assign,misc]
            sp.model.sample_sequential = sampler
            # compile only, as ShardedMarkovSampler.__init__ does: the first
            # launch must then compile nothing (a rank compiling mid-call
            # leaves its peers polling until they give up)
            sampler._launch(sp, rows, None, warmup=True)
            torch.distributed.barrier(group=cpu_group)
            cache = dm._sharded_markov_kernel.device_caches[rank][0]
            n_compiled = len(cache)
            _set_inputs([sp], rows, g, 1.0, pad_row=False)
            sp._sample_sequential(rows, hidden)
            torch.accelerator.synchronize()
            assert len(cache) == n_compiled, (rows, mode)
            assert not sampler.sharded.failed()
            for temp in (1.0, 0.6, 0.0):
                n_tok = n_eq = 0
                for trial in range(24):
                    _set_inputs([ref, sp], rows, g, temp, pad_row=trial % 4 == 3)
                    ref._sample_sequential(rows, hidden)
                    sp._sample_sequential(rows, hidden)
                    torch.accelerator.synchronize()
                    tr = ref.draft_tokens[:rows].cpu()
                    tf = sp.draft_tokens[:rows].cpu()
                    valid = ref.sample_idx_mapping[: rows * NSPEC : NSPEC].cpu() >= 0
                    n_tok += tr[valid].numel()
                    n_eq += (tr == tf)[valid].sum().item()
                    for r in range(rows):
                        if not valid[r]:
                            continue
                        for i in range(NSPEC):
                            d = (
                                ref.draft_logits.values[r, i].float()
                                - sp.draft_logits.values[r, i]
                            )
                            # the bias's last bf16 bit, at the logits' scale
                            assert d.abs().max().item() <= 0.25, (rows, mode, temp)
                            if tr[r, i] != tf[r, i]:
                                break  # later positions follow another token
                assert not sampler.sharded.failed()
                # a bias 1 bf16 bit off can flip a near tie: rare
                assert n_eq >= 0.98 * n_tok, (rows, mode, temp, n_eq, n_tok)

            # a CUDA graph of the sharded path replays to the eager result
            _set_inputs([sp], rows, g, 1.0, pad_row=False)
            sp._sample_sequential(rows, hidden)
            torch.accelerator.synchronize()
            want_tok = sp.draft_tokens.clone()
            want_logits = sp.draft_logits.values.clone()
            torch.distributed.barrier(group=cpu_group)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                sp._sample_sequential(rows, hidden)
            for _ in range(10):
                sp.draft_tokens.fill_(-7)
                sp.draft_logits.values.fill_(0)
                graph.replay()
                torch.accelerator.synchronize()
                assert torch.equal(sp.draft_tokens, want_tok)
                assert torch.equal(sp.draft_logits.values, want_logits)
            assert not sampler.sharded.failed()
            torch.distributed.barrier(group=cpu_group)


@pytest.mark.skipif(not _on_gfx950(), reason="needs gfx950 (MI355X)")
@multi_gpu_test(num_gpus=8)
def test_sharded_markov_matches_vllm_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _worker)
