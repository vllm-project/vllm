# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-sharded verification: owned-column assembly, exact top-p, and the
rejection sampler on owned blocks give the full-vocabulary results."""

import queue
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

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
MAX_REQS = 40


def _topp_reference(x: torch.Tensor, p: torch.Tensor):
    """Kept mask of exact top-p (float64), and each row's tolerance band: the
    reference cumulative masses next to the boundary, relative to p * Z."""
    xd = x.double()
    n, v = xd.shape
    idx = torch.arange(v, device=x.device).expand(n, v)
    # (logit desc, index asc)
    order = torch.argsort(idx, dim=1, stable=True)
    vals, order2 = torch.sort(xd.gather(1, order), dim=1, descending=True, stable=True)
    order = order.gather(1, order2)
    mx = vals[:, :1]
    e = torch.exp(vals - mx)
    e = torch.where(torch.isfinite(vals), e, torch.zeros_like(e))
    cum = torch.cumsum(e, 1)
    z = cum[:, -1:]
    target = p.double().unsqueeze(1) * z
    k = torch.searchsorted(cum, target).clamp(max=v - 1)  # first cum >= target
    kept_sorted = torch.arange(v, device=x.device).unsqueeze(0) <= k
    kept = torch.zeros_like(kept_sorted)
    kept.scatter_(1, order, kept_sorted)
    kept |= xd == mx
    kept[p >= 1.0] = True
    band = (cum.gather(1, k) - target).abs().squeeze(1) / z.squeeze(1)
    prev = cum.gather(1, (k - 1).clamp(min=0)).squeeze(1)
    band = torch.minimum(band, (prev - target.squeeze(1)).abs() / z.squeeze(1))
    return kept, band


def _logits(regime: str, n: int, v: int, g, device) -> torch.Tensor:
    x = torch.randn(n, v, device=device, generator=g)
    if regime == "lm":
        x = (x * 2.5).bfloat16().float()
        x[:, :50] += 8.0
    elif regime == "ties":
        x = torch.round(x * 4) / 4
    elif regime == "masked":
        x = (x * 3).bfloat16().float()
        x[torch.rand(n, v, device=device, generator=g) < 0.3] = float("-inf")
    elif regime == "peaked":
        x = x * 0.5
        x[:, 7] = 30.0
    elif regime == "flat":
        x = (x * 0.05).bfloat16().float()
    return x.contiguous()


def _rejection_inputs(num_reqs, vocab, g, device, greedy_rows=True):
    num_logits = num_reqs * (N_SPEC + 1)
    cu = torch.arange(num_reqs + 1, device=device, dtype=torch.int32) * (N_SPEC + 1)
    idx_mapping = torch.randperm(MAX_REQS, device=device, generator=g)[:num_reqs]
    idx_mapping = idx_mapping.to(torch.int32)
    expanded_idx_mapping = idx_mapping.repeat_interleave(N_SPEC + 1)
    expanded_local_pos = torch.arange(N_SPEC + 1, device=device, dtype=torch.int32)
    expanded_local_pos = expanded_local_pos.repeat(num_reqs)
    pos = torch.randint(10, 1 << 20, (num_logits,), device=device, generator=g)
    draft_logits = (
        torch.randn(MAX_REQS, N_SPEC, vocab, device=device, generator=g) * 2
    ).bfloat16()
    target = (torch.randn(num_logits, vocab, device=device, generator=g) * 2).float()
    # Correlate target and draft so that drafts are often accepted.
    tgt_view = target.view(num_reqs, N_SPEC + 1, vocab)
    tgt_view[:, :N_SPEC] += draft_logits[idx_mapping.long()].float()
    draft_tokens = draft_logits[idx_mapping.long()].float().argmax(-1)
    draft_sampled = torch.zeros(num_logits, dtype=torch.int64, device=device)
    draft_sampled.view(num_reqs, N_SPEC + 1)[:, 1:] = draft_tokens
    draft_sampled.view(num_reqs, N_SPEC + 1)[:, 0] = torch.randint(
        0, vocab, (num_reqs,), device=device, generator=g
    )
    if num_reqs > 2:
        draft_sampled.view(num_reqs, N_SPEC + 1)[2, 5:] = -1  # placeholders
    temperature = torch.ones(MAX_REQS, device=device)
    temperature[idx_mapping[0].long()] = 0.7
    if greedy_rows and num_reqs > 1:
        temperature[idx_mapping[1].long()] = 0.0
    seed = torch.randint(0, 1 << 40, (MAX_REQS,), device=device, generator=g)
    return dict(
        target=target,
        draft_logits=draft_logits,
        draft_sampled=draft_sampled,
        cu=cu,
        pos=pos,
        idx_mapping=idx_mapping,
        expanded_idx_mapping=expanded_idx_mapping,
        expanded_local_pos=expanded_local_pos,
        temperature=temperature,
        seed=seed,
    )


def _draft_cache(values: torch.Tensor):
    try:
        from vllm.v1.worker.gpu.spec_decode.draft_logits import (  # noqa: F401
            DraftLogitsCache,
        )
    except ImportError:  # older layout: rejection_sample takes the tensor
        return values
    return SimpleNamespace(values=values, row_mapping=None)


def _worker(local_rank, world_size, port, vocab, q):
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )
    from vllm.v1.worker.gpu.spec_decode.vocab_shard import VocabShard

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
        local = -(-vocab // (64 * world_size)) * 64
        try:
            vs = VocabShard(vocab, local, MAX_REQS * (N_SPEC + 1), MAX_REQS, device)
        except RuntimeError as e:  # no symmetric memory
            q.put(f"symmetric memory is unavailable: {e}")
            return
        c0, c1 = vs.col_start, vs.col_end
        r = vs.rank

        # 1. Owned columns from the TP shards.
        for rows, dtype in (
            (1, torch.bfloat16),
            (37, torch.bfloat16),
            (64, torch.float32),
        ):
            g = torch.Generator(device=device).manual_seed(rows)
            full = torch.randn(rows, local * world_size, device=device, generator=g)
            full = full.to(dtype)
            vs.begin_step()
            own = vs.owned_logits(full[:, r * local : (r + 1) * local])
            vs.end_step()
            assert torch.equal(own, full[:, c0:c1].float()), (rows, dtype)

        # 2. Exact top-p.
        for ri, regime in enumerate(("lm", "ties", "masked", "peaked", "flat")):
            for rows in (1, 8, 64, 200):
                g = torch.Generator(device=device).manual_seed(1000 * ri + rows)
                x = _logits(regime, rows, vocab, g, device)
                p = torch.rand(rows, device=device, generator=g) * 0.98 + 0.01
                if rows > 4:
                    p[::5] = 1.0
                ref, band = _topp_reference(x, p)
                got = None
                for _ in range(2):  # deterministic
                    own = x[:, c0:c1].clone()
                    vs.begin_step()
                    vs.top_p(own, p)
                    vs.end_step()
                    assert got is None or torch.equal(own, got)
                    got = own
                kept = got > float("-inf")
                finite = x[:, c0:c1] > float("-inf")
                diff = (kept != (ref[:, c0:c1] & finite)).any(1)
                bad = diff & (band > 1e-6)
                assert not bad.any(), (regime, rows, bad.nonzero()[:4].tolist())
                # kept values are untouched
                assert torch.equal(got[kept], x[:, c0:c1][kept])

        # 3. Rejection sampling on the owned blocks == on the full vocabulary.
        rates = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], device=device)
        for mode in ("block", "standard", "synthetic"):
            for num_reqs in (1, 3, 32):
                g = torch.Generator(device=device).manual_seed(num_reqs + len(mode))
                inp = _rejection_inputs(num_reqs, vocab, g, device)
                kwargs = dict(
                    use_block_verification=mode == "block",
                    synthetic_conditional_rates=rates if mode == "synthetic" else None,
                )
                args = (
                    _draft_cache(inp["draft_logits"]),
                    inp["draft_sampled"],
                    inp["cu"],
                    inp["pos"],
                    inp["idx_mapping"],
                    inp["expanded_idx_mapping"],
                    inp["expanded_local_pos"],
                    inp["temperature"],
                    inp["seed"],
                    N_SPEC,
                )
                ref_s, ref_n = rejection_sample(inp["target"], *args, **kwargs)
                vs.begin_step()
                got_s, got_n = rejection_sample(
                    inp["target"][:, c0:c1].contiguous(),
                    *args,
                    vocab_shard=vs,
                    **kwargs,
                )
                vs.end_step()
                assert torch.equal(ref_n, got_n), (mode, num_reqs)
                valid = (
                    torch.arange(N_SPEC + 1, device=device)[None, :] < ref_n[:, None]
                )
                assert torch.equal(ref_s[valid], got_s[valid]), (mode, num_reqs)
        torch.accelerator.synchronize()
        assert vs.errors.item() == 0


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("vocab", [200064, 151936])
def test_vocab_sharded_verification(world_size, vocab):
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
