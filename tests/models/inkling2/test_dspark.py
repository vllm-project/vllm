# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 DSpark draft components against torch ports of the reference
implementation."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(100):
    pytest.skip("Inkling2 requires Blackwell", allow_module_level=True)

from vllm.models.inkling2.dspark import (  # noqa: E402
    Inkling2MarkovHead,
)
from vllm.models.inkling2.ops.attention import block_decode_attention  # noqa: E402
from vllm.models.inkling2.ops.dspark import (  # noqa: E402
    Inkling2DSparkArgmax,
    add_softcap,
    dspark_argmax,
    markov_gate,
    mix_rmsnorm,
)
from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax  # noqa: E402

DEVICE = "cuda"


@pytest.mark.parametrize("rows,taps,width", [(1, 3, 128), (17, 8, 4608)])
@pytest.mark.parametrize("strided", [False, True])
def test_mix_rmsnorm_rounds_once(rows, taps, width, strided):
    g = torch.Generator().manual_seed(851)
    features = torch.randn(rows, taps, width * 2, generator=g).bfloat16().to(DEVICE)
    features = features[..., ::2] if strided else features[..., :width]
    alpha = torch.randn(taps, generator=g).softmax(-1).bfloat16().to(DEVICE)
    weight = (0.5 + torch.rand(width, generator=g)).bfloat16().to(DEVICE)
    mixed = (features.float() * alpha.float()[:, None]).sum(1)
    ref = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
    ref = (ref * weight.float()).bfloat16()
    out = mix_rmsnorm(features, alpha, weight, 1e-6)
    torch.testing.assert_close(out, ref, atol=0, rtol=1e-2)


def test_markov_head_fold_and_softcapped_logits(dist_init, default_vllm_config):
    """The rank-r latent goes through `hidden_out` and the LM head folded
    once in fp64; step logits are softcap(bf16(base + bias)); the recurrent
    gating rounds as the reference's torch ops."""
    g = torch.Generator().manual_seed(852)
    width, rank, vocab, cap = 128, 32, 256, 30.0

    def normal(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).bfloat16().to(DEVICE)

    config = SimpleNamespace(
        hidden_size=width, markov_rank=rank, padded_vocab_size=vocab, rms_norm_eps=1e-6
    )
    with torch.device(DEVICE):
        head = Inkling2MarkovHead(config).bfloat16()
    head.hidden_out.weight.data = normal(width, rank, scale=0.03)
    head.hidden_out.bias.data = normal(width, scale=0.03)
    lm_head = normal(vocab, width, scale=0.03)
    head.fold(lm_head)
    projection = lm_head.double() @ head.hidden_out.weight.double()
    assert torch.equal(head.fold_weight, projection.bfloat16())
    assert torch.equal(
        head.fold_bias, (lm_head.double() @ head.hidden_out.bias.double()).bfloat16()
    )

    base, bias = normal(3, vocab, scale=20.0), normal(3, vocab)
    ref = ((base + bias).float() / cap).tanh().mul(cap).bfloat16()
    torch.testing.assert_close(add_softcap(base, bias, cap), ref, atol=0, rtol=1e-2)

    # The fused gating matches the reference's bf16 torch ops bitwise.
    raw = torch.randn(512, 3 * rank, generator=g).mul(4).to(DEVICE)
    state = normal(512, rank)
    gate_raw, cand_raw, out_raw = raw.bfloat16().chunk(3, dim=-1)
    gate = torch.sigmoid(gate_raw)
    new_state, latent = markov_gate(raw, state)
    assert torch.equal(new_state, gate * state + (1.0 - gate) * torch.tanh(cand_raw))
    assert torch.equal(latent, torch.tanh(out_raw))


def _sharded_argmax_reference(base, bias, cap, world, valid):
    """`greedy_argmax` of the gathered softcapped logits, NaN and each
    shard's columns from `valid` on as -inf."""
    logits = add_softcap(base, bias, cap).float()
    cols = base.shape[1] // world
    mask = (torch.arange(base.shape[1], device=DEVICE) % cols) >= torch.tensor(
        valid, device=DEVICE
    ).repeat_interleave(cols)
    logits[mask | logits.isnan()] = float("-inf")
    return greedy_argmax(logits)


def _sharded_argmax_worker(rank: int, world: int, port: int, queue) -> None:
    import vllm._C_stable_libtorch  # noqa: F401

    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    group = dist.new_group(backend="gloo")
    failures = []

    def check(name, workspace, base, bias, cap, vocab):
        cols = workspace.num_cols
        valid = [min(cols, vocab - r * cols) for r in range(world)]
        shard = slice(rank * cols, (rank + 1) * cols)
        out = dspark_argmax(
            workspace, base[:, shard], bias[:, shard], cap, valid[rank], shard.start
        )
        expected = _sharded_argmax_reference(base, bias, cap, world, valid)
        if not torch.equal(out, expected):
            bad = (out != expected).nonzero().flatten()[:4].tolist()
            failures.append(f"{name}: rows {bad} {out[bad]} != {expected[bad]}")
        return out, expected

    # Real-size shards; the last one is partly padding.
    cols, vocab, cap = 50256, 200058, 30.0
    big = Inkling2DSparkArgmax(group, 64, cols)
    g = torch.Generator().manual_seed(853)  # shared by all ranks

    def rand(rows):
        base = torch.randn(rows, world * cols, generator=g).bfloat16().to(DEVICE)
        bias = torch.randn(rows, world * cols, generator=g).bfloat16().to(DEVICE)
        return base, bias

    base, bias = rand(9)
    planted = [5, cols, cols + 3, cols + 900, 2 * cols + 1, 3 * cols - 1]
    base[:3, planted], bias[:3, planted] = 90.0, 0.0
    base[0, [5, cols + 3, cols + 900, 3 * cols - 1]] = 0.0  # Across shards.
    base[1, [5, cols, 2 * cols + 1, 3 * cols - 1]] = 0.0  # Within a shard.
    base[3, vocab:] = 90.0  # Padding.
    base[4], bias[4] = 1.0, 0.0  # All tied.
    base[5], bias[5] = -0.0, -0.0  # A -0.0 maximum (the Lamport sentinel).
    base[6], bias[6] = -1.0, 0.0
    base[6, 2 * cols + 7], bias[6, 2 * cols + 7] = -0.0, -0.0
    base[7] = float("nan")  # All NaN: index 0.
    base[8, : 3 * cols] = float("nan")
    out, expected = check("planted", big, base, bias, cap, vocab)
    if expected[:3].tolist() != [cols, cols + 3, 5]:
        failures.append(f"planted reference {expected[:3].tolist()}")
    # Changing row counts flip the rows' buffers unevenly; strided rows.
    for i, rows in enumerate([1, 7, 64, 2, 33, 64, 5]):
        base, bias = rand(rows)
        base = base.view(rows, 1, -1).expand(rows, 3, -1).contiguous()[:, 1]
        check(f"call {i}", big, base, bias, 5.0 + i, vocab)
    # Replayed in a CUDA graph.
    static = rand(5)
    shard = slice(rank * cols, (rank + 1) * cols)
    valid = min(cols, vocab - rank * cols)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = dspark_argmax(
            big, static[0][:, shard], static[1][:, shard], cap, valid, shard.start
        )
    for i in range(4):
        base, bias = rand(5)
        base[0, [i, 2 * cols + i]] = 90.0
        static[0].copy_(base)
        static[1].copy_(bias)
        graph.replay()
        valid_all = [min(cols, vocab - r * cols) for r in range(world)]
        expected = _sharded_argmax_reference(base, bias, cap, world, valid_all)
        if not torch.equal(out, expected):
            failures.append(f"graph replay {i}")
    # Every bf16 input next to its neighbors, on two ranks in either order:
    # the softcap's rounding of every input matches add_softcap's.
    small = Inkling2DSparkArgmax(group, 1024, 8)
    x = torch.arange(1 << 16, dtype=torch.int32).short().view(torch.bfloat16)
    for step in (1, 2):
        lo, hi = x.to(DEVICE), x.roll(-step).to(DEVICE)
        for first, second in ((lo, hi), (hi, lo)):
            base = torch.full((1 << 16, world * 8), float("nan"), device=DEVICE)
            base = base.bfloat16()
            base[:, 3], base[:, 13] = first, second
            for start in range(0, 1 << 16, 1024):
                rows = slice(start, start + 1024)
                check(
                    f"neighbors {step} at {start}",
                    small,
                    base[rows],
                    torch.zeros_like(base[rows]),
                    cap,
                    world * 8,
                )
    torch.accelerator.synchronize()
    queue.put((rank, failures[:20]))
    dist.destroy_process_group()


def test_sharded_argmax_matches_full_vocab():
    """`dspark_argmax` at TP4 equals the first argmax of the gathered
    softcapped logits: ties within and across shards, padding columns holding
    the largest value, NaN, -0.0 maxima, repeated calls with changing row
    counts, CUDA graph replay, and every bf16 input's softcap."""
    world = 4
    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    queue = mp.get_context("spawn").Queue()
    mp.spawn(_sharded_argmax_worker, args=(world, 29650, queue), nprocs=world)
    results = dict(queue.get() for _ in range(world))
    assert all(not f for f in results.values()), results


@pytest.mark.parametrize("is_local", [True, False])
@pytest.mark.parametrize("num_splits", [1, 3])
def test_block_attention_anchored_window(is_local, num_splits):
    """Every query of a block sees the whole block and, for local layers, the
    `window` positions before the block; a padding request (no keys) comes
    out zero."""
    g = torch.Generator().manual_seed(854)
    block, window, num_heads, num_kv_heads, head_dim, page = 8, 64, 6, 2, 128, 16
    context_lens, num_pages = [5, 70, 150, 300], 20

    def rand(*shape):
        return torch.randn(*shape, generator=g).bfloat16().to(DEVICE)

    num_reqs = len(context_lens) + 1
    # The model's [blocks, block_size, kv_heads, (K | V)] cache layout.
    kv = rand(num_reqs * num_pages, page, num_kv_heads, 2 * head_dim)
    key_cache, value_cache = kv.split(head_dim, dim=-1)
    block_table = torch.arange(num_reqs * num_pages, dtype=torch.int32, device=DEVICE)
    seq_lens = [n + block for n in context_lens] + [0]
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    q = rand(block * num_reqs, num_heads, head_dim)
    out = torch.full((q.shape[0], num_heads * head_dim), float("nan"), device=DEVICE)
    out = out.bfloat16()
    block_decode_attention(
        q,
        key_cache,
        value_cache,
        out,
        block_table=block_table.view(num_reqs, num_pages),
        cache_seqlens=seq_lens,
        cu_seqlens_q=torch.arange(num_reqs + 1, device=DEVICE).int() * block,
        block=block,
        window=window + block if is_local else 0,
        num_splits=num_splits,
    )
    out = out.view(-1, num_heads, head_dim)

    group = num_heads // num_kv_heads
    for i, n in enumerate(context_lens):
        start = max(0, n - window) if is_local else 0
        flat = slice(i * num_pages * page + start, i * num_pages * page + n + block)
        keys = key_cache.flatten(0, 1)[flat].float().repeat_interleave(group, 1)
        vals = value_cache.flatten(0, 1)[flat].float().repeat_interleave(group, 1)
        rows = slice(i * block, (i + 1) * block)
        scores = torch.einsum("qhd,khd->hqk", q[rows].float(), keys) * head_dim**-0.5
        ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vals)
        torch.testing.assert_close(out[rows].float(), ref, atol=2e-2, rtol=2e-2)
    assert (out[len(context_lens) * block :] == 0).all()


def _all_reduce_norm_worker(rank: int, world: int, port: int, queue) -> None:
    from tests.utils import init_test_distributed_environment
    from vllm.distributed import (
        get_tp_group,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )
    from vllm.models.inkling2.dspark import _RMSNorm, all_reduce_add_rmsnorm

    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(world, 1, rank, str(port), rank)
    comm = get_tp_group().device_communicator
    assert isinstance(comm, CudaCommunicator) and comm.fi_ar_comm is not None
    dim = 4608
    norm = _RMSNorm(dim, 1e-6).to(DEVICE, torch.bfloat16)
    failures, stats = [], {}
    # One shot, two shots, and above the workspace (unfused).
    for rows in (8, 64, 256, 8192):
        g = torch.Generator().manual_seed(rows)  # shared by all ranks
        residual = torch.randn(rows, dim, generator=g).bfloat16().to(DEVICE)
        norm.weight.copy_(0.5 + torch.rand(dim, generator=g))
        g.manual_seed(rows + 1 + rank)
        x = torch.randn(rows, dim, generator=g).bfloat16().to(DEVICE)
        fused_res, ref_res = residual.clone(), residual.clone()
        out = all_reduce_add_rmsnorm(x.clone(), fused_res, norm)
        reduced = tensor_model_parallel_all_reduce(x)
        exact = reduced.double() + residual.double()
        ref = norm(reduced, ref_res)
        exact = exact * torch.rsqrt(exact.square().mean(-1, keepdim=True) + 1e-6)
        exact = exact * norm.weight.double()
        ulp = torch.finfo(torch.bfloat16).eps * exact.abs().clamp_min(1e-30)
        stats[rows] = [
            ((o.double() - exact).abs() / ulp).mean().item() for o in (out, ref)
        ]
        gathered = tensor_model_parallel_all_gather(out[None], 0)
        checks = {
            "fused iff it fits": comm.fi_ar_comm.should_use_fi_ar(x) == (rows < 8192),
            "same on all ranks": all(torch.equal(o, out) for o in gathered),
            "residual": torch.equal(fused_res, ref_res),
            # The norm reads the bf16-rounded residual.
            "norm": torch.allclose(out.float(), ref.float(), atol=0, rtol=2**-7),
        }
        failures += [f"rows={rows}: {k}" for k, ok in checks.items() if not ok]
    queue.put((rank, failures, stats))


@pytest.mark.parametrize("world", [2, 4])
def test_all_reduce_add_rmsnorm_matches_unfused(world):
    """The fused all-reduce + residual add + RMSNorm keeps the unfused bf16
    residual bits and gives the same norm output on every rank."""
    import torch.multiprocessing as mp

    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    queue = mp.get_context("spawn").Queue()
    mp.spawn(_all_reduce_norm_worker, args=(world, 29620 + world, queue), nprocs=world)
    results = [queue.get() for _ in range(world)]
    assert all(not failures for _, failures, _ in results), results
