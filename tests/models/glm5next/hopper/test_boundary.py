# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Hopper fused TP all-reduce + mHC boundary and the MTP draft's
all-reduce + RMSNorm against an all-gather and an FP64 reference, with the
stock kernels' error on the same inputs as the bar."""

import tempfile

import pytest
import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.platforms import current_platform

HIDDEN, HC, MIX, SINKHORN = 4096, 4, 24, 20
RMS_EPS = NORM_EPS = 1e-5
HC_EPS = 1e-6
# The planned decode steps: plain decode (1-8) and MTP5 verify (6/12/18/24),
# plus counts that end the cluster groups short.
TOKENS = (1, 2, 3, 4, 6, 8, 12, 15, 16, 17, 18, 24, 31, 32)


def _rank(rank, tp, store, check):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=tp,
            rank=rank,
            distributed_init_method=f"file://{store}",
            local_rank=rank,
        )
        ensure_model_parallel_initialized(tp, 1)
    try:
        check(rank, torch.device("cuda", rank))
    finally:
        cleanup_dist_env_and_memory()


def run_tp(tp, check):
    """Runs `check(rank, device)` on `tp` spawned ranks, one GPU each."""
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"Requires {tp} GPUs with NVLink multicast")
    with tempfile.TemporaryDirectory() as tmp:
        torch.multiprocessing.spawn(_rank, args=(tp, f"{tmp}/store", check), nprocs=tp)


requires_sm90 = pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="Requires SM90"
)


def _inputs(tokens, rank, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, generator=g):
        return torch.randn(*shape, generator=generator, device=device)

    x = randn(
        tokens, HIDDEN, generator=torch.Generator(device).manual_seed(seed + rank)
    )
    # Packed +0/-0 pairs collide with the Lamport sentinel.
    x[:, :16] = 0
    x[:, 9:16:2] = -0.0
    residual = randn(tokens, HC, HIDDEN)
    residual[:, :, 100:104] *= 50  # outlier channels
    comb = torch.rand(tokens, HC, HC, generator=g, device=device) + 0.05
    for _ in range(4):
        comb = comb / comb.sum(-1, keepdim=True)
        comb = comb / comb.sum(-2, keepdim=True)
    return dict(
        x=(0.5 * x).bfloat16(),
        residual=residual.bfloat16(),
        post=torch.rand(tokens, HC, 1, generator=g, device=device) * 2,
        comb=comb,
        fn=(randn(MIX, HC * HIDDEN) * 0.02).bfloat16(),
        scale=torch.tensor([0.5, 0.5, 1.0], device=device),
        base=randn(MIX) * 0.1,
        norm=(1 + 0.1 * randn(HIDDEN)).bfloat16(),
    )


def _reference(x, i):
    """FP64 mHC post, pre and RMSNorm of the BF16 all-reduced `x`, rounding
    to BF16 where the model does."""
    r = torch.einsum("mij,mih->mjh", i["comb"].double(), i["residual"].double())
    r = (r + i["post"].double() * x.double().unsqueeze(1)).bfloat16()
    flat = r.double().flatten(1)
    mixes = flat @ i["fn"].double().t()
    mixes = mixes * torch.rsqrt(flat.square().mean(-1, keepdim=True) + RMS_EPS)
    s, b = i["scale"].double(), i["base"].double()
    pre = torch.sigmoid(mixes[:, :HC] * s[0] + b[:HC]) + HC_EPS
    post = torch.sigmoid(mixes[:, HC : 2 * HC] * s[1] + b[HC : 2 * HC]) * 2.0
    comb = mixes[:, 2 * HC :].view(-1, HC, HC) * s[2] + b[2 * HC :].view(1, HC, HC)
    comb = torch.softmax(comb, -1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    v = (pre.unsqueeze(-1) * r.double()).sum(1).bfloat16().double()
    normed = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + NORM_EPS)
    return r, post.unsqueeze(-1), comb, (normed * i["norm"].double()).bfloat16()


def _stock(x, i):
    import vllm.model_executor.layers.mhc  # noqa: F401  (registers the op)

    r, p, c, li = torch.ops.vllm.mhc_fused_post_pre_tilelang(
        x, i["residual"], i["post"], i["comb"], i["fn"].float(),
        i["scale"], i["base"], RMS_EPS, HC_EPS, HC_EPS, 2.0, SINKHORN, 1, 1,
        i["norm"], NORM_EPS,
    )  # fmt: skip
    m = x.shape[0]
    return r, p.view(m, HC, 1), c.view(m, HC, HC), li


def _mixes_feed_boundary_kernel(graph) -> bool:
    """Whether a captured graph has an edge from the side-stream mixes kernel
    straight into a boundary kernel. Such an edge is programmatic (the boundary
    kernel launches with PDL), so the boundary kernel could read `post` and
    `comb` before its PDL wait and before the mixes kernel's writes land.
    Asserts that the graph has both kernels, so a renamed kernel cannot make
    the answer False by matching nothing."""
    from cuda.bindings import driver

    def name(node):
        _, kind = driver.cuGraphNodeGetType(node)
        if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            return ""
        _, params = driver.cuGraphKernelNodeGetParams(node)
        _, func = driver.cuFuncGetName(params.func)
        return func.decode() if isinstance(func, bytes) else str(func)

    def is_mixes(node):
        return "MixesDevice" in name(node)

    def is_boundary(node):
        return "_boundary_kernels" in name(node)

    raw = driver.CUgraph(graph.raw_cuda_graph())
    count = driver.cuGraphGetNodes(raw, 0)[-1]
    _, nodes, _ = driver.cuGraphGetNodes(raw, count)
    assert any(map(is_mixes, nodes)) and any(map(is_boundary, nodes))
    count = driver.cuGraphGetEdges(raw, 0)[-1]
    _, sources, targets, *_ = driver.cuGraphGetEdges(raw, count)
    return any(is_mixes(a) and is_boundary(b) for a, b in zip(sources, targets))


def _error(out, ref):
    return ((out.double() - ref.double()).norm() / ref.double().norm()).item()


# Relative L2 error allowed above 1.25x the stock kernels' own: a few BF16
# outputs one rounding apart (FMA and rank summation order), FP32 mixes within
# the approximate reciprocals of the Sinkhorn.
SLACK = {"residual": 1e-4, "post": 1e-6, "comb": 1e-6, "layer_input": 1e-4}


def _boundary(rank, device):
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.boundary import (
        DEFAULT_MODES,
        HopperBoundary,
    )

    group = get_tp_group()
    tp = group.world_size
    # Every input, with one to three tokens per cluster at the various counts
    # (short last groups included), and the default table.
    tables = [
        {m: ("lamport", 1 + m % 3) for m in TOKENS},
        {m: ("reduced", 1 + (m + 1) % 3) for m in TOKENS},
        DEFAULT_MODES,
    ]
    for modes in tables:
        op = HopperBoundary(
            hidden_size=HIDDEN, max_num_tokens=32, rms_eps=RMS_EPS, hc_eps=HC_EPS,
            post_mult=2.0, sinkhorn_repeat=SINKHORN, norm_eps=NORM_EPS,
            device=device, modes=modes,
        )  # fmt: skip
        # Every rank dispatches each count the same way.
        dispatch = [None] * tp
        torch.distributed.all_gather_object(
            dispatch, [op.modes.get(m) for m in TOKENS], group=group.cpu_group
        )
        assert all(d == dispatch[0] for d in dispatch)

        def check(outputs, i):
            x = group.all_reduce(i["x"].clone())
            ref = _reference(x, i)
            stock = _stock(x, i)
            names = ("residual", "post", "comb", "layer_input")
            for name, out, s, r in zip(names, outputs, stock, ref):
                assert out.shape == r.shape and out.dtype == s.dtype, name
                assert _error(out, r) <= 1.25 * _error(s, r) + SLACK[name], name
            torch.testing.assert_close(outputs[0], ref[0], atol=2**-6, rtol=2**-7)
            torch.testing.assert_close(
                outputs[1].double(), ref[1], atol=1e-4, rtol=1e-4
            )
            torch.testing.assert_close(
                outputs[2].double(), ref[2], atol=1e-4, rtol=1e-4
            )
            torch.testing.assert_close(outputs[3], ref[3], atol=2**-5, rtol=2**-7)

        # Changing shapes cover shrinking and growing batches in one mailbox.
        for tokens in (1, 6, 32, 3, 17, 2, 24, 31, 8, 12, 4, 15, 16, 18):
            if not op.supports(tokens):
                continue
            i = _inputs(tokens, rank, device, seed=tokens)
            args = [i[k] for k in ("x", "residual", "post", "comb", "fn")]
            args += [i[k] for k in ("scale", "base", "norm")]
            eager = op(*args)
            op.join()
            check(eager, i)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                for _ in range(4):
                    op(*args)
                captured = op(*args)
                op.join()
            assert not _mixes_feed_boundary_kernel(graph)
            for _ in range(10):
                graph.replay()
            check(captured, i)
            # Replays pick up in-place input changes.
            i["x"].mul_(0.5)
            i["residual"].mul_(2)
            i["comb"].copy_(i["comb"].flip(-1))
            graph.replay()
            check(captured, i)


def _draft(rank, device):
    from vllm.models.glm5next.nvidia.ops.fused_eh_norm import fused_eh_norm
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.boundary import (
        HopperDraftAllReduceNorm,
    )

    group = get_tp_group()
    tp = group.world_size
    op = HopperDraftAllReduceNorm(
        hidden_size=HIDDEN, max_num_tokens=32, top_k=8, eps=NORM_EPS, device=device
    )
    g = torch.Generator(device=device).manual_seed(0)
    weight = (1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).bfloat16()

    def rms_norm(v, w):
        v = v.double()
        return (
            v * torch.rsqrt(v.square().mean(-1, keepdim=True) + NORM_EPS) * w.double()
        )

    def check(normed, summed, x, residual):
        # The kernel sums the ranks in FP32 in rank order and rounds once.
        peers = group.all_gather(x, dim=0).view(tp, -1, HIDDEN).float()
        acc = peers[0]
        for peer in peers[1:]:
            acc = acc + peer
        total = (acc.bfloat16().float() + residual.float()).bfloat16()
        assert torch.equal(summed, total)
        ref = rms_norm(total, weight)
        torch.testing.assert_close(normed.double(), ref, atol=2**-6, rtol=2**-7)

    for tokens in (1, 4, 2, 3, 24, 6):
        x = torch.randn(
            tokens, HIDDEN, device=device,
            generator=torch.Generator(device).manual_seed(10 * tokens + rank),
        ).bfloat16()  # fmt: skip
        x[:, 9:16:2] = -0.0
        residual = torch.randn(tokens, HIDDEN, device=device, generator=g).bfloat16()
        check(*op(x, residual, weight), x, residual)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(3):
                op(x, residual, weight)
            captured = op(x, residual, weight)
        for _ in range(5):
            graph.replay()
        check(*captured, x, residual)
        x.mul_(0.5)
        graph.replay()
        check(*captured, x, residual)

    # The vocab-parallel embedding all-reduce fused with enorm / hnorm.
    vocab, shard = 1000, 512 if tp == 2 else 256
    table = torch.randn(tp * shard, HIDDEN, generator=g, device=device).bfloat16()
    table[3, :16] = -0.0
    start = rank * shard
    rows = min(shard, vocab - start)
    embedding = table[start : start + shard].clone()
    embedding[rows:] = float("nan")  # padding rows are never read
    enorm = (1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).bfloat16()
    hnorm = (1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).bfloat16()
    op.prepare_embedding(embedding, start, rows, NORM_EPS)
    for tokens in (1, 6, 24, 4):
        ids = torch.randint(0, vocab, (tokens,), generator=g, device=device).int()
        ids[0] = 3
        positions = torch.randint(0, 9, (tokens,), generator=g, device=device)
        previous = torch.randn(tokens, HIDDEN, generator=g, device=device).bfloat16()
        out = op.embed_norm(ids, positions, previous, embedding, enorm, hnorm)
        rows_e = torch.where(
            (positions == 0).unsqueeze(-1), 0.0, table[ids.long()].double()
        )
        ref = torch.cat([rms_norm(rows_e, enorm), rms_norm(previous, hnorm)], -1)
        torch.testing.assert_close(out.double(), ref, atol=2**-7, rtol=2**-7)
        stock = fused_eh_norm(
            positions, table[ids.long()], previous, enorm, hnorm, NORM_EPS
        )
        assert (out.double() - ref).abs().max() <= (
            stock.double() - ref
        ).abs().max() * 1.05


@pytest.mark.parametrize("tp", [2, 4])
@requires_sm90
def test_boundary_matches_reference(tp):
    run_tp(tp, _boundary)


@pytest.mark.parametrize("tp", [2, 4])
@requires_sm90
def test_draft_all_reduce_norm_matches_reference(tp):
    run_tp(tp, _draft)


@requires_sm90
def test_parse_mode_rejects_empty_clusters():
    """Policy choices parse to modes; zero tokens per cluster fails at parse
    time instead of in the kernel's grid computation."""
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.boundary import (
        parse_mode,
    )

    assert parse_mode("lamport") == ("lamport", 1)
    assert parse_mode("lamport2") == ("lamport", 2)
    assert parse_mode("reduced3") == ("reduced", 3)
    for choice in ("lamport0", "reduced-1", "mailbox1"):
        with pytest.raises(ValueError):
            parse_mode(choice)
