# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 DSpark drafter kernels (``vllm/models/minimax_m3/amd/mono/drafter``).

- Deployment checks that need no GPU: the kernels are built for 7 query rows a
  request at TP4 on gfx950, and any other deployment, or a ROCm install without
  FlyDSL, keeps the stock draft layers.
- TP4 numerics on four gfx950 GPUs: ``DrafterRunner`` against the ops of the
  stock draft layers (vLLM's MXFP8 linear, the Gemma norms,
  fused_minimax_m3_qknorm_rope_kv_insert, Triton unified attention non-causal
  within the query block over a 4096-key window, NCCL all-reduces) on random
  weights at the real shapes: every request's final hidden states and every
  layer's new K/V rows, for draft steps of 1, 2, 3, 8 and 10 requests (two
  launches, the last request padded as a FULL graph replay pads it), and a CUDA
  graph replay of the 10-request step.

The kernel package imports flydsl at module import, so these tests run in the
ROCm images (where FlyDSL is installed); elsewhere they are skipped.
"""

import math
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("flydsl")

from vllm.models.minimax_m3.amd.mono.drafter import dispatch  # noqa: E402


def _vllm_config(tp=4, dp=1, pp=1, method="dspark", k=7):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, data_parallel_size=dp, pipeline_parallel_size=pp
        ),
        speculative_config=SimpleNamespace(method=method, num_speculative_tokens=k),
    )


@pytest.fixture(autouse=True)
def _gfx950(monkeypatch):
    monkeypatch.setattr(dispatch, "on_gfx950", lambda: True)
    monkeypatch.setattr(dispatch, "_flydsl_available", lambda: True)


def test_supported_deployment():
    assert dispatch._config_refusal(_vllm_config()) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"tp": 8},
        {"dp": 2},
        {"pp": 2},
        {"method": "eagle3"},
        {"k": 3},
    ],
)
def test_other_deployments_keep_the_stock_layers(overrides):
    assert dispatch._config_refusal(_vllm_config(**overrides)) is not None


def test_no_speculative_config():
    config = _vllm_config()
    config.speculative_config = None
    assert dispatch._config_refusal(config) is not None


def test_not_gfx950(monkeypatch):
    monkeypatch.setattr(dispatch, "on_gfx950", lambda: False)
    assert dispatch._config_refusal(_vllm_config()) is not None


def test_no_flydsl(monkeypatch):
    monkeypatch.setattr(dispatch, "_flydsl_available", lambda: False)
    assert dispatch._config_refusal(_vllm_config()) is not None


# ---------------------------------------------------------------- TP4 numerics
H, QH, HD, IR, LAYERS = 6144, 16, 128, 3072, 5  # a rank's shard at TP4
QS = QH * HD
ROT, THETA, WINDOW, BLOCK, EPS = 64, 1e7, 4096, 128, 1e-6
# Context lengths of the step's requests: windows that do and do not fill the
# 4096 keys, a single key and partial blocks.
CTX = [9000, 300, 4200, 1, 7000, 130, 2047, 15000, 600, 5000]
REQS = [1, 2, 3, 8, 10]
FP8 = torch.float8_e4m3fn
# Two paths that round differently drift apart by about one e4m3 step wherever
# MXFP8 re-quantizes an activation, however small the first difference, and
# the drift depends on the request (a short context averages fewer keys). So
# each request is held to the stock layers' own spread: the kernels may be at
# most SPREAD times as far from the stock layers as the stock layers are from
# themselves when the embeddings move by about one bf16 ulp (measured: final
# hidden 0.92-0.95, K/V rows up to 1.08).
SPREAD = 1.5


def _supported() -> bool:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm() or torch.accelerator.device_count() < 4:
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def _cos_sin(max_pos: int) -> torch.Tensor:
    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROT, 2, dtype=torch.float) / ROT))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to("cuda", torch.bfloat16)


def _draft_model(rank: int, num_blocks: int) -> SimpleNamespace:
    """The parts of one rank's MiniMaxM3DSparkModel that DrafterRunner reads:
    this rank's shard of the MXFP8 projections (e4m3 [N, K], e8m0 scales
    [N, K / 32] around 2^-6, and around 2^-11 for down_proj so that, as in a
    trained model, a layer's MLP adds a fraction of the residual rather than
    several times it), fp8 caches holding random context K/V, and the Gemma
    norm weights, which every rank holds the same."""
    shared = torch.Generator(device="cuda").manual_seed(1234)
    own = torch.Generator(device="cuda").manual_seed(100 + rank)

    def norm(n):
        w = torch.randn(n, generator=shared, device="cuda") * 0.1
        return SimpleNamespace(weight=w.to(torch.bfloat16))

    def linear(n, k, scale=121):
        w = torch.randn(n, k, generator=own, device="cuda").to(FP8)
        s = torch.randint(
            scale - 1, scale + 2, (n, k // 32), generator=own, device="cuda"
        )
        return SimpleNamespace(weight=w, weight_scale=s.to(torch.uint8))

    layers = []
    for i in range(LAYERS):
        kv = torch.randn(num_blocks, 1, BLOCK, 2 * HD, generator=own, device="cuda")
        attn = SimpleNamespace(
            qkv_proj=linear(QS + 2 * HD, H),
            o_proj=linear(H, QS),
            q_norm=norm(HD),
            k_norm=norm(HD),
            attn=SimpleNamespace(
                kv_cache=(kv * 0.5).to(FP8).view(torch.uint8), layer_name=f"d.{i}"
            ),
        )
        mlp = SimpleNamespace(
            gate_up_proj=linear(2 * IR, H), down_proj=linear(H, IR, scale=116)
        )
        layers.append(
            SimpleNamespace(
                self_attn=attn,
                mlp=mlp,
                input_layernorm=norm(H),
                post_attention_layernorm=norm(H),
            )
        )
    return SimpleNamespace(
        layers=layers, norm=norm(H), _rope_cos_sin_cache=_cos_sin(max(CTX) + 64)
    )


def _step(n_reqs: int, num_blocks: int):
    """A draft step's metadata, the same on every rank: 7 query rows a request
    at positions ctx .. ctx + 6 that see keys < ctx + 7, each request on its own
    blocks. With 10 requests the last one is padded as prepare_dflash_inputs
    pads a FULL graph's rows: position 0, seq_len 0, slot -1, a stale table row."""
    g = torch.Generator().manual_seed(n_reqs)
    ctx = CTX[:n_reqs]
    width = max(math.ceil((c + 7) / BLOCK) for c in ctx)
    perm = torch.randperm(num_blocks, generator=g)
    bt = torch.zeros(n_reqs, width, dtype=torch.int32)
    pos, slots, start = [], [], 0
    for r, c in enumerate(ctx):
        nb = math.ceil((c + 7) / BLOCK)
        bt[r, :nb] = perm[start : start + nb]
        start += nb
        p = torch.arange(c, c + 7)
        pos.append(p)
        slots.append(bt[r, p // BLOCK].long() * BLOCK + p % BLOCK)
    seq = torch.tensor([c + 7 for c in ctx], dtype=torch.int32)
    pos, slots = torch.cat(pos), torch.cat(slots)
    padded = n_reqs == 10
    if padded:
        pos[-7:], seq[-1], slots[-7:] = 0, 0, -1
        bt[-1] = bt[0]
    embeds = torch.randn(n_reqs * 7, H, generator=g).to(torch.bfloat16)
    cuda = lambda t: t.to("cuda")  # noqa: E731
    return cuda(embeds), cuda(pos), cuda(slots), cuda(bt), cuda(seq), padded


def _jitter(x: torch.Tensor) -> torch.Tensor:
    """X moved by about one bf16 ulp."""
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(0))
    return (x.float() * (1 + 2**-8 * noise.to(x.device))).to(torch.bfloat16)


def _reference(model, x, pos, slots, caches, bt, seq, group):
    """The stock draft layers' ops on this rank's shard (final norm included)."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        _mxfp8_dot_scaled_linear as mx,
    )
    from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
    )
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    n, res = x.shape[0], None
    one = torch.ones(1, 1, dtype=torch.float32, device="cuda")
    for layer, cache in zip(model.layers, caches):
        a, m = layer.self_attn, layer.mlp
        if res is None:
            res, h = x, gemma_rmsnorm(x, layer.input_layernorm.weight, EPS)
        else:
            h, res = gemma_fused_add_rmsnorm(x, res, layer.input_layernorm.weight, EPS)
        qkv = mx(h, a.qkv_proj.weight, a.qkv_proj.weight_scale)
        q8 = torch.empty(n, QS, dtype=FP8, device="cuda")
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv, a.q_norm.weight, a.k_norm.weight, model._rope_cos_sin_cache, pos,
            QH, 1, ROT, EPS, slot_mapping=slots, kv_cache=cache, block_size=BLOCK,
            kv_cache_dtype="fp8", q_fp8_out=q8, q_fp8_scale=1.0,
        )  # fmt: skip
        k, v = cache.transpose(1, 2).split(HD, dim=-1)
        out = torch.empty(n, QH, HD, dtype=torch.bfloat16, device="cuda")
        unified_attention(
            q=q8.view(n, QH, HD), k=k.view(FP8), v=v.view(FP8), out=out,
            cu_seqlens_q=torch.arange(0, n + 1, 7, dtype=torch.int32, device="cuda"),
            max_seqlen_q=7, seqused_k=seq, max_seqlen_k=int(seq.max()),
            softmax_scale=HD**-0.5, causal=False, window_size=(WINDOW - 1, 0),
            block_table=bt, softcap=0, q_descale=one[0], k_descale=one, v_descale=one,
        )  # fmt: skip
        o = mx(out.view(n, QS), a.o_proj.weight, a.o_proj.weight_scale)
        dist.all_reduce(o, group=group)
        h, res = gemma_fused_add_rmsnorm(
            o, res, layer.post_attention_layernorm.weight, EPS
        )
        gu = mx(h, m.gate_up_proj.weight, m.gate_up_proj.weight_scale).float()
        gate, up = gu[:, :IR].clamp(max=7.0), gu[:, IR:].clamp(-7.0, 7.0)
        act = (gate * torch.sigmoid(gate * 1.702) * (up + 1.0)).to(torch.bfloat16)
        x = mx(act, m.down_proj.weight, m.down_proj.weight_scale)
        dist.all_reduce(x, group=group)
    return gemma_fused_add_rmsnorm(x, res, model.norm.weight, EPS)[0]


def _spread(got: torch.Tensor, ref: torch.Tensor, jittered: torch.Tensor) -> float:
    """The worst request's error of got against ref over ref's own spread; the
    leading dim is the request."""

    def rel(a):
        diff = (a.float() - ref.float()).flatten(1).norm(dim=-1)
        return diff / ref.float().flatten(1).norm(dim=-1)

    return float((rel(got) / rel(jittered).clamp(min=1e-6)).max())


def _new_rows(cache: torch.Tensor, slots: torch.Tensor, n_reqs: int) -> torch.Tensor:
    """The K/V rows at slots as e4m3, one row of 7 x 256 a request."""
    return cache.view(-1, 2 * HD)[slots].view(FP8).view(n_reqs, -1)


def _numerics_rank(rank: int, port: int, results) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import set_forward_context
    from vllm.models.minimax_m3.amd.mono.drafter.runner import DrafterRunner

    torch.accelerator.set_device_index(rank)
    config = VllmConfig()
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=4,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=4)
    group = get_tp_group().device_group
    num_blocks = sum(math.ceil((c + 7) / BLOCK) for c in CTX) + 8
    model = _draft_model(rank, num_blocks)
    attns = [layer.self_attn.attn for layer in model.layers]
    config.compilation_config.static_forward_context.update(
        {a.layer_name: a for a in attns}
    )
    runner = DrafterRunner(model)
    caches = [a.kv_cache for a in attns]
    initial = [c.clone() for c in caches]

    def attention(slots, bt, seq):
        md = SimpleNamespace(block_table=bt, seq_lens=seq)
        return set_forward_context(
            {a.layer_name: md for a in attns},
            config,
            slot_mapping={a.layer_name: slots for a in attns},
        )

    def kernel_step(embeds, pos, slots, bt, seq):
        for c, c0 in zip(caches, initial):
            c.copy_(c0)
        with attention(slots, bt, seq):
            return runner.forward(pos, embeds)

    def stock_step(embeds, pos, slots, bt, seq):
        ref_caches = [c.clone() for c in initial]
        out = _reference(model, embeds, pos, slots, ref_caches, bt, seq, group)
        return out, ref_caches

    out: dict = {}
    for n_reqs in REQS:
        embeds, pos, slots, bt, seq, padded = _step(n_reqs, num_blocks)
        got = kernel_step(embeds, pos, slots, bt, seq).clone()
        real = n_reqs - padded
        rows = real * 7
        meta = (pos[:rows], slots[:rows], bt[:real], seq[:real])
        ref, ref_caches = stock_step(embeds[:rows], *meta)
        jittered, jittered_caches = stock_step(_jitter(embeds[:rows]), *meta)
        new = slots[:rows]
        kv = [
            _spread(*(_new_rows(c, new, real) for c in layer_caches))
            for layer_caches in zip(caches, ref_caches, jittered_caches)
        ]
        untouched = []
        for c, c0 in zip(caches, initial):
            rest = c.clone().view(-1, 2 * HD)
            rest[new] = c0.view(-1, 2 * HD)[new]
            untouched.append(torch.equal(rest, c0.view(-1, 2 * HD)))
        mine = got[:rows].contiguous()
        everyone = [torch.empty_like(mine) for _ in range(4)]
        dist.all_gather(everyone, mine, group=group)
        out[n_reqs] = dict(
            hidden=_spread(got[:rows].view(real, -1), ref.view(real, -1),
                           jittered.view(real, -1)),
            kv=max(kv),
            rest_of_cache_untouched=all(untouched),
            same_on_every_rank=all(torch.equal(everyone[0], e) for e in everyone),
        )  # fmt: skip
        if padded:
            # the padded 10-request step replayed from a CUDA graph
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                kernel_step(embeds, pos, slots, bt, seq)
                graph = torch.cuda.CUDAGraph()
                with attention(slots, bt, seq), torch.cuda.graph(graph, stream=stream):
                    replayed = runner.forward(pos, embeds)
            for c, c0 in zip(caches, initial):
                c.copy_(c0)
            graph.replay()
            torch.accelerator.synchronize()
            out["graph_equals_eager"] = torch.equal(replayed[:rows], got[:rows])
    torch.accelerator.synchronize()
    out["health"] = runner.health.tolist()
    results[rank] = out
    dist.barrier()
    runner.close()
    dist.destroy_process_group()


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(not _supported(), reason="needs 4 gfx950 GPUs")
def test_drafter_kernels_match_the_stock_layers():
    """On every rank and for every real request, the final hidden states and
    each layer's new K/V rows are within SPREAD of the stock layers' own spread,
    the rest of every cache is untouched (the padded request writes nothing),
    every rank returns the same rows, the CUDA graph replay is bit for bit the
    eager launch, and no wait in the kernels timed out."""
    from vllm.utils.network_utils import get_open_port

    results = mp.Manager().dict()
    ranks = mp.start_processes(
        _numerics_rank, args=(get_open_port(), results), nprocs=4, join=False
    )
    deadline = time.monotonic() + 600
    while not ranks.join(timeout=10):
        if time.monotonic() > deadline:
            for p in ranks.processes:
                p.kill()
            pytest.fail("TP4 drafter ranks still running after 600 s")
    for rank in range(4):
        r = results[rank]
        assert r["health"] == [0, 0, 0, 0], f"rank {rank}: a wait timed out {r}"
        assert r["graph_equals_eager"], f"rank {rank}"
        for n_reqs in REQS:
            c = r[n_reqs]
            msg = f"rank {rank}, {n_reqs} requests: {c}"
            assert c["hidden"] <= SPREAD and c["kv"] <= SPREAD, msg
            assert c["rest_of_cache_untouched"], msg
            assert c["same_on_every_rank"], msg
