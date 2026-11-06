# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized DSpark drafter (specialized_decode/drafter): the fused Markov
sampling against vLLM's own loop (``DSparkSpeculator._sample_sequential``), and the
drafter GEMM engine against an FP32 reference and cuBLAS."""

import types

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(100):
    pytest.skip("specialized drafter kernels need SM100", allow_module_level=True)

from vllm.models.kimi_k3.nvidia.specialized_decode.drafter import markov  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.linears import (  # noqa: E402
    DraftGemm,
)
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (  # noqa: E402
    DSparkSpeculator,
)

HIDDEN = 256


@pytest.fixture(autouse=True)
def _fp32_cublas_reductions():
    # vLLM's loop runs the bias GEMV on cuBLAS; keep its split-K reductions in
    # FP32 so the exactly representable cases compare bit for bit.
    old = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    yield
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old


class UnquantizedEmbeddingMethod:  # the class name the fused path checks
    pass


def _weights(vocab, rank, exact, g):
    if exact:
        # Multiples of 1/64 in [-2, 2]: every product and partial sum of the
        # 256-term bias is exact in FP32, so any summation order rounds alike.
        def w(*s):
            return (torch.randint(-128, 129, s, generator=g, device="cuda") / 64.0).to(
                torch.bfloat16
            )

        return w(vocab, rank), w(vocab, rank)
    return (
        (torch.randn(vocab, rank, generator=g, device="cuda") * 0.3).bfloat16(),
        (torch.randn(vocab, rank, generator=g, device="cuda") * 0.3).bfloat16(),
    )


def _setup(R, k, vocab, rank, dtype, exact, fp64, seed=0, max_reqs=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    max_reqs = max_reqs or R + 3
    w1, w2 = _weights(vocab, rank, exact, g)
    lm = (torch.randn(vocab, HIDDEN, generator=g, device="cuda") * 0.2).bfloat16()

    head = types.SimpleNamespace(
        markov_w1=types.SimpleNamespace(weight=w1),
        markov_w2=types.SimpleNamespace(
            weight=w2, quant_method=UnquantizedEmbeddingMethod()
        ),
    )

    def compute_draft_logits(h):
        if dtype == torch.float32:
            return torch.mm(h, lm.t(), out_dtype=torch.float32)
        return F.linear(h, lm)

    def markov_bias(e):
        if dtype == torch.float32:
            return torch.mm(e, w2.t(), out_dtype=torch.float32)
        return F.linear(e, w2)

    model = types.SimpleNamespace(
        model=types.SimpleNamespace(markov_head=head),
        logits_processor=types.SimpleNamespace(scale=1.0, soft_cap=None),
        compute_draft_logits=compute_draft_logits,
        markov_embed=lambda t: F.embedding(t, w1),
        markov_bias=markov_bias,
        map_draft_to_target=lambda t: t,
    )

    num = R * k
    num_tokens = num + 8
    # Request slots shuffled; one padded row (-1) when R > 2.
    slots = torch.randperm(max_reqs, generator=g, device="cuda")[:R]
    idx = slots.repeat_interleave(k).clone()
    if R > 2:
        idx[(R - 1) * k :] = -1
    temp = torch.rand(max_reqs, generator=g, device="cuda") * 1.5 + 0.2
    temp[slots[0]] = 0.0  # one greedy request
    spec = types.SimpleNamespace(
        model=model,
        num_speculative_steps=k,
        max_num_reqs=max_reqs,
        sample_indices=torch.randperm(num_tokens, generator=g, device="cuda")[:num],
        sample_idx_mapping=idx,
        sample_pos=torch.randint(5, 10_000, (num,), generator=g, device="cuda"),
        input_buffers=types.SimpleNamespace(
            input_ids=torch.randint(0, vocab, (num_tokens,), generator=g, device="cuda")
        ),
        _anchor_idx=torch.arange(max_reqs, device="cuda") * k,
        temperature=temp,
        seeds=torch.randint(0, 2**62, (max_reqs,), generator=g, device="cuda"),
        draft_tokens=torch.zeros(max_reqs, k, dtype=torch.int64, device="cuda"),
        draft_logits=DraftLogitsCache(
            max_reqs, k, vocab, dtype=dtype, fill=0.0, device="cuda"
        ),
        _step_cols=torch.arange(k, dtype=torch.int32, device="cuda"),
        _d2t_scatter_index=None,
        _draft_topk=None,
        draft_watermarker=None,
        acceptance_estimator=None,
        use_confidence_head=False,
        use_fp64_gumbel=fp64,
    )
    spec._sample_logits = types.MethodType(DSparkSpeculator._sample_logits, spec)
    spec._maybe_predict_acceptance = types.MethodType(
        DSparkSpeculator._maybe_predict_acceptance, spec
    )
    head_hidden = (
        torch.randn(num_tokens, HIDDEN, generator=g, device="cuda") * 0.5
    ).bfloat16()
    return spec, model, head_hidden


def _run_both(R, k, vocab, rank, dtype, exact, fp64, seed=0):
    spec, model, hh = _setup(R, k, vocab, rank, dtype, exact, fp64, seed)
    DSparkSpeculator._sample_sequential(spec, R, hh)
    ref_tokens = spec.draft_tokens.clone()
    ref_cache = spec.draft_logits.values.clone()
    spec.draft_tokens.fill_(-7)
    spec.draft_logits.values.fill_(0)
    assert markov.sample_sequential(spec, model, R, hh)
    return spec, ref_tokens, ref_cache


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("R,k", [(1, 4), (4, 4), (12, 5), (33, 7), (70, 4)])
def test_fused_markov_matches_vllm_loop_exactly(R, k, dtype):
    """Exactly representable bias products: every token and cached logit equal."""
    spec, ref_tokens, ref_cache = _run_both(R, k, 20480 + 448, 256, dtype, True, False)
    rows = spec.sample_idx_mapping.view(R, k)[:, 0]
    live = rows >= 0
    assert torch.equal(spec.draft_tokens[:R][live], ref_tokens[:R][live])
    assert torch.equal(spec.draft_logits.values, ref_cache)


def test_fused_markov_fp64_gumbel_exact():
    spec, ref_tokens, ref_cache = _run_both(
        5, 4, 8192 + 64, 256, torch.bfloat16, True, True
    )
    assert torch.equal(spec.draft_tokens[:4], ref_tokens[:4])
    assert torch.equal(spec.draft_logits.values, ref_cache)


@pytest.mark.parametrize("R,k", [(4, 4), (16, 5)])
def test_fused_markov_random_weights(R, k):
    """Random weights at the K3.1 vocabulary: the bias may differ from cuBLAS in its
    last BF16 bit (summation order), nothing more."""
    spec, ref_tokens, ref_cache = _run_both(
        R, k, 163840, 256, torch.bfloat16, False, False
    )
    got = spec.draft_logits.values.float()
    ref = ref_cache.float()
    ulp = ref.abs().clamp_min(2**-60) * 2**-7
    assert ((got - ref).abs() <= ulp).all()
    frac_equal = (got == ref).float().mean().item()
    assert frac_equal > 0.97, frac_equal
    rows = spec.sample_idx_mapping.view(R, k)[:, 0] >= 0
    same = (spec.draft_tokens[:R][rows] == ref_tokens[:R][rows]).float().mean().item()
    assert same > 0.9, same


def test_fused_markov_cuda_graph():
    """Captured once, replayed with new anchors, temperatures and seeds."""
    R, k, vocab = 6, 4, 32768
    spec, model, hh = _setup(R, k, vocab, 256, torch.bfloat16, True, False, max_reqs=8)
    assert markov.sample_sequential(spec, model, R, hh)  # warmup (eager)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(graph):
        markov.sample_sequential(spec, model, R, hh)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.Generator(device="cuda").manual_seed(7)
    spec.input_buffers.input_ids.copy_(
        torch.randint(
            0, vocab, spec.input_buffers.input_ids.shape, generator=g, device="cuda"
        )
    )
    spec.seeds.copy_(
        torch.randint(0, 2**62, spec.seeds.shape, generator=g, device="cuda")
    )
    spec.temperature.mul_(0.7)
    hh.copy_((torch.randn(hh.shape, generator=g, device="cuda") * 0.5).bfloat16())
    graph.replay()
    torch.cuda.synchronize()
    got = spec.draft_tokens.clone()
    got_cache = spec.draft_logits.values.clone()
    spec.draft_logits.values.fill_(0)
    DSparkSpeculator._sample_sequential(spec, R, hh)
    live = spec.sample_idx_mapping.view(R, k)[:, 0] >= 0
    assert torch.equal(got[:R][live], spec.draft_tokens[:R][live])
    assert torch.equal(got_cache, spec.draft_logits.values)


@pytest.mark.parametrize("R,k", [(1, 4), (3, 5), (8, 4)])
def test_fused_markov_precomputed_noise_bit_equal(R, k, monkeypatch):
    """The noise computed ahead (side stream, beside the lm_head) and read back
    gives the same keys as hashing it in the step kernels: tokens and cached
    logits bit-identical, eager and under graph capture."""
    outs = []
    for rows in (0, markov.NOISE_ROWS):
        monkeypatch.setattr(markov, "NOISE_ROWS", rows)
        markov._FUSED.clear()
        spec, model, hh = _setup(R, k, 163840, 256, torch.bfloat16, False, False, 3)
        assert markov.sample_sequential(spec, model, R, hh)
        fused = next(iter(markov._FUSED.values()))
        assert (fused.noise is not None) == (rows > 0)
        eager = spec.draft_tokens.clone(), spec.draft_logits.values.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s), torch.cuda.graph(graph):
            markov.sample_sequential(spec, model, R, hh)
        torch.cuda.current_stream().wait_stream(s)
        spec.draft_tokens.fill_(-7)
        spec.draft_logits.values.fill_(0)
        spec.seeds.add_(1)
        graph.replay()
        torch.cuda.synchronize()
        outs.append((eager, (spec.draft_tokens.clone(), spec.draft_logits.values)))
    markov._FUSED.clear()
    for (a_tok, a_cache), (b_tok, b_cache) in zip(outs[0], outs[1]):
        assert torch.equal(a_tok[:R], b_tok[:R])
        assert torch.equal(a_cache, b_cache)


@pytest.mark.parametrize("rows", [0, markov.NOISE_ROWS])
@pytest.mark.parametrize("R,k", [(1, 4), (3, 5), (8, 4)])
def test_fused_markov_gumbel_keys_match_stock(R, k, rows, monkeypatch):
    """The Gumbel keys themselves, not only their argmax: every step's per-block
    maxima (blocks of ``VB`` tokens) bit-equal vLLM's ``_gumbel_sample_kernel`` run
    on the cached logits with the same block size, with the noise hashed in the
    steps and precomputed. A token compares equal unless two keys tie within an
    ulp, so a token-only test misses a one-ulp difference in the keys."""
    from vllm.v1.worker.gpu.sample import gumbel

    monkeypatch.setattr(markov, "NOISE_ROWS", rows)
    markov._FUSED.clear()
    maxima = []
    select = markov._markov_select_kernel

    class Recorder:
        def __getitem__(self, grid):
            def launch(out_max, *args, **kwargs):
                maxima.append(out_max.clone())
                return select[grid](out_max, *args, **kwargs)

            return launch

    monkeypatch.setattr(markov, "_markov_select_kernel", Recorder())
    vocab = 163840
    spec, model, hh = _setup(R, k, vocab, 256, torch.bfloat16, False, False, 11)
    g = torch.Generator(device="cuda").manual_seed(5)
    spec.temperature.copy_(
        torch.rand(spec.temperature.shape, generator=g, device="cuda") + 0.05
    )
    assert markov.sample_sequential(spec, model, R, hh)
    assert (next(iter(markov._FUSED.values())).noise is not None) == (rows > 0)
    markov._FUSED.clear()
    nb = (vocab + markov.VB - 1) // markov.VB
    for step in range(k):
        idx = spec.sample_idx_mapping[: R * k].view(R, k)[:, step].contiguous()
        pos = (spec.sample_pos[: R * k].view(R, k)[:, step] - 1).contiguous()
        logits = spec.draft_logits.values[idx.clamp(min=0), step, :vocab].contiguous()
        ref = torch.empty(R, nb, dtype=torch.float32, device="cuda")
        arg = torch.empty(R, nb, dtype=torch.int64, device="cuda")
        gumbel._gumbel_sample_kernel[(R, nb)](
            arg, arg.stride(0), ref, ref.stride(0),
            None, 0, 0, None, None, 0, None,
            logits, logits.stride(0), idx, spec.seeds ^ (step << 32), pos,
            spec.temperature, vocab,
            BLOCK_SIZE=markov.VB, IS_DRAFTING=True, APPLY_TEMPERATURE=True,
            USE_FP64=False, PER_TOKEN_COL=False,
        )  # fmt: skip
        valid = idx >= 0
        got = maxima[step][:R][valid]
        assert torch.equal(got.view(torch.int32), ref[valid].view(torch.int32))


# Per-rank drafter shapes (N, K): TP8 / DCP8 and TP1 (DEP).
GEMM_SHAPES = [
    (2112, 7168),  # fused_qkv_a (replicated)
    (1536, 1536),  # q_b TP8
    (7168, 1024),  # o_proj TP8
    (3584, 7168),  # gate_up TP8
    (7168, 1792),  # down TP8
    (896, 35840),  # context_proj TP8 shard (K > 8192: cuBLAS)
    (7168, 8192),  # o_proj TP1
    (7168, 14336),  # down TP1 (K tiles of 256)
]


@pytest.mark.parametrize("n,k", GEMM_SHAPES)
@pytest.mark.parametrize("m", [1, 4, 16, 20, 32, 48])
def test_draft_gemm_matches_reference(n, k, m):
    g = torch.Generator(device="cuda").manual_seed(n + k + m)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.02).bfloat16()
    x = torch.randn(m, k, generator=g, device="cuda").bfloat16()
    gemm = DraftGemm(w)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    gemm.run(x, out)
    ref = x.float() @ w.float().t()
    cub = (x @ w.t()).float()
    err = ((out.float() - ref).norm() / ref.norm()).item()
    err_cublas = ((cub - ref).norm() / ref.norm()).item()
    assert err <= 1.5 * err_cublas + 1e-4, (err, err_cublas)
    eligible = k <= 16384 and m <= 32
    assert gemm.kind(m) == ("warp" if eligible else "cublas")
    # Graph replay with new activations.
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(graph):
        gemm.run(x, out)
    torch.cuda.current_stream().wait_stream(s)
    x.copy_(torch.randn(m, k, generator=g, device="cuda").bfloat16())
    graph.replay()
    ref = x.float() @ w.float().t()
    err = ((out.float() - ref).norm() / ref.norm()).item()
    assert err <= 1.5 * err_cublas + 1e-4


@pytest.mark.parametrize("head_dtype", [None, torch.float32])
def test_draft_logits_lm_head(head_dtype):
    """The drafter's lm_head shard on DraftGemm for BF16 logits; an FP32 head
    keeps the model's own compute_logits."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.linears import (
        draft_logits,
    )

    g = torch.Generator(device="cuda").manual_seed(3)
    vocab, hidden = 20480, 7168
    w = (torch.randn(vocab, hidden, generator=g, device="cuda") * 0.02).bfloat16()
    lp = types.SimpleNamespace(
        head_dtype=head_dtype,
        logits_as_input=False,
        org_vocab_size=vocab - 64,
        soft_cap=None,
        scale=1.0,
        _gather_logits=lambda x: x,
    )
    head = types.SimpleNamespace(
        weight=w, quant_method=UnquantizedEmbeddingMethod(), tp_size=1
    )

    def compute_logits(h):
        if head_dtype == torch.float32:
            return torch.mm(h, w.t(), out_dtype=torch.float32)[..., : vocab - 64]
        return F.linear(h, w)[..., : vocab - 64]

    model = types.SimpleNamespace(
        logits_processor=lp, lm_head=head, compute_logits=compute_logits
    )
    for rows in (4, 16, 20):
        h = torch.randn(rows, hidden, generator=g, device="cuda").bfloat16()
        got = draft_logits(model, h)
        ref = compute_logits(h)
        assert got.shape == ref.shape and got.dtype == ref.dtype
        if head_dtype == torch.float32:
            assert torch.equal(got, ref)
        else:
            exact = h.float() @ w.float().t()
            exact = exact[..., : vocab - 64]
            err = ((got.float() - exact).norm() / exact.norm()).item()
            err_ref = ((ref.float() - exact).norm() / exact.norm()).item()
            assert err <= 1.5 * err_ref + 1e-4
    assert (model._draft_lm_head_gemm[1] is None) == (head_dtype is not None)
