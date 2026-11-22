# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU checks of the Kimi-K3.1 non-layer amd_kernels routes against the stock
paths (vllm/models/kimi_k3/amd/akr_*.py): the fused Markov chain behind
`DSparkSpeculator._sample_sequential` (synthetic and block rejection interfaces: the
draft tokens and the draft logits cache, with and without a row-mapped cache), the
`split_top_p_override`, the drafter's dense MLP and the tile packer.

Run: HIP_VISIBLE_DEVICES=0 pytest tests/kernels/models/amd/test_akr_nonlayer_1gpu.py
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="amd_kernels routes are for ROCm gfx950"
)

VOCAB, RANK, MAX_REQS = 163840, 512, 8


@pytest.fixture(autouse=True)
def _vllm_config(default_vllm_config):
    yield


def _need_akr():
    pytest.importorskip("amd_kernels")
    from vllm.models.kimi_k3.amd import akr_config

    if not akr_config._available():
        pytest.skip("amd_kernels / gfx950 not available")


# ---------------------------------------------------------------- Markov chain
class _MarkovW2(torch.nn.Module):
    def __init__(self, w):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.quant_method = UnquantizedEmbeddingMethod()


def _model(w1, w2, base):
    from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM

    class _Model(K3DSparkForCausalLM):
        def __init__(self):
            torch.nn.Module.__init__(self)
            head = torch.nn.Module()
            head.markov_w1 = torch.nn.Embedding(VOCAB, RANK, _weight=w1)
            head.markov_w2 = _MarkovW2(w2)
            self.model = SimpleNamespace(markov_head=head)
            self.logits_processor = SimpleNamespace(
                scale=1.0, soft_cap=None, head_dtype=torch.bfloat16
            )
            self.base = base

        def compute_draft_logits(self, hidden):
            return self.base[: hidden.shape[0]].clone()

        def markov_embed(self, ids):
            return self.model.markov_head.markov_w1(ids)

        def markov_bias(self, e):
            from vllm.model_executor.layers.utils import rocm_unquantized_gemm

            w2 = self.model.markov_head.markov_w2
            return rocm_unquantized_gemm(w2, e, w2.weight, None)

        def map_draft_to_target(self, ids):
            return ids

    return _Model()


def _speculator(temp: float, seed: int, row_mapped: bool, n_spec: int):
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    dev = torch.device("cuda")
    bf16 = torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(seed)
    w1 = (torch.randn(VOCAB, RANK, generator=g, device=dev) * 0.5).to(bf16)
    w2 = (torch.randn(VOCAB, RANK, generator=g, device=dev) * 0.05).to(bf16)
    base = torch.randn(MAX_REQS * n_spec, VOCAB, generator=g, device=dev) * 3
    s = DSparkSpeculator.__new__(DSparkSpeculator)
    s.device = dev
    s.vocab_size = VOCAB
    s.model = _model(w1, w2, base.to(bf16))
    s.num_speculative_steps = n_spec
    s.max_num_reqs = MAX_REQS
    s._draft_topk = None
    s.use_confidence_head = False
    s.acceptance_estimator = None
    s.draft_watermarker = None
    s._d2t_scatter_index = None
    s._draft_scatter_buf = None
    s.use_fp64_gumbel = False
    s.draft_logits = DraftLogitsCache(
        MAX_REQS, n_spec, VOCAB, dtype=bf16, fill=0.0, device=dev
    )
    if row_mapped:
        s.draft_logits.row_mapping = torch.tensor(
            [3, -1, 0, -1, 2, -1, 1, -1], dtype=torch.int32, device=dev
        )
    s.temperature = torch.full((MAX_REQS,), temp, dtype=torch.float32, device=dev)
    s.seeds = torch.randint(0, 2**62, (MAX_REQS,), generator=g, device=dev)
    s._step_cols = torch.arange(n_spec, dtype=torch.int32, device=dev)
    s.sample_indices = torch.arange(MAX_REQS * n_spec, dtype=torch.int64, device=dev)
    slots = torch.randperm(MAX_REQS, generator=g, device=dev).to(torch.int32)
    if MAX_REQS == 8:
        slots = torch.tensor([2, 0, 3, 1, 6, 5, 4, 7], dtype=torch.int32, device=dev)
    s.sample_idx_mapping = slots.repeat_interleave(n_spec).to(torch.int64)
    ctx = torch.randint(5, 800000, (MAX_REQS,), generator=g, device=dev)
    if MAX_REQS == 8:
        ctx = torch.tensor(
            [360000, 40000, 7, 99999, 131072, 5, 777777, 12],
            dtype=torch.int64,
            device=dev,
        )
    steps = torch.arange(n_spec, device=dev)
    s.sample_pos = (ctx[:, None] + 1 + steps[None, :]).flatten()
    s.input_buffers = SimpleNamespace(
        input_ids=torch.randint(
            0, VOCAB, (MAX_REQS * n_spec,), generator=g, device=dev, dtype=torch.int32
        )
    )
    s._anchor_idx = torch.arange(MAX_REQS, dtype=torch.int64, device=dev) * n_spec
    s.draft_tokens = torch.zeros(MAX_REQS, n_spec, dtype=torch.int64, device=dev)
    s._akr_markov = None  # type: ignore[attr-defined]
    return s


def _run(s, num_reqs):
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    s.draft_tokens.zero_()
    s.draft_logits.values.zero_()
    hidden = torch.zeros(MAX_REQS * s.num_speculative_steps, 8, device="cuda")
    DSparkSpeculator._sample_sequential(s, num_reqs, hidden)
    torch.accelerator.synchronize()
    return s.draft_tokens.clone(), s.draft_logits.values.clone()


@pytest.mark.parametrize("n_spec", [3, 7])
@pytest.mark.parametrize("row_mapped", [False, True])
@pytest.mark.parametrize("temp", [0.0, 0.6, 1.0])
@pytest.mark.parametrize("num_reqs", [1, 5, 8])
def test_markov_chain_matches_vllm_loop(num_reqs, temp, row_mapped, n_spec):
    _need_akr()
    from vllm.models.kimi_k3.amd import akr_sampling
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    s = _speculator(
        temp, seed=int(temp * 10) + num_reqs, row_mapped=row_mapped, n_spec=n_spec
    )
    ref_tok, ref_cache = _run(s, num_reqs)
    akr_sampling.setup_speculator(s)
    assert s._akr_markov is not None
    got_tok, got_cache = _run(s, num_reqs)

    slots = s.sample_idx_mapping[: num_reqs * n_spec : n_spec].long()
    rows = (
        slots
        if s.draft_logits.row_mapping is None
        else s.draft_logits.row_mapping[slots].long()
    )
    local = rows >= 0
    written = got_cache.view(-1, VOCAB).abs().sum(-1) != 0
    assert (
        int(written.sum()) == int(local.sum()) * n_spec
    )  # other ranks' rows untouched
    lr = rows[local]
    step_diff = (
        ref_cache[lr].view(torch.int16) != got_cache[lr].view(torch.int16)
    ).sum(-1)
    if step_diff.numel():
        assert int(step_diff.max()) <= VOCAB // 1000, step_diff.tolist()
    tok_diff = ref_tok[:num_reqs] != got_tok[:num_reqs]
    for r_i, r in enumerate(local.nonzero().flatten().tolist()):
        for k in range(n_spec):
            if tok_diff[r, k]:
                assert bool(tok_diff[r, :k].any()) or bool(
                    (step_diff[r_i, : k + 1] > 0).any()
                ), f"draft token ({r}, {k}) differs without a cause"
    # exact sampling from the cached logits: vLLM's sampler on the fused cache row
    idx_map = s.sample_idx_mapping[: num_reqs * n_spec].view(num_reqs, n_spec)
    pos = s.sample_pos[: num_reqs * n_spec].view(num_reqs, n_spec)
    base = s.model.base[: num_reqs * n_spec].view(num_reqs, n_spec, VOCAB)
    anchors = s.input_buffers.input_ids[s._anchor_idx[:num_reqs]]
    for k in range(n_spec):
        prev = anchors if k == 0 else got_tok[:num_reqs, k - 1]
        logits_k = base[:, k] + s.model.markov_bias(s.model.markov_embed(prev))
        if bool(local.any()):
            logits_k[local] = got_cache[lr, k]
        drawn = DSparkSpeculator._sample_logits(
            s, logits_k, idx_map[:, k], pos[:, k], k
        )
        assert torch.equal(drawn[local], got_tok[:num_reqs, k][local]), (
            k,
            drawn,
            got_tok,
        )


def test_markov_gates():
    _need_akr()
    from vllm.models.kimi_k3.amd import akr_sampling

    s = _speculator(1.0, 3, False, 3)
    assert akr_sampling._markov_reason(s) is None
    s.draft_watermarker = object()
    assert akr_sampling._markov_reason(s) is not None
    s.draft_watermarker = None
    cache, s.draft_logits = s.draft_logits, None
    assert akr_sampling._markov_reason(s) is not None
    s.draft_logits = cache
    s.use_fp64_gumbel = True
    assert akr_sampling._markov_reason(s) is not None
    s.use_fp64_gumbel = False
    # more steps than the kernel takes: the entry's reason at set-up
    s.num_speculative_steps = 9
    akr_sampling.setup_speculator(s)
    assert s._akr_markov is None


def test_markov_over_48_rows_falls_back():
    _need_akr()
    from vllm.models.kimi_k3.amd import akr_sampling

    s = _speculator(1.0, 5, False, 3)
    akr_sampling.setup_speculator(s)
    hidden = torch.zeros(49 * 3, 8, device="cuda")
    assert akr_sampling.markov_sequential(s, 49, hidden) is False


def test_markov_graph_replay():
    """The draft step captured in a graph and replayed with new base logits."""
    _need_akr()
    from vllm.models.kimi_k3.amd import akr_sampling

    s = _speculator(1.0, 7, False, 3)
    akr_sampling.setup_speculator(s)
    hidden = torch.zeros(MAX_REQS * 3, 8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            assert akr_sampling.markov_sequential(s, 5, hidden)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        akr_sampling.markov_sequential(s, 5, hidden)
    gen = torch.Generator(device="cuda").manual_seed(1)
    for _ in range(20):
        s.model.base.copy_(
            (torch.randn(s.model.base.shape, generator=gen, device="cuda") * 3).to(
                torch.bfloat16
            )
        )
        graph.replay()
        got = s.draft_tokens.clone()
        assert akr_sampling.markov_sequential(s, 5, hidden)
        torch.accelerator.synchronize()
        assert torch.equal(got[:5], s.draft_tokens[:5])
        assert bool(((got[:5] >= 0) & (got[:5] < VOCAB)).all())


# ---------------------------------------------------------------- top-p
@pytest.mark.parametrize("rows", [1, 3, 12, 24, 64])
@pytest.mark.parametrize("temp", [0.7, 1.0])
def test_top_p_override_bit_equal(rows, temp):
    _need_akr()
    from vllm.models.kimi_k3.amd import akr_sampling
    from vllm.v1.sample.ops import topk_topp_triton as tt

    g = torch.Generator(device="cuda").manual_seed(rows)
    logits = torch.randn(rows, VOCAB, generator=g, device="cuda") * 3 / temp
    p = torch.rand(rows, generator=g, device="cuda") * 0.6 + 0.4
    p[::5] = 1.0
    tt.split_top_p_override = None
    ref = tt.apply_top_k_top_p_triton(logits.clone(), None, p)
    assert akr_sampling.install_top_p(VOCAB)
    try:
        calls = []
        real = tt.split_top_p_override
        assert real is not None

        def spy(*a):
            calls.append(real(*a))
            return calls[-1]

        tt.split_top_p_override = spy
        got = tt.apply_top_k_top_p_triton(logits.clone(), None, p)
    finally:
        tt.split_top_p_override = None
    assert calls == [True]
    assert torch.equal(ref, got)


def test_top_p_override_declines():
    _need_akr()
    from vllm.models.kimi_k3.amd import akr_sampling
    from vllm.v1.sample.ops import topk_topp_triton as tt

    assert akr_sampling.install_top_p(VOCAB)
    try:
        ov = tt.split_top_p_override
        assert ov is not None
        x = torch.randn(4, VOCAB, device="cuda")
        p = torch.full((4,), 0.9, device="cuda")
        assert ov(x, p, float("-inf")) is True
        assert ov(x, p, -1e9) is False
        assert ov(x.to(torch.bfloat16), p, float("-inf")) is False
        assert (
            ov(
                torch.randn(65, VOCAB, device="cuda"),
                torch.full((65,), 0.9, device="cuda"),
                float("-inf"),
            )
            is False
        )
    finally:
        tt.split_top_p_override = None


# ---------------------------------------------------------------- drafter MLP
def test_tile_packer_matches_library():
    """`pack_tiles` against the packed row layout the entry reads ([tiles, K / 32,
    64, 8]: 16-row tiles, 4 x 8 columns of each 32-column K step)."""
    _need_akr()
    from amd_kernels.decoder.bf16_dense_mlp_common import dense_mlp_packed_tiles

    from vllm.models.kimi_k3.amd.akr_draft_ops import pack_tiles

    down = torch.empty(7168, 1792, dtype=torch.bfloat16, device="cuda")
    for rows in (1, 12, 16, 17, 60, 144, 192):
        x = torch.randn(rows, 7168, device="cuda").to(torch.bfloat16)
        tiles = dense_mlp_packed_tiles(rows, down=down)
        out = torch.full(
            (tiles * 224 * 512,), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        pack_tiles(x, out, tiles)
        xp = torch.zeros(tiles * 16, 7168, dtype=x.dtype, device=x.device)
        xp[:rows] = x
        ref = (
            xp.view(tiles, 16, 224, 4, 8)
            .permute(0, 2, 3, 1, 4)
            .reshape(tiles, 224, 64, 8)
        )
        assert torch.equal(out.view(tiles, 224, 64, 8), ref), rows


@pytest.mark.parametrize("rows", [12, 32, 60, 144])
def test_dense_mlp_vs_stock(rows):
    _need_akr()
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.models.kimi_k3.amd import akr_draft

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(rows)
    gu = (torch.randn(3584, 7168, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    dn = (torch.randn(7168, 1792, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    x = torch.randn(rows, 7168, generator=g, device=dev).to(torch.bfloat16)
    stock = (SiluAndMul()(x @ gu.t()) @ dn.t()).float()
    h = x.float() @ gu.float().t()
    act = (torch.nn.functional.silu(h[:, :1792]) * h[:, 1792:]).to(torch.bfloat16)
    ref = (
        act.float() @ dn.float().t()
    )  # the contract: one bf16 rounding of the intermediate

    class _Proj:
        def __init__(self, w):
            self.weight = SimpleNamespace(data=w)

    mlp = SimpleNamespace(gate_up_proj=_Proj(gu.clone()), down_proj=_Proj(dn.clone()))
    from amd_kernels.decoder.bf16_dense_mlp_common import dense_mlp_workspace

    m = akr_draft._Mlp(mlp, dense_mlp_workspace(dev, hidden=7168, inter=1792), dev)
    out = m(x).float()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    rel = (out - ref).norm() / ref.norm()
    assert float(cos) > 0.99999 and float(rel) < 2e-3, (float(cos), float(rel))
    assert (
        float((out - stock).norm() / stock.norm()) < 8e-3
    )  # stock rounds gate / up too
    # the stock weights were replaced in place by the packed layout (same shape)
    assert mlp.gate_up_proj.weight.data.shape == gu.shape


# ---------------------------------------------------------------- final attn_res + norm
def _fake_comm():
    """A one-rank stand-in of the 8-rank communicator `attnres_comm` builds (no
    group: the final norm runs with reduce=False), made from the class the entry
    routes (its `_SHAPE_HOSTS` module)."""
    import importlib

    from amd_kernels.norm.attnres_rmsnorm_common import host as entry

    host = importlib.import_module(entry._SHAPE_HOSTS[(7168, 8)])
    c = object.__new__(host.AttnResComm)
    MAX_ROWS = host.MAX_ROWS
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    c.rank, c.device, c.group = 0, dev, None
    c.bufs = [0] * 8
    c.buf_table = torch.zeros(8, dtype=torch.int64, device=dev)
    c.counts = torch.zeros(MAX_ROWS, dtype=torch.int32, device=dev)
    return c


@pytest.mark.parametrize("rows", [1, 12, 80, 192, 256, 300])
def test_final_norm_vs_stock(rows):
    """The fused final attn_res + final norm against the stock pair (Triton attn_res,
    then the model's RMSNorm): per-row cosine >= 0.99999, within 2^-7 |y| of the
    stock result, the all-in-one rows > 256 take the stock pair itself."""
    _need_akr()
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.kimi_k3.amd import akr_final
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    dev, bf16 = torch.device("cuda"), torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(rows)
    nb = 8
    prefix = torch.randn(rows, 7168, generator=g, device=dev).to(bf16)
    delta = torch.randn(rows, 7168, generator=g, device=dev).to(bf16)
    bank = (torch.randn(rows, nb, 7168, generator=g, device=dev) * 2).to(bf16)
    nw = (torch.rand(7168, generator=g, device=dev) + 0.5).to(bf16)
    qk = (torch.randn(7168, generator=g, device=dev) * 0.05).to(bf16)
    fw = (torch.rand(7168, generator=g, device=dev) + 0.5).to(bf16)
    norm = RMSNorm(7168, eps=1e-5).to(bf16).to(dev)
    norm.weight.data.copy_(fw)
    res_norm = SimpleNamespace(weight=nw, variance_epsilon=1e-6)
    proj = SimpleNamespace(weight=qk.unsqueeze(0))
    model = SimpleNamespace(
        norm=norm,
        output_attn_res_norm=res_norm,
        output_attn_res_proj=proj,
        _akr_final=_fake_comm(),
    )

    def stock():
        mix = attn_res(prefix.clone(), delta, bank, nw, qk, None, nb, -1, 1e-6, 0.0)
        return norm(mix, None)

    ref = stock()
    out = akr_final.final_hidden(
        model,
        lambda: attn_res(prefix.clone(), delta, bank, nw, qk, None, nb, -1, 1e-6, 0.0),
        prefix.clone(),
        bank,
        delta,
        nb,
    )
    assert out.shape == ref.shape and out.dtype == bf16
    cos = torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=-1)
    assert float(cos.min()) > 0.99999, float(cos.min())
    err = (out.float() - ref.float()).abs()
    assert bool(
        (err <= ref.float().abs() * 2**-6 + 1e-3 * ref.float().abs().max()).all()
    )
    if rows > 256:
        assert torch.equal(out, ref)


# ------------------------------------------------------- drafter dense MLA decode
@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.uint8])
@pytest.mark.parametrize("reqs,qlen", [(8, 4), (20, 3), (5, 3)])
def test_dense_mla_wrapper_vs_aiter(reqs, qlen, cache_dtype):
    """`akr_draft_mla._try` (the wrapper's kernel branch, fed with the backend's
    metadata fields) against AITER's persistent `mla_decode_fwd` as vLLM's CPRR path
    calls it: the latent output of the 96 heads and the natural-log LSE."""
    _need_akr()
    from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
    from aiter.mla import mla_decode_fwd

    from vllm.models.kimi_k3.amd import akr_draft_mla
    from vllm.v1.attention.backends.mla import rocm_aiter_mla as backend

    FP8, W, local_len = torch.float8_e4m3fn, 8, 3000
    sm = 192**-0.5
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(reqs * 10 + qlen)
    total = reqs * local_len
    cache = torch.randn((total + 64, 576), generator=g, device=dev).to(FP8)
    idx = torch.randperm(total, generator=g, device=dev).to(torch.int32)
    ip = torch.arange(
        0, (reqs + 1) * local_len, local_len, dtype=torch.int32, device=dev
    )
    gip = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=dev),
            (ip[1:] - ip[:-1]).mul(W).cumsum(0).to(torch.int32),
        ]
    )
    q = torch.randn((reqs * qlen, 96, 576), generator=g, device=dev).to(FP8)
    qs = torch.tensor([0.5], dtype=torch.float32, device=dev)
    ks = torch.tensor([0.25], dtype=torch.float32, device=dev)

    qp = q.repeat(1, 2, 1)[:, :128].contiguous()
    qo = torch.arange(0, (reqs + 1) * qlen, qlen, dtype=torch.int32, device=dev)
    last = torch.ones(reqs, dtype=torch.int32, device=dev)
    info = get_mla_metadata_info_v1(
        reqs,
        qlen,
        128,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=False,
        fast_mode=True,
        max_split_per_batch=256,
    )
    wm, wip, wis, rip, rfm, rpm = [
        torch.empty(s, dtype=t, device=dev) for (s, t) in info
    ]
    get_mla_metadata_v1(
        qo,
        ip,
        last,
        128,
        1,
        False,
        wm,
        wis,
        wip,
        rip,
        rfm,
        rpm,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=qlen,
        uni_seqlen_qo=qlen,
        fast_mode=True,
        dtype_q=dtypes.fp8,
        dtype_kv=dtypes.fp8,
        max_split_per_batch=256,
        intra_batch_mode=False,
        is_cp_round_robin=True,
    )
    o = torch.empty(reqs * qlen, 128, 512, dtype=torch.bfloat16, device=dev)
    _, lse_ref = mla_decode_fwd(
        qp,
        cache.view(-1, 1, 1, 576),
        o,
        qo,
        ip,
        idx,
        last,
        qlen,
        sm_scale=sm,
        return_lse=True,
        causal=True,
        work_meta_data=wm,
        work_indptr=wip,
        work_info_set=wis,
        reduce_indptr=rip,
        reduce_final_map=rfm,
        reduce_partial_map=rpm,
        q_scale=qs,
        kv_scale=ks,
        g_kv_indptr=gip,
        cp_world_size=W,
        cp_rank=3,
        num_kv_splits=256,
        intra_batch_mode=False,
    )
    kv_cache = cache.view(-1, 1, 576)
    if cache_dtype == torch.uint8:
        kv_cache = kv_cache.view(torch.uint8)
    decode = SimpleNamespace(
        use_gluon_decode=False,
        use_gluon_verify=False,
        dcp_route=backend._DCPDecodeRoute.CPRR,
        g_kv_indptr=gip,
        max_qo_len=qlen,
        paged_kv_indptr=ip,
        paged_kv_indices=idx,
        cp_rank=3,
        attn_out_dtype=torch.bfloat16,
    )
    got = akr_draft_mla._try(
        SimpleNamespace(dcp_world_size=8, scale=sm),
        q,
        kv_cache,
        SimpleNamespace(decode=decode, causal=True),
        SimpleNamespace(_q_scale=qs, _k_scale=ks),
    )
    assert got is not None, "the kernel branch declined"
    out, lse = got
    assert out.shape == (reqs * qlen, 96, 512) and out.dtype == torch.bfloat16
    ref = o[:, :96].float()
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(0, 1), ref.flatten(0, 1), dim=-1
    )
    assert float(cos.min()) > 0.995, float(cos.min())
    lref = lse_ref[:, :96] if lse_ref.dim() == 2 else lse_ref.flatten(1)[:, :96]
    assert lse.shape == lref.shape, (lse.shape, lse_ref.shape)
    assert float((lse - lref).abs().max()) < 0.05, float((lse - lref).abs().max())
    # not the kernel's case: a bf16 query, a non-causal block and a non-CPRR route fall
    # back
    assert (
        akr_draft_mla._try(
            SimpleNamespace(dcp_world_size=8, scale=sm),
            q.to(torch.bfloat16),
            kv_cache,
            SimpleNamespace(decode=decode, causal=True),
            SimpleNamespace(_q_scale=qs, _k_scale=ks),
        )
        is None
    )
    assert (
        akr_draft_mla._try(
            SimpleNamespace(dcp_world_size=8, scale=sm),
            q,
            kv_cache,
            SimpleNamespace(decode=decode, causal=False),
            SimpleNamespace(_q_scale=qs, _k_scale=ks),
        )
        is None
    )
