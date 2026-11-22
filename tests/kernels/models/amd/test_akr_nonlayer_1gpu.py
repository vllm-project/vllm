# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU checks of the Kimi-K3.1 non-layer amd_kernels routes against the stock
paths (vllm/models/kimi_k3/amd/akr_*.py): the fused Markov chain behind
`DSparkSpeculator._sample_sequential` (synthetic and block rejection interfaces: the
draft tokens and the draft logits cache, with and without a row-mapped cache; the
preview's and the release's draft class), the `split_top_p_override`, the drafter's
dense MLP (both drafts' MLP modules, eager, torch.compile and graph replay), the tile
packer, and the release draft's fused q / k norm + RoPE + KV insert and context K/V
store (`akr_draft_gqa`).

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


def _draft_cls(draft: str):
    if draft == "release":
        from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

        return Qwen3DSparkForCausalLM
    from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM

    return K3DSparkForCausalLM


def _model(w1, w2, base, draft: str = "preview"):
    class _Model(_draft_cls(draft)):  # type: ignore[misc]
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


def _speculator(
    temp: float, seed: int, row_mapped: bool, n_spec: int, draft: str = "preview"
):
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
    s.model = _model(w1, w2, base.to(bf16), draft)
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


@pytest.mark.parametrize(
    "draft,n_spec,row_mapped,temp,num_reqs",
    [
        ("preview", n, rm, t, r)
        for n in (3, 7)
        for rm in (False, True)
        for t in (0.0, 0.6, 1.0)
        for r in (1, 5, 8)
    ]
    # the release (nst 4 at c16..c48, 7 at c1, 5 at c4): the same head and loop
    + [
        ("release", n, rm, 1.0, r)
        for n in (4, 7)
        for rm in (False, True)
        for r in (1, 8)
    ],
)
def test_markov_chain_matches_vllm_loop(draft, num_reqs, temp, row_mapped, n_spec):
    _need_akr()
    from vllm.models.kimi_k3.amd import akr_sampling
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    s = _speculator(
        temp,
        seed=int(temp * 10) + num_reqs,
        row_mapped=row_mapped,
        n_spec=n_spec,
        draft=draft,
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
    for rows in (1, 12, 16, 17, 60, 144, 192, 193, 240, 256):
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


@pytest.mark.parametrize("rows", [12, 32, 60, 144, 145, 192, 240, 256])
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

    stage = torch.zeros(
        akr_draft._x_numel(dn, akr_draft.MLP_MAX_ROWS), dtype=torch.bfloat16, device=dev
    )
    m = akr_draft._Mlp(mlp, dense_mlp_workspace(dev, hidden=7168, inter=1792), stage)
    out = m(x).float()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    rel = (out - ref).norm() / ref.norm()
    assert float(cos) > 0.99999 and float(rel) < 2e-3, (float(cos), float(rel))
    assert (
        float((out - stock).norm() / stock.norm()) < 8e-3
    )  # stock rounds gate / up too
    # the stock weights were replaced in place by the packed layout (same shape)
    assert mlp.gate_up_proj.weight.data.shape == gu.shape


def _draft_mlp(draft: str):
    """One layer's MLP of the draft class at the TP8 shard (7168, 1792): the preview's
    `KimiMLP` (its all-reduce is the next norm's), the release's `Qwen2MLP` (its down
    projection all-reduces; a no-op at TP 1)."""
    if draft == "preview":
        from vllm.models.kimi_k3.nvidia.model import KimiMLP

        mlp = KimiMLP(7168, 1792, "silu", reduce_results=False)
    else:
        from vllm.model_executor.models.qwen2 import Qwen2MLP

        mlp = Qwen2MLP(7168, 1792, "silu")
    mlp = mlp.to(device="cuda", dtype=torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(11)
    for p in mlp.parameters():
        p.data.copy_(torch.randn(p.shape, generator=g, device="cuda") * 0.02)
    return mlp


@pytest.mark.parametrize("draft", ["preview", "release"])
def test_dense_mlp_module_both_drafts(dist_init, draft):
    """`_install_mlp` on the draft's own layers: the layer's `mlp` becomes the op
    `vllm.kimi_akr_draft_dense_mlp` (+ the stock all-reduce when the MLP had one), the
    same in eager, under torch.compile (fullgraph: the release's backbone is compiled)
    and in a graph replay."""
    _need_akr()
    from vllm.models.kimi_k3.amd import akr_draft

    layer = torch.nn.Module()
    layer.mlp = _draft_mlp(draft)
    reduces = draft == "release"
    assert layer.mlp.down_proj.reduce_results == reduces
    xs = {
        rows: torch.randn(rows, 7168, device="cuda").to(torch.bfloat16)
        for rows in (1, 5, 16, 17, 32, 80, 129, 144)
    }
    stock = {rows: layer.mlp(x).float() for rows, x in xs.items()}
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    spec = SimpleNamespace(device=torch.device("cuda"), max_num_reqs=36)
    assert akr_draft._install_mlp(spec, model) == 1
    assert isinstance(layer.mlp, akr_draft.AkrDenseMlp)
    assert layer.mlp.all_reduce is False  # TP 1
    compiled = torch.compile(
        lambda x: layer.mlp(x), fullgraph=True, dynamic=True, backend="aot_eager"
    )
    for rows, x in xs.items():
        for out in (layer.mlp(x), compiled(x)):
            rel = float((out.float() - stock[rows]).norm() / stock[rows].norm())
            assert out.shape == x.shape and rel < 8e-3, (draft, rows, rel)
    x = xs[32].clone()
    layer.mlp(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = layer.mlp(x)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(out, layer.mlp(x))


def test_dense_mlp_layers_share_staging(dist_init):
    """Two installed layers share one workspace and one activation staging, and the
    two-layer chain (each layer's packed rows consumed before the next packs) is the
    stock chain's, eager and in a graph replay."""
    _need_akr()
    from vllm.models.kimi_k3.amd import akr_draft

    layers = []
    for _ in range(2):
        layer = torch.nn.Module()
        layer.mlp = _draft_mlp("release")
        layers.append(layer)
    x = torch.randn(80, 7168, device="cuda").to(torch.bfloat16)
    stock = layers[1].mlp(layers[0].mlp(x)).float()
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    spec = SimpleNamespace(device=torch.device("cuda"), max_num_reqs=20)
    first = len(akr_draft._MLPS)
    assert akr_draft._install_mlp(spec, model) == 2
    a, b = akr_draft._MLPS[first], akr_draft._MLPS[first + 1]
    assert a.x_stage is b.x_stage and a.ws is b.ws

    def chain(x):
        return layers[1].mlp(layers[0].mlp(x))

    out = chain(x)
    rel = float((out.float() - stock).norm() / stock.norm())
    assert rel < 1.2e-2, rel
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = chain(x)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(replayed, out)


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


# ------------------------------------------------------------ release draft q / K / V
NQ, NKV, HD, PAGE, PAGES = 12, 2, 64, 64, 96


def _triton_attn(kind: str, idx: int = 0):
    """A TRITON_ATTN draft attention layer: impl (cache update), cache, scales."""
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    impl = TritonAttentionImpl.__new__(TritonAttentionImpl)
    impl.attn_type = AttentionType.DECODER
    impl.kv_cache_dtype = "fp8" if kind == "fp8" else "auto"
    impl._is_per_token_head_quant = False
    impl.head_size = HD
    impl.fp8_dtype = current_platform.fp8_dtype()
    impl._set_kv_ptr_range = False
    g = torch.Generator(device="cuda").manual_seed(100 + idx)
    shape = (PAGES, NKV, PAGE, 2 * HD)
    if kind == "fp8":
        cache = torch.randint(0, 120, shape, generator=g, device="cuda").to(torch.uint8)
    else:
        cache = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
    return SimpleNamespace(
        impl=impl,
        kv_cache=cache,
        _k_scale=torch.tensor(0.0625 * (1 + idx), device="cuda"),
        _v_scale=torch.tensor(0.03 * (1 + idx), device="cuda"),
        kv_sharing_target_layer_name=None,
        attn_backend=SimpleNamespace(
            forward_includes_kv_cache_update=False, get_name=lambda: "TRITON_ATTN"
        ),
        layer_name=f"model.layers.{idx}.self_attn.attn",
    )


def _rope():
    from vllm.model_executor.layers.rotary_embedding import get_rope

    return get_rope(
        HD,
        max_position=1 << 16,
        is_neox_style=True,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        dtype=torch.bfloat16,
    )


def _gqa_layer(kind: str):
    """The release draft's `DFlashQwen3Attention` pieces the routes read (TP8 shard)."""
    from vllm.model_executor.layers.layernorm import RMSNorm

    a = torch.nn.Module()
    a.num_heads, a.num_kv_heads, a.head_dim = NQ, NKV, HD
    a.q_size, a.kv_size, a.v_scale = NQ * HD, NKV * HD, None
    g = torch.Generator(device="cuda").manual_seed(7)
    for name in ("q_norm", "k_norm"):
        norm = RMSNorm(HD, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)
        norm.weight.data.copy_(torch.rand(HD, generator=g, device="cuda") + 0.5)
        setattr(a, name, norm)
    a.rotary_emb = _rope().cuda()
    a.attn = _triton_attn(kind)
    return a


def _slots(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randperm(PAGES * PAGE, generator=g, device="cuda")[:n].clone()
    s[::5] = -1  # padded rows write nothing
    return s


def _kv_close(got: torch.Tensor, ref: torch.Tensor, kind: str, exact: bool) -> None:
    """Cache: V bit-equal; K bit-equal (`exact`) or within one bf16 ulp / e4m3 step of
    the stock ops or 2^-6 of its head's maximum (RoPE cancels; the eager stock ops round
    the normed K before RoPE, the kernel and the compiled layer do not)."""
    assert torch.equal(got[..., HD:], ref[..., HD:])
    a, b = got[..., :HD], ref[..., :HD]
    if exact:
        assert torch.equal(a, b)
        return
    if kind == "fp8":
        a, b = (t.view(torch.float8_e4m3fn).float() for t in (a, b))
        step = 2.0**-3
    else:
        a, b = a.float(), b.float()
        step = 2.0**-7
    tol = (
        torch.maximum(b.abs() * step, b.abs().amax(dim=-1, keepdim=True) * 2.0**-6)
        + 2.0**-9
    )
    assert bool(((a - b).abs() <= tol).all()), float(((a - b).abs() - tol).max())


@pytest.mark.parametrize("kind", ["fp8", "bf16"])
@pytest.mark.parametrize("rows", [1, 7, 32, 80, 192, 240, 2048])
def test_draft_qkv_norm_rope_insert_vs_stock(monkeypatch, kind, rows):
    """The op behind `DQKV` (`akr_draft_gqa._qkv_op`) against the stock ops it replaces
    (q / k RMSNorm, NeoX RoPE, TRITON_ATTN's cache update): q within one bf16 ulp of
    the row maximum, the cache within one step; no slot (profile run): q only."""
    _need_akr()
    from vllm.model_executor.layers.attention import attention as A
    from vllm.models.kimi_k3.amd import akr_draft_gqa as G

    a = _gqa_layer(kind)
    assert G._layer_reason(a) is None
    key = len(G._LAYERS)
    G._LAYERS[key] = a
    g = torch.Generator(device="cuda").manual_seed(rows)
    qkv = torch.randn(rows, (NQ + 2 * NKV) * HD, generator=g, device="cuda")
    qkv = qkv.to(torch.bfloat16)
    pos = torch.randint(0, 1 << 16, (rows,), generator=g, device="cuda")
    slots = _slots(rows, rows)
    ref_cache = a.attn.kv_cache.clone()
    ref_q = G._qkv_stock(
        a,
        SimpleNamespace(**{**vars(a.attn), "kv_cache": ref_cache}),
        qkv,
        pos,
        ref_cache,
        slots,
    )
    ctx = {"slots": slots}
    monkeypatch.setattr(
        A,
        "get_attention_context",
        lambda name: (None, a.attn, a.attn.kv_cache, ctx["slots"]),
    )
    q = G._qkv_op(qkv, pos, key, a.attn.layer_name)
    tol = ref_q.float().abs().amax(dim=-1, keepdim=True) * 2**-6
    assert bool(((q.float() - ref_q.float()).abs() <= tol).all())
    _kv_close(a.attn.kv_cache, ref_cache, kind, exact=False)
    ctx["slots"] = None
    before = a.attn.kv_cache.clone()
    q2 = G._qkv_op(qkv, pos, key, a.attn.layer_name)
    assert torch.equal(q2, q) and torch.equal(a.attn.kv_cache, before)


def _ctx_model(kind: str, hidden: int = 512):
    """A `DFlashQwen3Model` shell with the release draft's context K/V buffers at TP8
    (5 layers, 2 KV heads of 64) and TRITON_ATTN caches."""
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    nl = 5
    m = DFlashQwen3Model.__new__(DFlashQwen3Model)
    torch.nn.Module.__init__(m)
    g = torch.Generator(device="cuda").manual_seed(3)
    m._num_attn_layers, m._head_dim, m._num_kv_heads = nl, HD, NKV
    m._kv_size, m._rms_norm_eps = NKV * HD, 1e-5
    m._k_norm_weights = (torch.rand(nl, HD, generator=g, device="cuda") + 0.5).to(
        torch.bfloat16
    )
    rope = _rope().cuda()
    m._rope_cos_sin_cache, m._rope_head_size, m._rope_is_neox = (
        rope.cos_sin_cache,
        HD,
        True,
    )
    w = torch.randn(nl * 2 * NKV * HD, hidden, generator=g, device="cuda") * 0.05
    m._fused_kv_weight, m._fused_kv_bias, m._context_kv_proj = w.bfloat16(), None, None
    m._hidden_norm_weight = (torch.rand(hidden, generator=g, device="cuda") + 0.5).to(
        torch.bfloat16
    )
    m.__dict__["layers"] = [SimpleNamespace(self_attn=SimpleNamespace(v_scale=None))]
    m._attn_layers = [_triton_attn(kind, i) for i in range(nl)]
    return m


@pytest.mark.parametrize("kind", ["fp8", "bf16"])
@pytest.mark.parametrize("rows", [1, 12, 40, 100, 240, 1000])
@pytest.mark.parametrize("per_layer", [False, True])
def test_draft_context_kv_store_vs_stock(kind, rows, per_layer):
    """`DCTXKV` (`akr_draft_gqa.ContextKv`) against the stock
    `precompute_and_store_context_kv` on the same caches: every layer's cache bytes;
    a graph replay of the route equals its eager call."""
    _need_akr()
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    from vllm.models.kimi_k3.amd import akr_draft_gqa as G

    m = _ctx_model(kind)
    model = SimpleNamespace(model=m)
    assert G._ctx_reason(model) is None
    g = torch.Generator(device="cuda").manual_seed(rows)
    states = torch.randn(rows, 512, generator=g, device="cuda").to(torch.bfloat16)
    pos = torch.randint(0, 1 << 16, (rows,), generator=g, device="cuda")
    slots = [_slots(rows, rows + (i if per_layer else 0)) for i in range(5)]
    slot_arg = slots if per_layer else slots[0]
    start = [a.kv_cache.clone() for a in m._attn_layers]
    DFlashQwen3Model.precompute_and_store_context_kv(m, states, pos, slot_arg)
    ref = [a.kv_cache.clone() for a in m._attn_layers]
    for a, c in zip(m._attn_layers, start):
        a.kv_cache.copy_(c)
    route = G.ContextKv(m)
    route(states, pos, slot_arg)
    for a, r in zip(m._attn_layers, ref):
        _kv_close(a.kv_cache, r, kind, exact=True)
    if rows != 100:
        return
    eager = [a.kv_cache.clone() for a in m._attn_layers]
    for a, c in zip(m._attn_layers, start):
        a.kv_cache.copy_(c)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        route(states, pos, slot_arg)
    graph.replay()
    torch.accelerator.synchronize()
    for a, e in zip(m._attn_layers, eager):
        assert torch.equal(a.kv_cache, e)
