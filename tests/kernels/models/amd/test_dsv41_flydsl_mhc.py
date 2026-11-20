# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's mHC seams and step-end tail on AITER's FlyDSL kernels
(VLLM_ROCM_USE_AITER_FLYDSL_MHC).

The first routing tests run anywhere, without the kernels (CI images do not
ship them yet): `mhc_pre_delayed_flydsl` hands the seams outside AITER's rule
to the seam it replaces with the same arguments. `_RuleOnlyOps` stands in for
AITER there, with AITER's documented rule and no kernel.

The other tests need the kernels (gfx950 and an AITER that ships
`flydsl_mhc_post_pre_rmsnorm`): the call sites against the torch reference of
the path they replace, and a real `DeepseekV4DecoderLayer` and
`DeepseekV4Model` with the flag on against the same ones with it off. The layer
is built as in `tests/kernels/mhc/test_mhc_kernels.py` (its attention and FFN
are plain callables) and the model as in
`tests/models/test_deepseek_v41_amd_replay_forward.py`; both then run their
real `forward`.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.mhc.torch import (
    mhc_post_torch,
    mhc_pre_delayed_torch,
)
from vllm.models.deepseek_v41.amd import mhc
from vllm.utils.torch_utils import set_random_seed

HIDDEN, HC = 5120, 4
MIX = (2 + HC) * HC
GATES = (1e-6, 1e-6, 1e-6, 2.0, 20)


class _RuleOnlyOps:
    """AITER's documented eligibility and dispatch rule, without the kernel:
    the seam takes 129-256 rows, or 1-1024 rows when it also returns the
    stream mean."""

    def __init__(self):
        self.calls: list[dict] = []

    def flydsl_mhc_post_pre_rmsnorm_supported(self, num_tokens, hidden, hc_mult):
        return hidden == HIDDEN and hc_mult == HC and 0 <= num_tokens <= 4096

    def flydsl_mhc_post_pre_rmsnorm_preferred(self, num_tokens, aux_mean=False):
        if aux_mean:
            return 0 < num_tokens <= 1024
        return 129 <= num_tokens <= 256

    def flydsl_mhc_post_pre_rmsnorm(self, residual, *args, **kwargs):
        self.calls.append(kwargs)
        t = residual.shape[0]
        out = (
            residual,
            torch.zeros(t, HC, 1),
            torch.zeros(t, HC, HC),
            torch.zeros(t, HIDDEN, dtype=torch.bfloat16),
            torch.zeros(t, HC),
        )
        if kwargs["aux_mean"]:
            out += (torch.zeros(t, HIDDEN, dtype=torch.bfloat16),)
        return out


def _seam_inputs(num_tokens: int, device: str = "cpu", post: bool = True):
    residual = torch.randn(num_tokens, HC, HIDDEN, device=device).to(torch.bfloat16)
    fn = torch.randn(MIX, HC * HIDDEN, device=device) * 0.01
    args = (
        residual,
        fn,
        torch.rand(3, device=device),
        torch.randn(MIX, device=device) * 0.1,
        *GATES,
    )
    kwargs = dict(
        pre_mix=torch.softmax(torch.randn(num_tokens, HC, device=device), -1),
        norm_weight=(torch.rand(HIDDEN, device=device) + 0.5).to(torch.bfloat16),
        norm_eps=1e-6,
    )
    if post:
        kwargs.update(
            sublayer_out=torch.randn(num_tokens, HIDDEN, device=device).to(
                torch.bfloat16
            ),
            post_layer_mix=torch.rand(num_tokens, HC, 1, device=device),
            comb_res_mix=torch.rand(num_tokens, HC, HC, device=device),
        )
    return args, kwargs


@pytest.mark.parametrize(
    ("num_tokens", "aux", "flydsl"),
    [
        # Mega-MHC ties at up to 128 rows and wins from 288 rows.
        (128, False, False),
        (129, False, True),
        (256, False, True),
        (288, False, False),
        # The mean of the new streams also replaces an mhc_post + mean.
        (16, True, True),
        (1024, True, True),
    ],
)
def test_seam_routing(num_tokens, aux, flydsl):
    """Seams outside AITER's rule run the seam they replace, unchanged; with
    `aux_out` (only given where the rule takes it) the mean is appended."""
    ops = _RuleOnlyOps()
    fallback_calls = []

    def fallback(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return "default seam"

    args, kwargs = _seam_inputs(num_tokens)
    aux_out: list[torch.Tensor] | None = [] if aux else None
    out = mhc.mhc_pre_delayed_flydsl(
        fallback,
        *args,
        **kwargs,
        ops=ops,
        fn_flydsl=torch.empty(0),
        mxfp8=False,
        aux_out=aux_out,
    )
    assert (len(ops.calls), len(fallback_calls)) == ((1, 0) if flydsl else (0, 1))
    if flydsl:
        assert ops.calls[0]["aux_mean"] == aux
        assert aux_out is None or len(aux_out) == 1
    else:
        assert out == "default seam"
        assert fallback_calls[0] == (args, {**kwargs, "x": None, "fn_packed": None})


def test_seam_keeps_layer0_and_unfused_norm():
    """The model-entry seam (`x`) and seams without the fused bf16 norm stay
    on the default path even inside the row window."""
    ops = _RuleOnlyOps()
    args, kwargs = _seam_inputs(192, post=False)

    def run(**overrides):
        return mhc.mhc_pre_delayed_flydsl(
            lambda *a, **k: "default seam",
            *args,
            **{**kwargs, **overrides},
            ops=ops,
            fn_flydsl=torch.empty(0),
            mxfp8=False,
        )

    assert run(x=torch.zeros(192, HIDDEN)) == "default seam"
    assert run(norm_weight=None) == "default seam"
    assert run(norm_weight=kwargs["norm_weight"].float()) == "default seam"
    assert not ops.calls


@pytest.fixture
def flydsl_ops(monkeypatch):
    """The real kernels, or skip."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FLYDSL_MHC", "1")
    rocm_aiter_ops.refresh_env_variables()
    ops = mhc.get_flydsl_mhc_ops(HIDDEN, HC)
    if ops is None:
        rocm_aiter_ops.refresh_env_variables()
        pytest.skip("AITER's FlyDSL mHC kernels need gfx950 and a newer AITER")
    yield ops
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def test_flydsl_mhc_ops_needs_flag_and_shape(flydsl_ops, monkeypatch):
    """Only the flag and this model's shape turn the kernels on."""
    assert mhc.get_flydsl_mhc_ops(HIDDEN, HC) is flydsl_ops
    assert mhc.get_flydsl_mhc_ops(7168, HC) is None
    assert mhc.get_flydsl_mhc_ops(HIDDEN, 8) is None
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FLYDSL_MHC", "0")
    rocm_aiter_ops.refresh_env_variables()
    assert mhc.get_flydsl_mhc_ops(HIDDEN, HC) is None


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(torch.bfloat16)


def _assert_mxfp8_of(layer_input: torch.Tensor):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.deepseek_v4.amd.model import mxfp8_input

    qa = mxfp8_input(layer_input)
    assert qa is not None
    ref_values, ref_scales = mxfp8_e4m3_quantize(layer_input)
    assert torch.equal(qa.data.view(torch.uint8), ref_values.view(torch.uint8))
    assert torch.equal(qa.scale.view(torch.uint8), ref_scales.view(torch.uint8))


@pytest.mark.parametrize(
    ("num_tokens", "aux", "mxfp8", "entry"),
    [
        (160, False, False, "post"),
        (256, False, True, "post"),
        (48, True, True, "post"),
        (1024, True, False, "post"),
        # An Engram layer's or a pipeline entry's attention seam.
        (192, False, True, "no_post"),
        # The first DSpark layer's attention seam (identity pre-mix).
        (192, False, False, "identity_pre"),
    ],
)
def test_seam_matches_reference(flydsl_ops, num_tokens, aux, mxfp8, entry):
    """The FlyDSL seam against the reference of the post-mix + delayed pre +
    norm it replaces; its MXFP8 copy is the quantized bf16 layer input and the
    aux mean is the mean of its own new residual."""
    set_random_seed(0)
    args, kwargs = _seam_inputs(num_tokens, "cuda", post=entry == "post")
    if entry == "identity_pre":
        kwargs["pre_mix"] = None
    residual, fn = args[:2]
    aux_out: list[torch.Tensor] | None = [] if aux else None
    out = mhc.mhc_pre_delayed_flydsl(
        lambda *a, **k: pytest.fail("fell back to the default seam"),
        *args,
        **kwargs,
        ops=flydsl_ops,
        fn_flydsl=flydsl_ops.flydsl_mhc_pack_fn(fn),
        mxfp8=mxfp8,
        aux_out=aux_out,
    )
    new_residual, post_mix, comb_mix, layer_input, next_pre = out

    ref_residual = (
        mhc_post_torch(
            kwargs["sublayer_out"],
            residual,
            kwargs["post_layer_mix"],
            kwargs["comb_res_mix"],
        )
        if entry == "post"
        else residual
    )
    ref_post, ref_comb, ref_input, ref_pre = mhc_pre_delayed_torch(
        ref_residual, fn, args[2], args[3], *GATES, pre_mix=kwargs["pre_mix"]
    )
    ref_input = _rms_norm(ref_input, kwargs["norm_weight"], kwargs["norm_eps"])
    torch.testing.assert_close(new_residual, ref_residual, atol=1e-2, rtol=1e-2)
    for got, ref in ((post_mix, ref_post), (comb_mix, ref_comb), (next_pre, ref_pre)):
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(layer_input, ref_input, atol=3e-2, rtol=2e-2)
    if mxfp8:
        _assert_mxfp8_of(layer_input)
    if aux_out is not None:
        (mean,) = aux_out
        ref_mean = new_residual.float().mean(dim=1).to(torch.bfloat16)
        torch.testing.assert_close(mean, ref_mean, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", [1, 192, 1536])
@pytest.mark.parametrize("post", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_tail_matches_reference(flydsl_ops, num_tokens, post, keep):
    """The tail against `mhc_post` + `hc_collapse` + final RMSNorm, and the
    residual it hands to the MTP buffer."""
    set_random_seed(0)
    args, kwargs = _seam_inputs(num_tokens, "cuda", post=post)
    residual, pre_mix = args[0], kwargs["pre_mix"]
    norm_weight, eps = kwargs["norm_weight"], kwargs["norm_eps"]
    sublayer_out = kwargs.get("sublayer_out")
    post_mix, comb_mix = kwargs.get("post_layer_mix"), kwargs.get("comb_res_mix")
    mtp = torch.full((num_tokens + 3, HC * HIDDEN), 7.0, device="cuda").to(
        torch.bfloat16
    )
    residual_out = mtp[:num_tokens].view(residual.shape) if keep else None
    out = mhc.mhc_tail_flydsl(
        flydsl_ops,
        residual,
        pre_mix,
        norm_weight,
        eps,
        sublayer_out,
        post_mix,
        comb_mix,
        residual_out=residual_out,
    )

    ref_residual = (
        mhc_post_torch(sublayer_out, residual, post_mix, comb_mix) if post else residual
    )
    collapsed = (pre_mix.unsqueeze(-1) * ref_residual.float()).sum(dim=1)
    ref = _rms_norm(collapsed.to(torch.bfloat16), norm_weight, eps)
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=2e-2)
    if keep:
        torch.testing.assert_close(
            mtp[:num_tokens].view(residual.shape), ref_residual, atol=1e-2, rtol=1e-2
        )
    assert mtp[num_tokens:].eq(7.0).all()


def test_seam_cuda_graph_replay(flydsl_ops):
    """One eager seam on the capture stream is enough to capture it; a replay
    on new inputs matches an eager call."""
    set_random_seed(0)
    num_tokens = 192
    args, kwargs = _seam_inputs(num_tokens, "cuda")
    fn_flydsl = flydsl_ops.flydsl_mhc_pack_fn(args[1])

    def seam():
        return mhc.mhc_pre_delayed_flydsl(
            lambda *a, **k: pytest.fail("fell back to the default seam"),
            *args,
            **kwargs,
            ops=flydsl_ops,
            fn_flydsl=fn_flydsl,
            mxfp8=False,
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        seam()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graphed = seam()
    for t in (args[0], kwargs["sublayer_out"]):
        t.copy_(torch.randn_like(t, dtype=torch.float32).to(t.dtype))
    graph.replay()
    eager = seam()
    torch.accelerator.synchronize()
    for got, ref in zip(graphed, eager):
        assert torch.equal(got, ref)


class _FakeEngram(nn.Module):
    """An Engram injection that only shifts the residual."""

    layer_hash_index = 0

    def forward(self, residual, hashes, mask):
        return residual + 0.125


class _Attention(nn.Module):
    """Attention that scales its input. Its first GEMM does not read the
    seam's MXFP8 layer input."""

    def __init__(self):
        super().__init__()
        self.fused_wqa_wkv = nn.Identity()

    def forward(self, positions, x, _):
        return x * 0.5


class _FFN(nn.Module):
    """An FFN that scales its input, without MegaMoE or a shared expert."""

    use_mega_moe = False
    shared_experts = None

    def forward(self, x, input_ids):
        return x * 0.25


def _decoder_layer(seed: int, engram: bool = False):
    """A real `DeepseekV4DecoderLayer` on the Mega-MHC path with plain
    attention and FFN callables, loaded with seeded weights. It takes the
    FlyDSL seams when the flag is on when it is built."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.mhc import MHCPostOp, MHCPreDelayedOp
    from vllm.models.deepseek_v41.amd.model import DeepseekV4DecoderLayer

    layer = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
    nn.Module.__init__(layer)
    layer.mono = None
    layer.engram = _FakeEngram() if engram else None
    layer.mhc_overlap = None
    layer.mhc_seam = None
    layer.use_sequence_parallel = False
    layer.fuse_seam_norm = layer.use_mega_mhc = True
    layer.hidden_size, layer.hc_mult = HIDDEN, HC
    layer.hc_sinkhorn_iters, layer.hc_eps, layer.hc_post_alpha = 20, 1e-6, 2.0
    layer.rms_norm_eps = 1e-6
    layer.mhc_pre_delayed = MHCPreDelayedOp()
    layer.mhc_post = MHCPostOp()
    layer.flydsl_mhc_ops = mhc.get_flydsl_mhc_ops(HIDDEN, HC)
    layer.hc_attn_fn_packed = layer.hc_ffn_fn_packed = None
    layer.hc_attn_fn_flydsl = layer.hc_ffn_fn_flydsl = None

    layer.attn, layer.ffn = _Attention(), _FFN()
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.device("cuda"):
        for name in ("attn", "ffn"):
            fn = torch.randn(MIX, HC * HIDDEN, generator=g, device="cuda") * 0.01
            setattr(layer, f"hc_{name}_fn", nn.Parameter(fn, requires_grad=False))
            setattr(
                layer, f"hc_{name}_scale", torch.rand(3, generator=g, device="cuda")
            )
            setattr(
                layer,
                f"hc_{name}_base",
                torch.randn(MIX, generator=g, device="cuda") * 0.1,
            )
            norm = RMSNorm(HIDDEN, 1e-6).to(torch.bfloat16)
            norm.weight.data.copy_(torch.rand(HIDDEN, generator=g, device="cuda") + 0.5)
            setattr(layer, f"{name}_norm", norm)
    layer.process_weights_after_loading()
    return layer


def _record_seams(monkeypatch, ops, layers) -> list[tuple]:
    """Record each seam `layers` run: (layer index, sublayer, kernel,
    aux_mean, out_dtype, has_post), passing the calls through. Seams of other
    layers are not recorded."""
    seams: list[tuple] = []

    def which(fn_packed, attn, ffn):
        for i, layer in enumerate(layers):
            if fn_packed is getattr(layer, attn):
                return i, "attn"
            if fn_packed is getattr(layer, ffn):
                return i, "ffn"
        return None

    flydsl = ops.flydsl_mhc_post_pre_rmsnorm

    # Both kernels take (residual, fn_packed, hc_scale, hc_base, 4 gate
    # constants, sinkhorn_repeat, pre_mix, sublayer_out, ...).
    def flydsl_seam(residual, fn_packed, *args, **kwargs):
        seam = which(fn_packed, "hc_attn_fn_flydsl", "hc_ffn_fn_flydsl")
        if seam is not None:
            seams.append(
                (
                    *seam,
                    "flydsl",
                    kwargs["aux_mean"],
                    kwargs["out_dtype"],
                    args[8] is not None,
                )
            )
        return flydsl(residual, fn_packed, *args, **kwargs)

    monkeypatch.setattr(ops, "flydsl_mhc_post_pre_rmsnorm", flydsl_seam)

    def mega(name: str, out_dtype: str):
        kernel = getattr(rocm_aiter_ops, name)

        def mega_seam(residual, fn_packed, *args):
            seam = which(fn_packed, "hc_attn_fn_packed", "hc_ffn_fn_packed")
            if seam is not None:
                seams.append((*seam, "mega", False, out_dtype, args[8] is not None))
            return kernel(residual, fn_packed, *args)

        monkeypatch.setattr(rocm_aiter_ops, name, mega_seam)

    mega("mhc_mega_post_pre_delayed_rms_norm", "bf16")
    mega("mhc_mega_post_pre_delayed_rms_norm_mxfp8", "mxfp8")
    return seams


def _flag_off(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FLYDSL_MHC", "0")
    rocm_aiter_ops.refresh_env_variables()


def _layer_inputs(num_tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    return dict(
        x=randn(num_tokens, HIDDEN).to(torch.bfloat16),
        positions=torch.arange(num_tokens, device="cuda"),
        input_ids=None,
        pre_mix=torch.softmax(randn(num_tokens, HC), -1),
        post_mix=torch.rand(num_tokens, HC, 1, generator=g, device="cuda"),
        res_mix=torch.rand(num_tokens, HC, HC, generator=g, device="cuda") / HC,
        residual=randn(num_tokens, HC, HIDDEN).to(torch.bfloat16),
    )


def _assert_layer_outputs_close(got, ref):
    x, residual, post_mix, res_mix, pre = got
    ref_x, ref_residual, ref_post, ref_res, ref_pre = ref
    torch.testing.assert_close(residual, ref_residual, atol=1e-2, rtol=1e-2)
    for a, b in ((post_mix, ref_post), (res_mix, ref_res), (pre, ref_pre)):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(x, ref_x, atol=1e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [128, 192, 288])
@pytest.mark.parametrize("aux", [False, True])
@pytest.mark.parametrize("mxfp8", [False, True], ids=["bf16", "attn_mxfp8"])
def test_decoder_layer_seams(
    flydsl_ops, monkeypatch, default_vllm_config, num_tokens, aux, mxfp8
):
    """`DeepseekV4DecoderLayer.forward` takes the FlyDSL seam with the
    attention pack for its attention seam and the FFN pack for its FFN seam
    where AITER's rule takes them (129-256 rows), and Mega-MHC elsewhere; with
    `aux_out`, the attention seam alone takes the aux-mean variant at any of
    these sizes. Each seam keeps its layer input format. The outputs match the
    same layer with the flag off, and the aux mean is the stock `mhc_post` +
    mean."""
    on = _decoder_layer(seed=1)
    on.attn_seam_mxfp8 = mxfp8
    assert on.takes_aux_mean(num_tokens)
    seams = _record_seams(monkeypatch, flydsl_ops, [on])
    inputs = _layer_inputs(num_tokens, seed=2)
    aux_out: list[torch.Tensor] | None = [] if aux else None
    got = on(**inputs, aux_out=aux_out)
    torch.accelerator.synchronize()

    in_window = 129 <= num_tokens <= 256
    attn_dtype = "mxfp8" if mxfp8 else "bf16"
    assert seams == [
        (0, "attn", "flydsl" if aux or in_window else "mega", aux, attn_dtype, True),
        (0, "ffn", "flydsl" if in_window else "mega", False, "bf16", True),
    ]

    _flag_off(monkeypatch)
    off = _decoder_layer(seed=1)
    off.attn_seam_mxfp8 = mxfp8
    assert off.hc_attn_fn_flydsl is None and not off.takes_aux_mean(num_tokens)
    ref = off(**inputs)
    _assert_layer_outputs_close(got, ref)
    if aux_out is not None:
        ref_mean = off.mhc_post(
            inputs["x"], inputs["residual"], inputs["post_mix"], inputs["res_mix"]
        ).mean(dim=1)
        (mean,) = aux_out
        torch.testing.assert_close(mean, ref_mean, atol=1e-2, rtol=1e-2)


def test_decoder_layer_engram_seam(flydsl_ops, monkeypatch, default_vllm_config):
    """An Engram layer's attention seam runs after the injection, without a
    post-mix, on the FlyDSL seam; the layer never takes the aux mean."""
    num_tokens = 192
    on = _decoder_layer(seed=1, engram=True)
    assert not on.takes_aux_mean(num_tokens)
    seams = _record_seams(monkeypatch, flydsl_ops, [on])
    inputs = _layer_inputs(num_tokens, seed=2)
    hashes = torch.zeros(num_tokens, 1, 1, dtype=torch.int32, device="cuda")
    got = on(**inputs, engram_hashes=hashes)
    torch.accelerator.synchronize()
    assert seams == [
        (0, "attn", "flydsl", False, "bf16", False),
        (0, "ffn", "flydsl", False, "bf16", True),
    ]
    _flag_off(monkeypatch)
    ref = _decoder_layer(seed=1, engram=True)(**inputs, engram_hashes=hashes)
    _assert_layer_outputs_close(got, ref)


def test_takes_aux_mean_yields_to_gate_overlap(
    flydsl_ops, monkeypatch, default_vllm_config
):
    """Inside a FULL graph capture of at most 128 rows the layer runs the mHC
    gate overlap, so it does not take the aux mean there; it does above."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context

    layer = _decoder_layer(seed=1)
    layer.mhc_overlap = mhc.MHCOverlapStream()
    stream = torch.cuda.Stream()
    with (
        set_forward_context(
            None, default_vllm_config, cudagraph_runtime_mode=CUDAGraphMode.FULL
        ),
        torch.cuda.stream(stream),
    ):
        scratch = torch.zeros(1, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            scratch.add_(1)
            eligible = {t: layer.takes_aux_mean(t) for t in (128, 129, 192)}
    assert eligible == {128: False, 129: True, 192: True}
    assert layer.takes_aux_mean(128)


def _model(monkeypatch, layers, num_tokens):
    """`DeepseekV4Model` on the last pipeline rank over `layers`, with an MTP
    hidden buffer."""
    import vllm.models.deepseek_v41.amd.model as amd
    from vllm.config import VllmConfig, set_current_vllm_config

    monkeypatch.setattr(
        amd,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    monkeypatch.setattr(amd.EngramLayout, "from_config", lambda config: None)
    monkeypatch.setattr(
        amd, "make_layers", lambda n, factory, prefix: (0, n, list(layers))
    )
    config = SimpleNamespace(
        vocab_size=32, hc_eps=1e-6, hc_mult=HC, hidden_size=HIDDEN, rms_norm_eps=1e-6,
        index_topk=4, kv_source_layer_ids=[1], num_hidden_layers=len(layers),
        sliding_window=128, engram_layer_ids=(), candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
    )  # fmt: skip
    # An EAGLE drafter reads the pre-collapse residual (the MTP hidden buffer).
    eagle = SimpleNamespace(
        use_eagle=lambda: True,
        uses_draft_model=lambda: False,
        use_dspark=lambda: False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, dtype=torch.bfloat16),
        quant_config=None,
        kernel_config=SimpleNamespace(moe_backend="auto"),
        cache_config=SimpleNamespace(swa_bounded_replay=False),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, use_ubatching=False,
            enable_expert_parallel=False,
        ),
        speculative_config=eagle,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=num_tokens + 5),
    )  # fmt: skip
    # The model runner builds the model on its device.
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        model = amd.DeepseekV4Model(vllm_config=vllm_config)
    model.norm.to(torch.bfloat16).weight.data.copy_(
        torch.linspace(0.5, 1.5, HIDDEN, device="cuda")
    )
    model.process_weights_after_loading()
    return model


@pytest.mark.parametrize("num_tokens", [192, 288])
@pytest.mark.parametrize("aux_layers", [(2, 3), (2, 4)])
def test_model_aux_in_seam_and_tail(
    flydsl_ops, monkeypatch, default_vllm_config, num_tokens, aux_layers
):
    """`DeepseekV4Model.forward` over real decoder layers: each auxiliary
    hidden state taken before a layer is left to that layer's attention seam
    (aux-mean variant), the one after the last layer (aux id 4) keeps the stock
    `mhc_post` + mean, and the step ends with the FlyDSL tail. The aux list,
    the normed output and the MTP hidden buffer match the same model with the
    flag off."""
    from vllm.sequence import IntermediateTensors

    num_layers = 4
    g = torch.Generator(device="cuda").manual_seed(3)
    hidden = torch.randn(num_tokens, HC, HIDDEN, generator=g, device="cuda").to(
        torch.bfloat16
    )
    pre_mix = torch.softmax(torch.randn(num_tokens, HC, generator=g, device="cuda"), -1)
    positions = torch.arange(num_tokens, device="cuda")

    def run(model):
        model.aux_hidden_state_layers = aux_layers
        out, aux = model(
            input_ids=positions.to(torch.int32),
            positions=positions,
            intermediate_tensors=IntermediateTensors(
                {"hidden_states": hidden, "pre_mix": pre_mix}
            ),
        )
        torch.accelerator.synchronize()
        mtp = model._mtp_hidden_buffer[:num_tokens].clone()
        return out, aux, mtp

    on_layers = [_decoder_layer(seed=10 + i) for i in range(num_layers)]
    seams = _record_seams(monkeypatch, flydsl_ops, on_layers)
    on = _model(monkeypatch, on_layers, num_tokens)
    assert on.flydsl_mhc_tail_ops is flydsl_ops
    out, aux, mtp = run(on)

    # Aux id 2 is the state before layer 2, taken by layer 2's attention seam.
    deferred = [i for i in aux_layers if i < num_layers]
    assert [s[0] for s in seams if s[3]] == deferred
    assert all(s[1] == "attn" for s in seams if s[3])

    _flag_off(monkeypatch)
    off = _model(
        monkeypatch,
        [_decoder_layer(seed=10 + i) for i in range(num_layers)],
        num_tokens,
    )
    assert off.flydsl_mhc_tail_ops is None
    ref_out, ref_aux, ref_mtp = run(off)
    # The residual streams pass up to eight bf16 post-mixes, which the two
    # paths round in a different order: a few elements differ by a few ulp.
    assert len(aux) == len(ref_aux) == len(aux_layers)
    for got, ref in zip(aux, ref_aux):
        torch.testing.assert_close(got, ref, atol=1e-2, rtol=2e-2)
    torch.testing.assert_close(out, ref_out, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(mtp, ref_mtp, atol=1e-2, rtol=2e-2)
