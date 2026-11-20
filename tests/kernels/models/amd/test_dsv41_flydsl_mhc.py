# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's mHC seams and step-end tail on AITER's FlyDSL kernels
(VLLM_ROCM_USE_AITER_FLYDSL_MHC).

The routing tests run anywhere, without the kernels: which seams go to the
FlyDSL kernel and which keep the default seam. The numerics tests need the
kernels (gfx950 and an AITER that ships `flydsl_mhc_post_pre_rmsnorm`) and
check the call sites against the torch reference of the path they replace.
"""

import pytest
import torch

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


def test_flydsl_mhc_needs_flag_and_shape(monkeypatch):
    """Without the flag, or for another hidden size, nothing changes."""
    monkeypatch.delenv("VLLM_ROCM_USE_AITER_FLYDSL_MHC", raising=False)
    rocm_aiter_ops.refresh_env_variables()
    assert mhc.flydsl_mhc(HIDDEN, HC) is None
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FLYDSL_MHC", "1")
    rocm_aiter_ops.refresh_env_variables()
    try:
        assert mhc.flydsl_mhc(7168, HC) is None
        assert mhc.flydsl_mhc(HIDDEN, 8) is None
    finally:
        monkeypatch.delenv("VLLM_ROCM_USE_AITER_FLYDSL_MHC")
        rocm_aiter_ops.refresh_env_variables()


@pytest.fixture
def flydsl_ops(monkeypatch):
    """The real kernels, or skip."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FLYDSL_MHC", "1")
    rocm_aiter_ops.refresh_env_variables()
    ops = mhc.flydsl_mhc(HIDDEN, HC)
    if ops is None:
        rocm_aiter_ops.refresh_env_variables()
        pytest.skip("AITER's FlyDSL mHC kernels need gfx950 and a newer AITER")
    yield ops
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


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
    ("num_tokens", "aux", "mxfp8"),
    [(160, False, False), (256, False, True), (48, True, True), (1024, True, False)],
)
def test_seam_matches_reference(flydsl_ops, num_tokens, aux, mxfp8):
    """The FlyDSL seam against the reference of the post-mix + delayed pre +
    norm it replaces; its MXFP8 copy is the quantized bf16 layer input and the
    aux mean is the mean of its own new residual."""
    set_random_seed(0)
    args, kwargs = _seam_inputs(num_tokens, "cuda")
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

    ref_residual = mhc_post_torch(
        kwargs["sublayer_out"],
        residual,
        kwargs["post_layer_mix"],
        kwargs["comb_res_mix"],
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


class _CountingPost(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *args):
        self.calls += 1
        return mhc_post_torch(*args)


class _StandInLayer(torch.nn.Module):
    """A decoder layer with fixed outputs whose attention seam applies the
    carried post-mix (and, with `aux_out`, appends the stream mean), as
    `DeepseekV4DecoderLayer` does."""

    def __init__(self, num_tokens: int, aux_in_seam: bool):
        super().__init__()
        self.aux_in_seam = aux_in_seam
        self.got_aux_out: list[bool] = []
        self.out = (
            torch.randn(num_tokens, HIDDEN, device="cuda").to(torch.bfloat16),
            torch.rand(num_tokens, HC, 1, device="cuda"),
            torch.rand(num_tokens, HC, HC, device="cuda") / HC,
            torch.softmax(torch.randn(num_tokens, HC, device="cuda"), -1),
        )

    def takes_aux_mean(self, num_tokens: int) -> bool:
        return self.aux_in_seam

    def forward(self, x, positions, input_ids, pre_mix, post_mix, res_mix, residual,
                engram_hashes, engram_mask, aux_out=None):  # fmt: skip
        residual = (
            x if residual is None else mhc_post_torch(x, residual, post_mix, res_mix)
        )
        self.got_aux_out.append(aux_out is not None)
        if aux_out is not None:
            aux_out.append(residual.float().mean(dim=1).to(torch.bfloat16))
        x, post_mix, res_mix, pre_mix = self.out
        return x, residual, post_mix, res_mix, pre_mix


@pytest.mark.parametrize("aux_layers", [(2, 3), (2, 4)])
def test_model_aux_in_seam_and_tail(flydsl_ops, monkeypatch, aux_layers):
    """`DeepseekV4Model.forward` leaves an auxiliary hidden state to the next
    layer's seam where that layer takes it, keeps the order of the aux list,
    and ends the step with the FlyDSL tail: the last post-mix (or the final
    aux residual), collapse and final norm, with the residual in the MTP
    buffer."""
    from types import SimpleNamespace

    import vllm.models.deepseek_v41.amd.model as amd
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.sequence import IntermediateTensors

    set_random_seed(0)
    num_tokens, num_layers = 192, 4
    layers = [_StandInLayer(num_tokens, aux_in_seam=i == 2) for i in range(num_layers)]
    monkeypatch.setattr(
        amd,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    monkeypatch.setattr(amd.EngramLayout, "from_config", lambda config: None)
    monkeypatch.setattr(amd, "make_layers", lambda n, factory, prefix: (0, n, layers))
    config = SimpleNamespace(
        vocab_size=32, hc_eps=1e-6, hc_mult=HC, hidden_size=HIDDEN, rms_norm_eps=1e-6,
        index_topk=4, kv_source_layer_ids=[1], num_hidden_layers=num_layers,
        sliding_window=128, engram_layer_ids=(), candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
    )  # fmt: skip
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, dtype=torch.bfloat16),
        quant_config=None,
        kernel_config=SimpleNamespace(moe_backend="auto"),
        cache_config=SimpleNamespace(swa_bounded_replay=False),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, use_ubatching=False,
            enable_expert_parallel=False,
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=num_tokens),
    )  # fmt: skip
    with set_current_vllm_config(VllmConfig()):
        model = amd.DeepseekV4Model(vllm_config=vllm_config).cuda()
    assert model.flydsl_mhc_tail is flydsl_ops
    model.aux_hidden_state_layers = aux_layers
    model.norm.to(torch.bfloat16).weight.data.copy_(torch.rand(HIDDEN) + 0.5)
    mtp = torch.zeros(num_tokens + 5, HC * HIDDEN, device="cuda").to(torch.bfloat16)
    model._mtp_hidden_buffer = mtp
    model.mhc_post = _CountingPost()

    hidden = torch.randn(num_tokens, HC, HIDDEN, device="cuda").to(torch.bfloat16)
    pre_mix = torch.softmax(torch.randn(num_tokens, HC, device="cuda"), -1)
    positions = torch.arange(num_tokens, device="cuda")
    out, aux = model(
        input_ids=positions.to(torch.int32),
        positions=positions,
        intermediate_tensors=IntermediateTensors(
            {"hidden_states": hidden, "pre_mix": pre_mix}
        ),
    )

    residuals = [hidden]
    for layer in layers:
        residuals.append(mhc_post_torch(layer.out[0], residuals[-1], *layer.out[1:3]))
    final = residuals[-1]
    # Layer 2's seam takes its aux; the other one is the stock post + mean.
    assert [layer.got_aux_out for layer in layers] == [[i == 2] for i in range(4)]
    assert model.mhc_post.calls == 1
    assert len(aux) == 2
    for got, i in zip(aux, aux_layers):
        ref = residuals[i].float().mean(dim=1).to(torch.bfloat16)
        torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)
    collapsed = (layers[-1].out[3].unsqueeze(-1) * final.float()).sum(dim=1)
    ref_out = _rms_norm(collapsed.to(torch.bfloat16), model.norm.weight, 1e-6)
    torch.testing.assert_close(out, ref_out, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(
        mtp[:num_tokens].view(final.shape), final, atol=1e-2, rtol=1e-2
    )
    assert mtp[num_tokens:].eq(0).all()
