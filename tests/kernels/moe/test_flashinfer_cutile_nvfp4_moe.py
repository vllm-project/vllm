# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-forward W4A4/W4A16 selection for NVFP4 routed experts on SM12x.

Both precisions must read the same prepared weights, select from the token
count, and stay correct across the cutoff inside captured CUDA graphs.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip("requires SM120/SM121", allow_module_level=True)

from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutile_moe import (
    CuTileNvfp4DynamicMoE,
    has_flashinfer_cutile_nvfp4,
    prepare_cutile_nvfp4_weights,
)

if not has_flashinfer_cutile_nvfp4("cutile"):
    pytest.skip("FlashInfer lacks cuTile NVFP4 W4A4/W4A16", allow_module_level=True)

# (W4A4, W4A16) backend pairs
BACKENDS = [
    pytest.param(
        pair,
        id="-".join(pair),
        marks=pytest.mark.skipif(
            not has_flashinfer_cutile_nvfp4(pair[1], pair[0]),
            reason=f"FlashInfer lacks the {pair} backends",
        ),
    )
    for pair in (("cutile", "cutile"), ("cutile", "sm12x"), ("sm12x", "sm12x"))
]

E, TOPK, H, INTER = 8, 2, 256, 128
CUTOFF = 4
_E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1 = torch.tensor(_E2M1_POS + [-v for v in _E2M1_POS])


def _dequant(w, scale, gscale):
    codes = torch.stack((w & 0xF, w >> 4), dim=-1).flatten(-2).long()
    vals = _E2M1.to(w.device)[codes]
    return vals * scale.float().repeat_interleave(16, dim=-1) * gscale.view(-1, 1, 1)


def _weights(activation, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = 2 * INTER if activation == MoEActivation.SILU else INTER

    def codes(*shape):
        return torch.randint(0, 256, shape, device="cuda", generator=g).to(torch.uint8)

    def scales(*shape):
        s = torch.rand(shape, device="cuda", generator=g) * 1.5 + 0.25
        return s.to(torch.float8_e4m3fn)

    w13, w2 = codes(E, rows, H // 2), codes(E, H, INTER // 2)
    s13, s2 = scales(E, rows, H // 16), scales(E, H, INTER // 16)
    g13 = torch.rand(E, device="cuda", generator=g) * 0.004 + 0.004
    g2 = torch.rand(E, device="cuda", generator=g) * 0.02 + 0.02
    return w13, s13, g13, w2, s2, g2


def _reference(x, ids, weights, w13, s13, g13, w2, s2, g2, activation):
    dw13, dw2 = _dequant(w13, s13, g13), _dequant(w2, s2, g2)
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for k in range(TOPK):
            e = int(ids[t, k])
            h = x[t].float() @ dw13[e].T
            if activation == MoEActivation.SILU:
                a = torch.nn.functional.silu(h[:INTER]) * h[INTER:]
            else:
                a = torch.relu(h).square()
            out[t] += weights[t, k] * (a.bfloat16().float() @ dw2[e].T)
    return out


def _dispatch(
    activation, a16_max_tokens=CUTOFF, backends=("cutile", "cutile"), max_num_tokens=256
):
    raw = _weights(activation)
    dispatch = CuTileNvfp4DynamicMoE(
        num_experts=E,
        top_k=TOPK,
        intermediate_size=INTER,
        activation=activation,
        max_num_tokens=max_num_tokens,
        a16_max_tokens=a16_max_tokens,
        w4a16_backend=backends[1],
        device=torch.device("cuda"),
        w4a4_backend=backends[0],
    )
    view = prepare_cutile_nvfp4_weights(*raw, activation)
    dispatch.set_weights(view)
    return dispatch, raw, view


def _routing(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, H, device="cuda", generator=g).bfloat16()
    logits = torch.randn(m, E, device="cuda", generator=g)
    vals, ids = torch.topk(logits, TOPK, dim=-1)
    return x, ids.to(torch.int32), torch.softmax(vals, dim=-1)


@pytest.mark.parametrize("backends", BACKENDS)
def test_selects_by_token_count_over_one_weight_copy(backends):
    from flashinfer.fused_moe import MoEActivationPack

    dispatch, _, view = _dispatch(MoEActivation.SILU, backends=backends)
    assert dispatch.select(1) is dispatch.w4a16
    assert dispatch.select(CUTOFF) is dispatch.w4a16
    assert dispatch.select(CUTOFF + 1) is dispatch.w4a4
    x, ids, w = _routing(CUTOFF, 0)
    act = MoEActivationPack(
        hidden_states_q=x, hidden_states_scale=None, topk_ids=ids, topk_weights=w
    )
    keys = ("w1", "w1_scale", "w1_global_scale", "w2", "w2_scale", "w2_global_scale")
    expected = [view[k].data_ptr() for k in keys]
    for path in (dispatch.w4a4, dispatch.w4a16):
        # Packed inputs are [out, x, ids, weights, *weight view, *extras].
        inputs = path.runner.pack_inputs(act, dispatch._weights)
        assert [t.data_ptr() for t in inputs[4:10]] == expected


@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
@pytest.mark.parametrize("m", [1, 3, CUTOFF, CUTOFF + 1, 64])
@pytest.mark.parametrize("backends", BACKENDS)
@torch.inference_mode()
def test_matches_reference_on_both_sides_of_cutoff(activation, m, backends):
    dispatch, raw, _ = _dispatch(activation, backends=backends)
    x, ids, w = _routing(m, 1)
    out = torch.empty_like(x)
    dispatch.run(out, x, ids, w)
    w13, s13, g13, w2, s2, g2 = raw
    ref = _reference(x, ids, w, w13, s13, g13, w2, s2, g2, activation)
    # W4A4 also quantizes activations to FP4, so its bound is looser.
    tol = 3e-2 if m <= CUTOFF else 2.5e-1
    err = (out.float() - ref).abs() / (ref.abs().mean() + 1e-6)
    assert err.mean() < tol, f"m={m} mean rel err {err.mean():.3e}"


@pytest.mark.parametrize("m", [1, CUTOFF, 64])
@pytest.mark.parametrize("backends", BACKENDS)
@torch.inference_mode()
def test_padding_rows_routed_to_invalid_expert(m, backends):
    """CUDA-graph padding rows carry expert id -1 (VLLM_MOE_SKIP_PADDING); they
    must not perturb the real rows. With m=1 every row is padding, as in the
    cudagraph memory-profiling forward."""
    dispatch, _, _ = _dispatch(MoEActivation.SILU, backends=backends)
    x, ids, w = _routing(m, 4)
    valid = m // 2
    ids[valid:] = -1
    out = torch.empty_like(x)
    dispatch.run(out, x, ids, w)
    assert torch.isfinite(out).all()
    if valid:
        clean = torch.empty_like(x[:valid])
        dispatch.run(clean, x[:valid], ids[:valid], w[:valid])
        err = (out[:valid].float() - clean.float()).abs().max()
        assert err <= 1e-2 * clean.float().abs().mean(), f"m={m} max err {err:.3e}"


@pytest.mark.parametrize("backends", BACKENDS)
@torch.inference_mode()
def test_cuda_graph_replay_across_cutoff(backends):
    dispatch, _, _ = _dispatch(MoEActivation.SILU, backends=backends)
    for m in (3, CUTOFF, CUTOFF + 1):
        x, ids, w = _routing(m, 2)
        out = torch.empty_like(x)
        dispatch.run(out, x, ids, w)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dispatch.run(out, x, ids, w)
        new_x, new_ids, new_w = _routing(m, 3)
        for live, new in ((x, new_x), (ids, new_ids), (w, new_w)):
            live.copy_(new)
        expected = torch.empty_like(out)
        dispatch.run(expected, x, ids, w)
        out.zero_()
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(out, expected), f"m={m}"


def test_zero_cutoff_never_builds_w4a16():
    dispatch, _, _ = _dispatch(MoEActivation.SILU, a16_max_tokens=0)
    assert dispatch.w4a16 is None
    assert dispatch.select(1) is dispatch.w4a4


@pytest.mark.parametrize("backends", [p for p in BACKENDS if "sm12x" in p.values[0]])
@torch.inference_mode()
def test_tuning_compiles_every_sm12x_token_bucket(backends, monkeypatch):
    """SM12x kernels compile per token bucket; tuning must cover every bucket
    so that serving never compiles on the first use of an unseen one."""
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

    from vllm.model_executor.layers.fused_moe.experts import flashinfer_cutile_moe

    max_num_tokens = 32
    monkeypatch.setattr(flashinfer_cutile_moe, "_WARMED_BUCKETS", set())
    dispatch, _, _ = _dispatch(
        MoEActivation.SILU, backends=backends, max_num_tokens=max_num_tokens
    )
    seen: dict[str, list[int]] = {}
    for name in ("w4a4", "w4a16"):
        runner = getattr(dispatch, name).runner
        if not runner.backend_key.startswith("sm12x"):
            continue
        calls = seen.setdefault(name, [])

        def record(inputs, *args, _forward=runner.forward, _calls=calls, **kwargs):
            # Packed inputs are [out, x, ids, weights, ...].
            _calls.append(inputs[1].shape[0])
            return _forward(inputs, *args, **kwargs)

        monkeypatch.setattr(runner, "forward", record)

    x, ids, w = _routing(1, 0)
    dispatch.run(torch.empty_like(x), x, ids, w)
    assert all(calls in ([], [1]) for calls in seen.values())

    with autotune(True):
        dispatch.run(torch.empty_like(x), x, ids, w)
    limits = {"w4a4": max_num_tokens, "w4a16": CUTOFF}
    for name, calls in seen.items():
        assert set(get_hybrid_num_tokens_buckets(limits[name])) <= set(calls), name


@torch.inference_mode()
def test_sm12x_w4a4_layers_share_bucket_scratch():
    """SM12x W4A4 rewrites its scratch on every call, so layers of one geometry
    share it per token bucket instead of holding one copy per layer."""
    backends = ("sm12x", "sm12x")
    if not has_flashinfer_cutile_nvfp4(backends[1], backends[0]):
        pytest.skip("FlashInfer lacks the SM12x backends")
    first, (w13, s13, g13, w2, s2, g2), _ = _dispatch(
        MoEActivation.SILU, backends=backends
    )
    second, _, _ = _dispatch(MoEActivation.SILU, backends=backends)
    other, _, _ = _dispatch(MoEActivation.RELU2_NO_MUL, backends=backends)
    assert first.w4a4.runner._workspaces is second.w4a4.runner._workspaces
    assert first.w4a4.runner._workspaces is not other.w4a4.runner._workspaces
    x, ids, w = _routing(CUTOFF + 3, 3)
    ref = _reference(x, ids, w, w13, s13, g13, w2, s2, g2, MoEActivation.SILU)
    outs = [d.run(torch.empty_like(x), x, ids, w) for d in (first, second, first)]
    err = (outs[0].float() - ref).abs() / (ref.abs().mean() + 1e-6)
    assert err.mean() < 2.5e-1, f"mean rel err {err.mean():.3e}"
    assert torch.equal(outs[0], outs[2])
