# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpine draft coverage: code bridge, serial refinement, loading.

The float64 helpers port tencent-dspine/tests/ref_utils.py to vLLM's
batched [B, S, ...] layout: ``_ref_bridge_write`` replays
``DSpineCodeBridge._forward_eager`` and ``_ref_refine_table`` rebuilds the
[B, T-1, K, K] transition logits of the redone last write (the tencent ref
indexes the flattened residual stream by block row; vLLM's
``score_transitions`` takes the proposal rows already gathered).
"""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import set_current_vllm_config
from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
from vllm.model_executor.models.qwen3_dspine import (
    BRIDGE_PREFIX,
    DSpineCodeBridge,
    DSpineQwen3ForCausalLM,
    build_serial_tables,
    resolve_dspine_head_topk,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspine.speculator import DSpineSpeculator

pytestmark = pytest.mark.skip_global_cleanup


def _ref_bridge_write(
    bridge: DSpineCodeBridge,
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    input_ids: torch.Tensor,
    layer: int,
    next_norm: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """float64 replay of ``DSpineCodeBridge._forward_eager``."""
    t, r, c = bridge.block_size, bridge.rank, bridge.code_size
    raw = hidden.double() if residual is None else hidden.double() + residual.double()
    rms = raw.square().mean(-1).clamp_min(torch.finfo(torch.float64).tiny).sqrt()
    normalized = raw / (rms[:, None] + bridge.eps)
    projected = normalized @ bridge.layer_read_weight(layer).double().T
    codes = (
        math.sqrt(c) * projected / (projected.norm(dim=-1, keepdim=True) + bridge.eps)
    )
    anchors = input_ids.reshape(-1, t)[:, 0]
    anchor_codes = math.sqrt(c) * bridge.codebook.double()[anchors]
    codes = torch.cat((anchor_codes[:, None], codes.view(-1, t, c)[:, 1:]), dim=1).view(
        -1, c
    )
    packed = codes @ bridge.packed_weight.double().T
    q, s, v = packed.view(-1, t, 3 * r).split(r, dim=-1)
    gate = torch.sigmoid(q[:, 1:] + s[:, :-1] + bridge.gate.bias.double())
    message = torch.cat((torch.zeros_like(q[:, :1]), gate * v[:, :-1]), dim=1)
    delta = message.reshape(-1, r) @ bridge.feature_write_weights[layer].double().T
    not_anchor = (torch.arange(raw.shape[0]) % t != 0).double()[:, None]
    out = raw + (rms[:, None] * delta) * not_anchor
    normed = (
        out
        * torch.rsqrt(out.square().mean(-1, keepdim=True) + next_norm.variance_epsilon)
        * next_norm.weight.double()
    )
    return normed, out, raw, rms, packed


def _ref_refine_table(
    candidate_ids: torch.Tensor,
    raw: torch.Tensor,
    rms: torch.Tensor,
    query: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    *,
    codebook: torch.Tensor,
    pred_gate_w: torch.Tensor,
    prev_w: torch.Tensor,
    gate_b: torch.Tensor,
    write_w: torch.Tensor,
    norm_w: torch.Tensor,
    lm_head_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """float64 [B, S, K, K] transition logits of the redone last write.

    Ported from tencent ref_refine_table / serial_rs_table: sender s (the
    anchor on the first proposal row, the previous row's candidate lane
    otherwise) gets e = sqrt(C) * codebook[s] and
    x = raw + rms * write_w @ (sigmoid(query + pred_gate_w @ e + gate_b)
                               * (prev_w @ e)).
    """
    b, s, k = candidate_ids.shape
    senders = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(b, 1, k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    e = math.sqrt(codebook.shape[1]) * codebook.double()[senders]
    gate = torch.sigmoid(
        query.double()[:, :, None, :] + e @ pred_gate_w.double().T + gate_b.double()
    )
    msg = gate * (e @ prev_w.double().T)
    x = raw.double()[:, :, None, :] + rms.double()[:, :, None, None] * (
        msg @ write_w.double().T
    )
    normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    normed = normed * norm_w.double()
    return torch.einsum("bskh,bsch->bskc", normed, lm_head_w.double()[candidate_ids])


def _bridge_hf_config(**dflash_overrides) -> SimpleNamespace:
    """Config namespace for a standalone ``DSpineCodeBridge``."""
    dflash_config = {
        "selected_prefix_terminal": False,
        "selected_prefix_feature_write_mode": "independent",
        "selected_prefix_feature_writes": True,
        "draft_block_causal": True,
        "selected_prefix_feature_layers": [1, 2],
        "selected_prefix_code_bridge": {
            "code_size": 32,
            "rank": 16,
            "eps": 1e-6,
        },
        "dspine_bridge_impl": "eager",
    }
    dflash_config.update(dflash_overrides)
    return SimpleNamespace(
        hidden_size=64,
        num_hidden_layers=2,
        block_size=4,
        vocab_size=256,
        dflash_config=dflash_config,
    )


def _bridge_checkpoint(
    hidden: int = 64,
    layers: int = 2,
    vocab: int = 256,
    code: int = 32,
    rank: int = 16,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Random bridge tensors in the tencent checkpoint layout."""
    g = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    return {
        "codebook": randn(vocab, code),
        "early_read_weights": randn(layers - 1, code, hidden),
        "feature_write_weights": randn(layers, hidden, rank),
        "gate.bias": randn(rank),
        "gate.weight": randn(rank, code),
        "independent_write_version": torch.ones((), dtype=torch.int64),
        "out.weight": torch.zeros(code, rank),
        "predecessor_gate.weight": randn(rank, code),
        "prev.weight": randn(rank, code),
        "read_weight": randn(code, hidden),
        "terminal_disabled_version": torch.ones((), dtype=torch.int64),
    }


def _load_bridge(bridge: DSpineCodeBridge, weights: dict[str, torch.Tensor]) -> None:
    bridge.begin_load()
    for name, tensor in weights.items():
        bridge.load_weight(name, tensor)
    bridge.prepare()


# ---------------------------------------------------------------------------
# resolve_dspine_head_topk
# ---------------------------------------------------------------------------


def test_head_topk_reads_exact_sender_refine() -> None:
    config = SimpleNamespace(
        num_hidden_layers=5,
        dflash_config={
            "selected_prefix_last_write_refine": {
                "kind": "exact_sender",
                "head_topk": 16,
            }
        },
    )
    assert resolve_dspine_head_topk(config) == 16


def test_head_topk_reads_serial_table_second_pass() -> None:
    config = SimpleNamespace(
        num_hidden_layers=5,
        code_bridge_second_pass={
            "mode": "serial_table",
            "first_exact": 4,
            "head_topk": 8,
        },
        dflash_config={},
    )
    assert resolve_dspine_head_topk(config) == 8
    with pytest.raises(ValueError, match="first_exact"):
        resolve_dspine_head_topk(config, 4)


def test_head_topk_fallback_keys() -> None:
    for key in ("dspine_top_k", "transition_top_k"):
        config = SimpleNamespace(num_hidden_layers=5, dflash_config={key: 32})
        assert resolve_dspine_head_topk(config) == 32


def test_head_topk_requires_a_source() -> None:
    config = SimpleNamespace(num_hidden_layers=5, dflash_config={})
    with pytest.raises(ValueError, match="no refinement width"):
        resolve_dspine_head_topk(config)


@pytest.mark.parametrize("head_topk", [1, 3, 12, 128])
def test_head_topk_rejects_out_of_range_or_non_power_of_two(
    head_topk: int,
) -> None:
    config = SimpleNamespace(
        num_hidden_layers=5,
        dflash_config={
            "selected_prefix_last_write_refine": {
                "kind": "exact_sender",
                "head_topk": head_topk,
            }
        },
    )
    with pytest.raises(ValueError, match="power of two"):
        resolve_dspine_head_topk(config)


def test_head_topk_rejects_disagreeing_sources() -> None:
    config = SimpleNamespace(
        num_hidden_layers=5,
        code_bridge_second_pass={
            "mode": "serial_table",
            "first_exact": 4,
            "head_topk": 16,
        },
        dflash_config={
            "selected_prefix_last_write_refine": {
                "kind": "exact_sender",
                "head_topk": 8,
            }
        },
    )
    with pytest.raises(ValueError, match="disagree"):
        resolve_dspine_head_topk(config)


@pytest.mark.parametrize(
    "second_pass",
    [
        {"mode": "dense", "first_exact": 4, "head_topk": 16},
        {"mode": "serial_table", "first_exact": 4, "head_topk": 16, "x": 1},
    ],
)
def test_head_topk_rejects_bad_second_pass(second_pass: dict) -> None:
    config = SimpleNamespace(
        num_hidden_layers=5,
        code_bridge_second_pass=second_pass,
        dflash_config={},
    )
    with pytest.raises(ValueError):
        resolve_dspine_head_topk(config)


def test_head_topk_rejects_wrong_refine_kind() -> None:
    config = SimpleNamespace(
        num_hidden_layers=5,
        dflash_config={
            "selected_prefix_last_write_refine": {
                "kind": "nearest",
                "head_topk": 16,
            }
        },
    )
    with pytest.raises(ValueError, match="exact_sender"):
        resolve_dspine_head_topk(config)


# ---------------------------------------------------------------------------
# DSpineCodeBridge config / weight validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"selected_prefix_terminal": True}, "terminal"),
        ({"selected_prefix_feature_write_mode": "shared"}, "independent"),
        ({"selected_prefix_feature_writes": False}, "feature_writes"),
        ({"draft_block_causal": False}, "causal"),
        ({"selected_prefix_code_bridge": None}, "selected_prefix_code_bridge"),
        ({"selected_prefix_feature_layers": None}, "feature_layers"),
        ({"selected_prefix_feature_layers": [1]}, "last layer"),
        ({"selected_prefix_feature_layers": [1, 2, 3]}, "out of range"),
        ({"selected_prefix_feature_reliability": {"k": 1}}, "Reliability"),
        ({"selected_prefix_feature_next_token_rows": True}, "anchor-row"),
        ({"dspine_bridge_impl": "bogus"}, "Unknown DSPINE bridge"),
    ],
)
def test_bridge_config_validation(overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DSpineCodeBridge(_bridge_hf_config(**overrides))


def test_bridge_prepare_requires_all_tensors() -> None:
    bridge = DSpineCodeBridge(_bridge_hf_config())
    weights = _bridge_checkpoint()
    bridge.begin_load()
    missing = next(iter(weights))
    for name, tensor in weights.items():
        if name != missing:
            bridge.load_weight(name, tensor)
    with pytest.raises(ValueError, match="Missing DSPINE code-bridge"):
        bridge.prepare()


def test_bridge_prepare_rejects_nonzero_terminal_projection() -> None:
    bridge = DSpineCodeBridge(_bridge_hf_config())
    weights = _bridge_checkpoint()
    weights["out.weight"] = torch.randn_like(weights["out.weight"])
    bridge.begin_load()
    for name, tensor in weights.items():
        bridge.load_weight(name, tensor)
    with pytest.raises(ValueError, match="nonzero terminal projection"):
        bridge.prepare()


@pytest.mark.parametrize(
    "version_name",
    ["independent_write_version", "terminal_disabled_version"],
)
def test_bridge_prepare_rejects_unsupported_version(version_name: str) -> None:
    bridge = DSpineCodeBridge(_bridge_hf_config())
    weights = _bridge_checkpoint()
    weights[version_name] = torch.full((), 2, dtype=torch.int64)
    bridge.begin_load()
    for name, tensor in weights.items():
        bridge.load_weight(name, tensor)
    with pytest.raises(ValueError, match=f"Unsupported DSPINE {version_name}"):
        bridge.prepare()


def test_bridge_load_weight_rejects_unknown_and_mismatched() -> None:
    bridge = DSpineCodeBridge(_bridge_hf_config())
    bridge.begin_load()
    with pytest.raises(ValueError, match="Unexpected DSPINE code-bridge"):
        bridge.load_weight("bogus.weight", torch.zeros(1))
    with pytest.raises(ValueError, match="shape mismatch"):
        bridge.load_weight("codebook", torch.zeros(4, 4))


# ---------------------------------------------------------------------------
# DSpineCodeBridge forward numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bridge_impl", ["eager", "fused"])
def test_bridge_forward_matches_float64_reference(bridge_impl: str) -> None:
    """Eager (and on CPU the fused fallback) match the fp64 reference.

    ``dspine_bridge_impl="fused"`` silently degrades to eager off CUDA, so
    both names must produce identical CPU output.
    """
    torch.manual_seed(0)
    config = _bridge_hf_config(dspine_bridge_impl=bridge_impl)
    bridge = DSpineCodeBridge(config, params_dtype=torch.float32)
    weights = _bridge_checkpoint()
    _load_bridge(bridge, weights)
    t, h = config.block_size, config.hidden_size
    next_norm = SimpleNamespace(weight=torch.randn(h), variance_epsilon=1e-6)
    input_ids = torch.randint(config.vocab_size, (2 * t,))

    for use_residual in (True, False):
        hidden = torch.randn(2 * t, h)
        residual = torch.randn(2 * t, h) if use_residual else None
        normed, out, raw, rms, packed = bridge(
            hidden, residual, input_ids, 1, next_norm
        )
        e_normed, e_out, e_raw, e_rms, e_packed = _ref_bridge_write(
            bridge, hidden, residual, input_ids, 1, next_norm
        )
        torch.testing.assert_close(normed.double(), e_normed, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out.double(), e_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(raw.double(), e_raw, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(rms.double(), e_rms, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(packed.double(), e_packed, rtol=1e-4, atol=1e-4)

        # Anchor rows read the vocabulary codebook and skip the write.
        anchor_rows = torch.arange(0, 2 * t, t)
        anchors = input_ids.view(-1, t)[:, 0]
        torch.testing.assert_close(
            packed[anchor_rows].double(),
            math.sqrt(bridge.code_size)
            * weights["codebook"].double()[anchors]
            @ bridge.packed_weight.double().T,
            rtol=1e-4,
            atol=1e-4,
        )
        assert torch.equal(out[anchor_rows], raw[anchor_rows])


def test_bridge_layer_read_weight_routes_last_layer() -> None:
    """Layers 0..L-2 read through early_read_weights, L-1 through read_weight."""
    config = _bridge_hf_config()
    bridge = DSpineCodeBridge(config, params_dtype=torch.float32)
    weights = _bridge_checkpoint()
    _load_bridge(bridge, weights)
    torch.testing.assert_close(
        bridge.layer_read_weight(0), weights["early_read_weights"][0]
    )
    torch.testing.assert_close(
        bridge.layer_read_weight(config.num_hidden_layers - 1),
        weights["read_weight"],
    )


def test_bridge_forward_requires_prepared_and_whole_blocks() -> None:
    config = _bridge_hf_config()
    bridge = DSpineCodeBridge(config)
    next_norm = SimpleNamespace(weight=torch.ones(64), variance_epsilon=1e-6)
    ids = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="not prepared"):
        bridge(torch.randn(4, 64), None, ids, 1, next_norm)
    _load_bridge(bridge, _bridge_checkpoint())
    with pytest.raises(ValueError, match="whole request blocks"):
        bridge(torch.randn(3, 64), None, ids[:3], 1, next_norm)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="fused kernels need CUDA")
def test_bridge_fused_matches_eager_on_cuda() -> None:
    """The Triton kernels and the eager path agree, bf16 inputs."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = _bridge_checkpoint()
    config = _bridge_hf_config(dspine_bridge_impl="fused")
    fused = DSpineCodeBridge(config, params_dtype=torch.bfloat16).to(device)
    _load_bridge(fused, {k: v.to(device) for k, v in weights.items()})
    eager = DSpineCodeBridge(
        _bridge_hf_config(dspine_bridge_impl="eager"),
        params_dtype=torch.bfloat16,
    ).to(device)
    _load_bridge(eager, {k: v.to(device) for k, v in weights.items()})
    t, h = config.block_size, config.hidden_size
    next_norm = SimpleNamespace(
        weight=torch.randn(h, device=device, dtype=torch.bfloat16),
        variance_epsilon=1e-6,
    )
    hidden = torch.randn(2 * t, h, device=device, dtype=torch.bfloat16)
    residual = torch.randn(2 * t, h, device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(config.vocab_size, (2 * t,), device=device)

    got = fused(hidden, residual, input_ids, 1, next_norm)
    want = eager(hidden, residual, input_ids, 1, next_norm)
    for actual, expected in zip(got, want):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=1e-2, atol=1e-2
        )
    anchor_rows = torch.arange(0, 2 * t, t, device=device)
    assert torch.equal(got[1][anchor_rows], got[2][anchor_rows])


# ---------------------------------------------------------------------------
# build_serial_tables / score_transitions
# ---------------------------------------------------------------------------


def _score_model() -> tuple[
    DSpineQwen3ForCausalLM, dict[str, torch.Tensor], torch.Tensor
]:
    """A bare ``DSpineQwen3ForCausalLM`` with a real prepared fp32 bridge."""
    config = _bridge_hf_config()
    bridge = DSpineCodeBridge(config, params_dtype=torch.float32)
    weights = _bridge_checkpoint()
    _load_bridge(bridge, weights)
    model = DSpineQwen3ForCausalLM.__new__(DSpineQwen3ForCausalLM)
    nn.Module.__init__(model)
    model.config = config
    model.model = SimpleNamespace(
        selected_prefix_bridge=bridge,
        layers=[None] * config.num_hidden_layers,
        norm=SimpleNamespace(
            weight=torch.randn(config.hidden_size), variance_epsilon=1e-6
        ),
    )
    lm_head = torch.randn(config.vocab_size, config.hidden_size)
    model.materialize_inference_buffers(lm_head)
    return model, weights, lm_head


def test_build_serial_tables_matches_direct_formulas() -> None:
    torch.manual_seed(0)
    v, h, r, c = 12, 8, 4, 6
    lm_head = torch.randn(v, h)
    norm_w = torch.randn(h)
    write_w = torch.randn(h, r)
    pv_w = torch.randn(2 * r, c)
    codebook = torch.randn(v, c)

    tg, tsv = build_serial_tables(lm_head, norm_w, write_w, pv_w, codebook)

    torch.testing.assert_close(tg, (lm_head * norm_w) @ write_w)
    torch.testing.assert_close(tsv, (math.sqrt(c) * codebook) @ pv_w.T)
    with pytest.raises(ValueError, match="shape contract"):
        build_serial_tables(lm_head, norm_w, torch.randn(r, h), pv_w, codebook)


def test_materialize_inference_buffers_checks_lm_head_shape() -> None:
    model, _, _ = _score_model()
    with pytest.raises(ValueError, match="lm_head_weight"):
        model.materialize_inference_buffers(torch.randn(7, 64))


def test_score_transitions_matches_float64_refine_table() -> None:
    torch.manual_seed(0)
    model, weights, lm_head = _score_model()
    b, s, k = 2, 3, 8
    g = torch.Generator().manual_seed(1)
    candidate_ids = torch.stack(
        [torch.randperm(model.config.vocab_size, generator=g)[:k] for _ in range(b * s)]
    ).view(b, s, k)
    raw = torch.randn(b, s, model.config.hidden_size)
    rms = raw.square().mean(-1).sqrt()
    query = torch.randn(b, s, 16)
    anchors = torch.randint(model.config.vocab_size, (b,))

    got = model.score_transitions(candidate_ids, raw, rms, query, anchors)

    assert got.dtype == torch.float32
    assert got.shape == (b, s, k, k)
    expected = _ref_refine_table(
        candidate_ids,
        raw,
        rms,
        query,
        anchors,
        codebook=weights["codebook"],
        pred_gate_w=weights["predecessor_gate.weight"],
        prev_w=weights["prev.weight"],
        gate_b=weights["gate.bias"],
        write_w=weights["feature_write_weights"][-1],
        norm_w=model.model.norm.weight,
        lm_head_w=lm_head,
        eps=model.norm_eps,
    )
    torch.testing.assert_close(got.double(), expected, rtol=1e-3, atol=1e-3)


def test_score_transitions_step_zero_uses_anchor_for_all_senders() -> None:
    """Row i=0 has no predecessor candidate: every sender lane is the anchor."""
    torch.manual_seed(0)
    model, _, _ = _score_model()
    b, s, k = 2, 3, 8
    g = torch.Generator().manual_seed(2)
    candidate_ids = torch.stack(
        [torch.randperm(model.config.vocab_size, generator=g)[:k] for _ in range(b * s)]
    ).view(b, s, k)
    raw = torch.randn(b, s, model.config.hidden_size)
    rms = raw.square().mean(-1).sqrt()
    query = torch.randn(b, s, 16)
    anchors = torch.randint(model.config.vocab_size, (b,))

    scores = model.score_transitions(candidate_ids, raw, rms, query, anchors)

    # Identical senders -> identical sender rows.
    step0 = scores[:, 0]
    assert torch.equal(step0, step0[:, :1].expand_as(step0))


def test_score_transitions_unnormalized_shares_greedy_argmax() -> None:
    """The normalize=False table only drops a positive per-(sender) scale."""
    torch.manual_seed(0)
    model, _, _ = _score_model()
    b, s, k = 2, 3, 8
    g = torch.Generator().manual_seed(3)
    candidate_ids = torch.stack(
        [torch.randperm(model.config.vocab_size, generator=g)[:k] for _ in range(b * s)]
    ).view(b, s, k)
    raw = torch.randn(b, s, model.config.hidden_size)
    rms = raw.square().mean(-1).sqrt()
    query = torch.randn(b, s, 16)
    anchors = torch.randint(model.config.vocab_size, (b,))

    normalized = model.score_transitions(candidate_ids, raw, rms, query, anchors)
    unnorm = model.score_transitions(
        candidate_ids, raw, rms, query, anchors, normalize=False
    )
    assert torch.equal(unnorm.argmax(-1), normalized.argmax(-1))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_score_transitions_matches_float64_refine_table_on_cuda() -> None:
    """The R-space fp32 algebra holds on GPU with BF16 checkpoint weights."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    config = _bridge_hf_config()
    weights = _bridge_checkpoint()
    bridge = DSpineCodeBridge(config, params_dtype=torch.bfloat16).to(device)
    _load_bridge(bridge, {k: v.to(device) for k, v in weights.items()})
    model = DSpineQwen3ForCausalLM.__new__(DSpineQwen3ForCausalLM)
    nn.Module.__init__(model)
    model.config = config
    model.model = SimpleNamespace(
        selected_prefix_bridge=bridge,
        layers=[None] * config.num_hidden_layers,
        norm=SimpleNamespace(
            weight=torch.randn(config.hidden_size, device=device, dtype=torch.bfloat16),
            variance_epsilon=1e-6,
        ),
    )
    lm_head = torch.randn(
        config.vocab_size,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    model.materialize_inference_buffers(lm_head)

    b, s, k = 2, 3, 8
    g = torch.Generator(device=device).manual_seed(1)
    candidate_ids = torch.stack(
        [
            torch.randperm(config.vocab_size, generator=g, device=device)[:k]
            for _ in range(b * s)
        ]
    ).view(b, s, k)
    raw = torch.randn(b, s, config.hidden_size, device=device, dtype=torch.bfloat16)
    rms = raw.float().square().mean(-1).sqrt()
    query = torch.randn(b, s, bridge.rank, device=device, dtype=torch.bfloat16)
    anchors = torch.randint(config.vocab_size, (b,), device=device)

    got = model.score_transitions(candidate_ids, raw, rms, query, anchors)
    expected = _ref_refine_table(
        candidate_ids,
        raw,
        rms,
        query,
        anchors,
        codebook=weights["codebook"].to(device),
        pred_gate_w=weights["predecessor_gate.weight"].to(device),
        prev_w=weights["prev.weight"].to(device),
        gate_b=weights["gate.bias"].to(device),
        write_w=weights["feature_write_weights"][-1].to(device),
        norm_w=model.model.norm.weight,
        lm_head_w=lm_head,
        eps=model.norm_eps,
    )
    torch.testing.assert_close(got.double(), expected, rtol=1e-2, atol=1e-2)
    assert torch.equal(
        got.argmax(-1),
        model.score_transitions(
            candidate_ids, raw, rms, query, anchors, normalize=False
        ).argmax(-1),
    )


# ---------------------------------------------------------------------------
# Model construction, weight loading, causality
# ---------------------------------------------------------------------------


def _dspine_hf_config(
    *,
    hidden: int = 64,
    layers: int = 2,
    vocab: int = 256,
    block_size: int = 4,
    code: int = 32,
    rank: int = 16,
    head_topk: int = 8,
    heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 16,
    ffn: int = 128,
    aux_layers: tuple[int, ...] = (0, 1),
) -> SimpleNamespace:
    """HF-style config matching tencent/Qwen3-8B-DSpine-b16's fields."""
    return SimpleNamespace(
        model_type="qwen3",
        architectures=["DSpineDraftModel"],
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        intermediate_size=ffn,
        hidden_act="silu",
        is_causal=None,
        layer_types=["full_attention"] * layers,
        sliding_window=None,
        block_size=block_size,
        mask_token_id=vocab - 1,
        dflash_config={
            "mask_token_id": vocab - 1,
            "target_layer_ids": list(aux_layers),
            "sample_from_anchor": False,
            "draft_block_causal": True,
            "selected_prefix_terminal": False,
            "selected_prefix_feature_write_mode": "independent",
            "selected_prefix_feature_writes": True,
            "selected_prefix_feature_layers": list(range(1, layers + 1)),
            "selected_prefix_code_bridge": {
                "code_size": code,
                "rank": rank,
                "eps": 1e-6,
            },
            "selected_prefix_last_write_refine": {
                "kind": "exact_sender",
                "head_topk": head_topk,
            },
            "dspine_bridge_impl": "eager",
        },
    )


def _dspine_checkpoint(
    config: SimpleNamespace, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Checkpoint tensors named like the tencent export (no ``model.`` prefix)."""
    g = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    h, layers = config.hidden_size, config.num_hidden_layers
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    head_dim, ffn = config.head_dim, config.intermediate_size
    bridge = config.dflash_config["selected_prefix_code_bridge"]
    n_aux = len(config.dflash_config["target_layer_ids"])
    weights = {
        "fc.weight": randn(h, n_aux * h),
        "hidden_norm.weight": randn(h),
        "norm.weight": randn(h),
        "draft_block_causal_version": torch.ones((), dtype=torch.int64),
        "last_write_refine_version": torch.ones((), dtype=torch.int64),
    }
    for name, tensor in _bridge_checkpoint(
        h, layers, config.vocab_size, bridge["code_size"], bridge["rank"]
    ).items():
        weights[f"{BRIDGE_PREFIX}{name}"] = tensor
    for i in range(layers):
        weights[f"layers.{i}.input_layernorm.weight"] = randn(h)
        weights[f"layers.{i}.post_attention_layernorm.weight"] = randn(h)
        weights[f"layers.{i}.self_attn.q_proj.weight"] = randn(heads * head_dim, h)
        weights[f"layers.{i}.self_attn.k_proj.weight"] = randn(kv_heads * head_dim, h)
        weights[f"layers.{i}.self_attn.v_proj.weight"] = randn(kv_heads * head_dim, h)
        weights[f"layers.{i}.self_attn.o_proj.weight"] = randn(h, heads * head_dim)
        weights[f"layers.{i}.self_attn.q_norm.weight"] = randn(head_dim)
        weights[f"layers.{i}.self_attn.k_norm.weight"] = randn(head_dim)
        weights[f"layers.{i}.mlp.gate_proj.weight"] = randn(ffn, h)
        weights[f"layers.{i}.mlp.up_proj.weight"] = randn(ffn, h)
        weights[f"layers.{i}.mlp.down_proj.weight"] = randn(h, ffn)
    return weights


def _mock_draft_configs(
    monkeypatch: pytest.MonkeyPatch,
    hf_config: SimpleNamespace,
    *,
    vocab: int,
    num_target_layers: int,
    num_speculative_tokens: int,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """The current and draft vLLM configs for building a draft model on CPU."""
    current_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            kv_cache_dtype_skip_layers=[],
            cache_dtype="auto",
            sliding_window=None,
            enable_prefix_caching=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend=None,
            backend_per_kind={},
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            compile_custom_ops=False,
            custom_ops="all",
            enabled_custom_ops=set(),
            static_forward_context={},
            mode=0,  # CompilationMode.NONE is 0
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
            rswa_window=None,
            head_dtype=None,
        ),
        kernel_config=SimpleNamespace(
            linear_backend="auto",
        ),
    )
    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        lambda *args, **kwargs: (
            "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
        ),
    )

    class MockGroup:
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        MockGroup(),
    )

    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=hf_config,
                quantization=None,
                # A local dir: mask_embedding.pt lookup stays off the network.
                model="/tmp",
                revision=None,
            ),
            num_speculative_tokens=num_speculative_tokens,
            enable_adaptive_verification=False,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
            get_total_num_hidden_layers=lambda: num_target_layers,
            get_vocab_size=lambda: vocab,
        ),
        load_config=SimpleNamespace(
            quantization=None,
            quantization_param_path=None,
        ),
    )
    current_config.speculative_config = vllm_config.speculative_config
    vllm_config.compilation_config = current_config.compilation_config
    return current_config, vllm_config


def _build_dspine_model(
    monkeypatch: pytest.MonkeyPatch,
    hf_config: SimpleNamespace | None = None,
    *,
    vocab: int = 256,
    num_target_layers: int = 36,
    num_speculative_tokens: int = 3,
) -> DSpineQwen3ForCausalLM:
    if hf_config is None:
        hf_config = _dspine_hf_config()
    current_config, vllm_config = _mock_draft_configs(
        monkeypatch,
        hf_config,
        vocab=vocab,
        num_target_layers=num_target_layers,
        num_speculative_tokens=num_speculative_tokens,
    )
    with set_current_vllm_config(current_config):
        return DSpineQwen3ForCausalLM(vllm_config=vllm_config)


def test_meta_shapes_match_qwen3_8b_dspine_b16(monkeypatch) -> None:
    """The tencent/Qwen3-8B-DSpine-b16 parameter contract, on meta."""
    hf_config = _dspine_hf_config(
        hidden=4096,
        layers=5,
        vocab=151936,
        block_size=16,
        code=1024,
        rank=512,
        head_topk=16,
        heads=32,
        kv_heads=8,
        head_dim=128,
        ffn=12288,
        aux_layers=(1, 9, 17, 25, 33),
    )
    current_config, vllm_config = _mock_draft_configs(
        monkeypatch,
        hf_config,
        vocab=151936,
        num_target_layers=36,
        num_speculative_tokens=15,
    )
    with set_current_vllm_config(current_config), torch.device("meta"):
        model = DSpineQwen3ForCausalLM(vllm_config=vllm_config)

    bridge = model.model.selected_prefix_bridge
    assert bridge.codebook.shape == (151936, 1024)
    assert bridge.read_weight.shape == (1024, 4096)
    assert bridge.early_read_weights.shape == (4, 1024, 4096)
    assert bridge.feature_write_weights.shape == (5, 4096, 512)
    assert bridge.gate.weight.shape == (512, 1024)
    assert bridge.predecessor_gate.weight.shape == (512, 1024)
    assert bridge.prev.weight.shape == (512, 1024)
    assert bridge.out.weight.shape == (1024, 512)
    assert bridge.gate.bias.shape == (512,)
    assert model.model.fc.weight.shape == (4096, 5 * 4096)
    assert model.head_topk == 16
    assert model.get_draft_kv_cache_layer_names() == [
        f"model.layers.{36 + i}.self_attn.attn" for i in range(5)
    ]
    assert model.get_draft_attn_causal() == [True] * 5
    assert dflash_has_any_non_causal(hf_config) is False


def test_dspine_loads_checkpoint_and_resolves_causality(monkeypatch) -> None:
    hf_config = _dspine_hf_config()
    model = _build_dspine_model(monkeypatch, hf_config)
    checkpoint = _dspine_checkpoint(hf_config)

    model.load_weights(iter(checkpoint.items()))

    bridge = model.model.selected_prefix_bridge
    assert bridge.packed_weight is not None
    torch.testing.assert_close(
        bridge.packed_weight,
        torch.cat(
            (
                checkpoint[f"{BRIDGE_PREFIX}gate.weight"],
                checkpoint[f"{BRIDGE_PREFIX}predecessor_gate.weight"],
                checkpoint[f"{BRIDGE_PREFIX}prev.weight"],
            )
        ),
    )
    torch.testing.assert_close(bridge.codebook, checkpoint[f"{BRIDGE_PREFIX}codebook"])
    assert not torch.isnan(model.model.norm.weight).any()
    # Optional weights omitted by the checkpoint are zeroed, not left NaN.
    assert torch.equal(model.lm_head.weight, torch.zeros_like(model.lm_head.weight))
    assert torch.equal(
        model.model.embed_tokens.weight,
        torch.zeros_like(model.model.embed_tokens.weight),
    )
    assert model.head_topk == 8
    # draft_block_causal resolved to causal attention in every draft layer.
    assert model.get_draft_attn_causal() == [True, True]
    assert dflash_has_any_non_causal(hf_config) is False
    # Draft layers are numbered after the 36-layer target.
    assert model.get_draft_kv_cache_layer_names() == [
        "model.layers.36.self_attn.attn",
        "model.layers.37.self_attn.attn",
    ]


def test_compute_candidates_returns_sorted_topk(monkeypatch) -> None:
    torch.manual_seed(0)
    hf_config = _dspine_hf_config()
    model = _build_dspine_model(monkeypatch, hf_config)
    model.load_weights(iter(_dspine_checkpoint(hf_config).items()))
    with torch.no_grad():
        model.lm_head.weight.copy_(
            torch.randn(model.config.vocab_size, model.config.hidden_size)
        )

    hidden = torch.randn(6, model.config.hidden_size)
    candidate_ids, unary = model.compute_candidates(hidden)

    logits = hidden @ model.lm_head.weight.T
    expected_ids = logits.topk(model.head_topk).indices.sort(dim=-1).values
    assert torch.equal(candidate_ids, expected_ids)
    torch.testing.assert_close(unary, logits.gather(-1, candidate_ids))


@pytest.mark.parametrize(
    "version_name",
    ["draft_block_causal_version", "last_write_refine_version"],
)
def test_load_rejects_unsupported_checkpoint_version(
    monkeypatch, version_name: str
) -> None:
    hf_config = _dspine_hf_config()
    model = _build_dspine_model(monkeypatch, hf_config)
    checkpoint = _dspine_checkpoint(hf_config)
    checkpoint[version_name] = torch.full((), 2, dtype=torch.int64)
    with pytest.raises(ValueError, match="checkpoint version"):
        model.load_weights(iter(checkpoint.items()))


def test_load_rejects_missing_bridge_tensor(monkeypatch) -> None:
    hf_config = _dspine_hf_config()
    model = _build_dspine_model(monkeypatch, hf_config)
    checkpoint = _dspine_checkpoint(hf_config)
    del checkpoint[f"{BRIDGE_PREFIX}prev.weight"]
    with pytest.raises(ValueError, match="Missing DSPINE code-bridge"):
        model.load_weights(iter(checkpoint.items()))


def test_load_rejects_missing_layer_weight(monkeypatch) -> None:
    hf_config = _dspine_hf_config()
    model = _build_dspine_model(monkeypatch, hf_config)
    checkpoint = _dspine_checkpoint(hf_config)
    del checkpoint["norm.weight"]
    with pytest.raises(ValueError, match="left parameters unloaded"):
        model.load_weights(iter(checkpoint.items()))


def test_construction_rejects_non_causal_draft(monkeypatch) -> None:
    hf_config = _dspine_hf_config()
    hf_config.is_causal = False
    with pytest.raises(ValueError, match="causal draft attention"):
        _build_dspine_model(monkeypatch, hf_config)


def test_construction_rejects_block_size_mismatch(monkeypatch) -> None:
    hf_config = _dspine_hf_config()
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        _build_dspine_model(monkeypatch, hf_config, num_speculative_tokens=4)


# ---------------------------------------------------------------------------
# DSpineSpeculator
# ---------------------------------------------------------------------------


def _stub_speculator(
    monkeypatch: pytest.MonkeyPatch, draft_logits: torch.Tensor | None
) -> DSpineSpeculator:
    """DSpineSpeculator with the DFlashSpeculator base init stubbed out."""

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=5,
                dflash_config={
                    "selected_prefix_last_write_refine": {
                        "kind": "exact_sender",
                        "head_topk": 8,
                    }
                },
            )
        )
        self.max_num_reqs = 3
        self.num_query_per_req = 5
        self.num_speculative_steps = 4
        self.vocab_size = 64
        self.draft_tokens = torch.empty((3, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits
        self.input_buffers = SimpleNamespace(input_ids=torch.arange(15, device=device))

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)
    return DSpineSpeculator(None, torch.device("cpu"))


def test_dspine_draft_logits_spec_is_fp32_neg_inf() -> None:
    dtype, fill = DSpineSpeculator.draft_logits_spec(None, None)
    assert dtype is torch.float32
    assert fill == float("-inf")


def test_speculator_resolves_top_k_and_greedy_keeps_no_logits(
    monkeypatch,
) -> None:
    speculator = _stub_speculator(monkeypatch, draft_logits=None)
    assert speculator._speculator_name == "DSpine"
    assert speculator.top_k == 8
    assert speculator.candidate_sampler.top_k == 8
    assert speculator.draft_logits is None


def test_score_candidates_routes_anchors_and_normalize(monkeypatch) -> None:
    """Greedy requests the unnormalized table; probabilistic normalizes."""
    greedy = _stub_speculator(monkeypatch, draft_logits=None)
    probabilistic = _stub_speculator(monkeypatch, draft_logits=torch.zeros(3, 4, 8))
    calls = []

    def fake_score_transitions(cand, raw, rms, query, anchors, *, normalize):
        calls.append((normalize, anchors.clone()))
        return torch.zeros(*cand.shape, cand.shape[-1])

    b, s, k = 3, 4, 8
    cand = torch.zeros(b, s, k, dtype=torch.int64)
    raw = torch.randn(b, s, 64)
    rms = torch.ones(b, s)
    query = torch.randn(b, s, 16)
    for speculator in (greedy, probabilistic):
        speculator.model = SimpleNamespace(score_transitions=fake_score_transitions)
        speculator._score_candidates(cand, torch.zeros(b, s, k), raw, rms, query)

    assert calls[0][0] is False
    assert calls[1][0] is True
    # Anchors are the committed token at each request's block start.
    assert torch.equal(calls[0][1], torch.tensor([0, 5, 10]))
