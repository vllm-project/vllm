# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpine draft coverage: code bridge, serial refinement, config, loading."""

import logging
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
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspine.config import DSpineSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.dspine.speculator import DSpineSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    DraftModelSpeculatorConfig,
    _resolve_checkpoint_config_class,
    get_method_config_class,
    get_speculator_config_class,
)

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
    """float64 [B, S, K, K] transition logits of the redone last write."""
    b = candidate_ids.shape[0]
    k = candidate_ids.shape[2]
    senders = torch.cat(
        (anchor_token_ids[:, None, None].expand(b, 1, k), candidate_ids[:, :-1]),
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
    normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm_w.double()
    return torch.einsum("bskh,bsch->bskc", normed, lm_head_w.double()[candidate_ids])


def _bridge_hf_config(**dflash_overrides) -> SimpleNamespace:
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
            mode=0,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
            rswa_window=None,
            head_dtype=None,
        ),
        kernel_config=SimpleNamespace(linear_backend="auto"),
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

    monkeypatch.setattr("vllm.distributed.parallel_state._TP", MockGroup())
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=hf_config,
                quantization=None,
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
        load_config=SimpleNamespace(quantization=None, quantization_param_path=None),
        compilation_config=current_config.compilation_config,
    )
    current_config.speculative_config = vllm_config.speculative_config
    with set_current_vllm_config(current_config):
        return DSpineQwen3ForCausalLM(vllm_config=vllm_config)


def _score_model() -> tuple[
    DSpineQwen3ForCausalLM, dict[str, torch.Tensor], torch.Tensor
]:
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


@pytest.mark.parametrize("bridge_impl", ["eager", "fused"])
def test_bridge_forward_matches_float64_reference(bridge_impl: str) -> None:
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

        anchor_rows = torch.arange(0, 2 * t, t)
        assert torch.equal(out[anchor_rows], raw[anchor_rows])


@pytest.mark.skipif(not current_platform.is_cuda(), reason="fused kernels need CUDA")
def test_bridge_fused_matches_eager_on_cuda() -> None:
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
    # Row 0 has the anchor as sender for every lane; unnormalized keeps argmax.
    assert torch.equal(got[:, 0], got[:, 0, :1].expand_as(got[:, 0]))
    unnorm = model.score_transitions(
        candidate_ids, raw, rms, query, anchors, normalize=False
    )
    assert torch.equal(unnorm.argmax(-1), got.argmax(-1))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_score_transitions_matches_float64_refine_table_on_cuda() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    config = _bridge_hf_config()
    weights = _bridge_checkpoint()
    bf16_weights = {
        k: v.to(device=device, dtype=torch.bfloat16)
        if v.is_floating_point()
        else v.to(device)
        for k, v in weights.items()
    }
    bridge = DSpineCodeBridge(config, params_dtype=torch.bfloat16).to(device)
    _load_bridge(bridge, bf16_weights)
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
        codebook=bf16_weights["codebook"],
        pred_gate_w=bf16_weights["predecessor_gate.weight"],
        prev_w=bf16_weights["prev.weight"],
        gate_b=bf16_weights["gate.bias"],
        write_w=bf16_weights["feature_write_weights"][-1],
        norm_w=model.model.norm.weight,
        lm_head_w=lm_head,
        eps=model.norm_eps,
    )
    torch.testing.assert_close(got.double(), expected, rtol=1e-2, atol=1e-2)


def test_dspine_loads_checkpoint_and_computes_candidates(monkeypatch) -> None:
    torch.manual_seed(0)
    hf_config = _dspine_hf_config()
    model = _build_dspine_model(monkeypatch, hf_config)
    checkpoint = _dspine_checkpoint(hf_config)
    model.load_weights(iter(checkpoint.items()))

    assert model.head_topk == 8
    assert model.get_draft_attn_causal() == [True, True]
    assert dflash_has_any_non_causal(hf_config) is False
    assert model.get_draft_kv_cache_layer_names() == [
        "model.layers.36.self_attn.attn",
        "model.layers.37.self_attn.attn",
    ]
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


def test_construction_warns_on_block_size_mismatch_and_uses_runtime_stride(
    monkeypatch, caplog
) -> None:
    hf_config = _dspine_hf_config(block_size=4)
    with caplog.at_level(logging.WARNING):
        model = _build_dspine_model(monkeypatch, hf_config, num_speculative_tokens=2)
    assert "block_size=4" in caplog.text
    bridge = model.model.selected_prefix_bridge
    assert bridge.block_size == 3
    model.load_weights(iter(_dspine_checkpoint(hf_config).items()))
    hidden = torch.randn(6, hf_config.hidden_size)
    input_ids = torch.randint(hf_config.vocab_size, (6,))
    normed, out, *_ = bridge(
        hidden, None, input_ids, 0, model.model.layers[1].input_layernorm
    )
    assert normed.shape == (6, hf_config.hidden_size)
    assert out.shape == (6, hf_config.hidden_size)


def test_dspine_speculator_config_handles_dispatch_and_block_size_warning(
    caplog,
) -> None:
    hf_config = _dspine_hf_config(block_size=16)
    draft_cfg = SimpleNamespace(
        model="tencent/Qwen3-8B-DSpine-b16",
        architectures=["DSpineDraftModel"],
        hf_config=hf_config,
        speculator_config_class=(
            "vllm.v1.worker.gpu.spec_decode.dspine.config.DSpineSpeculatorConfig"
        ),
    )
    assert (
        get_method_config_class(SimpleNamespace(method="dspine", custom_method=None))
        is DSpineSpeculatorConfig
    )
    # DFlashSpeculatorConfig stays untouched and delegates to DSpineSpeculatorConfig.
    spec = SimpleNamespace(
        method="dflash",
        custom_method=None,
        draft_model_config=draft_cfg,
        num_speculative_tokens=None,
        draft_tensor_parallel_size=1,
        kv_cache_dtype="auto",
        parallel_drafting=False,
        model=draft_cfg.model,
        update_arch_=lambda: None,
    )
    assert (
        DFlashSpeculatorConfig.matches(
            SimpleNamespace(
                draft_model_config=SimpleNamespace(model="tencent/custom-draft")
            )
        )
        is False
    )
    assert get_speculator_config_class(spec) is DSpineSpeculatorConfig

    spec.method = None
    draft_cfg.speculator_config_class = None
    assert (
        _resolve_checkpoint_config_class(spec, DraftModelSpeculatorConfig)
        is DSpineSpeculatorConfig
    )

    DSpineSpeculatorConfig.resolve_num_speculative_tokens(spec)
    assert spec.num_speculative_tokens == 15

    spec.num_speculative_tokens = 7
    with caplog.at_level(logging.WARNING):
        DSpineSpeculatorConfig.resolve_num_speculative_tokens(spec)
    assert "block_size=16" in caplog.text
    assert spec.num_speculative_tokens == 7

    spec.draft_tensor_parallel_size = 2
    with pytest.raises(ValueError, match="tensor_parallel_size == 1"):
        DSpineSpeculatorConfig.apply_options(spec)


def test_score_candidates_routes_anchors_and_normalize(monkeypatch) -> None:
    def make_speculator(draft_logits: torch.Tensor | None) -> DSpineSpeculator:
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
            self.input_buffers = SimpleNamespace(
                input_ids=torch.arange(15, device=device)
            )

        monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)
        return DSpineSpeculator(None, torch.device("cpu"))

    greedy = make_speculator(draft_logits=None)
    probabilistic = make_speculator(draft_logits=torch.zeros(3, 4, 8))
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
    assert torch.equal(calls[0][1], torch.tensor([0, 5, 10]))
