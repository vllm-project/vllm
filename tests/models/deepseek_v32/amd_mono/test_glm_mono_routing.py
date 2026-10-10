# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the GLM-5 MonoKernel integration: refusals, the per-step
decision, routing, weight-view checks, cache re-checks and fail-stop."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v32.amd.mono import kernel as K
from vllm.models.deepseek_v32.amd.mono import spec
from vllm.models.deepseek_v32.amd.mono import weights as Wt

H = 16


def make_config(**overrides):
    """A VllmConfig stand-in that the kernel supports; ``overrides`` breaks it."""
    o = dict(
        model_type="glm_moe_dsa",
        tp=4,
        pp=1,
        dp=1,
        ep=False,
        eplb=False,
        dcp=1,
        pcp=1,
        spec=None,
        lora=None,
        kv_transfer=None,
        routed_experts=False,
        prompt_embeds=False,
        sleep=False,
        offload_gb=0,
        offload_group=0,
        kv_dtype="fp8",
        rope_interleave=True,
        v2=True,
        graphs=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    o.update(overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=o["tp"],
            pipeline_parallel_size=o["pp"],
            data_parallel_size=o["dp"],
            enable_expert_parallel=o["ep"],
            enable_eplb=o["eplb"],
            decode_context_parallel_size=o["dcp"],
            prefill_context_parallel_size=o["pcp"],
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type=o["model_type"], indexer_rope_interleave=o["rope_interleave"]
            ),
            enable_prompt_embeds=o["prompt_embeds"],
            enable_sleep_mode=o["sleep"],
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=o["graphs"]),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=o["offload_gb"]),
            prefetch=SimpleNamespace(offload_group_size=o["offload_group"]),
        ),
        cache_config=SimpleNamespace(cache_dtype=o["kv_dtype"]),
        speculative_config=o["spec"],
        lora_config=o["lora"],
        kv_transfer_config=o["kv_transfer"],
        aux_output_config=SimpleNamespace(
            enable_return_routed_experts=o["routed_experts"]
        ),
        use_v2_model_runner=o["v2"],
    )


def test_supported_config_passes(monkeypatch):
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    assert spec.refuse_config(make_config(), True, 256) == []
    assert spec.refuse_config(make_config(tp=8), True, 256) == []


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        (dict(model_type="deepseek_v32"), "glm_moe_dsa"),
        (dict(tp=2), "tensor parallel size"),
        (dict(pp=2), "pipeline parallel"),
        (dict(dp=2), "data parallel"),
        (dict(ep=True), "expert parallel"),
        (dict(eplb=True), "EPLB"),
        (dict(dcp=2), "decode context parallel"),
        (dict(pcp=2), "prefill context parallel"),
        (dict(spec=object()), "speculative decoding"),
        (dict(lora=object()), "LoRA"),
        (dict(kv_transfer=object()), "KV transfer"),
        (dict(routed_experts=True), "routed-expert capture"),
        (dict(prompt_embeds=True), "prompt embeds"),
        (dict(sleep=True), "sleep mode"),
        (dict(offload_gb=4), "CPU weight offload"),
        (dict(kv_dtype="auto"), "kv-cache-dtype fp8"),
        (dict(rope_interleave=False), "indexer_rope_interleave"),
        (dict(v2=False), "Model Runner V2"),
        (dict(graphs=CUDAGraphMode.PIECEWISE), "piecewise"),
    ],
)
def test_each_unsupported_feature_is_refused(monkeypatch, overrides, fragment):
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    reasons = spec.refuse_config(make_config(**overrides), True, 256)
    assert len(reasons) == 1 and fragment in reasons[0]


def test_every_reason_is_reported_at_once(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    reasons = spec.refuse_config(make_config(tp=2, lora=object()), False, 192)
    joined = " | ".join(reasons)
    for fragment in ("gfx950", "compute units", "tensor parallel", "LoRA", "breakable"):
        assert fragment in joined
    assert len(reasons) == 5


def test_refusal_raises_with_every_reason(monkeypatch):
    monkeypatch.setattr(K.current_platform, "is_rocm", lambda: False)
    with pytest.raises(ValueError, match="gfx950.*compute units"):
        K.refuse_config(make_config())


def _metadata(max_query_len=1):
    return {"attn": SimpleNamespace(max_query_len=max_query_len)}


@pytest.mark.parametrize("tokens", spec.STEP_TOKENS)
def test_pure_decode_at_each_width_takes_the_kernel(tokens):
    assert spec.step_reason(tokens, _metadata(), "attn") is None


@pytest.mark.parametrize(
    "tokens, metadata, piecewise",
    [
        (3, _metadata(), False),
        (16, _metadata(), False),
        (4, _metadata(max_query_len=7), False),
        (4, None, False),
        (4, {"other": SimpleNamespace(max_query_len=1)}, False),
        (4, _metadata(), True),
    ],
)
def test_other_steps_take_vllm(tokens, metadata, piecewise):
    assert spec.step_reason(tokens, metadata, "attn", piecewise) is not None


class _Layer:
    def __init__(self):
        self.calls = 0

    def __call__(self, positions, hidden_states, residual):
        self.calls += 1
        return hidden_states + 1, hidden_states if residual is None else residual


def _mono(monkeypatch, layers=3, md=None, mode=CUDAGraphMode.FULL):
    mono = K.GlmMono.__new__(K.GlmMono)
    mono._layers = [_Layer() for _ in range(layers)]
    mono._first, mono._last = 0, layers - 1
    mono._layer_name = "attn"
    mono._ops = {2: [None] * layers}
    mono._runtimes = {w: [] for w in (2, 4, 8)}
    mono._zeros = torch.zeros(8, H, dtype=torch.bfloat16)
    mono.active, mono._rows, mono._step = False, 0, {}
    launched = []

    def launch(idx, layer, state):
        launched.append(idx)
        if state.shape[0] == 1:
            state = torch.cat([state, torch.zeros_like(state)])
        return state * 2

    monkeypatch.setattr(mono, "_launch", launch)
    monkeypatch.setattr(mono, "_check_caches", lambda md: None)
    ctx = SimpleNamespace(
        attn_metadata=_metadata() if md is None else md,
        cudagraph_runtime_mode=mode,
        slot_mapping={"attn": torch.arange(8)},
    )
    md_ns = ctx.attn_metadata["attn"] if isinstance(ctx.attn_metadata, dict) else None
    if md_ns is not None:
        md_ns.paged_kv_indptr = torch.arange(9, dtype=torch.int32)
        md_ns.req_id_per_token = torch.arange(8, dtype=torch.int32)
    monkeypatch.setattr(K, "get_forward_context", lambda: ctx)
    return mono, launched


def _forward(mono, tokens):
    hidden = torch.ones(tokens, H, dtype=torch.bfloat16)
    positions = torch.zeros(tokens, dtype=torch.int64)
    residual = None
    for i, layer in enumerate(mono._layers):
        hidden, residual = K.forward_layer(mono, i, layer, positions, hidden, residual)
    return hidden, residual


@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
def test_decode_step_runs_every_layer_in_the_kernel(monkeypatch, tokens):
    mono, launched = _mono(monkeypatch)
    hidden, residual = _forward(mono, tokens)
    assert mono.active and launched == [0, 1, 2]
    assert all(layer.calls == 0 for layer in mono._layers)
    # the kernel layer returns a zero partial and the full residual stream
    rows = spec.width_for(tokens)
    assert hidden.shape == (rows, H) and not hidden.any()
    assert torch.equal(residual, torch.full((rows, H), 8.0, dtype=torch.bfloat16))
    norm = lambda x: x  # noqa: E731
    assert mono.final_norm(norm, residual).shape == (tokens, H)


@pytest.mark.parametrize(
    "case",
    [
        dict(tokens=3),
        dict(tokens=16),
        dict(tokens=4, md=_metadata(max_query_len=3)),
        dict(tokens=4, mode=CUDAGraphMode.PIECEWISE),
    ],
)
def test_other_steps_run_vllm_layers(monkeypatch, case):
    tokens = case.pop("tokens")
    mono, launched = _mono(monkeypatch, **case)
    _forward(mono, tokens)
    assert not mono.active and launched == []
    assert all(layer.calls == 1 for layer in mono._layers)


def test_switch_off_calls_the_layer():
    layer = _Layer()
    out = K.forward_layer(None, 0, layer, None, torch.ones(2, H), None)
    assert layer.calls == 1 and torch.equal(out[1], torch.ones(2, H))


def test_mono_adds_no_module_parameters():
    assert not issubclass(K.GlmMono, torch.nn.Module)


def _experts(w13_shape, w2_shape):
    return SimpleNamespace(
        experts=SimpleNamespace(
            routed_experts=SimpleNamespace(
                w13_weight=torch.zeros(w13_shape, dtype=torch.uint8),
                w2_weight=torch.zeros(w2_shape, dtype=torch.uint8),
                w13_weight_scale=torch.zeros(1, dtype=torch.uint8),
                w2_weight_scale=torch.zeros(1, dtype=torch.uint8),
            )
        ),
        gate=SimpleNamespace(
            weight=torch.zeros(4, 8), e_score_correction_bias=torch.zeros(4)
        ),
    )


CFG = SimpleNamespace(n_experts=4, num_shared_experts=1, inter=32, hidden=64)


def test_expert_views_accept_the_tp_shard():
    tensors, physical = Wt._moe_weights(_experts((5, 64, 32), (5, 64, 16)), CFG)
    assert physical == 5 and tensors["w_ug"].dtype is torch.uint8


@pytest.mark.parametrize(
    "w13, w2",
    [
        ((4, 64, 32), (4, 64, 16)),
        ((5, 128, 32), (5, 64, 16)),
        ((5, 64, 32), (5, 64, 32)),
    ],
)
def test_expert_views_reject_wrong_shapes(w13, w2):
    with pytest.raises(ValueError, match="MXFP4 bytes"):
        Wt._moe_weights(_experts(w13, w2), CFG)


def _cache_mono(kv_width=576):
    mono = K.GlmMono.__new__(K.GlmMono)
    attn = SimpleNamespace(kv_cache=torch.zeros(4, kv_width, dtype=torch.uint8))
    mono._layers = [SimpleNamespace(self_attn=attn)]
    mono._fused = [False]
    mono._first = 0
    mono._cfg = SimpleNamespace(kv_lora=512, pe_dim=64)
    mono._index_options = {}
    mono._cache_key = None
    return mono, attn


def test_moved_cache_is_rechecked_and_rebound():
    mono, attn = _cache_mono()
    md = SimpleNamespace(block_table=torch.zeros(2, 8, dtype=torch.int32))
    mono._check_caches(md)
    first = mono._cache_key
    attn.kv_cache = torch.zeros(6, 576, dtype=torch.uint8)
    mono._check_caches(md)
    assert mono._cache_key != first


def test_cache_with_another_layout_is_refused():
    mono, attn = _cache_mono(kv_width=512)
    md = SimpleNamespace(block_table=torch.zeros(2, 8, dtype=torch.int32))
    with pytest.raises(ValueError, match="KV cache rows"):
        mono._check_caches(md)


STAGES = ("qkv_a", "q_b", "split", "o", "router", "ug", "down")


def _watch(words, xflags=None, unsafe=False):
    watch = K._HealthWatch.__new__(K._HealthWatch)
    watch._stages, watch._rank = STAGES, 0
    watch._words = torch.tensor(words, dtype=torch.int32)
    watch._host = watch._words.clone()
    watch._xrank = None if xflags is None else torch.tensor(xflags, dtype=torch.int32)
    watch._host_x = None if xflags is None else watch._xrank.clone()
    watch._unsafe_warn = unsafe
    return watch


def test_expired_poll_fails_stop(monkeypatch):
    stops = []
    monkeypatch.setattr(K, "_fail_stop", lambda msg, rank: stops.append(msg))
    words = [0] * len(STAGES)
    words[-1] = 1
    _watch(words, xflags=[0, 1, 0, 0])._check()
    assert len(stops) == 1 and "down" in stops[0] and "[1]" in stops[0]


def test_clean_step_does_not_fail_stop(monkeypatch):
    stops = []
    monkeypatch.setattr(K, "_fail_stop", lambda msg, rank: stops.append(msg))
    _watch([0] * len(STAGES), xflags=[0, 0, 0, 0])._check()
    assert stops == []


def test_unsafe_warn_logs_and_clears(monkeypatch):
    stops = []
    monkeypatch.setattr(K, "_fail_stop", lambda msg, rank: stops.append(msg))
    watch = _watch([1] + [0] * (len(STAGES) - 1), xflags=[0, 0, 0, 0], unsafe=True)
    watch._check()
    assert stops == [] and not watch._words.any()
