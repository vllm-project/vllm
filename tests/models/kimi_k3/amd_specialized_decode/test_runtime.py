# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's specialized decode on CPU: the switch, its widths and layers, its
refusals (config, loaded weights, caches) and plan() routing."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.models.kimi_k3.amd.specialized_decode import runtime
from vllm.models.kimi_k3.amd.specialized_decode.runtime import (
    FFN,
    WHOLE,
    DecodeConfig,
    layer_kinds,
    specialized_decode_config,
)
from vllm.models.kimi_k3.amd.specialized_decode.runtime import decode as decode_mod
from vllm.models.kimi_k3.amd.specialized_decode.runtime.decode import (
    SpecializedDecode,
    step_reason,
)
from vllm.models.kimi_k3.amd.specialized_decode.runtime.guards import (
    cache_reasons,
    weight_reasons,
)


def _vllm_config(**over) -> SimpleNamespace:
    """A supported config: TP8, DCP8, 7 speculative tokens, FULL graphs, MRV2.
    ``over`` sets dotted paths, e.g. ``{"parallel_config.data_parallel_size": 2}``."""
    spec = SimpleNamespace(
        num_speculative_tokens=7,
        enable_adaptive_verification=False,
        dynamic=False,
    )
    spec.uses_dynamic_speculative_decoding = lambda: spec.dynamic
    c = SimpleNamespace(
        additional_config={"decode_backend": "latency_specialized"},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            decode_context_parallel_size=8,
            prefill_context_parallel_size=1,
            use_ubatching=False,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enable_sleep_mode=False,
            hf_text_config=SimpleNamespace(
                hidden_size=7168,
                num_experts=896,
                attn_res_block_size=12,
                linear_attn_config={
                    "num_heads": 96,
                    "head_dim": 128,
                    "short_conv_kernel_size": 4,
                },
            ),
        ),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto", use_replayssm=False, use_kda_recoverssm=False
        ),
        lora_config=None,
        kv_transfer_config=None,
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        weight_transfer_config=None,
        speculative_config=spec,
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
    )
    for path, v in over.items():
        obj = c
        *head, last = path.split(".")
        for h in head:
            obj = getattr(obj, h)
        setattr(obj, last, v)
    return c


@pytest.fixture
def on_mi355x(monkeypatch):
    monkeypatch.setattr(runtime, "platform_reasons", lambda: [])


@pytest.mark.parametrize("k, want", [(7, (8,)), (3, (4, 8)), (1, (2, 4, 6, 8))])
def test_widths_are_whole_requests_up_to_eight_rows(k, want):
    assert DecodeConfig(q=k + 1).widths == want


def test_regular_decode_widths_are_one_to_eight_rows():
    assert DecodeConfig(q=1).widths == tuple(range(1, 9))


def _k3_text_config():
    """Kimi-K3's layer pattern: 93 layers, MLA every 4th, AttnRes blocks of 12,
    layer 0 dense."""
    mla = {i for i in range(93) if i % 4 == 3} | {92}
    return SimpleNamespace(
        num_hidden_layers=93,
        attn_res_block_size=12,
        num_experts=896,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        is_kda_layer=lambda i: i not in mla,
    )


def test_kernel_layers_follow_the_k3_pattern():
    kinds = layer_kinds(_k3_text_config())
    assert list(kinds) == list(range(1, 93))
    assert kinds[1] == WHOLE and kinds[12] == WHOLE
    assert kinds[3] == FFN and kinds[92] == FFN


def test_auto_runs_the_stock_decode():
    """Without the switch nothing is checked, even on an unsupported deployment."""
    cfg = _vllm_config(
        additional_config={}, **{"parallel_config.tensor_parallel_size": 4}
    )
    assert specialized_decode_config(cfg) is None
    cfg.additional_config = {"decode_backend": "auto"}
    assert specialized_decode_config(cfg) is None


def test_an_unknown_backend_is_refused():
    cfg = _vllm_config(additional_config={"decode_backend": "fast"})
    with pytest.raises(ValueError, match="decode_backend must be"):
        specialized_decode_config(cfg)


def test_supported_config_is_accepted(on_mi355x):
    assert specialized_decode_config(_vllm_config()) == DecodeConfig(q=8)


def test_regular_decode_config_is_accepted(on_mi355x):
    cfg = _vllm_config(speculative_config=None)
    assert specialized_decode_config(cfg) == DecodeConfig(q=1)


@pytest.mark.parametrize(
    "over, reason",
    [
        (
            {
                "parallel_config.tensor_parallel_size": 4,
                "parallel_config.decode_context_parallel_size": 4,
            },
            "TP 4",
        ),
        ({"parallel_config.pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"parallel_config.decode_context_parallel_size": 2}, "decode context"),
        ({"kv_transfer_config": object()}, "KV transfer connectors"),
        ({"speculative_config.num_speculative_tokens": 8}, "more than 7"),
        ({"speculative_config.enable_adaptive_verification": True}, "adaptive"),
        ({"speculative_config.dynamic": True}, "dynamic speculative"),
        ({"cache_config.mamba_cache_dtype": "float32"}, "KDA cache dtypes"),
        ({"cache_config.use_replayssm": True}, "ReplaySSM"),
        ({"use_v2_model_runner": False}, "V1 model runner"),
        (
            {"compilation_config.mode": CompilationMode.STOCK_TORCH_COMPILE},
            "torch.compile",
        ),
        ({"model_config.hf_text_config.hidden_size": 4096}, "model shapes"),
    ],
)
def test_refusals(on_mi355x, over, reason):
    reasons = runtime.unsupported(_vllm_config(**over))
    assert len(reasons) == 1 and reason in reasons[0], reasons


def test_every_refusal_is_reported_at_once(monkeypatch):
    monkeypatch.setattr(
        runtime, "platform_reasons", lambda: ["this GPU (gfx950 / MI355X only)"]
    )
    cfg = _vllm_config(
        **{"parallel_config.tensor_parallel_size": 4, "use_v2_model_runner": False}
    )
    with pytest.raises(ValueError) as e:
        specialized_decode_config(cfg)
    for reason in ("TP 4", "V1 model runner", "this GPU", "decode_backend=auto"):
        assert reason in str(e.value)


def test_piecewise_graphs_only_warn(on_mi355x):
    """They are not refused: plan() runs every PIECEWISE step stock."""
    cfg = _vllm_config(**{"compilation_config.cudagraph_mode": CUDAGraphMode.PIECEWISE})
    assert runtime.piecewise_graphs_only(cfg)
    assert specialized_decode_config(cfg) is not None


def test_cache_refusal_names_the_cache():
    pytest.importorskip("flydsl")
    conv = torch.zeros(2, 4, 3, dtype=torch.float32)
    rstate = torch.zeros(2, 12, 128, 128, dtype=torch.bfloat16)
    reasons = cache_reasons({"layers.5.attn": (conv, rstate)})
    assert len(reasons) == 2 and all(r.startswith("layers.5.attn:") for r in reasons)


def _kda_metadata(**md):
    """One request of 4 tokens unless overridden."""
    m = SimpleNamespace(
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=1,
        spec_state_indices_tensor=torch.zeros(1, 4, dtype=torch.int32),
        non_spec_state_indices_tensor=None,
        uniform_spec_sequence_length=4,
    )
    for k, v in md.items():
        setattr(m, k, v)
    return {"kda": m}


@pytest.mark.parametrize(
    "width, md, why",
    [
        (4, {}, None),
        (8, {"num_spec_decodes": 2}, None),
        (4, {"num_prefills": 1}, "not pure decode"),
        (4, {"num_decodes": 1}, "not pure decode"),
        (4, {"spec_sequence_masks": None}, "not pure decode"),
        (8, {}, "not uniform"),
        (4, {"uniform_spec_sequence_length": None}, "not uniform"),
    ],
)
def test_steps_must_be_pure_uniform_spec_verify(width, md, why):
    assert step_reason(_kda_metadata(**md), "kda", width, q=4) == why


@pytest.mark.parametrize("width", [1, 4, 8])
def test_steps_accept_pure_regular_decode(width):
    md = _kda_metadata(
        spec_sequence_masks=None,
        num_decodes=width,
        num_spec_decodes=0,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=torch.arange(width, dtype=torch.int32),
        uniform_spec_sequence_length=None,
    )
    assert step_reason(md, "kda", width, q=1) is None


def test_regular_decode_falls_back_when_speculation_is_configured():
    md = _kda_metadata(
        spec_sequence_masks=None,
        num_decodes=4,
        num_spec_decodes=0,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=torch.arange(4, dtype=torch.int32),
        uniform_spec_sequence_length=None,
    )
    assert step_reason(md, "kda", 4, q=4) == "not spec-verify"


@pytest.mark.parametrize(
    "width, state_indices",
    [
        (4, torch.arange(3, dtype=torch.int32)),
        (4, torch.arange(4, dtype=torch.int32).view(2, 2)),
        (4, None),
    ],
)
def test_regular_decode_requires_one_state_slot_per_row(width, state_indices):
    md = _kda_metadata(
        spec_sequence_masks=None,
        num_decodes=width,
        num_spec_decodes=0,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=state_indices,
        uniform_spec_sequence_length=None,
    )
    assert step_reason(md, "kda", width, q=1) == "not uniform"


def test_regular_decode_rejects_graph_padding():
    md = _kda_metadata(
        spec_sequence_masks=None,
        num_decodes=4,
        num_spec_decodes=0,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=torch.arange(8, dtype=torch.int32),
        uniform_spec_sequence_length=None,
    )
    assert step_reason(md, "kda", 8, q=1) == "not uniform"


def test_regular_decode_k1_metadata_uses_one_accepted_token_per_row():
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.layers import _k1_metadata

    state_indices = torch.tensor([7, 11, 13, 17], dtype=torch.int32)
    accepted = torch.ones(8, dtype=torch.int32)
    m = SimpleNamespace(
        num_spec_decodes=0,
        non_spec_state_indices_tensor=state_indices,
    )
    qlen, state_view, num_accepted = _k1_metadata(
        m, SimpleNamespace(regular_num_accepted=accepted)
    )
    assert qlen == 1
    assert state_view.shape == (4, 1)
    assert state_view[:, 0].tolist() == state_indices.tolist()
    assert num_accepted.data_ptr() == accepted.data_ptr()
    assert num_accepted.tolist() == [1, 1, 1, 1]


def test_spec_decode_k1_metadata_is_unchanged():
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.layers import _k1_metadata

    state_indices = torch.arange(8, dtype=torch.int32).view(2, 4)
    accepted = torch.tensor([2, 4], dtype=torch.int32)
    m = SimpleNamespace(
        num_spec_decodes=2,
        spec_state_indices_tensor=state_indices,
        num_accepted_tokens=accepted,
    )
    qlen, state_view, num_accepted = _k1_metadata(m, SimpleNamespace())
    assert qlen == 4
    assert state_view is state_indices
    assert num_accepted is accepted


def test_steps_need_metadata():
    assert step_reason(None, "kda", 4, q=4) == "no metadata"


class _Kernel:
    def launch(self, step, positions, hidden_states, residual, prefix_delta):
        return "kernel", step.width


def _decode(monkeypatch, mode=CUDAGraphMode.FULL, md=None, q=4):
    """An executor with kernels on layers 1 and 2 at widths 4 and 8, bound to a
    KV cache, under a forward context of ``mode``."""
    decode = object.__new__(SpecializedDecode)
    decode.config = DecodeConfig(q=q)
    decode.runner = SimpleNamespace(kda_prefix="kda")
    decode.kernels = {w: {1: _Kernel(), 2: _Kernel()} for w in (4, 8)}
    decode._kv = (1, 2)
    monkeypatch.setattr(SpecializedDecode, "_kv_key", lambda self: (1, 2))
    context = SimpleNamespace(
        cudagraph_runtime_mode=mode,
        attn_metadata=_kda_metadata() if md is None else md,
    )
    monkeypatch.setattr(decode_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(decode_mod, "get_forward_context", lambda: context)
    return decode


def test_plan_runs_the_kernels_of_its_width(monkeypatch):
    step = _decode(monkeypatch).plan(torch.zeros(4), None)
    assert step is not None and step.width == 4
    assert step.layer(1)(None, None, None, None) == ("kernel", 4)
    assert step.layer(0) is None and step.layer(3) is None


def test_plan_runs_regular_decode(monkeypatch):
    md = _kda_metadata(
        spec_sequence_masks=None,
        num_decodes=4,
        num_spec_decodes=0,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=torch.arange(4, dtype=torch.int32),
        uniform_spec_sequence_length=None,
    )
    step = _decode(monkeypatch, md=md, q=1).plan(torch.zeros(4), None)
    assert step is not None and step.width == 4


@pytest.mark.parametrize(
    "mode, tokens, md, intermediate",
    [
        (CUDAGraphMode.PIECEWISE, 4, None, None),
        (CUDAGraphMode.FULL, 16, None, None),  # no kernels that wide
        (CUDAGraphMode.FULL, 4, _kda_metadata(num_prefills=1), None),
        (CUDAGraphMode.FULL, 4, None, object()),  # a pipeline stage's input
    ],
)
def test_plan_runs_other_steps_stock(monkeypatch, mode, tokens, md, intermediate):
    decode = _decode(monkeypatch, mode, md)
    assert decode.plan(torch.zeros(tokens), intermediate) is None


def test_plan_runs_stock_before_a_kv_cache_is_bound(monkeypatch):
    decode = _decode(monkeypatch)
    decode._kv = None
    assert decode.plan(torch.zeros(4), None) is None


def test_plan_refuses_a_kv_cache_bound_behind_its_back(monkeypatch):
    decode = _decode(monkeypatch)
    monkeypatch.setattr(SpecializedDecode, "_kv_key", lambda self: (3, 4))
    with pytest.raises(RuntimeError, match="without prepare_kv"):
        decode.plan(torch.zeros(4), None)


def test_weight_views_reject_every_mismatch():
    """weight_reasons names each layer whose weights the kernels cannot read as
    they are, and a MoE backend that lays them out otherwise."""
    pytest.importorskip("flydsl")
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import moe

    bad_moe = SimpleNamespace(
        use_latent_moe=True,
        shared_experts=object(),
        routed_scaling_factor=2.0,
        experts=SimpleNamespace(quant_method=SimpleNamespace()),
    )
    no_latent = SimpleNamespace(use_latent_moe=False, shared_experts=None)
    layers = [None] + [SimpleNamespace(mlp=no_latent)] * 92
    layers[5] = SimpleNamespace(mlp=bad_moe)
    model = SimpleNamespace(
        config=_k3_text_config(), layers=layers, start_layer=0, end_layer=93
    )
    reasons = weight_reasons(model)
    assert "layer 1: needs a latent MoE with shared experts" in reasons
    assert "layer 5: routed_scaling_factor 2.0" in reasons
    assert any(r.startswith("layer 5: MoE backend must be") for r in reasons)
    assert moe.E == 896
