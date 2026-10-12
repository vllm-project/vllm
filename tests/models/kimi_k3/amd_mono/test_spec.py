# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's MonoSpec on CPU: its layers and widths, its refusals (config,
loaded weights, caches) and its per-step eligibility."""

from types import SimpleNamespace

import pytest
import torch

from tests.models.common.mono.kit import fake_vllm_config
from vllm.config import CUDAGraphMode
from vllm.models.common.mono import guards
from vllm.models.common.mono.spec import MonoStep
from vllm.models.kimi_k3.amd.mono import spec as k3
from vllm.models.kimi_k3.amd.mono.spec import MONO_SPEC


def _k3_config(**over):
    cfg = {
        "speculative_config.num_speculative_tokens": 7,
        "parallel_config.decode_context_parallel_size": 8,
    }
    return fake_vllm_config(**{**cfg, **over})


@pytest.mark.parametrize("k, want", [(7, (8,)), (3, (4, 8)), (1, (2, 4, 6, 8))])
def test_widths_are_whole_requests_up_to_eight_rows(k, want):
    cfg = _k3_config(**{"speculative_config.num_speculative_tokens": k})
    assert MONO_SPEC.widths(cfg) == want


def test_regular_decode_widths_are_one_to_eight_rows():
    cfg = _k3_config(speculative_config=None)
    assert MONO_SPEC.widths(cfg) == tuple(range(1, 9))


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
    assert MONO_SPEC.layers(_k3_text_config()) == list(range(1, 93))
    kinds = k3.layer_kinds(_k3_text_config())
    assert kinds[1] == k3.WHOLE and kinds[12] == k3.WHOLE
    assert kinds[3] == k3.FFN and kinds[92] == k3.FFN


def _refuse(monkeypatch, platform=(), **over):
    monkeypatch.setattr(k3, "refuse_platform", lambda: list(platform))
    return guards.refuse_config(MONO_SPEC, _k3_config(**over))


def test_supported_config_is_accepted(monkeypatch):
    """TP8, DCP8, speculative decoding, FULL graphs."""
    assert _refuse(monkeypatch) == []


def test_regular_decode_config_is_accepted(monkeypatch):
    assert _refuse(monkeypatch, speculative_config=None) == []


@pytest.mark.parametrize(
    "over, reason",
    [
        ({"speculative_config.num_speculative_tokens": 8}, "exceed 8"),
        ({"parallel_config.tensor_parallel_size": 4}, "needs TP8"),
        ({"kv_transfer_config": object()}, "kv_connector"),
        (
            {"compilation_config.cudagraph_mode": CUDAGraphMode.PIECEWISE},
            "decode needs FULL graphs",
        ),
    ],
)
def test_refusals(monkeypatch, over, reason):
    reasons = _refuse(monkeypatch, **over)
    assert len(reasons) == 1 and reason in reasons[0], reasons


def test_the_platform_is_refused_with_the_rest(monkeypatch):
    reasons = _refuse(
        monkeypatch,
        platform=["needs gfx950 (MI355X)"],
        **{"parallel_config.tensor_parallel_size": 4},
    )
    assert reasons == ["needs gfx950 (MI355X)", "needs TP8, got TP4"]


def test_bind_names_the_cache():
    conv = torch.zeros(2, 4, 3, dtype=torch.float32)
    rstate = torch.zeros(2, 12, 128, 128, dtype=torch.bfloat16)
    reasons = MONO_SPEC.bind(None, {"layers.5.attn": (conv, rstate)})
    assert len(reasons) == 2 and all(r.startswith("layers.5.attn:") for r in reasons)


def _step(width: int, **md) -> MonoStep:
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
    return MonoStep(width, width, torch.zeros(width), {"kda": m}, CUDAGraphMode.FULL)


def _ctx(speculative: bool):
    return SimpleNamespace(
        state=SimpleNamespace(kda_prefix="kda"),
        vllm_config=SimpleNamespace(
            speculative_config=object() if speculative else None
        ),
    )


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
def test_eligible_is_pure_uniform_spec_verify(width, md, why):
    assert MONO_SPEC.eligible(_ctx(speculative=True), _step(width, **md)) == why


@pytest.mark.parametrize("width", [1, 4, 8])
def test_eligible_accepts_pure_regular_decode(width):
    state_indices = torch.arange(width, dtype=torch.int32)
    step = _step(
        width,
        spec_sequence_masks=None,
        num_decodes=width,
        num_spec_decodes=0,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=state_indices,
        uniform_spec_sequence_length=None,
    )
    assert MONO_SPEC.eligible(_ctx(speculative=False), step) is None


def test_regular_decode_falls_back_when_speculation_is_configured():
    state_indices = torch.arange(4, dtype=torch.int32)
    step = _step(
        4,
        spec_sequence_masks=None,
        num_decodes=4,
        num_spec_decodes=0,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=state_indices,
        uniform_spec_sequence_length=None,
    )
    assert MONO_SPEC.eligible(_ctx(speculative=True), step) == "not spec-verify"


@pytest.mark.parametrize(
    "width, state_indices",
    [
        (4, torch.arange(3, dtype=torch.int32)),
        (4, torch.arange(4, dtype=torch.int32).view(2, 2)),
        (4, None),
    ],
)
def test_regular_decode_requires_one_state_slot_per_row(width, state_indices):
    step = _step(
        width,
        spec_sequence_masks=None,
        num_decodes=width,
        num_spec_decodes=0,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=state_indices,
        uniform_spec_sequence_length=None,
    )
    assert MONO_SPEC.eligible(_ctx(speculative=False), step) == "not uniform"


def test_regular_decode_rejects_graph_padding():
    step = _step(
        8,
        spec_sequence_masks=None,
        num_decodes=4,
        num_spec_decodes=0,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=torch.arange(8, dtype=torch.int32),
        uniform_spec_sequence_length=None,
    )
    assert MONO_SPEC.eligible(_ctx(speculative=False), step) == "not uniform"


def test_regular_decode_k1_metadata_uses_one_accepted_token_per_row():
    from vllm.models.kimi_k3.amd.mono.kernel import _k1_metadata

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
    from vllm.models.kimi_k3.amd.mono.kernel import _k1_metadata

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


def test_eligible_needs_metadata():
    step = MonoStep(4, 4, torch.zeros(4), None, CUDAGraphMode.FULL)
    assert MONO_SPEC.eligible(_ctx(speculative=False), step) == "no metadata"


def test_weight_views_reject_every_mismatch():
    """refuse_model names each layer whose weights the kernels cannot read as
    they are, and a MoE backend that lays them out otherwise."""
    pytest.importorskip("flydsl")
    from vllm.models.kimi_k3.amd.ops.mono.stages import moe

    bad_moe = SimpleNamespace(
        use_latent_moe=True,
        shared_experts=object(),
        routed_scaling_factor=2.0,
        experts=SimpleNamespace(quant_method=SimpleNamespace()),
    )
    no_latent = SimpleNamespace(use_latent_moe=False, shared_experts=None)
    layers = [None] + [SimpleNamespace(mlp=no_latent)] * 92
    layers[5] = SimpleNamespace(mlp=bad_moe)
    model = SimpleNamespace(config=_k3_text_config(), layers=layers)
    reasons = k3.refuse_model(model)
    assert "layer 1: needs a latent MoE with shared experts" in reasons
    assert "layer 5: routed_scaling_factor 2.0" in reasons
    assert any(r.startswith("layer 5: MoE backend must be") for r in reasons)
    assert moe.E == 896
