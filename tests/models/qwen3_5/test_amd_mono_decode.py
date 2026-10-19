# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen mono decodes' host side, no GPU:
deployments they cannot run are refused with every reason at once, and only
pure decode steps (with MTP, verify steps) of at most ``MAX_TOKENS`` rows take
the kernels, never a piecewise cudagraph step."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("the mono decode is ROCm only", allow_module_level=True)

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_attn  # noqa: E402
import vllm.platforms.rocm as rocm  # noqa: E402
from vllm.models.qwen3_5.amd import mono_base as B  # noqa: E402
from vllm.models.qwen3_5.amd import mono_decode as qwen38  # noqa: E402

PROBE = "model.layers.0.linear_attn"
MODELS = {"qwen38": qwen38.MonoDecode}


@pytest.fixture(params=list(MODELS))
def cls(request):
    return MODELS[request.param]


def _config(**kw):
    pc = dict(
        enable_expert_parallel=False,
        enable_eplb=False,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    pc.update(kw.pop("parallel", {}))
    c = dict(lora_config=None, speculative_config=None, kv_transfer_config=None)
    c.update(kw)
    return SimpleNamespace(parallel_config=SimpleNamespace(**pc), **c)


def _mtp(n=3):
    return SimpleNamespace(method="mtp", num_speculative_tokens=n)


@pytest.fixture
def deployment(monkeypatch, cls):
    """A supported deployment; tests override one part of it."""
    d = SimpleNamespace(cdna=4, tp=cls.TP_SIZES[0], pp=1, cus=256)
    monkeypatch.setattr(rocm, "get_cdna_version", lambda: d.cdna)
    monkeypatch.setattr(B, "get_tp_group", lambda: SimpleNamespace(world_size=d.tp))
    monkeypatch.setattr(B, "get_pp_group", lambda: SimpleNamespace(world_size=d.pp))
    monkeypatch.setattr(
        B.current_platform, "num_compute_units", lambda device_id=0: d.cus
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "1")
    monkeypatch.setattr(gdn_attn, "GDN_AITER_TRITON_AVAILABLE", True)
    return d


def _refusals(cls, config):
    return object.__new__(cls).refusals(config)


def test_supported(deployment, cls):
    assert _refusals(cls, _config()) == []


def test_every_reason_at_once(deployment, cls):
    deployment.cdna, deployment.tp, deployment.pp, deployment.cus = 3, 4, 2, 240
    why = _refusals(
        cls,
        _config(
            parallel=dict(enable_expert_parallel=True, decode_context_parallel_size=2),
            lora_config=object(),
            speculative_config=SimpleNamespace(method="eagle3"),
            kv_transfer_config=object(),
        ),
    )
    assert len(why) == 9, why
    tp = "/".join(map(str, cls.TP_SIZES))
    for part in (
        "CDNA4",
        f"tensor parallel size {tp}",
        "pipeline",
        "expert or data",
        "context",
        "LoRA",
        "speculative",
        "KV connector",
        "240",
    ):
        assert any(part in w for w in why), (part, why)


@pytest.mark.parametrize(
    "parallel",
    [
        dict(enable_eplb=True),
        dict(data_parallel_size=2),
        dict(prefill_context_parallel_size=2),
    ],
)
def test_parallelism_refused(deployment, cls, parallel):
    assert len(_refusals(cls, _config(parallel=parallel))) == 1


def test_mtp(deployment, cls):
    why = _refusals(cls, _config(speculative_config=_mtp()))
    assert why == ([] if cls.MTP else ["needs no speculative decoding"])
    if cls.MTP:
        many = _mtp(cls.MAX_TOKENS)
        assert len(_refusals(cls, _config(speculative_config=many))) == 1


def _mono(monkeypatch, cls, piecewise=False, spec=False, **meta):
    """A MonoDecode past its checks, and a step's GDN metadata."""
    m = dict(
        spec_sequence_masks=None,
        num_prefills=0,
        num_spec_decodes=0,
        num_decodes=4,
        non_spec_state_indices_tensor=torch.arange(8, dtype=torch.int32),
    )
    m.update(meta)
    monkeypatch.setattr(
        B,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={PROBE: SimpleNamespace(**m)}),
    )
    monkeypatch.setattr(B, "in_piecewise_cudagraph", lambda: piecewise)
    mono = object.__new__(cls)
    mono.ok = True
    mono.spec = spec
    mono._probe = PROBE
    mono._weights_checked = True
    mono.model = SimpleNamespace(aux_hidden_state_layers=())
    return mono


def _ids(s):
    return torch.zeros(s, dtype=torch.long)


def _eligible(mono, s, **kw):
    if isinstance(mono, qwen38.MonoDecode):
        return mono.eligible(_ids(s), None, kw.get("inputs_embeds"))
    return mono.eligible(_ids(s), None)


@pytest.mark.parametrize("s", range(1, 9))
def test_decode_widths_take_the_kernels(monkeypatch, cls, s):
    assert _eligible(_mono(monkeypatch, cls), s)


@pytest.mark.parametrize("s", range(1, 9))
def test_piecewise_steps_take_vllm(monkeypatch, cls, s):
    # The piecewise graphs of these sizes serve short prefills too, so they
    # must be captured from the stock model.
    assert not _eligible(_mono(monkeypatch, cls, piecewise=True), s)


@pytest.mark.parametrize(
    "meta",
    [
        dict(num_prefills=1),
        dict(num_spec_decodes=1),
        dict(num_decodes=0),
        dict(spec_sequence_masks=torch.ones(1)),
        dict(non_spec_state_indices_tensor=None),
        dict(non_spec_state_indices_tensor=torch.arange(8)),  # int64
        dict(non_spec_state_indices_tensor=torch.arange(2, dtype=torch.int32)),
    ],
)
def test_other_steps_take_vllm(monkeypatch, cls, meta):
    assert not _eligible(_mono(monkeypatch, cls, **meta), 4)


@pytest.mark.parametrize(
    "meta, ok",
    [
        (dict(num_decodes=0, num_spec_decodes=2), True),
        (dict(num_decodes=1, num_spec_decodes=1), True),
        (dict(num_decodes=0, num_spec_decodes=0), False),
        (dict(num_prefills=1, num_spec_decodes=1), False),
    ],
)
def test_verify_steps(monkeypatch, cls, meta, ok):
    mono = _mono(monkeypatch, cls, spec=True, spec_sequence_masks=1, **meta)
    assert _eligible(mono, 8) == ok


def test_over_width_takes_vllm(monkeypatch, cls):
    assert not _eligible(_mono(monkeypatch, cls), cls.MAX_TOKENS + 1)


def test_qwen38_embeds_take_vllm(monkeypatch):
    mono = _mono(monkeypatch, qwen38.MonoDecode)
    assert not _eligible(mono, 4, inputs_embeds=torch.zeros(4, 8))


def test_no_metadata_takes_vllm(monkeypatch, cls):
    mono = _mono(monkeypatch, cls)
    monkeypatch.setattr(
        B, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    )
    assert not _eligible(mono, 4)
