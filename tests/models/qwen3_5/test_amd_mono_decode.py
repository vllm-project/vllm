# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.8 mono decode's host side, no GPU: deployments it cannot run are
refused with every reason at once, and only pure decode steps of at most
``MAX_TOKENS`` rows take the kernels."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("the mono decode is ROCm only", allow_module_level=True)

import vllm.platforms.rocm as rocm  # noqa: E402
from vllm.models.qwen3_5.amd import mono_decode as md  # noqa: E402
from vllm.models.qwen3_5.amd.mono import layout as L  # noqa: E402

PROBE = "model.layers.0.linear_attn"


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


@pytest.fixture
def deployment(monkeypatch):
    """A supported deployment; tests override one part of it."""
    d = SimpleNamespace(cdna=4, tp=L.TP, pp=1, cus=256)
    monkeypatch.setattr(rocm, "get_cdna_version", lambda: d.cdna)
    monkeypatch.setattr(md, "get_tp_group", lambda: SimpleNamespace(world_size=d.tp))
    monkeypatch.setattr(md, "get_pp_group", lambda: SimpleNamespace(world_size=d.pp))
    monkeypatch.setattr(
        md.current_platform, "num_compute_units", lambda device_id=0: d.cus
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    return d


def test_supported(deployment):
    assert md._refusals(_config()) == []


def test_every_reason_at_once(deployment):
    deployment.cdna, deployment.tp, deployment.pp, deployment.cus = 3, 4, 2, 240
    why = md._refusals(
        _config(
            parallel=dict(enable_expert_parallel=True, decode_context_parallel_size=2),
            lora_config=object(),
            speculative_config=object(),
            kv_transfer_config=object(),
        )
    )
    assert len(why) == 9, why
    for part in (
        "CDNA4",
        "tensor parallel size 8",
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
def test_parallelism_refused(deployment, parallel):
    assert len(md._refusals(_config(parallel=parallel))) == 1


def _mono(monkeypatch, **meta):
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
        md,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={PROBE: SimpleNamespace(**m)}),
    )
    mono = object.__new__(md.MonoDecode)
    mono.ok = True
    mono._probe = PROBE
    mono._weights_checked = True
    mono.model = SimpleNamespace(aux_hidden_state_layers=())
    return mono


def _ids(s):
    return torch.zeros(s, dtype=torch.long)


@pytest.mark.parametrize("s", range(1, L.MAX_TOKENS + 1))
def test_decode_widths_take_the_kernels(monkeypatch, s):
    assert _mono(monkeypatch).eligible(_ids(s), None, None, None)


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
def test_other_steps_take_vllm(monkeypatch, meta):
    assert not _mono(monkeypatch, **meta).eligible(_ids(4), None, None, None)


def test_over_width_takes_vllm(monkeypatch):
    mono = _mono(monkeypatch)
    assert not mono.eligible(_ids(L.MAX_TOKENS + 1), None, None, None)
    assert not mono.eligible(_ids(4), None, None, torch.zeros(4, L.HIDDEN))


def test_no_metadata_takes_vllm(monkeypatch):
    mono = _mono(monkeypatch)
    monkeypatch.setattr(
        md, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    )
    assert not mono.eligible(_ids(4), None, None, None)
