# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.8-Flash-Next mono decode's step gate, no GPU: decode steps of at
most ``MAX_TOKENS`` rows take the kernels, except in piecewise cudagraphs."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("the mono decode is ROCm only", allow_module_level=True)

from vllm.models.qwen4_exp.amd import mono_decode as md  # noqa: E402
from vllm.models.qwen4_exp.amd.mono import layout as L  # noqa: E402

PROBE = "model.layers.0.linear_attn"


def _mono(monkeypatch, piecewise=False, **meta):
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
    monkeypatch.setattr(md, "in_piecewise_cudagraph", lambda: piecewise)
    mono = object.__new__(md.MonoDecode)
    mono.ok = True
    mono._probe = PROBE
    mono._weights_checked = True
    return mono


def _ids(s):
    return torch.zeros(s, dtype=torch.long)


@pytest.mark.parametrize("s", range(1, L.MAX_TOKENS + 1))
def test_decode_widths_take_the_kernels(monkeypatch, s):
    assert _mono(monkeypatch).eligible(_ids(s), None)


@pytest.mark.parametrize("s", range(1, L.MAX_TOKENS + 1))
def test_piecewise_steps_take_vllm(monkeypatch, s):
    # The piecewise graphs of these sizes serve short prefills too, so they
    # must be captured from the stock model.
    assert not _mono(monkeypatch, piecewise=True).eligible(_ids(s), None)
