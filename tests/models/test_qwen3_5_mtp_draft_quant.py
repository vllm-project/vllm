# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression tests for the GPTQ dynamic-MTP draft exclusion.

MTP-preserved GPTQ checkpoints keep the ``mtp.*`` draft weights unquantized
and advertise that via ``quantization_config.dynamic`` entries keyed with a
``-:`` (negative) pattern matching ``mtp``. ``Qwen3_5MultiTokenPredictor``
detects that and must build the draft layers with ``quant_config is None``;
otherwise the BF16 ``mtp.*`` weights are loaded against GPTQ-packed params
and checkpoint loading fails at serve time. These tests lock that contract —
and its boundaries — with no GPU and no weights.
"""

import types

import torch
from transformers import PretrainedConfig

import vllm.model_executor.models.qwen3_5_mtp as qwen3_5_mtp_module
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor

import pytest


class _FakeQuantConfig:
    """Minimal stand-in for a quantization config (only ``get_name`` is used
    by the dynamic-exclusion branch)."""

    def __init__(self, name: str = "gptq"):
        """Store the quantization method name reported by ``get_name``."""
        self._name = name

    def get_name(self) -> str:
        """Return the quantization method name given at construction."""
        return self._name


def _text_config(**overrides) -> PretrainedConfig:
    """Build a small Qwen3.5-MTP text ``PretrainedConfig``; keyword
    arguments override the defaults."""
    cfg = dict(
        model_type="qwen3_5_moe_text",
        architectures=["Qwen3_5MoeMTP"],
        num_hidden_layers=8,
        mtp_num_hidden_layers=1,
        hidden_size=32,
        vocab_size=128,
        rms_norm_eps=1e-6,
    )
    cfg.update(overrides)
    return PretrainedConfig(**cfg)


def _build_draft(monkeypatch: pytest.MonkeyPatch, dynamic):
    """Run the real Qwen3_5MultiTokenPredictor constructor on CPU with the
    dist/kernel-heavy module members mocked out; return
    (vllm_config, quant_cfg, quant_config_seen_per_draft_layer).

    ``quant_config_seen_per_draft_layer`` records ``vllm_config.quant_config``
    *at draft-layer construction time*: the constructor restores
    ``vllm_config.quant_config`` after building the draft layers, so any
    post-construction inspection would hide the bypass.
    """
    quant_seen: list[object] = []

    class FakeDecoderLayer(torch.nn.Module):
        def __init__(self, vllm_config, **kwargs):
            """Stand-in draft layer that records the ``quant_config`` it is
            constructed with."""
            super().__init__()
            quant_seen.append(getattr(vllm_config, "quant_config"))

    hf_config = _text_config()
    if dynamic is not None:
        hf_config.quantization_config = {"quant_method": "gptq", "dynamic": dynamic}

    quant_cfg = _FakeQuantConfig("gptq")
    from vllm.config import CompilationMode

    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_text_config=_text_config(), hf_config=hf_config
        ),
        quant_config=quant_cfg,
        # @support_torch_compile reads compilation_config.mode in its __init__
        # wrapper; NONE => do_not_compile, no dynamo machinery.
        compilation_config=types.SimpleNamespace(mode=CompilationMode.NONE),
    )

    monkeypatch.setattr(qwen3_5_mtp_module, "Qwen3_5DecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(
        qwen3_5_mtp_module, "VocabParallelEmbedding", lambda *a, **k: torch.nn.Module()
    )
    monkeypatch.setattr(
        qwen3_5_mtp_module, "ColumnParallelLinear", lambda *a, **k: torch.nn.Module()
    )
    monkeypatch.setattr(
        qwen3_5_mtp_module,
        "is_model_fused_shared_expert_compatible",
        lambda *a, **k: False,
    )
    # Qwen3_5RMSNorm is a CustomOp whose ctor reads get_current_vllm_config();
    # the contract is about the quant_config threaded into the *decoder
    # layers*, so the norms are stubbed like the other heavy classes.
    monkeypatch.setattr(
        qwen3_5_mtp_module, "Qwen3_5RMSNorm", lambda *a, **k: torch.nn.Module()
    )

    Qwen3_5MultiTokenPredictor(vllm_config=vllm_config, prefix="mtp")
    assert quant_seen, "no draft layers were constructed"
    return vllm_config, quant_cfg, quant_seen


def test_dynamic_mtp_exclusion_builds_draft_unquantized(monkeypatch):
    """A `-:`-prefixed dynamic pattern mentioning mtp => draft layers see
    quant_config is None; the original config is restored afterwards (no
    leak to sibling modules)."""
    vllm_config, quant_cfg, quant_seen = _build_draft(
        monkeypatch, dynamic={"-:.*mtp.*": {}}
    )
    assert all(q is None for q in quant_seen), (
        "MTP draft layers must be built unquantized when the checkpoint "
        "declares a -:...mtp... dynamic exclusion"
    )
    assert vllm_config.quant_config is quant_cfg, (
        "quant_config must be restored after building the draft layers"
    )


def test_no_quantization_config_keeps_draft_quantized(monkeypatch):
    """Without any checkpoint ``quantization_config`` the unquantized-draft
    bypass must not trigger: every draft layer is built with the model's
    quantization config, and ``vllm_config`` keeps it."""
    vllm_config, quant_cfg, quant_seen = _build_draft(monkeypatch, dynamic=None)
    assert all(q is quant_cfg for q in quant_seen)
    assert vllm_config.quant_config is quant_cfg


@pytest.mark.parametrize(
    "dynamic",
    [
        # No `-:` prefix: a positive pattern must not disable quantization.
        {"mtp:.*": {}},
        # Negative but not mtp-related: must not disable quantization either.
        {"-:.*lm_head.*": {}},
        # Empty dynamic: must not disable quantization.
        {},
    ],
)
def test_non_mtp_negative_patterns_keep_draft_quantized(monkeypatch, dynamic):
    """``dynamic`` entries that are not ``-:``-prefixed mtp exclusions (a
    positive mtp pattern, a non-mtp negative pattern, an empty dict) must
    not bypass quantization."""
    _, _, quant_seen = _build_draft(monkeypatch, dynamic=dynamic)
    assert all(isinstance(q, _FakeQuantConfig) for q in quant_seen), (
        "only `-:`-prefixed patterns mentioning mtp may bypass quantization"
    )
