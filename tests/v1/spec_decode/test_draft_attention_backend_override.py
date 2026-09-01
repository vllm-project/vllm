# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft must honour ``attention_backend`` from --speculative-config.

The V1 proposer sets the draft's attention backend from the speculative config
and never inherits the target's, because draft and target attention shapes
differ and not every backend supports both. The V2 base speculator returned the
target's config unchanged, so the setting was silently dropped for every
speculator family that does not override the property itself.

The property is exercised through ``fget`` on a minimal holder: constructing a
real speculator would require a GPU, a loaded model and a KV cache, none of
which this contract depends on.
"""

from dataclasses import dataclass

from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


@dataclass
class _AttentionConfig:
    backend: str | None = None


@dataclass
class _SpeculativeConfig:
    attention_backend: str | None = None


@dataclass
class _VllmConfig:
    attention_config: _AttentionConfig
    speculative_config: _SpeculativeConfig


@dataclass
class _Holder:
    """Only the two attributes ``attn_vllm_config`` actually reads."""

    vllm_config: _VllmConfig
    speculative_config: _SpeculativeConfig


def _holder(target_backend: str, draft_backend: str | None) -> _Holder:
    spec = _SpeculativeConfig(attention_backend=draft_backend)
    return _Holder(
        vllm_config=_VllmConfig(
            attention_config=_AttentionConfig(backend=target_backend),
            speculative_config=spec,
        ),
        speculative_config=spec,
    )


def _draft_config(holder: _Holder) -> _VllmConfig:
    return DraftModelSpeculator.attn_vllm_config.fget(holder)


def test_draft_attention_backend_overrides_the_target():
    """An explicit draft backend must reach the draft's attention config."""
    assert _draft_config(
        _holder("FLASHINFER", "TRITON_ATTN")
    ).attention_config.backend == ("TRITON_ATTN")


def test_draft_inherits_target_when_unset():
    """No draft backend means unchanged behaviour: the target's own config."""
    h = _holder("FLASHINFER", None)
    assert _draft_config(h) is h.vllm_config


def test_override_does_not_mutate_the_target_config():
    """The target keeps its own backend after the draft config is derived."""
    h = _holder("FLASHINFER", "TRITON_ATTN")
    _draft_config(h)
    assert h.vllm_config.attention_config.backend == "FLASHINFER"
