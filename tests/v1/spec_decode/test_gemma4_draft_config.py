# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Gemma4 MTP draft must not be swapped into the target's VllmConfig.

Building a VllmConfig whose model_config is the dense draft while
parallel_config is the target's re-runs verify_with_parallel_config on that
mismatched pair. Under --enable-expert-parallel it rejects the expert-less
draft with "Number of experts in the model must be greater than 0", so a MoE
Gemma4 target cannot be served with MTP and expert parallelism together.
"""

from dataclasses import dataclass

from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator


@dataclass
class _AttentionConfig:
    backend: str | None = None


@dataclass
class _SpeculativeConfig:
    draft_model_config: object = None


@dataclass
class _VllmConfig:
    model_config: object
    attention_config: _AttentionConfig
    speculative_config: _SpeculativeConfig


def _speculator(target_backend: str | None) -> Gemma4Speculator:
    """A speculator holding only what _create_draft_vllm_config reads.

    __new__ sidesteps BaseSpeculator.__init__, which allocates device buffers.
    """
    spec = Gemma4Speculator.__new__(Gemma4Speculator)
    spec.vllm_config = _VllmConfig(
        model_config="target-model-config",
        attention_config=_AttentionConfig(backend=target_backend),
        speculative_config=_SpeculativeConfig(
            draft_model_config="draft-model-config",
        ),
    )
    spec.speculative_config = spec.vllm_config.speculative_config
    return spec


def test_draft_config_keeps_the_target_model_config():
    """The draft must never land in the VllmConfig handed to get_model().

    get_model() takes the draft model_config as an explicit argument and
    Gemma4MTPModel reads its own config from speculative_config, so swapping
    it in here buys nothing and breaks --enable-expert-parallel.
    """
    draft_vllm_config = _speculator("TRITON_ATTN")._create_draft_vllm_config()
    assert draft_vllm_config.model_config == "target-model-config"


def test_target_attention_backend_is_carried_through():
    """The method's actual purpose: draft layers keep the target's backend."""
    draft_vllm_config = _speculator("TRITON_ATTN")._create_draft_vllm_config()
    assert draft_vllm_config.attention_config.backend == "TRITON_ATTN"


def test_no_target_backend_leaves_the_backend_unset():
    draft_vllm_config = _speculator(None)._create_draft_vllm_config()
    assert draft_vllm_config.attention_config.backend is None


def test_carrying_the_backend_does_not_mutate_the_target():
    """The target must keep its own attention config after the draft is built."""
    spec = _speculator("TRITON_ATTN")
    target_attention_config = spec.vllm_config.attention_config
    spec._create_draft_vllm_config()
    assert spec.vllm_config.attention_config is target_attention_config
