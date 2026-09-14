# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CP impl checks must apply to the target model's layers only.

The draft model's layers share the static forward context but a draft
collapses PCP and DCP, so its KV is never CP-sharded.
"""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility


class _Backend:
    @staticmethod
    def supports_pcp() -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "FAKE"


class _CPCapableImpl:
    supports_mtp_with_cp_non_trivial_interleave_size = True
    need_to_return_lse_for_decode = True


class _PlainImpl:
    """A full-attention draft impl: declares nothing about CP."""

    supports_mtp_with_cp_non_trivial_interleave_size = False
    need_to_return_lse_for_decode = False


class _Layer(AttentionLayerBase):
    def __init__(self, impl):
        self.impl = impl

    def get_attn_backend(self):
        return _Backend

    def get_kv_cache_spec(self, vllm_config):
        return None


def _config(layers, *, pcp=8, dcp=8, interleave=64):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=interleave,
        ),
        speculative_config=SimpleNamespace(method="dspark"),
        compilation_config=SimpleNamespace(static_forward_context=layers),
    )


def test_draft_layers_are_not_held_to_target_interleave():
    layers = {
        "model.layers.0.attn": _Layer(_CPCapableImpl()),
        "draft.layers.0.attn": _Layer(_PlainImpl()),
    }
    # The caller scopes the checks to the target's layers; the draft's
    # full-attention impl must not trip the interleave or LSE assertions.
    check_attention_cp_compatibility(_config(layers), {"model.layers.0.attn"})


def test_unscoped_call_still_checks_every_layer():
    layers = {
        "model.layers.0.attn": _Layer(_CPCapableImpl()),
        "draft.layers.0.attn": _Layer(_PlainImpl()),
    }
    with pytest.raises(AssertionError, match="_PlainImpl"):
        check_attention_cp_compatibility(_config(layers), None)


def test_target_layer_is_still_checked():
    # Scoping must not turn the check off for layers that are in the set.
    layers = {"model.layers.0.attn": _Layer(_PlainImpl())}
    with pytest.raises(AssertionError, match="_PlainImpl"):
        check_attention_cp_compatibility(_config(layers), {"model.layers.0.attn"})


def test_dcp_lse_check_is_scoped_too():
    layers = {
        "model.layers.0.attn": _Layer(_CPCapableImpl()),
        "draft.layers.0.attn": _Layer(_PlainImpl()),
    }
    # No speculative config: only the DCP LSE assertion is reachable, and it
    # must likewise skip the draft's layer.
    cfg = _config(layers)
    cfg.speculative_config = None
    check_attention_cp_compatibility(cfg, {"model.layers.0.attn"})
    with pytest.raises(AssertionError, match="_PlainImpl"):
        check_attention_cp_compatibility(cfg, None)
