# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Qwen3.5 MTP checkpoint weight-name remapping.

Checkpoints differ in how they ship the draft model's output head: some carry a
dedicated ``mtp.lm_head.*``, others expect the draft to reuse the base model's
top-level ``lm_head.*``. Both layouts must resolve onto ``Qwen3_5MTP.lm_head``.
"""

from unittest.mock import Mock, patch

BASE_LM_HEAD = "lm_head.weight"
DRAFT_LM_HEAD = "mtp.lm_head.weight"
EMBED_TOKENS = "model.language_model.embed_tokens.weight"
MTP_LAYER = "mtp.layers.0.input_layernorm.weight"
BASE_MODEL_LAYER = "model.language_model.layers.0.input_layernorm.weight"


def _remap(names: list[str]) -> list[tuple[str, str]]:
    """Return the ``(remapped_name, source_name)`` stream ``load_weights`` builds.

    ``load_weights`` only uses ``self`` to construct the loader, so a Mock stands
    in for the module. Each weight carries its own checkpoint name as its value,
    which lets a test assert *which* checkpoint tensor reached a parameter.
    """
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    captured: list[tuple[str, str]] = []

    class _CapturingLoader:
        def __init__(self, module):
            pass

        def load_weights(self, weights, **kwargs):
            captured.extend(weights)
            return set()

    with patch(
        "vllm.model_executor.models.qwen3_5_mtp.AutoWeightsLoader", _CapturingLoader
    ):
        Qwen3_5MTP.load_weights(Mock(), [(name, name) for name in names])

    return captured


def _source_of(remapped: list[tuple[str, str]], name: str) -> list[str]:
    return [source for target, source in remapped if target == name]


def test_dedicated_draft_head_preferred():
    """A checkpoint carrying both heads must load the draft head, once."""
    remapped = _remap([BASE_LM_HEAD, MTP_LAYER, DRAFT_LM_HEAD])
    assert _source_of(remapped, BASE_LM_HEAD) == [DRAFT_LM_HEAD]


def test_dedicated_draft_head_preferred_before_base_head():
    """The choice must not depend on the order weights arrive in."""
    remapped = _remap([DRAFT_LM_HEAD, MTP_LAYER, BASE_LM_HEAD])
    assert _source_of(remapped, BASE_LM_HEAD) == [DRAFT_LM_HEAD]


def test_base_head_used_when_no_draft_head():
    """Without a dedicated draft head, the draft reuses the base model's head."""
    remapped = _remap([BASE_LM_HEAD, MTP_LAYER])
    assert _source_of(remapped, BASE_LM_HEAD) == [BASE_LM_HEAD]


def test_draft_head_scales_also_remapped():
    """Quantization scales alongside the draft head follow the same path."""
    remapped = _remap(["mtp.lm_head.weight_scale", "mtp.lm_head.input_scale"])
    assert dict(remapped) == {
        "lm_head.weight_scale": "mtp.lm_head.weight_scale",
        "lm_head.input_scale": "mtp.lm_head.input_scale",
    }


def test_mtp_prefix_and_embed_tokens_remap_unchanged():
    """Existing remaps must be preserved: mtp.* -> model.*, and the base
    model's input embedding is shared with the draft."""
    remapped = dict(_remap([MTP_LAYER, EMBED_TOKENS]))
    assert remapped["model.layers.0.input_layernorm.weight"] == MTP_LAYER
    assert remapped["model.embed_tokens.weight"] == EMBED_TOKENS


def test_base_model_weights_dropped():
    """Target-model weights are loaded by the target, not the draft."""
    assert _remap([BASE_MODEL_LAYER]) == []
