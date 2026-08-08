# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 conversions of DeepSeek-V4 quantize only the main stack; the draft
module (runtime layer ids >= num_hidden_layers) keeps native MXFP4 experts.
The quant routing must therefore be prefix-aware: draft-module RoutedExperts
route to Mxfp4MoEMethod even when the checkpoint declares moe_quant_algo=NVFP4
(jasl/vllm#35 — prefix-blind routing decoded MXFP4 bits as NVFP4 noise and
collapsed draft acceptance to 0%)."""

from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config


def _config_with(num_hidden_layers: int | None) -> DeepseekV4FP8Config:
    cfg = DeepseekV4FP8Config.__new__(DeepseekV4FP8Config)
    cfg._resolved_num_hidden_layers = num_hidden_layers
    return cfg


def test_main_stack_prefixes_are_not_draft():
    cfg = _config_with(43)
    for prefix in ("layers.0.ffn.experts", "layers.42.ffn.experts",
                   "model.layers.42.ffn.experts"):
        assert not cfg._is_draft_module_prefix(prefix), prefix


def test_draft_module_prefixes_detected_for_all_backends():
    cfg = _config_with(43)
    for prefix in (
        "layers.43.ffn.experts",        # NVIDIA DSpark, empty root
        "layers.45.ffn.experts",        # third DSpark block
        "model.layers.43.ffn.experts",  # NVIDIA MTP root
    ):
        assert cfg._is_draft_module_prefix(prefix), prefix


def test_unresolvable_prefixes_stay_on_declared_algo():
    cfg = _config_with(43)
    assert not cfg._is_draft_module_prefix("lm_head")
    assert not _config_with(None)._is_draft_module_prefix("layers.43.ffn.experts")
