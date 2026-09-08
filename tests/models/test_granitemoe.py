# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GraniteMoe checkpoint weight name mapping."""

import pytest

from vllm.model_executor.models.granitemoe import GraniteMoeModel

PREFIX = "model.layers.0.block_sparse_moe"


def _map(name: str) -> str:
    return GraniteMoeModel.hf_to_vllm_mapper.apply_list([name])[0]


@pytest.mark.cpu_test
def test_fused_expert_weights_are_mapped():
    """The fused expert weights keep mapping onto the FusedMoE names."""
    assert _map(f"{PREFIX}.input_linear.weight") == f"{PREFIX}.experts.gate_up_proj"
    assert _map(f"{PREFIX}.output_linear.weight") == f"{PREFIX}.experts.down_proj"


@pytest.mark.cpu_test
def test_fused_expert_weight_scales_are_mapped():
    """FP8 checkpoints pair each fused expert weight with a ``weight_scale``.

    The suffix rules match with ``str.endswith``, so a rule keyed on
    ``.input_linear.weight`` never matches ``.input_linear.weight_scale``.
    Without a rule of its own the scale keeps its checkpoint name, which
    matches no FusedMoE parameter and fails to load.

    ``FusedMoE`` derives the parameter from the ``experts.gate_up_proj`` /
    ``experts.down_proj`` substring (see ``RoutedExperts.load_weights``), so
    the ``_scale`` suffix carries through to ``w13_weight_scale`` /
    ``w2_weight_scale``.
    """
    assert (
        _map(f"{PREFIX}.input_linear.weight_scale")
        == f"{PREFIX}.experts.gate_up_proj_scale"
    )
    assert (
        _map(f"{PREFIX}.output_linear.weight_scale")
        == f"{PREFIX}.experts.down_proj_scale"
    )
