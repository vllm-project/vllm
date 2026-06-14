# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import pytest
import torch

from vllm.model_executor.models.granitemoe import GraniteMoeModel


class _CapturingLoader:
    """Captures the rewritten weights ``GraniteMoeModel.load_weights`` emits.

    ``load_weights`` only uses ``self`` for the final ``self._load_weights``
    call, so this stand-in exercises the real renaming without a full model.
    """

    def __init__(self) -> None:
        self.captured: dict[str, torch.Tensor] = {}

    def _load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.captured = dict(weights)
        return set(self.captured)


@pytest.mark.cpu_test
def test_granitemoe_load_weights_routes_fp8_expert_scales():
    """FP8 expert ``weight_scale`` tensors are split and renamed per expert
    like the weights, instead of falling through as unknown ``KeyError`` keys.
    """
    num_experts, intermediate, hidden = 2, 4, 6
    prefix = "model.layers.0.block_sparse_moe"
    weights = [
        (
            f"{prefix}.input_linear.weight",
            torch.zeros(num_experts, 2 * intermediate, hidden),
        ),
        (
            f"{prefix}.input_linear.weight_scale",
            torch.zeros(num_experts, 2 * intermediate, 1),
        ),
        (
            f"{prefix}.output_linear.weight",
            torch.zeros(num_experts, hidden, intermediate),
        ),
        (
            f"{prefix}.output_linear.weight_scale",
            torch.zeros(num_experts, hidden, 1),
        ),
    ]

    stub = _CapturingLoader()
    GraniteMoeModel.load_weights(stub, weights)

    for e in range(num_experts):
        # New behavior: scales are routed to the per-expert shard slots.
        assert f"{prefix}.experts.{e}.w1.weight_scale" in stub.captured
        assert f"{prefix}.experts.{e}.w3.weight_scale" in stub.captured
        assert f"{prefix}.experts.{e}.w2.weight_scale" in stub.captured
        # Existing behavior: the plain weights are still routed.
        assert f"{prefix}.experts.{e}.w1.weight" in stub.captured
        assert f"{prefix}.experts.{e}.w3.weight" in stub.captured
        assert f"{prefix}.experts.{e}.w2.weight" in stub.captured

    # The raw fused scale names must not fall through unrewritten.
    assert f"{prefix}.input_linear.weight_scale" not in stub.captured
    assert f"{prefix}.output_linear.weight_scale" not in stub.captured
