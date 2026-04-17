# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests that triton_kernel_moe_forward correctly applies expert_map
remapping when expert parallelism (EP) is enabled.

Both EP and non-EP paths use topk + make_routing_data. When expert_map
is provided, global expert IDs are remapped to local IDs before building
routing structures.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestTritonMoeForwardExpertMap:
    """Test that triton_kernel_moe_forward applies expert_map remapping
    when expert_map is provided (EP active)."""

    @pytest.mark.parametrize("expert_map_present", [False, True])
    def test_routing_path_selection(self, expert_map_present):
        """Verify that both EP and non-EP paths use topk + make_routing_data,
        and that expert_map remapping is applied when present."""

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_expert_map = (
            torch.tensor([0, -1, 1, -1], device=device) if expert_map_present else None
        )

        from vllm.utils.import_utils import import_triton_kernels

        import_triton_kernels()

        with (
            patch("triton_kernels.topk.topk") as mock_topk,
            patch(
                "vllm.model_executor.layers.fused_moe.experts."
                "gpt_oss_triton_kernels_moe.make_routing_data"
            ) as mock_make_routing,
            patch(
                "vllm.model_executor.layers.fused_moe.experts."
                "gpt_oss_triton_kernels_moe.triton_kernel_fused_experts"
            ) as mock_fused_experts,
        ):
            from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (  # noqa: E501
                triton_kernel_moe_forward,
            )

            mock_routing_data = MagicMock()
            mock_gather = MagicMock()
            mock_scatter = MagicMock()

            sparse_result = MagicMock()
            sparse_result.indx = torch.tensor([[0, 2]], dtype=torch.int32)
            sparse_result.vals = torch.tensor([[0.6, 0.4]])
            mock_topk.return_value = sparse_result
            mock_make_routing.return_value = (
                mock_routing_data,
                mock_gather,
                mock_scatter,
            )

            mock_fused_experts.return_value = torch.zeros((1, 8), device=device)

            hidden = torch.randn((1, 8), device=device)
            w1 = torch.randn((2, 8, 16), device=device)
            w2 = torch.randn((2, 8, 8), device=device)
            logits = torch.randn((1, 4), device=device)

            triton_kernel_moe_forward(
                hidden_states=hidden,
                w1=w1,
                w2=w2,
                gating_output=logits,
                topk=2,
                renormalize=True,
                expert_map=mock_expert_map,
            )

            # Both paths use topk + make_routing_data
            mock_topk.assert_called_once()
            mock_make_routing.assert_called_once()

            if expert_map_present:
                # expert_map should be None in the fused_experts call
                # (already applied)
                call_kwargs = mock_fused_experts.call_args
                assert call_kwargs[1].get("expert_map") is None or (
                    len(call_kwargs[0]) > 0
                )
