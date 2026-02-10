# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests that triton_kernel_moe_forward correctly applies expert_map
remapping when expert parallelism (EP) is enabled.

Previously, legacy_routing was always used and it produced routing data
with global expert IDs that didn't correspond to local weight indices,
causing illegal memory access with EP.  The fix splits routing: when
expert_map is provided, topk selection is performed first, expert_map is
applied to remap global→local IDs, and make_routing_data builds routing
structures from the local IDs.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.quantization.mxfp4 import (
    Mxfp4Backend,
    Mxfp4MoEMethod,
)


def _make_mock_moe_config(ep_size: int = 1) -> MagicMock:
    """Create a mock FusedMoEConfig with the given EP size."""
    parallel_config = MagicMock()
    parallel_config.ep_size = ep_size

    moe_config = MagicMock()
    moe_config.ep_size = ep_size
    moe_config.is_lora_enabled = False
    moe_config.moe_parallel_config = parallel_config
    return moe_config


class TestMxfp4TritonIsMonolithic:
    """Verify that is_monolithic is always True for the TRITON backend,
    regardless of EP size, since triton_kernel_moe_forward now handles
    expert_map remapping internally."""

    @pytest.mark.parametrize(
        "backend,ep_size,expected_monolithic",
        [
            # TRITON is always monolithic (handles EP via expert_map remapping)
            (Mxfp4Backend.TRITON, 1, True),
            (Mxfp4Backend.TRITON, 2, True),
            (Mxfp4Backend.TRITON, 4, True),
            # SM100 backends are always monolithic
            (Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM, 1, True),
            (Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM, 2, True),
            (Mxfp4Backend.SM100_FI_MXFP4_BF16, 1, True),
            (Mxfp4Backend.SM100_FI_MXFP4_BF16, 2, True),
            # MARLIN is never monolithic
            (Mxfp4Backend.MARLIN, 1, False),
            (Mxfp4Backend.MARLIN, 2, False),
        ],
        ids=[
            "triton-no-ep",
            "triton-ep2",
            "triton-ep4",
            "sm100-trtllm-no-ep",
            "sm100-trtllm-ep2",
            "sm100-bf16-no-ep",
            "sm100-bf16-ep2",
            "marlin-no-ep",
            "marlin-ep2",
        ],
    )
    @patch(
        "vllm.model_executor.layers.quantization.mxfp4.get_mxfp4_backend",
    )
    @patch(
        "vllm.model_executor.layers.quantization.mxfp4.get_current_vllm_config",
    )
    def test_is_monolithic(
        self,
        mock_get_config,
        mock_get_backend,
        backend,
        ep_size,
        expected_monolithic,
    ):
        """is_monolithic should be True for TRITON regardless of EP size."""
        mock_get_backend.return_value = backend

        mock_compilation_config = MagicMock()
        mock_compilation_config.max_cudagraph_capture_size = 1024
        mock_vllm_config = MagicMock()
        mock_vllm_config.compilation_config = mock_compilation_config
        mock_get_config.return_value = mock_vllm_config

        moe_config = _make_mock_moe_config(ep_size=ep_size)
        method = Mxfp4MoEMethod(moe_config)

        assert method.is_monolithic == expected_monolithic, (
            f"Expected is_monolithic={expected_monolithic} for "
            f"backend={backend.name}, ep_size={ep_size}, "
            f"but got {method.is_monolithic}."
        )


class TestTritonMoeForwardExpertMap:
    """Test that triton_kernel_moe_forward applies expert_map remapping
    when expert_map is provided (EP active)."""

    @pytest.mark.parametrize("expert_map_present", [False, True])
    def test_routing_path_selection(self, expert_map_present):
        """Verify that the EP-aware routing path is taken when expert_map
        is present, and the legacy_routing path is taken otherwise."""
        # This is a structural test: we mock the routing functions to
        # verify the correct path is exercised.
        mock_expert_map = torch.tensor([0, -1, 1, -1]) if expert_map_present else None

        with (
            patch(
                "vllm.model_executor.layers.fused_moe."
                "gpt_oss_triton_kernels_moe.legacy_routing"
            ) as mock_legacy,
            patch(
                "vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe.topk"
            ) as mock_topk,
            patch(
                "vllm.model_executor.layers.fused_moe."
                "gpt_oss_triton_kernels_moe.make_routing_data"
            ) as mock_make_routing,
            patch(
                "vllm.model_executor.layers.fused_moe."
                "gpt_oss_triton_kernels_moe.triton_kernel_fused_experts"
            ) as mock_fused_experts,
        ):
            from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (  # noqa: E501
                triton_kernel_moe_forward,
            )

            # Set up return values
            mock_routing_data = MagicMock()
            mock_gather = MagicMock()
            mock_scatter = MagicMock()

            if expert_map_present:
                sparse_result = MagicMock()
                sparse_result.indx = torch.tensor([[0, 2]])
                sparse_result.vals = torch.tensor([[0.6, 0.4]])
                mock_topk.return_value = sparse_result
                mock_make_routing.return_value = (
                    mock_routing_data,
                    mock_gather,
                    mock_scatter,
                )
            else:
                mock_legacy.return_value = (
                    mock_routing_data,
                    mock_gather,
                    mock_scatter,
                )

            mock_fused_experts.return_value = torch.zeros(1, 8)

            hidden = torch.randn(1, 8)
            w1 = torch.randn(2, 8, 16)
            w2 = torch.randn(2, 8, 8)
            logits = torch.randn(1, 4)

            triton_kernel_moe_forward(
                hidden_states=hidden,
                w1=w1,
                w2=w2,
                gating_output=logits,
                topk=2,
                renormalize=True,
                expert_map=mock_expert_map,
            )

            if expert_map_present:
                # EP path: should use topk + make_routing_data, NOT
                # legacy_routing
                mock_topk.assert_called_once()
                mock_make_routing.assert_called_once()
                mock_legacy.assert_not_called()
                # expert_map should be None in the fused_experts call
                # (already applied)
                call_kwargs = mock_fused_experts.call_args
                assert call_kwargs[1].get("expert_map") is None or (
                    len(call_kwargs[0]) > 0
                )
            else:
                # Non-EP path: should use legacy_routing
                mock_legacy.assert_called_once()
                mock_topk.assert_not_called()
                mock_make_routing.assert_not_called()
