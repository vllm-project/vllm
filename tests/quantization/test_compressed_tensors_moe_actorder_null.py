# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that CompressedTensorsWNA16MarlinMoEMethod does not register g_idx
parameters when actorder is null. Regression test for #35303.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


class TestCompressedTensorsMoEActorderNull(unittest.TestCase):
    """Verify g_idx params are conditionally registered based on actorder."""

    def _make_method(self, actorder=None):
        """Create a CompressedTensorsWNA16MarlinMoEMethod with mocked deps."""
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
            CompressedTensorsWNA16MarlinMoEMethod,
        )

        weight_quant = MagicMock()
        weight_quant.symmetric = True
        weight_quant.num_bits = 4
        weight_quant.strategy = "group"
        weight_quant.group_size = 32
        weight_quant.actorder = actorder

        input_quant = None
        moe = MagicMock()
        moe.is_act_and_mul = True

        with (
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.get_marlin_input_dtype",
                return_value=None,
            ),
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.is_flashinfer_mxint4_moe_available",
                return_value=False,
            ),
        ):
            method = CompressedTensorsWNA16MarlinMoEMethod(
                moe=moe,
                weight_quant=weight_quant,
                input_quant=input_quant,
            )
        return method

    def _create_weights_on_layer(self, method):
        """Call create_weights and return the layer module."""
        layer = nn.Module()
        num_experts = 4
        hidden_size = 64
        intermediate_size = 32

        method.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=torch.float16,
            intermediate_size_full=intermediate_size,
        )
        return layer

    def test_actorder_null_no_g_idx_params(self):
        """When actorder is null, g_idx should NOT be registered as params."""
        method = self._make_method(actorder=None)
        layer = self._create_weights_on_layer(method)

        param_names = {name for name, _ in layer.named_parameters()}
        g_idx_names = {
            "w13_weight_g_idx",
            "w2_weight_g_idx",
            "w13_g_idx_sort_indices",
            "w2_g_idx_sort_indices",
        }
        registered_g_idx = param_names & g_idx_names
        self.assertEqual(
            registered_g_idx,
            set(),
            f"g_idx params should not be registered when actorder=null, "
            f"but found: {registered_g_idx}",
        )

    def test_actorder_group_has_g_idx_params(self):
        """When actorder='group', g_idx SHOULD be registered as params."""
        method = self._make_method(actorder="group")
        layer = self._create_weights_on_layer(method)

        param_names = {name for name, _ in layer.named_parameters()}
        g_idx_names = {
            "w13_weight_g_idx",
            "w2_weight_g_idx",
            "w13_g_idx_sort_indices",
            "w2_g_idx_sort_indices",
        }
        missing = g_idx_names - param_names
        self.assertEqual(
            missing,
            set(),
            f"g_idx params should be registered when actorder='group', "
            f"but missing: {missing}",
        )

    def test_process_weights_actorder_null(self):
        """process_weights_after_loading should work when actorder is null."""
        method = self._make_method(actorder=None)
        layer = self._create_weights_on_layer(method)

        # After process_weights, g_idx should be set as plain attributes
        # (empty tensors), not nn.Parameters
        with (
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.ops"
            ) as mock_ops,
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.marlin_moe_permute_scales",
                side_effect=lambda s, **kw: s,
            ),
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.marlin_make_workspace_new",
                return_value=torch.empty(0),
            ),
        ):
            mock_ops.gptq_marlin_moe_repack.side_effect = lambda w, *a, **kw: w
            method.process_weights_after_loading(layer)

        # g_idx attrs should exist as empty tensors
        self.assertTrue(hasattr(layer, "w13_weight_g_idx"))
        self.assertTrue(hasattr(layer, "w2_weight_g_idx"))
        self.assertEqual(layer.w13_weight_g_idx.shape[1], 0)
        self.assertEqual(layer.w2_weight_g_idx.shape[1], 0)

        # Verify g_idx are NOT registered as nn.Parameters
        param_names = {name for name, _ in layer.named_parameters()}
        self.assertNotIn("w13_weight_g_idx", param_names)
        self.assertNotIn("w2_weight_g_idx", param_names)
        self.assertNotIn("w13_g_idx_sort_indices", param_names)
        self.assertNotIn("w2_g_idx_sort_indices", param_names)


if __name__ == "__main__":
    unittest.main()
