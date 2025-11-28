# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest


class TestCreateDPPlacementGroupsErrors:
    """Integration tests that call the actual create_dp_placement_groups function."""

    @pytest.fixture
    def mock_vllm_config(self) -> MagicMock:
        """Create a mock VllmConfig with necessary attributes."""
        config = MagicMock()
        config.parallel_config.data_parallel_master_ip = "192.168.1.100"
        config.parallel_config.data_parallel_size = 2
        config.parallel_config.data_parallel_size_local = 4  # Request 4 DP ranks
        config.parallel_config.world_size = 8
        config.parallel_config.all2all_backend = None
        return config

    @patch("vllm.v1.engine.utils.current_platform")
    @patch("vllm.v1.engine.utils.envs")
    def test_dp_allocation_error_message_contains_values(
        self, mock_envs, mock_platform, mock_vllm_config
    ) -> None:
        """
        Test that ValueError from create_dp_placement_groups contains
        properly formatted values (not %s placeholders).
        """
        from vllm.v1.engine.utils import CoreEngineActorManager

        # Setup mocks
        mock_platform.ray_device_key = "GPU"
        mock_envs.VLLM_RAY_DP_PACK_STRATEGY = "strict"

        excepted_message = (
            "Not enough resources to allocate 4 DP ranks on DP "
            "master node 192.168.1.100, possible to fit 0 DP ranks."
        )

        # Mock Ray's available_resources_per_node to return limited resources
        # This will trigger the "Not enough resources" error
        mock_resources = {
            "node_1": {
                "node:192.168.1.100": 1.0,  # Master node
                "GPU": 2.0,  # Only 2 GPUs available
            }
        }

        with (
            patch(
                "ray._private.state.available_resources_per_node",
                return_value=mock_resources,
            ),
            patch("ray.util.placement_group"),
        ):
            # dp_size_available = 2 // 8 = 0, but dp_size_local = 4
            # This should raise ValueError

            with pytest.raises(ValueError) as exc_info:
                CoreEngineActorManager.create_dp_placement_groups(mock_vllm_config)

            error_message = str(exc_info.value)
            # AFTER FIX: These assertions should pass
            # The error message should contain actual values, not %s

            # Check that the message is properly formatted
            # (contains actual values, not placeholders)
            assert "%s" not in error_message, (
                f"Error message should not contain %s placeholder. Got: {error_message}"
            )

            assert excepted_message == error_message

    @patch("vllm.v1.engine.utils.current_platform")
    @patch("vllm.v1.engine.utils.envs")
    def test_assert_node_ip_keys_error_message(
        self, mock_envs, mock_platform, mock_vllm_config
    ) -> None:
        """
        Test that AssertionError contains properly formatted values.

        This test triggers the error when node has multiple IP keys.
        """
        from vllm.v1.engine.utils import CoreEngineActorManager

        mock_platform.ray_device_key = "GPU"
        mock_envs.VLLM_RAY_DP_PACK_STRATEGY = "strict"

        excepted_message = (
            "Zero or multiple node IP keys found in node resources: "
            "['node:192.168.1.100', 'node:192.168.1.101']"
        )

        # Mock resources with MULTIPLE node IP keys (triggers the assert)
        mock_resources = {
            "node_1": {
                "node:192.168.1.100": 1.0,
                "node:192.168.1.101": 1.0,  # Extra node key!
                "GPU": 8.0,
            }
        }

        with patch(
            "ray._private.state.available_resources_per_node",
            return_value=mock_resources,
        ):
            with pytest.raises(AssertionError) as exc_info:
                CoreEngineActorManager.create_dp_placement_groups(mock_vllm_config)

            error_message = str(exc_info.value)

            # AFTER FIX: The message should contain the actual node_ip_keys list
            # not a tuple with %s placeholder
            assert "%s" not in error_message, (
                f"Assert message should not contain %s placeholder. "
                f"Got: {error_message}"
            )

            assert excepted_message == error_message

    @patch("vllm.v1.engine.utils.current_platform")
    @patch("vllm.v1.engine.utils.envs")
    def test_dp_master_node_missing_error_message_contains_ip(
        self, mock_envs, mock_platform, mock_vllm_config
    ):
        """Test that AssertionError contains the actual IP address."""
        from vllm.v1.engine.utils import CoreEngineActorManager

        mock_platform.ray_device_key = "GPU"
        mock_envs.VLLM_RAY_DP_PACK_STRATEGY = "strict"

        dp_master_ip = "192.168.1.100"
        excepted_message = f"The DP master node (ip: {dp_master_ip}) is missing or dead"

        # Mock resources where the master node IP is NOT present
        # This triggers the assert at line 364-367
        mock_resources = {
            "node_1": {
                "node:10.0.0.1": 1.0,  # Different IP, not the master!
                "GPU": 8.0,
            }
        }

        with patch(
            "ray._private.state.available_resources_per_node",
            return_value=mock_resources,
        ):
            with pytest.raises(AssertionError) as exc_info:
                CoreEngineActorManager.create_dp_placement_groups(mock_vllm_config)

            error_message = str(exc_info.value)

            # AFTER FIX: These assertions should pass
            assert "%s" not in error_message, (
                f"Assert message should not contain %s placeholder. "
                f"Got: {error_message}"
            )

            assert excepted_message == error_message
