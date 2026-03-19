# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

NODE_A = "node_a"
NODE_B = "node_b"
NODE_C = "node_c"

IP_A = "10.0.0.1"
IP_B = "10.0.0.2"
IP_C = "10.0.0.3"

NODE_ID_TO_IP = {NODE_A: IP_A, NODE_B: IP_B, NODE_C: IP_C}

MOCK_RAY_NODES = [
    {"NodeID": NODE_A, "NodeManagerAddress": IP_A, "Alive": True},
    {"NodeID": NODE_B, "NodeManagerAddress": IP_B, "Alive": True},
    {"NodeID": NODE_C, "NodeManagerAddress": IP_C, "Alive": True},
]


@pytest.mark.parametrize(
    "bundles_to_node_id,bundle_specs,world_size,expected",
    [
        pytest.param(
            {0: NODE_C, 1: NODE_A, 2: NODE_B, 3: NODE_C, 4: NODE_A, 5: NODE_B},
            [{"GPU": 1}] * 6,
            6,
            [
                (1, NODE_A, IP_A),
                (4, NODE_A, IP_A),
                (2, NODE_B, IP_B),
                (5, NODE_B, IP_B),
                (0, NODE_C, IP_C),
                (3, NODE_C, IP_C),
            ],
        ),
        pytest.param(
            {0: NODE_B, 1: NODE_B, 2: NODE_A, 3: NODE_A},
            [{"GPU": 1}] * 4,
            4,
            [
                (2, NODE_A, IP_A),
                (3, NODE_A, IP_A),
                (0, NODE_B, IP_B),
                (1, NODE_B, IP_B),
            ],
        ),
        pytest.param(
            {0: NODE_C, 1: NODE_B, 2: NODE_C, 3: NODE_B},
            [{"GPU": 1}] * 4,
            4,
            [
                (1, NODE_B, IP_B),
                (3, NODE_B, IP_B),
                (0, NODE_C, IP_C),
                (2, NODE_C, IP_C),
            ],
        ),
        pytest.param(
            {0: NODE_A, 1: NODE_A, 2: NODE_A},
            [{"GPU": 1}] * 3,
            3,
            [(0, NODE_A, IP_A), (1, NODE_A, IP_A), (2, NODE_A, IP_A)],
        ),
        pytest.param(
            {0: NODE_A},
            [{"GPU": 1}],
            1,
            [(0, NODE_A, IP_A)],
        ),
        pytest.param(
            {},
            [],
            0,
            [],
        ),
        pytest.param(
            {0: NODE_A, 1: NODE_B, 2: NODE_A, 3: NODE_B},
            [{"GPU": 1}] * 4,
            2,
            # After sort-then-clip, driver node (NODE_A) bundles are prioritized
            [(0, NODE_A, IP_A), (2, NODE_A, IP_A)],
        ),
        pytest.param(
            {0: NODE_A, 1: NODE_B, 2: NODE_A},
            [{"CPU": 1}, {"GPU": 1}, {"GPU": 1}],
            2,
            [(2, NODE_A, IP_A), (1, NODE_B, IP_B)],
        ),
    ],
)
def test_get_bundles_sorted_by_node(
    bundles_to_node_id, bundle_specs, world_size, expected
):
    mock_pg = MagicMock()
    mock_pg.bundle_specs = bundle_specs

    mock_ctx = MagicMock()
    mock_ctx.get_node_id.return_value = NODE_A

    with (
        patch(
            "vllm.v1.executor.ray_utils.placement_group_table",
            return_value={"bundles_to_node_id": bundles_to_node_id},
        ),
        patch("vllm.v1.executor.ray_utils.ray") as mock_ray,
        patch("vllm.v1.executor.ray_utils.current_platform") as mock_platform,
    ):
        mock_ray.get_runtime_context.return_value = mock_ctx
        mock_ray.nodes.return_value = MOCK_RAY_NODES
        mock_platform.ray_device_key = "GPU"

        from vllm.v1.executor.ray_utils import get_bundles_sorted_by_node

        result = get_bundles_sorted_by_node(mock_pg, world_size)

    assert result == expected
