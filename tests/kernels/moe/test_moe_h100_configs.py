# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_candidate_device_names,
    get_moe_configs,
)

EXPECTED_BATCH_SIZES = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
]


def test_get_candidate_device_names():
    """Verify prioritized device-family fallback list for MoE config lookup."""
    assert get_candidate_device_names("NVIDIA_H100_80GB_HBM3") == [
        "NVIDIA_H100_80GB_HBM3",
        "NVIDIA_H100",
    ]
    assert get_candidate_device_names("NVIDIA_H100_PCIe") == [
        "NVIDIA_H100_PCIe",
        "NVIDIA_H100_80GB_HBM3",
        "NVIDIA_H100",
    ]
    assert get_candidate_device_names("NVIDIA_H100_NVL") == [
        "NVIDIA_H100_NVL",
        "NVIDIA_H100_80GB_HBM3",
        "NVIDIA_H100",
    ]
    assert get_candidate_device_names("NVIDIA_H100") == [
        "NVIDIA_H100",
        "NVIDIA_H100_80GB_HBM3",
    ]
    assert get_candidate_device_names("NVIDIA_H800") == [
        "NVIDIA_H800",
        "NVIDIA_H100_80GB_HBM3",
        "NVIDIA_H100",
    ]
    assert get_candidate_device_names("NVIDIA_H200") == ["NVIDIA_H200"]


@pytest.mark.parametrize("n_dim", [704, 352])
@pytest.mark.parametrize(
    "simulated_device",
    [
        "NVIDIA_H100_80GB_HBM3",
        "NVIDIA_H100_PCIe",
        "NVIDIA_H100_NVL",
        "NVIDIA_H100",
        "NVIDIA_H800",
    ],
)
def test_gemma4_h100_configs(n_dim: int, simulated_device: str):
    """Verify Gemma 4 TP=1 (N=704) and TP=2 (N=352) configs resolve across H100 SKUs."""
    get_moe_configs.cache_clear()
    with patch(
        "vllm.model_executor.layers.fused_moe.fused_moe.get_device_name_as_file_name",
        return_value=simulated_device,
    ):
        configs = get_moe_configs(E=128, N=n_dim, dtype=None)
        assert configs is not None, (
            f"Failed to resolve E=128, N={n_dim} configs for {simulated_device}"
        )
        for bs in EXPECTED_BATCH_SIZES:
            assert bs in configs, f"Batch size {bs} missing from N={n_dim} configs"
            cfg = configs[bs]
            for key in (
                "BLOCK_SIZE_M",
                "BLOCK_SIZE_N",
                "BLOCK_SIZE_K",
                "GROUP_SIZE_M",
                "num_warps",
                "num_stages",
            ):
                assert key in cfg
        assert configs[1]["BLOCK_SIZE_M"] == 16
        assert configs[1]["GROUP_SIZE_M"] == 1
