# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from benchmarks.kernels.benchmark_moe import (
    get_configs_compute_bound,
    prune_cuda_search_space,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_candidate_device_names,
    get_default_config,
    get_moe_configs,
)


def test_get_candidate_device_names():
    # Exact H100 SXM5
    candidates = get_candidate_device_names("NVIDIA_H100_80GB_HBM3")
    assert candidates == ["NVIDIA_H100_80GB_HBM3", "NVIDIA_H100"]

    # PCIe variant
    candidates = get_candidate_device_names("NVIDIA_H100_PCIe")
    assert candidates == ["NVIDIA_H100_PCIe", "NVIDIA_H100_80GB_HBM3", "NVIDIA_H100"]

    # NVL variant
    candidates = get_candidate_device_names("NVIDIA_H100_NVL")
    assert candidates == ["NVIDIA_H100_NVL", "NVIDIA_H100_80GB_HBM3", "NVIDIA_H100"]

    # Generic H100
    candidates = get_candidate_device_names("NVIDIA_H100")
    assert candidates == ["NVIDIA_H100", "NVIDIA_H100_80GB_HBM3"]

    # H800 interconnect-restricted variant
    candidates = get_candidate_device_names("NVIDIA_H800")
    assert candidates == ["NVIDIA_H800", "NVIDIA_H100_80GB_HBM3", "NVIDIA_H100"]

    # H200 family
    candidates = get_candidate_device_names("NVIDIA_H200")
    assert candidates == ["NVIDIA_H200"]


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
def test_gemma4_h100_configs_tp1(simulated_device):
    get_moe_configs.cache_clear()
    with patch(
        "vllm.model_executor.layers.fused_moe.fused_moe.get_device_name_as_file_name",
        return_value=simulated_device,
    ):
        configs = get_moe_configs(E=128, N=704, dtype=None)
        assert configs is not None, f"Failed to resolve configs for {simulated_device}"
        expected_batches = [
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
        for bs in expected_batches:
            assert bs in configs, f"Batch size {bs} missing from TP=1 configs"
            cfg = configs[bs]
            assert "BLOCK_SIZE_M" in cfg
            assert "BLOCK_SIZE_N" in cfg
            assert "BLOCK_SIZE_K" in cfg
            assert "GROUP_SIZE_M" in cfg
            assert "num_warps" in cfg
            assert "num_stages" in cfg

        # Small decode batches must use BLOCK_SIZE_M=16 to prevent zero-padding bloat
        assert configs[1]["BLOCK_SIZE_M"] == 16
        assert configs[1]["GROUP_SIZE_M"] == 1


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
def test_gemma4_h100_configs_tp2(simulated_device):
    get_moe_configs.cache_clear()
    with patch(
        "vllm.model_executor.layers.fused_moe.fused_moe.get_device_name_as_file_name",
        return_value=simulated_device,
    ):
        configs = get_moe_configs(E=128, N=352, dtype=None)
        assert configs is not None, f"Failed to resolve configs for {simulated_device}"
        expected_batches = [
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
        for bs in expected_batches:
            assert bs in configs, f"Batch size {bs} missing from TP=2 configs"
            cfg = configs[bs]
            assert "BLOCK_SIZE_M" in cfg
            assert "BLOCK_SIZE_N" in cfg
            assert "BLOCK_SIZE_K" in cfg
            assert "GROUP_SIZE_M" in cfg
            assert "num_warps" in cfg
            assert "num_stages" in cfg

        assert configs[1]["BLOCK_SIZE_M"] == 16
        assert configs[1]["GROUP_SIZE_M"] == 1


def test_get_default_config_hopper():
    # Test default fallback when pre-tuned config is absent
    with (
        patch(
            "vllm.platforms.current_platform.get_device_capability",
            return_value=(9, 0),
        ),
        patch(
            "vllm.platforms.current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "vllm.platforms.current_platform.is_rocm",
            return_value=False,
        ),
    ):
        cfg_decode = get_default_config(M=1, E=128, N=704, K=2816, topk=8, dtype=None)
        assert cfg_decode["BLOCK_SIZE_M"] == 16
        assert cfg_decode["num_stages"] == 4

        cfg_prefill = get_default_config(
            M=1024, E=128, N=704, K=2816, topk=8, dtype=None
        )
        assert cfg_prefill["BLOCK_SIZE_M"] == 128
        assert cfg_prefill["num_stages"] == 3


def test_prune_cuda_search_space():
    base_space = get_configs_compute_bound(use_fp16=False, block_quant_shape=None)
    assert len(base_space) == 1920

    # Fine-grained MoE decode pruning
    pruned = prune_cuda_search_space(
        num_tokens=1,
        num_experts=128,
        shard_intermediate_size=1408,
        hidden_size=2816,
        search_space=base_space,
        topk=8,
    )
    assert len(pruned) < 200
    for cfg in pruned:
        assert cfg["BLOCK_SIZE_M"] <= 32
        assert cfg["GROUP_SIZE_M"] == 1
