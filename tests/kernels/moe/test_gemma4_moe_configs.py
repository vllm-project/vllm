# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Gemma 4 MoE configurations, fallbacks, and tile invariants."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from unittest.mock import patch

from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_candidate_device_names,
    get_config_file_name,
    get_default_config,
    get_moe_configs,
)


class TestGemma4MoEConfigs:
    """Test suite for Gemma 4 26B-A4B MoE configurations and invariants."""

    def test_candidate_device_names_hierarchy(self) -> None:
        """Verify device family fallback candidates for H100, H200, H800, A100, A800."""
        # H200 family
        h200_cand = get_candidate_device_names("NVIDIA_H200")
        assert "NVIDIA_H200" in h200_cand

        h200_nvl = get_candidate_device_names("NVIDIA_H200_NVL")
        assert "NVIDIA_H200" in h200_nvl

        # H100 family
        h100_pcie = get_candidate_device_names("NVIDIA_H100_PCIe")
        assert "NVIDIA_H100_PCIe" in h100_pcie
        assert "NVIDIA_H100_80GB_HBM3" in h100_pcie or "NVIDIA_H100" in h100_pcie

        h800 = get_candidate_device_names("NVIDIA_H800")
        assert "NVIDIA_H800" in h800
        assert "NVIDIA_H100_80GB_HBM3" in h800 or "NVIDIA_H100" in h800

        # A100 / A800 family
        a100_pcie = get_candidate_device_names("NVIDIA_A100-PCIE-40GB")
        assert "NVIDIA_A100-PCIE-40GB" in a100_pcie
        assert (
            "NVIDIA_A100-SXM4-80GB" in a100_pcie or "NVIDIA_A100-SXM4-40GB" in a100_pcie
        )

        a800 = get_candidate_device_names("NVIDIA_A800-SXM4-80GB")
        assert "NVIDIA_A800-SXM4-80GB" in a800
        assert "NVIDIA_A100-SXM4-80GB" in a800 or "NVIDIA_A100-SXM4-40GB" in a800

    def test_get_config_file_name_normalization(self) -> None:
        """Verify config file naming handles device normalization."""
        fn = get_config_file_name(128, 704, "bf16", device_name="NVIDIA_H200")
        assert fn == "E=128,N=704,device_name=NVIDIA_H200,dtype=bf16.json"

        fn_h200_variant = get_config_file_name(
            128, 352, None, device_name="NVIDIA_H200_NVL"
        )
        assert fn_h200_variant == "E=128,N=352,device_name=NVIDIA_H200.json"

        fn_block = get_config_file_name(
            128,
            176,
            "fp8_w8a8",
            block_shape=[128, 128],
            device_name="NVIDIA_A100-SXM4-80GB",
        )
        expected_fn = (
            "E=128,N=176,device_name=NVIDIA_A100-SXM4-80GB,"
            "dtype=fp8_w8a8,block_shape=[128,128].json"
        )
        assert fn_block == expected_fn

    def test_get_moe_configs_lookup(self) -> None:
        """Verify get_moe_configs finds tuned configs on known devices."""
        get_moe_configs.cache_clear()
        with patch(
            "vllm.model_executor.layers.fused_moe.fused_moe.get_device_name_as_file_name",
            return_value="NVIDIA_A100-SXM4-80GB",
        ):
            # Test N in {704, 352, 176, 88}
            for n in (704, 352, 176, 88):
                cfg = get_moe_configs(128, n, None)
                assert cfg is not None, f"Expected config for E=128, N={n}"
                assert 1 in cfg
                assert 2048 in cfg
                assert "BLOCK_SIZE_M" in cfg[1]
                assert "BLOCK_SIZE_N" in cfg[1]
                assert "BLOCK_SIZE_K" in cfg[1]

    def test_get_default_config_gemma4_moe(self) -> None:
        """Verify default heuristic for E=128, topk=8 across Gemma 4 shapes."""
        # TP8 N=88, M=128: must use conservative tile preventing register spills
        cfg_tp8_m128 = get_default_config(
            M=128, E=128, N=88, K=2816, topk=8, dtype=None
        )
        assert cfg_tp8_m128["BLOCK_SIZE_N"] <= 64
        assert cfg_tp8_m128["BLOCK_SIZE_K"] == 64
        assert cfg_tp8_m128["GROUP_SIZE_M"] == 1

        # M <= 256 for E=128: tokens_per_expert <= 16, must use GROUP_SIZE_M == 1
        for m in (1, 4, 16, 64, 128, 256):
            cfg = get_default_config(M=m, E=128, N=704, K=2816, topk=8, dtype=None)
            assert cfg["GROUP_SIZE_M"] == 1

        # Small expert count (E=8, topk=2) must preserve exact upstream behavior
        cfg_e8_m1 = get_default_config(M=1, E=8, N=1024, K=2048, topk=2, dtype=None)
        assert cfg_e8_m1["BLOCK_SIZE_M"] == 16
        assert cfg_e8_m1["BLOCK_SIZE_N"] == 64
        assert cfg_e8_m1["BLOCK_SIZE_K"] == 128
        assert cfg_e8_m1["GROUP_SIZE_M"] == 1

        cfg_e8_m128 = get_default_config(M=128, E=8, N=1024, K=2048, topk=2, dtype=None)
        assert cfg_e8_m128["BLOCK_SIZE_M"] == 64
        assert cfg_e8_m128["BLOCK_SIZE_N"] == 128
        assert cfg_e8_m128["BLOCK_SIZE_K"] == 64

        cfg_e8_m1024 = get_default_config(
            M=1024, E=8, N=1024, K=2048, topk=2, dtype=None
        )
        assert cfg_e8_m1024["GROUP_SIZE_M"] == 1

        # E=64, topk=16, M=128: 64x128 tile must set num_warps=8 to avoid spilling
        cfg_e64_m128 = get_default_config(
            M=128, E=64, N=704, K=2816, topk=16, dtype=None
        )
        assert cfg_e64_m128["BLOCK_SIZE_M"] == 64
        assert cfg_e64_m128["BLOCK_SIZE_N"] == 128
        assert cfg_e64_m128["num_warps"] == 8

    def test_pr3_json_invariants(self) -> None:
        """Verify tile invariants across all 269 PR3 JSON configuration files."""
        upstream_files = {
            "E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json",
            "E=128,N=704,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json",
            (
                "E=128,N=704,device_name="
                "NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,"
                "dtype=fp8_w8a8.json"
            ),
            "E=128,N=352,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json",
        }
        configs_dir = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "vllm/model_executor/layers/fused_moe/configs"
        )
        all_candidates = []
        for n in (704, 352, 176, 88):
            all_candidates.extend(glob.glob(str(configs_dir / f"E=128,N={n},*.json")))

        files = [
            fp for fp in all_candidates if os.path.basename(fp) not in upstream_files
        ]
        assert len(files) == 269, f"Expected 269 PR3 config files, found {len(files)}"

        for fp in files:
            fn = os.path.basename(fp)
            is_n88 = "N=88," in fn
            with open(fp) as f:
                data = json.load(f)

            for key, tile in data.items():
                if key == "triton_version":
                    continue
                # Invariant: BLOCK_SIZE_N <= 128 (no 256)
                assert tile["BLOCK_SIZE_N"] <= 128, (
                    f"File {fn} key {key} has BLOCK_SIZE_N={tile['BLOCK_SIZE_N']} > 128"
                )
                # Invariant: on N=88, BLOCK_SIZE_N <= 64
                if is_n88:
                    assert tile["BLOCK_SIZE_N"] <= 64, (
                        f"File {fn} key {key} has "
                        f"BLOCK_SIZE_N={tile['BLOCK_SIZE_N']} > 64 for N=88"
                    )
                # Invariant: num_stages >= 2
                assert tile["num_stages"] >= 2, (
                    f"File {fn} key {key} has num_stages={tile['num_stages']} < 2"
                )
                # Invariant: for M="256", BLOCK_SIZE_M <= 32 and GROUP_SIZE_M == 1
                if key == "256":
                    assert tile["BLOCK_SIZE_M"] <= 32, (
                        f"File {fn} key {key} has "
                        f"BLOCK_SIZE_M={tile['BLOCK_SIZE_M']} > 32"
                    )
                    assert tile["GROUP_SIZE_M"] == 1, (
                        f"File {fn} key {key} has "
                        f"GROUP_SIZE_M={tile['GROUP_SIZE_M']} != 1"
                    )
                # Invariant: in NVIDIA configs, large tiles require num_warps >= 8
                if "device_name=NVIDIA_" in fn:
                    bm = tile["BLOCK_SIZE_M"]
                    bn = tile["BLOCK_SIZE_N"]
                    bk = tile["BLOCK_SIZE_K"]
                    if (
                        bm * bn >= 8192
                        or (bn == 128 and bk == 128)
                        or (bm == 64 and bk == 128)
                    ):
                        assert tile["num_warps"] >= 8, (
                            f"File {fn} key {key} with tile ({bm},{bn},{bk}) "
                            f"has num_warps={tile['num_warps']} < 8"
                        )
