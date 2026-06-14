# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for benchmarks/kernels/validate_moe_configs.py.

All tests run without a GPU — no torch.cuda calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.kernels.validate_moe_configs import (
    _is_power_of_two,
    build_coverage_matrix,
    parse_filename,
    run_validation,
    validate_config_entry,
    validate_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_entry(**overrides: int) -> dict[str, int]:
    """Return a minimal valid config entry with optional overrides."""
    base: dict[str, int] = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 4,
    }
    base.update(overrides)
    return base


def _write_config(
    directory: Path,
    filename: str,
    data: dict | str | None = None,
) -> Path:
    """Write a config file (dict serialised to JSON, or raw string)."""
    filepath = directory / filename
    if data is None:
        filepath.touch()
    elif isinstance(data, str):
        filepath.write_text(data)
    else:
        filepath.write_text(json.dumps(data, indent=2))
    return filepath


# ---------------------------------------------------------------------------
# _is_power_of_two
# ---------------------------------------------------------------------------


class TestIsPowerOfTwo:
    def test_powers(self) -> None:
        for exp in range(11):
            assert _is_power_of_two(2**exp)

    def test_non_powers(self) -> None:
        for val in [0, -1, 3, 5, 6, 7, 9, 10, 15, 17, 100]:
            assert not _is_power_of_two(val)


# ---------------------------------------------------------------------------
# parse_filename
# ---------------------------------------------------------------------------


class TestParseFilename:
    def test_basic(self) -> None:
        parsed = parse_filename("E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json")
        assert parsed is not None
        assert parsed["E"] == 8
        assert parsed["N"] == 7168
        assert parsed["device"] == "NVIDIA_H100_80GB_HBM3"
        assert parsed["dtype"] is None
        assert parsed["block_shape"] is None

    def test_with_dtype(self) -> None:
        parsed = parse_filename(
            "E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json"
        )
        assert parsed is not None
        assert parsed["dtype"] == "fp8_w8a8"

    def test_with_block_shape(self) -> None:
        parsed = parse_filename(
            "E=128,N=384,device_name=NVIDIA_H200,"
            "dtype=fp8_w8a8,block_shape=[128,128].json"
        )
        assert parsed is not None
        assert parsed["block_shape"] == [128, 128]

    def test_amd_device_name(self) -> None:
        parsed = parse_filename("E=128,N=1024,device_name=AMD_Instinct_MI300X.json")
        assert parsed is not None
        assert parsed["device"] == "AMD_Instinct_MI300X"

    def test_device_with_hyphen(self) -> None:
        parsed = parse_filename("E=128,N=192,device_name=NVIDIA_H20-3e.json")
        assert parsed is not None
        assert parsed["device"] == "NVIDIA_H20-3e"

    def test_invalid_filename(self) -> None:
        assert parse_filename("not-a-valid-name.json") is None
        assert parse_filename("README") is None
        assert parse_filename("") is None


# ---------------------------------------------------------------------------
# validate_config_entry
# ---------------------------------------------------------------------------


class TestValidateConfigEntry:
    def test_valid_entry(self) -> None:
        errors = validate_config_entry(_make_valid_entry(), "1", "test.json")
        assert errors == []

    def test_valid_entry_with_optional_keys(self) -> None:
        entry = _make_valid_entry(waves_per_eu=0, kpack=2, SPLIT_K=1)
        errors = validate_config_entry(entry, "1", "test.json")
        assert errors == []

    def test_missing_required_key(self) -> None:
        entry = _make_valid_entry()
        del entry["num_warps"]
        errors = validate_config_entry(entry, "1", "test.json")
        assert any("missing required keys" in e for e in errors)

    def test_unexpected_key(self) -> None:
        entry = _make_valid_entry(bogus_key=42)
        errors = validate_config_entry(entry, "1", "test.json")
        assert any("unexpected keys" in e for e in errors)

    def test_non_power_of_two_block_size(self) -> None:
        entry = _make_valid_entry(BLOCK_SIZE_M=48)
        errors = validate_config_entry(entry, "1", "test.json")
        assert any("not a power of 2" in e for e in errors)

    def test_num_warps_out_of_range(self) -> None:
        entry = _make_valid_entry(num_warps=64)
        errors = validate_config_entry(entry, "1", "test.json")
        assert any("num_warps=64 outside" in e for e in errors)

    def test_num_warps_not_power_of_two(self) -> None:
        entry = _make_valid_entry(num_warps=3)
        errors = validate_config_entry(entry, "1", "test.json")
        assert any("not a power of 2" in e for e in errors)

    def test_num_stages_out_of_range(self) -> None:
        entry = _make_valid_entry(num_stages=9)
        errors = validate_config_entry(entry, "1", "test.json")
        assert any("num_stages=9 outside" in e for e in errors)

    def test_wrong_type(self) -> None:
        entry = _make_valid_entry()
        entry["num_warps"] = "four"  # type: ignore[assignment]
        errors = validate_config_entry(entry, "1", "test.json")
        assert any("must be int" in e for e in errors)

    def test_entry_not_a_dict(self) -> None:
        errors = validate_config_entry(
            [1, 2, 3],
            "1",
            "test.json",  # type: ignore[arg-type]
        )
        assert any("not a dict" in e for e in errors)

    def test_negative_group_size(self) -> None:
        entry = _make_valid_entry(GROUP_SIZE_M=-1)
        errors = validate_config_entry(entry, "1", "test.json")
        assert any("negative" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_file
# ---------------------------------------------------------------------------


class TestValidateFile:
    def test_valid_file(self, tmp_path: Path) -> None:
        data = {"1": _make_valid_entry(), "2": _make_valid_entry()}
        fp = _write_config(tmp_path, "test.json", data)
        parsed, errors = validate_file(fp)
        assert errors == []
        assert parsed is not None

    def test_valid_file_with_triton_version(self, tmp_path: Path) -> None:
        data: dict[str, Any] = {
            "triton_version": "3.0.0",
            "1": _make_valid_entry(),
        }
        fp = _write_config(tmp_path, "test.json", data)
        _, errors = validate_file(fp)
        assert errors == []

    def test_empty_file(self, tmp_path: Path) -> None:
        fp = _write_config(tmp_path, "empty.json", None)
        _, errors = validate_file(fp)
        assert any("empty" in e for e in errors)

    def test_malformed_json(self, tmp_path: Path) -> None:
        fp = _write_config(tmp_path, "bad.json", "{not valid json")
        _, errors = validate_file(fp)
        assert any("malformed JSON" in e for e in errors)

    def test_top_level_not_dict(self, tmp_path: Path) -> None:
        fp = _write_config(tmp_path, "list.json", "[1, 2, 3]")
        _, errors = validate_file(fp)
        assert any("not a dict" in e for e in errors)

    def test_non_numeric_top_level_key(self, tmp_path: Path) -> None:
        data = {"foo": _make_valid_entry()}
        fp = _write_config(tmp_path, "test.json", data)
        _, errors = validate_file(fp)
        assert any("non-numeric" in e for e in errors)

    def test_triton_version_wrong_type(self, tmp_path: Path) -> None:
        data: dict[str, Any] = {
            "triton_version": 3,
            "1": _make_valid_entry(),
        }
        fp = _write_config(tmp_path, "test.json", data)
        _, errors = validate_file(fp)
        assert any("triton_version" in e for e in errors)

    def test_invalid_entry_in_file(self, tmp_path: Path) -> None:
        entry = _make_valid_entry()
        del entry["num_warps"]
        data = {"1": entry}
        fp = _write_config(tmp_path, "test.json", data)
        _, errors = validate_file(fp)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# build_coverage_matrix
# ---------------------------------------------------------------------------


class TestBuildCoverageMatrix:
    def test_basic(self) -> None:
        parsed_files = {
            "f1.json": {"E": 8, "N": 7168, "device": "NVIDIA_H100", "dtype": None},
            "f2.json": {
                "E": 8,
                "N": 7168,
                "device": "NVIDIA_H100",
                "dtype": "fp8_w8a8",
            },
            "f3.json": {"E": 8, "N": 7168, "device": "NVIDIA_A100", "dtype": None},
        }
        matrix = build_coverage_matrix(parsed_files)
        assert "NVIDIA_H100" in matrix
        assert "NVIDIA_A100" in matrix
        h100_dtypes = matrix["NVIDIA_H100"]["E=8,N=7168"]
        assert None in h100_dtypes
        assert "fp8_w8a8" in h100_dtypes

    def test_empty(self) -> None:
        matrix = build_coverage_matrix({})
        assert matrix == {}


# ---------------------------------------------------------------------------
# run_validation (integration)
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_empty_directory(self, tmp_path: Path) -> None:
        results = run_validation(str(tmp_path))
        assert results["total_files"] == 0
        assert results["valid_files"] == 0
        assert results["errors"] == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        results = run_validation(str(tmp_path / "does_not_exist"))
        assert len(results["errors"]) > 0

    def test_mixed_valid_and_invalid(self, tmp_path: Path) -> None:
        good = {"1": _make_valid_entry(), "4": _make_valid_entry()}
        _write_config(
            tmp_path,
            "E=8,N=7168,device_name=NVIDIA_H100.json",
            good,
        )

        bad_entry = _make_valid_entry()
        del bad_entry["num_stages"]
        _write_config(
            tmp_path,
            "E=16,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json",
            {"1": bad_entry},
        )

        _write_config(tmp_path, "bad.json", "{broken")

        results = run_validation(str(tmp_path), verbose=True)
        assert results["total_files"] == 3
        assert results["valid_files"] == 1
        assert results["invalid_files"] == 2
        assert len(results["errors"]) >= 2
        assert "bad.json" in results["file_details"]

    def test_strict_promotes_warnings(self, tmp_path: Path) -> None:
        good = {"1": _make_valid_entry()}
        _write_config(tmp_path, "weird_name.json", good)
        results = run_validation(str(tmp_path), strict=True)
        assert len(results["warnings"]) > 0
        assert len(results["errors"]) >= len(results["warnings"])

    def test_coverage_populated(self, tmp_path: Path) -> None:
        data = {"1": _make_valid_entry()}
        _write_config(
            tmp_path,
            "E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json",
            data,
        )
        _write_config(
            tmp_path,
            "E=128,N=512,device_name=NVIDIA_B200.json",
            data,
        )
        results = run_validation(str(tmp_path))
        assert "NVIDIA_B200" in results["devices_found"]
        assert "E=128,N=512" in results["coverage"]["NVIDIA_B200"]

    def test_real_configs_pass(self) -> None:
        """Smoke test: validate the actual shipped configs (if available)."""
        real_dir = Path(__file__).resolve().parents[2] / (
            "vllm/model_executor/layers/fused_moe/configs"
        )
        if not real_dir.is_dir():
            pytest.skip("Shipped config directory not found")
        results = run_validation(str(real_dir))
        assert results["total_files"] > 0
        assert results["invalid_files"] == 0, (
            f"Shipped configs have errors: {results['errors']}"
        )
