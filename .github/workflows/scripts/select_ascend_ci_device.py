#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select a healthy Ascend device with enough free HBM for a CI workload."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

EXIT_PROBE_ERROR = 2
EXIT_NO_ELIGIBLE_DEVICE = 3


@dataclass(frozen=True)
class DeviceMemory:
    logical_id: int
    free_memory_mb: int
    total_memory_mb: int
    source: str

    def required_memory_mb(self, min_free_ratio: float) -> int:
        return math.ceil(self.total_memory_mb * min_free_ratio)


def parse_logical_map(mapping_output: str) -> dict[tuple[str, str], int]:
    logical_map: dict[tuple[str, str], int] = {}
    for line in mapping_output.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        npu_id, chip_id, logical_id = parts[:3]
        if npu_id.isdigit() and chip_id.isdigit() and logical_id.isdigit():
            logical_map[npu_id, chip_id] = int(logical_id)
    return logical_map


def parse_device_memory(
    info_output: str,
    logical_map: dict[tuple[str, str], int],
) -> list[DeviceMemory]:
    hbm_usage_pattern = re.compile(r"(\d+)\s*/\s*(\d+)\s*$")
    devices: list[DeviceMemory] = []
    current_npu_id: str | None = None
    current_health: str | None = None

    for raw_line in info_output.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue

        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 3:
            continue

        left_column = parts[0].split()
        if (
            len(left_column) >= 2
            and left_column[0].isdigit()
            and parts[1]
            and ":" not in parts[1]
        ):
            current_npu_id = left_column[0]
            current_health = parts[1]
            continue

        if current_npu_id is None or current_health != "OK":
            continue
        if len(left_column) != 1 or not left_column[0].isdigit():
            continue

        chip_id = left_column[0]
        logical_id = logical_map.get((current_npu_id, chip_id))
        source = "logical-map"
        if logical_id is None:
            if chip_id != "0":
                continue
            logical_id = int(current_npu_id)
            source = "status-fallback"

        hbm_match = hbm_usage_pattern.search(parts[2])
        if hbm_match is None:
            continue
        used_memory_mb = int(hbm_match.group(1))
        total_memory_mb = int(hbm_match.group(2))
        if total_memory_mb <= 0 or used_memory_mb < 0:
            continue
        devices.append(
            DeviceMemory(
                logical_id=logical_id,
                free_memory_mb=max(0, total_memory_mb - used_memory_mb),
                total_memory_mb=total_memory_mb,
                source=source,
            )
        )

    return devices


def choose_device(
    devices: list[DeviceMemory],
    *,
    min_free_ratio: float,
    candidate_devices: set[int] | None = None,
    selection_attempt: int = 1,
) -> tuple[DeviceMemory | None, list[DeviceMemory]]:
    considered = [
        device
        for device in devices
        if candidate_devices is None or device.logical_id in candidate_devices
    ]
    considered.sort(
        key=lambda device: (
            -device.free_memory_mb,
            device.logical_id,
            device.source,
        )
    )
    eligible = [
        device
        for device in considered
        if device.free_memory_mb >= device.required_memory_mb(min_free_ratio)
    ]
    if not eligible:
        return None, considered
    selected = eligible[(max(1, selection_attempt) - 1) % len(eligible)]
    return selected, considered


def run_npu_smi(
    npu_smi_bin: str,
    *args: str,
) -> subprocess.CompletedProcess[str] | None:
    clean_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    }
    command = [npu_smi_bin, *args]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=clean_env,
        )
        if result.returncode == 0 or os.geteuid() == 0:
            return result

        sudo_bin = shutil.which("sudo", path=clean_env["PATH"])
        if sudo_bin is None:
            return result
        sudo_result = subprocess.run(
            [sudo_bin, "-n", *command],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=clean_env,
        )
        return sudo_result if sudo_result.returncode == 0 else result
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"npu-smi {' '.join(args)} failed: {exc}", file=sys.stderr)
        return None


def parse_candidate_devices(value: str) -> set[int] | None:
    if not value.strip():
        return None
    candidates: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item.isdigit():
            raise ValueError(
                "candidate devices must be a comma-separated list of "
                f"non-negative integers, got {value!r}"
            )
        candidates.add(int(item))
    return candidates


def select_from_system(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    mapping_result = run_npu_smi(args.npu_smi_bin, "info", "-m")
    logical_map = {}
    if mapping_result is not None and mapping_result.returncode == 0:
        logical_map = parse_logical_map(mapping_result.stdout)

    info_result = run_npu_smi(args.npu_smi_bin, "info")
    if info_result is None or info_result.returncode != 0:
        return (
            {
                "status": "probe_error",
                "reason": "npu_smi_info_failed",
                "min_free_ratio": args.min_free_ratio,
                "devices": [],
            },
            EXIT_PROBE_ERROR,
        )

    devices = parse_device_memory(info_result.stdout, logical_map)
    if not devices:
        return (
            {
                "status": "probe_error",
                "reason": "no_healthy_device_memory_records",
                "min_free_ratio": args.min_free_ratio,
                "devices": [],
            },
            EXIT_PROBE_ERROR,
        )

    try:
        candidate_devices = parse_candidate_devices(args.candidate_devices)
    except ValueError as exc:
        return (
            {
                "status": "probe_error",
                "reason": str(exc),
                "min_free_ratio": args.min_free_ratio,
                "devices": [],
            },
            EXIT_PROBE_ERROR,
        )

    selected, considered = choose_device(
        devices,
        min_free_ratio=args.min_free_ratio,
        candidate_devices=candidate_devices,
        selection_attempt=args.selection_attempt,
    )
    device_records = [
        {
            **asdict(device),
            "required_free_memory_mb": device.required_memory_mb(args.min_free_ratio),
            "eligible": device.free_memory_mb
            >= device.required_memory_mb(args.min_free_ratio),
        }
        for device in considered
    ]
    if selected is None:
        return (
            {
                "status": "unavailable",
                "reason": "no_device_meets_free_memory_requirement",
                "min_free_ratio": args.min_free_ratio,
                "devices": device_records,
            },
            EXIT_NO_ELIGIBLE_DEVICE,
        )

    return (
        {
            "status": "selected",
            "device_id": selected.logical_id,
            "source": selected.source,
            "free_memory_mb": selected.free_memory_mb,
            "total_memory_mb": selected.total_memory_mb,
            "required_free_memory_mb": selected.required_memory_mb(args.min_free_ratio),
            "min_free_ratio": args.min_free_ratio,
            "devices": device_records,
        },
        0,
    )


def emit_result(result: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(result, sort_keys=True))
        return
    if result["status"] == "selected":
        print(
            "\t".join(
                str(result[key])
                for key in (
                    "status",
                    "device_id",
                    "source",
                    "free_memory_mb",
                    "total_memory_mb",
                    "required_free_memory_mb",
                )
            )
        )
        return
    devices = result.get("devices")
    if result["status"] == "unavailable" and isinstance(devices, list) and devices:
        best = devices[0]
        print(
            "\t".join(
                str(value)
                for value in (
                    result["status"],
                    best["logical_id"],
                    best["source"],
                    best["free_memory_mb"],
                    best["total_memory_mb"],
                    best["required_free_memory_mb"],
                    result["reason"],
                )
            )
        )
        return
    print(f"{result['status']}\t-\t-\t-\t-\t-\t{result['reason']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npu-smi-bin",
        default=os.environ.get("NPU_SMI_BIN") or shutil.which("npu-smi") or "",
    )
    parser.add_argument("--min-free-ratio", type=float, default=0.92)
    parser.add_argument("--candidate-devices", default="")
    parser.add_argument("--selection-attempt", type=int, default=1)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    args = parser.parse_args(argv)
    if not args.npu_smi_bin:
        parser.error("npu-smi is not available")
    if not 0 < args.min_free_ratio <= 1:
        parser.error("--min-free-ratio must be in the interval (0, 1]")
    if args.selection_attempt <= 0:
        parser.error("--selection-attempt must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result, exit_code = select_from_system(args)
    emit_result(result, args.format)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
