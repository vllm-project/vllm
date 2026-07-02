# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CHIP_TO_SPEC = {
    "910B2": "docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json",
    "910B3": "docs/official-baselines/perfgate-ascend-qwen25-3b-910b3.json",
}


def detect_chip_model_from_text(text: str) -> str:
    normalized = text.upper().replace(" ", "")
    for chip_model in sorted(CHIP_TO_SPEC, reverse=True):
        if chip_model in normalized:
            return chip_model

    return ""


def detect_chip_model_from_npu_smi(npu_smi_bin: str) -> str:
    if not npu_smi_bin:
        return ""

    for args in (("info",), ("info", "-t", "board")):
        try:
            result = subprocess.run(
                [npu_smi_bin, *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            continue

        chip_model = detect_chip_model_from_text(result.stdout + "\n" + result.stderr)
        if chip_model:
            return chip_model

    return ""


def resolve_spec_file(
    *,
    explicit_spec_file: str,
    explicit_chip_model: str,
    npu_smi_bin: str,
) -> tuple[str, str]:
    if explicit_spec_file:
        chip_model = explicit_chip_model or detect_chip_model_from_text(
            explicit_spec_file
        )
        return explicit_spec_file, chip_model

    chip_model = explicit_chip_model or detect_chip_model_from_npu_smi(npu_smi_bin)
    chip_model = chip_model.upper().replace(" ", "")
    spec_file = CHIP_TO_SPEC.get(chip_model)
    if not spec_file:
        supported = ", ".join(sorted(CHIP_TO_SPEC))
        raise ValueError(
            "unable to resolve perfgate spec file for Ascend chip model "
            f"{chip_model or '<unknown>'}; supported: {supported}"
        )

    return spec_file, chip_model


def write_github_env(env_file: str, values: dict[str, str]) -> None:
    if not env_file:
        return

    with Path(env_file).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the PR perfgate same-spec file for the current Ascend runner."
        )
    )
    parser.add_argument(
        "--explicit-spec-file",
        default=os.environ.get("PERFGATE_SPEC_FILE", "").strip(),
    )
    parser.add_argument(
        "--explicit-chip-model",
        default=os.environ.get("HARDWARE_CHIP_MODEL", "").strip(),
    )
    parser.add_argument(
        "--npu-smi-bin",
        default=os.environ.get("NPU_SMI_BIN", "npu-smi").strip(),
    )
    parser.add_argument(
        "--benchmark-repo",
        default=os.environ.get("VLLM_HUST_BENCHMARK_REPO", "").strip(),
    )
    parser.add_argument(
        "--fallback-spec-file",
        default=os.environ.get("MAIN_SAME_SPEC_SPEC_FILE", "").strip(),
    )
    parser.add_argument(
        "--github-env",
        default=os.environ.get("GITHUB_ENV", "").strip(),
    )
    args = parser.parse_args()

    try:
        spec_file, chip_model = resolve_spec_file(
            explicit_spec_file=args.explicit_spec_file,
            explicit_chip_model=args.explicit_chip_model,
            npu_smi_bin=args.npu_smi_bin,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    spec_path = None
    if args.benchmark_repo:
        spec_path = Path(args.benchmark_repo) / spec_file
        if not spec_path.is_file():
            if args.explicit_spec_file or not args.fallback_spec_file:
                print(
                    f"resolved perfgate spec file not found: {spec_path}",
                    file=sys.stderr,
                )
                return 2

            fallback_path = Path(args.benchmark_repo) / args.fallback_spec_file
            if not fallback_path.is_file():
                print(
                    "resolved perfgate spec file not found: "
                    f"{spec_path}; fallback same-spec file not found: {fallback_path}",
                    file=sys.stderr,
                )
                return 2

            print(
                "resolved perfgate spec file not found: "
                f"{spec_path}; falling back to {fallback_path}",
                file=sys.stderr,
            )
            spec_file = args.fallback_spec_file
            spec_path = fallback_path

    values = {"PERFGATE_SPEC_FILE": spec_file}
    if spec_path is not None:
        values["SAME_SPEC_SPEC_FILE"] = str(spec_path)
    if chip_model:
        values["HARDWARE_CHIP_MODEL"] = chip_model
        values["SOC_VERSION"] = f"ascend{chip_model.lower()}"

    write_github_env(args.github_env, values)
    for key, value in values.items():
        print(f"{key}={value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
