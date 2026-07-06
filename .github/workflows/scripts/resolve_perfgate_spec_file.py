# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ASCEND_910B_CHIP_PATTERN = re.compile(r"910B\d+")


def detect_chip_model_from_text(text: str) -> str:
    normalized = text.upper().replace(" ", "")
    match = ASCEND_910B_CHIP_PATTERN.search(normalized)
    return match.group(0) if match else ""


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


def normalize_chip_model(chip_model: str) -> str:
    return chip_model.upper().replace(" ", "")


def load_shared_perfgate_resolver(benchmark_repo: Path):
    src_dir = benchmark_repo / "src"
    if not src_dir.is_dir():
        raise ValueError(f"benchmark repo src directory not found: {src_dir}")

    for module_name in list(sys.modules):
        if module_name == "vllm_hust_benchmark" or module_name.startswith(
            "vllm_hust_benchmark."
        ):
            sys.modules.pop(module_name, None)
    sys.path.insert(0, str(src_dir))
    try:
        from vllm_hust_benchmark import perfgate_specs
    except Exception as exc:
        raise ValueError(
            "failed to import vllm_hust_benchmark.perfgate_specs from "
            f"{src_dir}; ensure vllm-hust-benchmark contains the shared "
            "perfgate spec resolver"
        ) from exc

    return perfgate_specs.resolve_perfgate_spec_file


def repo_relative_spec_file(spec_path: Path, benchmark_repo: Path) -> str:
    resolved_spec_path = spec_path.resolve()
    resolved_repo = benchmark_repo.resolve()
    try:
        return resolved_spec_path.relative_to(resolved_repo).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"resolved perfgate spec escapes benchmark repo: {resolved_spec_path}"
        ) from exc


def resolve_spec_file(
    *,
    explicit_spec_file: str,
    scenario: str,
    explicit_chip_model: str,
    npu_smi_bin: str,
    benchmark_repo: str,
) -> tuple[str, str, Path | None]:
    if explicit_spec_file:
        chip_model = normalize_chip_model(
            explicit_chip_model or detect_chip_model_from_text(explicit_spec_file)
        )
        spec_path = None
        if benchmark_repo:
            explicit_path = Path(explicit_spec_file)
            if explicit_path.is_absolute():
                spec_path = explicit_path
            else:
                spec_path = Path(benchmark_repo) / explicit_spec_file
            if not spec_path.is_file():
                raise ValueError(f"explicit perfgate spec file not found: {spec_path}")
        return explicit_spec_file, chip_model, spec_path

    chip_model = explicit_chip_model or detect_chip_model_from_npu_smi(npu_smi_bin)
    chip_model = normalize_chip_model(chip_model)
    if not chip_model:
        raise ValueError(
            "unable to resolve perfgate spec file because Ascend chip model "
            "is unknown; set VLLM_HUST_PERFGATE_HARDWARE_CHIP_MODEL or ensure "
            "npu-smi reports the chip model"
        )
    if not benchmark_repo:
        raise ValueError(
            "VLLM_HUST_BENCHMARK_REPO is required to resolve perfgate spec "
            "from the shared registry"
        )

    benchmark_repo_path = Path(benchmark_repo)
    shared_resolver = load_shared_perfgate_resolver(benchmark_repo_path)
    spec_path = Path(
        shared_resolver(
            scenario=scenario,
            hardware_chip_model=chip_model,
            repo_root=benchmark_repo_path,
        )
    )
    if not spec_path.is_absolute():
        spec_path = benchmark_repo_path / spec_path
    if not spec_path.is_file():
        raise ValueError(f"resolved perfgate spec file not found: {spec_path}")

    spec_file = repo_relative_spec_file(spec_path, benchmark_repo_path)
    return spec_file, chip_model, spec_path


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
        "--scenario",
        default=os.environ.get("BENCH_SCENARIO", "random-online").strip(),
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
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--github-env",
        default=os.environ.get("GITHUB_ENV", "").strip(),
    )
    args = parser.parse_args()

    try:
        spec_file, chip_model, spec_path = resolve_spec_file(
            explicit_spec_file=args.explicit_spec_file,
            scenario=args.scenario,
            explicit_chip_model=args.explicit_chip_model,
            npu_smi_bin=args.npu_smi_bin,
            benchmark_repo=args.benchmark_repo,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

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
