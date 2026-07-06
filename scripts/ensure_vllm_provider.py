# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import importlib
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure that the active Python environment exposes the top-level "
            "vllm package from exactly one installed distribution."
        )
    )
    parser.add_argument(
        "--module-name",
        default="vllm",
        help="Top-level import name to validate (default: vllm).",
    )
    parser.add_argument(
        "--expected-distribution",
        default=None,
        help=(
            "Distribution name that is expected to provide the top-level "
            "module (default: the current checkout's pyproject.toml name)."
        ),
    )
    parser.add_argument(
        "--remove-conflicts",
        action="store_true",
        help=(
            "Uninstall extra distributions that also provide the same top-level "
            "module, keeping only --expected-distribution."
        ),
    )
    return parser.parse_args()


def providers_for_module(module_name: str) -> list[str]:
    return sorted(set(importlib_metadata.packages_distributions().get(module_name, [])))


def distribution_metadata_path(distribution_name: str) -> Path:
    distribution = importlib_metadata.distribution(distribution_name)
    return Path(str(distribution._path)).resolve()


def pyproject_distribution_name() -> str | None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    if tomllib is not None:
        with pyproject_path.open("rb") as handle:
            pyproject: dict[str, Any] = tomllib.load(handle)
        project_name = pyproject.get("project", {}).get("name")
        return project_name if isinstance(project_name, str) else None

    in_project_section = False
    for raw_line in pyproject_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[project]":
            in_project_section = True
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project_section = False
        if in_project_section and line.startswith("name"):
            _, value = line.split("=", 1)
            return value.strip().strip('"').strip("'") or None
    return None


def resolve_expected_distribution(
    requested_distribution: str | None,
    providers: list[str],
) -> str:
    repo_distribution = pyproject_distribution_name()
    if repo_distribution and repo_distribution in providers:
        if requested_distribution and requested_distribution != repo_distribution:
            print(
                f"Expected distribution {requested_distribution!r} was requested, "
                f"but current checkout builds {repo_distribution!r}; using "
                f"{repo_distribution!r}.",
                file=sys.stderr,
            )
        return repo_distribution
    if requested_distribution:
        return requested_distribution
    if repo_distribution:
        return repo_distribution
    raise RuntimeError(
        "Unable to resolve expected distribution from --expected-distribution "
        "or pyproject.toml"
    )


def filter_repo_local_shadow_metadata(
    providers: list[str],
    expected_distribution: str,
) -> list[str]:
    metadata_paths = {
        provider: distribution_metadata_path(provider) for provider in providers
    }
    expected_path = metadata_paths.get(expected_distribution)

    if expected_path is None or REPO_ROOT not in expected_path.parents:
        return providers

    filtered_providers = []
    for provider in providers:
        provider_path = metadata_paths[provider]
        if provider != expected_distribution and REPO_ROOT in provider_path.parents:
            continue
        filtered_providers.append(provider)

    return filtered_providers


def uninstall_distribution(distribution_name: str) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        distribution_name,
    ]
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    expected_distribution = resolve_expected_distribution(
        args.expected_distribution,
        providers_for_module(args.module_name),
    )
    providers = filter_repo_local_shadow_metadata(
        providers_for_module(args.module_name),
        expected_distribution,
    )

    if not providers:
        print(
            f"No installed distribution provides top-level {args.module_name!r} ",
            f"for interpreter {sys.executable}",
            file=sys.stderr,
            sep="",
        )
        return 1

    if expected_distribution not in providers:
        print(
            f"Top-level {args.module_name!r} is provided by {providers}, but ",
            f"expected distribution {expected_distribution!r} is missing",
            file=sys.stderr,
            sep="",
        )
        return 1

    conflicts = [
        provider for provider in providers if provider != expected_distribution
    ]
    if conflicts and args.remove_conflicts:
        for provider in conflicts:
            print(
                f"Removing conflicting distribution {provider!r} because it also "
                f"provides top-level {args.module_name!r}"
            )
            uninstall_distribution(provider)
        providers = filter_repo_local_shadow_metadata(
            providers_for_module(args.module_name),
            expected_distribution,
        )
        conflicts = [
            provider for provider in providers if provider != expected_distribution
        ]

    if conflicts:
        print(
            f"Conflicting distributions still provide top-level {args.module_name!r}: ",
            f"{providers}. Remove all but {expected_distribution!r}.",
            file=sys.stderr,
            sep="",
        )
        return 1

    module = importlib.import_module(args.module_name)
    module_path = getattr(module, "__file__", None)
    if not module_path:
        print(
            f"Imported {args.module_name!r} without a concrete module path",
            file=sys.stderr,
        )
        return 1

    print(f"module={args.module_name}")
    print(f"provider={expected_distribution}")
    print(f"import_path={module_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
