import argparse
import importlib
from importlib import metadata as importlib_metadata
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure that the active Python environment exposes the top-level "
            "vllm package from exactly one installed distribution."
        ))
    parser.add_argument(
        "--module-name",
        default="vllm",
        help="Top-level import name to validate (default: vllm).",
    )
    parser.add_argument(
        "--expected-distribution",
        default="vllm-hust",
        help=(
            "Distribution name that is expected to provide the top-level "
            "module (default: vllm-hust)."
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
    return sorted(
        set(importlib_metadata.packages_distributions().get(module_name, [])))


def distribution_metadata_path(distribution_name: str) -> Path:
    distribution = importlib_metadata.distribution(distribution_name)
    return Path(str(distribution._path)).resolve()


def filter_repo_local_shadow_metadata(
    providers: list[str],
    expected_distribution: str,
) -> list[str]:
    repo_root = Path(__file__).resolve().parent.parent
    metadata_paths = {
        provider: distribution_metadata_path(provider)
        for provider in providers
    }
    expected_path = metadata_paths.get(expected_distribution)

    if expected_path is None or repo_root not in expected_path.parents:
        return providers

    filtered_providers = []
    for provider in providers:
        provider_path = metadata_paths[provider]
        if provider != expected_distribution and repo_root in provider_path.parents:
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
    providers = filter_repo_local_shadow_metadata(
        providers_for_module(args.module_name),
        args.expected_distribution,
    )

    if not providers:
        print(
            f"No installed distribution provides top-level {args.module_name!r} ",
            f"for interpreter {sys.executable}",
            file=sys.stderr,
            sep="",
        )
        return 1

    if args.expected_distribution not in providers:
        print(
            f"Top-level {args.module_name!r} is provided by {providers}, but ",
            f"expected distribution {args.expected_distribution!r} is missing",
            file=sys.stderr,
            sep="",
        )
        return 1

    conflicts = [
        provider for provider in providers
        if provider != args.expected_distribution
    ]
    if conflicts and args.remove_conflicts:
        for provider in conflicts:
            print(
                f"Removing conflicting distribution {provider!r} because it also "
                f"provides top-level {args.module_name!r}")
            uninstall_distribution(provider)
        providers = filter_repo_local_shadow_metadata(
            providers_for_module(args.module_name),
            args.expected_distribution,
        )
        conflicts = [
            provider for provider in providers
            if provider != args.expected_distribution
        ]

    if conflicts:
        print(
            f"Conflicting distributions still provide top-level {args.module_name!r}: ",
            f"{providers}. Remove all but {args.expected_distribution!r}.",
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
    print(f"provider={args.expected_distribution}")
    print(f"import_path={module_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())