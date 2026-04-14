#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate ROCm Docker base-image and architecture compatibility."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.rocm_base"
MIN_ROCM_VERSION_BY_ARCH = {
    "gfx1150": (7, 2),
    "gfx1151": (7, 2),
}


def _extract_arg_default(dockerfile_text: str, arg_name: str) -> str | None:
    prefix = f"ARG {arg_name}="
    for line in dockerfile_text.splitlines():
        if not line.startswith(prefix):
            continue
        return line.removeprefix(prefix).strip().strip("\"'")
    return None


def _parse_rocm_version(base_image: str) -> tuple[int, int] | None:
    if ":" not in base_image:
        return None
    tag = base_image.rsplit(":", maxsplit=1)[1]
    version_token = tag.split("-", maxsplit=1)[0]
    version_parts = version_token.split(".")
    if len(version_parts) < 2:
        return None

    try:
        return int(version_parts[0]), int(version_parts[1])
    except ValueError:
        return None


def validate_rocm_docker_config(dockerfile_path: Path) -> list[str]:
    dockerfile_text = dockerfile_path.read_text()
    errors: list[str] = []

    base_image = _extract_arg_default(dockerfile_text, "BASE_IMAGE")
    if base_image is None:
        return [f"{dockerfile_path} is missing ARG BASE_IMAGE."]

    rocm_version = _parse_rocm_version(base_image)
    if rocm_version is None:
        return [f"Could not parse ROCm version from BASE_IMAGE={base_image!r}."]

    rocm_arch = _extract_arg_default(dockerfile_text, "PYTORCH_ROCM_ARCH")
    if rocm_arch is None:
        return [f"{dockerfile_path} is missing ARG PYTORCH_ROCM_ARCH."]

    advertised_arches = {arch.strip() for arch in rocm_arch.split(";") if arch.strip()}

    for arch, min_version in MIN_ROCM_VERSION_BY_ARCH.items():
        if arch not in advertised_arches:
            continue
        if rocm_version >= min_version:
            continue

        required = f"{min_version[0]}.{min_version[1]}"
        actual = f"{rocm_version[0]}.{rocm_version[1]}"
        errors.append(
            f"{dockerfile_path} advertises {arch} in PYTORCH_ROCM_ARCH but "
            f"BASE_IMAGE uses ROCm {actual}; ROCm {required}+ is required "
            f"for gfx1150/gfx1151 (see #31333)."
        )

    return errors


def main() -> int:
    errors = validate_rocm_docker_config(DOCKERFILE)
    if not errors:
        return 0

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
