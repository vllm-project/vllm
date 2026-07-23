# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from pathlib import Path

NIXL_PACKAGE = "nixl"
NIXL_CUDA12_PACKAGE = "nixl-cu12"
NIXL_CUDA13_PACKAGE = "nixl-cu13"
CUDA13_MAJOR_VERSION = 13
COMMENT_PREFIX = "#"


def get_runtime_specific_nixl_package(cuda_major: int) -> str:
    if cuda_major >= CUDA13_MAJOR_VERSION:
        return NIXL_CUDA13_PACKAGE
    return NIXL_CUDA12_PACKAGE


def split_inline_comment(line: str) -> tuple[str, str]:
    body, separator, comment = line.partition(COMMENT_PREFIX)
    if not separator:
        return line, ""
    return body, f"{separator}{comment}"


def resolve_requirement_line(line: str, cuda_major: int) -> str:
    body, comment = split_inline_comment(line)
    stripped_body = body.lstrip()
    if not stripped_body:
        return line

    if not stripped_body.startswith(NIXL_PACKAGE):
        return line

    suffix = stripped_body[len(NIXL_PACKAGE) :]
    if suffix and not (suffix[0].isspace() or suffix[0] in "<>!=~"):
        return line

    indentation = body[: len(body) - len(stripped_body)]
    package_name = get_runtime_specific_nixl_package(cuda_major)
    return f"{indentation}{package_name}{suffix}{comment}"


def resolve_kv_connector_requirements(
    requirements_text: str,
    cuda_major: int,
) -> str:
    resolved_lines: list[str] = []
    replaced_requirement = False

    for line in requirements_text.splitlines(keepends=True):
        resolved_line = resolve_requirement_line(line, cuda_major)
        if resolved_line != line:
            replaced_requirement = True
        resolved_lines.append(resolved_line)

    if not replaced_requirement:
        msg = "Could not find a nixl requirement to resolve."
        raise ValueError(msg)

    return "".join(resolved_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve kv connector requirements to a CUDA-runtime-specific NIXL wheel."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the kv connector requirements file.",
    )
    parser.add_argument(
        "--cuda-major",
        type=int,
        required=True,
        help="CUDA major version used to select nixl-cu12 or nixl-cu13.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the resolved requirements file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    resolved = resolve_kv_connector_requirements(
        input_path.read_text(),
        args.cuda_major,
    )
    output_path.write_text(resolved)


if __name__ == "__main__":
    main()
