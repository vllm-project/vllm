#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# do not complain about line length (for docstring)
# ruff: noqa: E501

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

if not sys.version_info >= (3, 10):
    raise RuntimeError("This script requires Python 3.10 or higher.")

INDEX_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
  <meta name="pypi:repository-version" content="1.0">
  <body>
{items}
  </body>
</html>
"""


@dataclass
class WheelFileInfo:
    package_name: str
    version: str
    build_tag: str | None
    python_tag: str
    abi_tag: str
    platform_tag: str
    variant: str | None
    filename: str


def parse_from_filename(file: str) -> WheelFileInfo:
    """
    Parse wheel file name to extract metadata.

    The format of wheel names:
        {package_name}-{version}(-{build_tag})?-{python_tag}-{abi_tag}-{platform_tag}.whl
    All versions could contain a variant like '+cu129' or '.cpu' or `.rocm` (or not).
    Example:
        vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl
        vllm-0.10.2rc2+cu129-cp38-abi3-manylinux2014_aarch64.whl
        vllm-0.11.1rc8.dev14+gaa384b3c0-cp38-abi3-manylinux2014_aarch64.whl
        vllm-0.11.1rc8.dev14+gaa384b3c0.cu130-cp38-abi3-manylinux1_x86_64.whl
    """
    wheel_file_re = re.compile(
        r"^(?P<package_name>.+)-(?P<version>[^-]+?)(-(?P<build_tag>[^-]+))?-(?P<python_tag>[^-]+)-(?P<abi_tag>[^-]+)-(?P<platform_tag>[^-]+)\.whl$"
    )
    match = wheel_file_re.match(file)
    if not match:
        raise ValueError(f"Invalid wheel file name: {file}")

    package_name = match.group("package_name")
    version = match.group("version")
    build_tag = match.group("build_tag")
    python_tag = match.group("python_tag")
    abi_tag = match.group("abi_tag")
    platform_tag = match.group("platform_tag")

    # extract variant from version
    variant = None
    if "dev" in version:
        ver_after_dev = version.split("dev")[-1]
        if "." in ver_after_dev:
            variant = ver_after_dev.split(".")[-1]
            version = version.removesuffix("." + variant)
    else:
        if "+" in version:
            version, variant = version.split("+")

    return WheelFileInfo(
        package_name=package_name,
        version=version,
        build_tag=build_tag,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
        variant=variant,
        filename=file,
    )


def generate_project_list(subdir_names: list[str]) -> str:
    """
    Generate project list HTML content linking to each project & variant sub-directory.
    """
    href_tags = []
    for name in sorted(subdir_names):
        name = name.strip("/").strip(".")
        href_tags.append(f'    <a href="{name}/">{name}/</a><br/>')
    return INDEX_HTML_TEMPLATE.format(items="\n".join(href_tags))


def generate_package_index_and_metadata(
    wheel_files: list[WheelFileInfo], wheel_base_dir: Path, index_base_dir: Path
) -> tuple[str, str]:
    """
    Generate package index HTML content for a specific package, linking to actual wheel files.
    """
    href_tags = []
    metadata = []
    for file in sorted(wheel_files, key=lambda x: x.filename):
        relative_path = (
            wheel_base_dir.relative_to(index_base_dir, walk_up=True) / file.filename
        )
        href_tags.append(
            f'    <a href="{quote(relative_path.as_posix())}">{file.filename}</a><br/>'
        )
        file_meta = asdict(file)
        file_meta["path"] = relative_path.as_posix()
        metadata.append(file_meta)
    index_str = INDEX_HTML_TEMPLATE.format(items="\n".join(href_tags))
    metadata_str = json.dumps(metadata, indent=2)
    return index_str, metadata_str


def generate_index_and_metadata(
    whl_files: list[str],
    wheel_base_dir: Path,
    index_base_dir: Path,
    default_variant: str | None = None,
    alias_to_default: str | None = None,
):
    """
    Generate index for all wheel files.

    Args:
        whl_files (list[str]): List of wheel files (must be directly under `wheel_base_dir`).
        wheel_base_dir (Path): Base directory for wheel files.
        index_base_dir (Path): Base directory to store index files.
        default_variant (str | None): The default variant name, if any.
        alias_to_default (str | None): Alias variant name for the default variant, if any.

    First, parse all wheel files to extract metadata.
    We need to collect all wheel files for each variant, and generate an index for it (in a sub-directory).
    The index for the default variant (if any) is generated in the root index directory.

    If `default_variant` is provided, all wheels must have variant suffixes, and the default variant index
    is purely a copy of the corresponding variant index, with only the links adjusted.
    Otherwise, all wheels without variant suffixes are treated as the default variant.

    If `alias_to_default` is provided, an additional alias sub-directory is created, it has the same content
    as the default variant index, but the links are adjusted accordingly.

    Index directory structure:
        index_base_dir/ (hosted at wheels.vllm.ai/{nightly,$commit,$version}/)
            index.html  # project list, linking to "vllm/" and other packages, and all variant sub-directories
            vllm/
                index.html # package index, pointing to actual files in wheel_base_dir (relative path)
                metadata.json # machine-readable metadata for all wheels in this package
            cpu/ # cpu variant sub-directory
                index.html
                vllm/
                    index.html
                    metadata.json
            cu129/ # cu129 is actually the alias to default variant
                index.html
                vllm/
                    index.html
                    metadata.json
            cu130/ # cu130 variant sub-directory
                index.html
                vllm/
                    index.html
                    metadata.json
            ...

    metadata.json stores a dump of all wheel files' metadata in a machine-readable format:
        [
            {
                "package_name": "vllm",
                "version": "0.10.2rc2",
                "build_tag": null,
                "python_tag": "cp38",
                "abi_tag": "abi3",
                "platform_tag": "manylinux2014_aarch64",
                "variant": "cu129",
                "filename": "vllm-0.10.2rc2+cu129-cp38-abi3-manylinux2014_aarch64.whl",
                "path": "../vllm-0.10.2rc2+cu129-cp38-abi3-manylinux2014_aarch64.whl" # to be concatenated with the directory URL
            },
            ...
        ]
    """

    parsed_files = [parse_from_filename(f) for f in whl_files]

    if not parsed_files:
        print("No wheel files found, skipping index generation.")
        return

    # Group by variant
    variant_to_files: dict[str, list[WheelFileInfo]] = {}
    for file in parsed_files:
        variant = file.variant or "default"
        if variant not in variant_to_files:
            variant_to_files[variant] = []
        variant_to_files[variant].append(file)

    print(f"Found variants: {list(variant_to_files.keys())}")

    # sanity check for default variant
    if default_variant:
        if "default" in variant_to_files:
            raise ValueError(
                "All wheel files must have variant suffixes when `default_variant` is specified."
            )
        if default_variant not in variant_to_files:
            raise ValueError(
                f"Default variant '{default_variant}' not found among wheel files."
            )

    if alias_to_default:
        if "default" not in variant_to_files:
            # e.g. only some wheels are uploaded to S3 currently
            print(
                "[WARN] Alias to default variant specified, but no default variant found."
            )
        elif alias_to_default in variant_to_files:
            raise ValueError(
                f"Alias variant name '{alias_to_default}' already exists among wheel files."
            )
        else:
            variant_to_files[alias_to_default] = variant_to_files["default"].copy()
            print(f"Alias variant '{alias_to_default}' created for default variant.")

    # Generate index for each variant
    subdir_names = set()
    for variant, files in variant_to_files.items():
        if variant == "default":
            variant_dir = index_base_dir
        else:
            variant_dir = index_base_dir / variant
            subdir_names.add(variant)

        variant_dir.mkdir(parents=True, exist_ok=True)

        # gather all package names in this variant
        packages = set(f.package_name for f in files)
        if variant == "default":
            # these packages should also appear in the "project list"
            # generate after all variants are processed
            subdir_names = subdir_names.union(packages)
        else:
            # generate project list for this variant directly
            project_list_str = generate_project_list(sorted(packages))
            with open(variant_dir / "index.html", "w") as f:
                f.write(project_list_str)

        for package in packages:
            # filter files belonging to this package only
            package_files = [f for f in files if f.package_name == package]
            package_dir = variant_dir / package
            package_dir.mkdir(parents=True, exist_ok=True)
            index_str, metadata_str = generate_package_index_and_metadata(
                package_files, wheel_base_dir, package_dir
            )
            with open(package_dir / "index.html", "w") as f:
                f.write(index_str)
            with open(package_dir / "metadata.json", "w") as f:
                f.write(metadata_str)

    # Generate top-level project list index
    project_list_str = generate_project_list(sorted(subdir_names))
    with open(index_base_dir / "index.html", "w") as f:
        f.write(project_list_str)


if __name__ == "__main__":
    """
    Arguments:
        --version <version> : version string for the current build (e.g., commit hash)
        --current-objects <path_to_json> : path to JSON file containing current S3 objects listing in this version directory
        --output-dir <output_directory> : directory to store generated index files
        --alias-to-default <alias_variant_name> : (optional) alias variant name for the default variant
    """

    parser = argparse.ArgumentParser(
        description="Process nightly build wheel files to generate indices."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version string for the current build (e.g., commit hash)",
    )
    parser.add_argument(
        "--current-objects",
        type=str,
        required=True,
        help="Path to JSON file containing current S3 objects listing in this version directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store generated index files",
    )
    parser.add_argument(
        "--alias-to-default",
        type=str,
        default=None,
        help="Alias variant name for the default variant",
    )

    args = parser.parse_args()

    version = args.version
    if "/" in version or "\\" in version:
        raise ValueError("Version string must not contain slashes.")
    current_objects_path = Path(args.current_objects)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Read current objects JSON
    with open(current_objects_path) as f:
        current_objects: dict[str, list[dict[str, Any]]] = json.load(f)

    # current_objects looks like from list_objects_v2 S3 API:
    """
    "Contents": [
        {
            "Key": "e2f56c309d2a28899c68975a7e104502d56deb8f/vllm-0.11.2.dev363+ge2f56c309-cp38-abi3-manylinux1_x86_64.whl",
            "LastModified": "2025-11-28T14:00:32+00:00",
            "ETag": "\"37a38339c7cdb61ca737021b968075df-52\"",
            "ChecksumAlgorithm": [
                "CRC64NVME"
            ],
            "ChecksumType": "FULL_OBJECT",
            "Size": 435649349,
            "StorageClass": "STANDARD"
        },
        ...
    ]
    """

    # Extract wheel file keys
    wheel_files = []
    for item in current_objects.get("Contents", []):
        key: str = item["Key"]
        if key.endswith(".whl"):
            wheel_files.append(key.split("/")[-1])  # only the filename is used

    print(f"Found {len(wheel_files)} wheel files for version {version}: {wheel_files}")

    # Generate index and metadata, assuming wheels and indices are stored as:
    # s3://vllm-wheels/{version}/<wheel files>
    # s3://vllm-wheels/<anything>/<index files>
    wheel_base_dir = Path(output_dir).parent / version
    index_base_dir = Path(output_dir)

    generate_index_and_metadata(
        whl_files=wheel_files,
        wheel_base_dir=wheel_base_dir,
        index_base_dir=index_base_dir,
        default_variant=None,
        alias_to_default=args.alias_to_default,
    )
    print(f"Successfully generated index and metadata in {output_dir}")
