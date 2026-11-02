#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt TheRock (ROCm 10) wheels for the vLLM ROCm wheel index.

external-links: list the TheRock wheels the index links to instead of hosting
    (ROCm SDK, device kernels, torchvision/torchaudio, rocm-bootstrap), with
    versions and GPU arches read from docker/Dockerfile.rocm_base. torch's
    device-<arch> extras can also pull in a GPU-family wheel (e.g.
    amd-torch-device-gfx110x), so those are read from the torch wheel itself.
"""

import argparse
import sys
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin

import regex as re


def dockerfile_args(dockerfile: Path) -> dict[str, str]:
    args = {}
    for m in re.finditer(r'^ARG (\w+)="?([^"\s]*)"?', dockerfile.read_text(), re.M):
        args.setdefault(m.group(1), m.group(2))
    return args


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs += [v for k, v in attrs if k == "href" and v]


def _project_files(index_url: str, project: str) -> list[str]:
    page = urljoin(index_url, f"{project}/")
    with urllib.request.urlopen(page, timeout=60) as resp:
        parser = _Links()
        parser.feed(resp.read().decode())
    return [urljoin(page, href.split("#")[0]) for href in parser.hrefs]


def _usable(url: str, version: str | None) -> bool:
    name = unquote(url.rsplit("/", 1)[-1])
    if (
        version is not None
        and f"-{version}-" not in name
        and f"-{version}." not in name
    ):
        return False
    if name.endswith(".tar.gz"):
        return True
    tags = name[: -len(".whl")].split("-")[-3:]
    return (
        name.endswith(".whl")
        and tags[0] in ("cp312", "py3")
        and (tags[2] == "any" or ("x86_64" in tags[2] and "win" not in tags[2]))
    )


def torch_device_requirements(
    torch_wheel: Path, arches: list[str]
) -> list[tuple[str, str]]:
    """(project, version) pairs that torch's device-<arch> extras require."""
    with zipfile.ZipFile(torch_wheel) as zf:
        metadata = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        text = zf.read(metadata).decode()
    wanted = {f"device-{arch}" for arch in arches}
    reqs = []
    for m in re.finditer(
        r'^Requires-Dist: ([\w.-]+)\s*==\s*(\S+)\s*;.*extra == "([^"]+)"', text, re.M
    ):
        if m.group(3) in wanted and (m.group(1), m.group(2)) not in reqs:
            reqs.append((m.group(1), m.group(2)))
    missing = wanted - set(re.findall(r'extra == "(device-[^"]+)"', text))
    if missing:
        raise SystemExit(f"torch has no extras for {sorted(missing)}")
    return reqs


def external_links(dockerfile: Path, torch_wheel: Path) -> list[str]:
    args = dockerfile_args(dockerfile)
    index = args["ROCM_RELEASE_WHEELS_MULTIARCH_URL"]
    sdk = args["ROCM_SDK_VERSION"]
    vision_v, audio_v = args["TORCHVISION_VERSION"], args["TORCHAUDIO_VERSION"]
    arches = [a for a in args["PYTORCH_ROCM_ARCH"].split(";") if a]
    wanted: list[tuple[str, str | None]] = [
        ("rocm", sdk),
        ("rocm-sdk-core", sdk),
        ("rocm-sdk-libraries", sdk),
        ("torchvision", vision_v),
        ("torchaudio", audio_v),
        ("rocm-bootstrap", None),
    ]
    for arch in arches:
        wanted += [
            (f"rocm-sdk-device-{arch}", sdk),
            (f"amd-torchvision-device-{arch}", vision_v),
        ]
    wanted += torch_device_requirements(torch_wheel, arches)
    urls = []
    for project, version in wanted:
        found = [u for u in _project_files(index, project) if _usable(u, version)]
        if not found:
            raise SystemExit(f"no usable {project}=={version} files on {index}")
        urls += found
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("external-links")
    p.add_argument(
        "--dockerfile", type=Path, default=Path("docker/Dockerfile.rocm_base")
    )
    p.add_argument("--torch-wheel", type=Path, required=True)
    args = parser.parse_args()

    if args.cmd == "external-links":
        print("\n".join(external_links(args.dockerfile, args.torch_wheel)))


if __name__ == "__main__":
    sys.exit(main())
