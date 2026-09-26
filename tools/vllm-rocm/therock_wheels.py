#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt TheRock (ROCm 10) wheels for the vLLM ROCm wheel index.

repack-torch: TheRock's torch wheel hard-pins the triton it was built with.
    vLLM ships a newer source-built triton, so rewrite that one Requires-Dist
    and the METADATA hash in RECORD. The version and filename stay the same,
    because torchvision and the amd-torch-device-* wheels pin torch exactly.

external-links: list the TheRock wheels the index links to instead of hosting
    (ROCm SDK, device kernels, torchvision/torchaudio, rocm-bootstrap), with
    versions and GPU arches read from docker/Dockerfile.rocm_base.

patch-amdsmi: rocm-sdk-core ships the amdsmi bindings under
    _rocm_sdk_core/share/amd_smi, and the loader finds libamd_smi relative to
    that directory. Once installed as its own package it would miss the SDK
    library, so also look it up through the installed _rocm_sdk_core package.
"""

import argparse
import base64
import hashlib
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin


def _record_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode()}"


def repack_torch(wheel: Path, triton_version: str, out_dir: Path) -> Path:
    with zipfile.ZipFile(wheel) as src:
        infos = src.infolist()
        metadata = next(i for i in infos if i.filename.endswith(".dist-info/METADATA"))
        record = next(i for i in infos if i.filename.endswith(".dist-info/RECORD"))

        text = src.read(metadata).decode()
        new_text, count = re.subn(
            r"^Requires-Dist: triton\s*==\s*\S+\s*$",
            f"Requires-Dist: triton=={triton_version}",
            text,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise SystemExit(
                f"expected exactly one pinned triton requirement, found {count}"
            )
        new_metadata = new_text.encode()

        lines = src.read(record).decode().splitlines()
        prefix = metadata.filename + ","
        replaced = [
            f"{metadata.filename},{_record_hash(new_metadata)},{len(new_metadata)}"
            if line.startswith(prefix)
            else line
            for line in lines
        ]
        if replaced == lines:
            raise SystemExit(f"{metadata.filename} not found in RECORD")
        new_record = ("\n".join(replaced) + "\n").encode()

        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / wheel.name
        with zipfile.ZipFile(out, "w") as dst:
            for info in infos:
                if info.filename == metadata.filename:
                    data = new_metadata
                elif info.filename == record.filename:
                    data = new_record
                else:
                    data = src.read(info)
                dst.writestr(info, data, compress_type=info.compress_type)
    return out


_AMDSMI_ANCHOR = "    possible_locations.append(libamd_smi_path)\n"
_AMDSMI_PATCH = """\
    # vLLM: installed as its own package, find the lib in TheRock's rocm-sdk-core.
    try:
        import importlib.util
        _sdk_core = importlib.util.find_spec("_rocm_sdk_core")
        if _sdk_core is not None and _sdk_core.origin:
            possible_locations.append(
                Path(_sdk_core.origin).parent / "lib/libamd_smi.so.27")
    except Exception:
        pass
"""


def patch_amdsmi(src_dir: Path, out_dir: Path) -> Path:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(src_dir, out_dir, ignore=shutil.ignore_patterns("tests"))
    wrapper = out_dir / "amdsmi" / "amdsmi_wrapper.py"
    text = wrapper.read_text()
    if text.count(_AMDSMI_ANCHOR) != 1:
        raise SystemExit(f"amdsmi loader anchor not found in {wrapper}")
    wrapper.write_text(text.replace(_AMDSMI_ANCHOR, _AMDSMI_ANCHOR + _AMDSMI_PATCH))
    return out_dir


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


def external_links(dockerfile: Path) -> list[str]:
    args = dockerfile_args(dockerfile)
    index = args["ROCM_RELEASE_WHEELS_MULTIARCH_URL"]
    sdk, torch_v = args["ROCM_SDK_VERSION"], args["TORCH_VERSION"]
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
            (f"amd-torch-device-{arch}", torch_v),
            (f"amd-torchvision-device-{arch}", vision_v),
        ]
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
    p = sub.add_parser("repack-torch")
    p.add_argument("wheel", type=Path)
    p.add_argument("--triton-version", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("external-links")
    p.add_argument(
        "--dockerfile", type=Path, default=Path("docker/Dockerfile.rocm_base")
    )
    p = sub.add_parser("patch-amdsmi")
    p.add_argument("src_dir", type=Path)
    p.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.cmd == "repack-torch":
        print(repack_torch(args.wheel, args.triton_version, args.out_dir))
    elif args.cmd == "external-links":
        print("\n".join(external_links(args.dockerfile)))
    else:
        out = args.out_dir or Path(tempfile.mkdtemp()) / "amd_smi"
        print(patch_amdsmi(args.src_dir, out))


if __name__ == "__main__":
    sys.exit(main())
