# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import platform
import tarfile
from pathlib import Path
from shutil import copy, copytree, which
from tempfile import TemporaryDirectory
from typing import Optional

from packaging.version import Version
from torch.utils.cpp_extension import CUDA_HOME


def is_sccache_available() -> bool:
    return which("sccache") is not None


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def is_url_available(url: str) -> bool:
    from urllib.request import urlopen

    status = None
    try:
        with urlopen(url) as f:
            status = f.status
    except Exception:
        return False
    return status == 200


def open_url(url: str, timeout: int = 300):
    from urllib.request import Request, urlopen
    headers = {
        'User-Agent':
        'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) '
        'Gecko/20100101 Firefox/119.0',
    }
    return urlopen(Request(url, headers=headers), timeout=timeout)


def download_extract_copy(url: str, install_paths: dict[Path, Path]):
    with tarfile.open(fileobj=open_url(url), mode="r|*") as file, \
            TemporaryDirectory() as tmp_path:
        file.extractall(path=tmp_path)

        for archive_path, output_path in install_paths.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            src_path = tmp_path / archive_path
            if os.path.isdir(src_path):
                copytree(src_path, output_path, dirs_exist_ok=True)
            else:
                copy(src_path, output_path)


def download_toolchain(nvcc_version: str, ptxas_version: str, dst_path: Path):
    system = platform.system().lower()
    arch = platform.machine()
    arch = {"arm64": "aarch64"}.get(arch, arch)

    download_extract_copy(
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/"
        f"{system}-{arch}/cuda_nvcc-{system}-{arch}-{nvcc_version}-archive.tar.xz",
        {
            Path(f"cuda_nvcc-{system}-{arch}-{nvcc_version}-archive/bin"):
            dst_path / "bin",
        },
    )
    download_extract_copy(
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/"
        f"{system}-{arch}/cuda_nvcc-{system}-{arch}-{ptxas_version}-archive.tar.xz",
        {
            Path(f"cuda_nvcc-{system}-{arch}-{ptxas_version}-archive/bin/ptxas"):
            dst_path / "bin",
            Path(f"cuda_nvcc-{system}-{arch}-{ptxas_version}-archive/nvvm/bin"):
            dst_path / "nvvm/bin",
        },
    )


def maybe_patch_compiler(
        cuda_version: Optional[Version] = None) -> dict[Path, Path]:
    if cuda_version is None or cuda_version != Version("12.6"):
        return {}
    assert CUDA_HOME is not None, "CUDA_HOME is not set"

    nvcc_version = "12.6.85"
    ptxas_version = "12.8.93"
    temp_path = Path(__file__).parent / ".deps/nvidia-toolchain"
    download_toolchain(nvcc_version, ptxas_version, temp_path)
    return {
        Path(CUDA_HOME) / "bin/ptxas": temp_path / "bin/ptxas",
        Path(CUDA_HOME) / "nvvm/bin/cicc": temp_path / "nvvm/bin/cicc",
    }


class OverrideFiles:

    def __init__(self, override_map: dict[Path, Path]):
        self.override_map = override_map

    def __enter__(self):
        for target, destination in self.override_map.items():
            target.rename(target.with_suffix('.backup'))
            target.symlink_to(destination)

    def __exit__(self, exc_type, exc_value, traceback):
        for target in self.override_map:
            target.unlink()
            target.with_suffix('.backup').rename(target)
