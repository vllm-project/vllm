# usage: python -m vllm.tools.install_nccl --cuda 11 --nccl 2.18.3
# after installation, files are available in `{sys.prefix}/vllm_nccl` directory

import os
import platform
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class DistInfo:
    cuda_version: str
    full_version: str
    public_version: str
    filename_linux: str

    def get_url(self, architecture: str) -> str:
        url_temp = "https://developer.download.nvidia.com/compute/redist/nccl/v{}/{}".format(
            self.public_version, self.filename_linux)
        return url_temp.replace("x86_64", architecture)


# taken from https://developer.download.nvidia.com/compute/redist/nccl/
available_dist_info = [
    # nccl 2.16.5
    DistInfo('11.8', '2.16.5', '2.16.5', 'nccl_2.16.5-1+cuda11.8_x86_64.txz'),
    DistInfo('12.0', '2.16.5', '2.16.5', 'nccl_2.16.5-1+cuda12.0_x86_64.txz'),
    # nccl 2.17.1
    DistInfo('11.0', '2.17.1', '2.17.1', 'nccl_2.17.1-1+cuda11.0_x86_64.txz'),
    DistInfo('12.0', '2.17.1', '2.17.1', 'nccl_2.17.1-1+cuda12.0_x86_64.txz'),
    # nccl 2.18.1
    DistInfo('11.0', '2.18.1', '2.18.1', 'nccl_2.18.1-1+cuda11.0_x86_64.txz'),
    DistInfo('12.0', '2.18.1', '2.18.1', 'nccl_2.18.1-1+cuda12.0_x86_64.txz'),
    # nccl 2.20.3
    DistInfo('11.0', '2.20.3', '2.20.3', 'nccl_2.20.3-1+cuda11.0_x86_64.txz'),
    DistInfo('12.2', '2.20.3', '2.20.3', 'nccl_2.20.3-1+cuda12.2_x86_64.txz'),
]

if __name__ == "__main__":

    from argparse import ArgumentParser

    args = ArgumentParser(description="Install NCCL package for VLLM")
    args.add_argument("--cuda",
                      type=str,
                      required=True,
                      default="12",
                      help="Major CUDA version",
                      choices=["11", "12"])
    args.add_argument("--nccl",
                      type=str,
                      required=True,
                      default="2.18",
                      help="Major NCCL version",
                      choices=["2.20", "2.18", "2.17", "2.16"])

    args = args.parse_args()

    architecture = platform.machine()
    if architecture not in ["x86_64", "aarch64", "ppc64le"]:
        print(
            f"Unsupported architecture: {architecture}, using x86_64 instead.")
        architecture = "x86_64"

    nccl_major_version, cuda_major_version = args.nccl, args.cuda

    url = None

    for each in available_dist_info:
        if each.cuda_version.split(
                ".")[0] == cuda_major_version and each.full_version.startswith(
                    nccl_major_version):
            url = each.get_url(architecture)
            break

    assert url is not None, \
        (
            "Could not find a suitable nccl package for cuda"
        f" {cuda_major_version} and nccl {nccl_major_version}"
        )

    print(f"Downloading nccl package from {url}")
    import sys

    dest_path = f"{sys.prefix}/vllm_nccl"
    os.makedirs(dest_path, exist_ok=True)

    file_path = f"{dest_path}/{each.filename_linux}"
    dir_path = file_path[:-4]

    # download from url
    if not os.path.exists(dir_path):
        import os
        import shutil

        import requests

        # download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)

        # extract the file
        import tarfile
        with tarfile.open(file_path) as f:
            f.extractall(dir_path)

    logger.info(f"NCCL package downloaded and extracted to {dir_path}")
