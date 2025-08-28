# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fnmatch
import os
import signal
from pathlib import Path
from typing import Optional

from vllm.utils import PlaceholderModule

try:
    import boto3
except ImportError:
    boto3 = PlaceholderModule("boto3")  # type: ignore[assignment]


def _filter_allow(paths: list[str], patterns: list[str]) -> list[str]:
    return [
        path for path in paths if any(
            fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]


def _filter_ignore(paths: list[str], patterns: list[str]) -> list[str]:
    return [
        path for path in paths
        if not any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]


def glob(s3=None,
         path: str = "",
         allow_pattern: Optional[list[str]] = None) -> list[str]:
    """
    List full file names from S3 path and filter by allow pattern.

    Args:
        s3: S3 client to use.
        path: The S3 path to list from.
        allow_pattern: A list of patterns of which files to pull.

    Returns:
        list[str]: List of full S3 paths allowed by the pattern
    """
    if s3 is None:
        s3 = boto3.client("s3")
    if not path.endswith("/"):
        path = path + "/"
    bucket_name, _, paths = list_files(s3,
                                       path=path,
                                       allow_pattern=allow_pattern)
    return [f"s3://{bucket_name}/{path}" for path in paths]


def list_files(
        s3,
        path: str,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None
) -> tuple[str, str, list[str]]:
    """
    List files from S3 path and filter by pattern.

    Args:
        s3: S3 client to use.
        path: The S3 path to list from.
        allow_pattern: A list of patterns of which files to pull.
        ignore_pattern: A list of patterns of which files not to pull.

    Returns:
        tuple[str, str, list[str]]: A tuple where:
            - The first element is the bucket name
            - The second element is string represent the bucket 
              and the prefix as a dir like string
            - The third element is a list of files allowed or 
              disallowed by pattern
    """
    parts = path.removeprefix('s3://').split('/')
    prefix = '/'.join(parts[1:])
    bucket_name = parts[0]

    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    paths = [obj['Key'] for obj in objects.get('Contents', [])]

    paths = _filter_ignore(paths, ["*/"])
    if allow_pattern is not None:
        paths = _filter_allow(paths, allow_pattern)

    if ignore_pattern is not None:
        paths = _filter_ignore(paths, ignore_pattern)

    return bucket_name, prefix, paths


# define global single instance
_s3 = None


def get_s3():
    global _s3
    if _s3 is None:
        _s3 = S3Model()
    return _s3


class S3Model:
    """
    A class representing a S3 model mirrored into a temporary directory.

    Attributes:
        s3: S3 client.
        dir: The temporary created directory.

    Methods:
        pull_files(): Pull model from S3 to the temporary directory.
    """
    _s3 = None

    def __init__(self) -> None:
        self.s3 = boto3.client('s3')
        for sig in (signal.SIGINT, signal.SIGTERM):
            existing_handler = signal.getsignal(sig)
            signal.signal(sig, self._close_by_signal(existing_handler))
        self.dir = "/tmp/s3/model"

    def _close_by_signal(self, existing_handler=None):

        def new_handler(signum, frame):
            if existing_handler:
                existing_handler(signum, frame)

        return new_handler

    def pull_files(self,
                   s3_model_path: str = "",
                   allow_pattern: Optional[list[str]] = None,
                   ignore_pattern: Optional[list[str]] = None) -> None:
        """
        Pull files from S3 storage into the temporary directory.

        Args:
            s3_model_path: The S3 path of the model.
            allow_pattern: A list of patterns of which files to pull.
            ignore_pattern: A list of patterns of which files not to pull.

        """
        if not s3_model_path.endswith("/"):
            s3_model_path = s3_model_path + "/"

        bucket_name, base_dir, files = list_files(self.s3, s3_model_path,
                                                  allow_pattern,
                                                  ignore_pattern)
        if len(files) == 0:
            return

        for file in files:
            destination_file = os.path.join(self.dir, file)
            local_dir = Path(destination_file).parent
            os.makedirs(local_dir, exist_ok=True)
            if os.path.exists(destination_file):
                continue
            self.s3.download_file(bucket_name, file, destination_file)

    def pull_file_name(self,
                       s3_model_path: str = "",
                       file_name: str = "") -> str:
        if not s3_model_path.endswith("/"):
            s3_model_path = s3_model_path + "/"

        bucket_name, base_dir, files = list_files(self.s3, s3_model_path, None,
                                                  None)
        destination_file = os.path.join(self.dir, base_dir, file_name)
        local_dir = Path(destination_file).parent
        os.makedirs(local_dir, exist_ok=True)
        if os.path.exists(destination_file):
            return destination_file
        self.s3.download_file(bucket_name, file_name, destination_file)
        return destination_file

    def get_model_path(self, repo_id: str):
        parts = repo_id.removeprefix('s3://').split('/')
        base_dir = '/'.join(parts[1:])
        destination_file = os.path.join(self.dir, base_dir)
        return destination_file

    def try_to_load_from_cache(self, repo_id: str, filename: str):
        parts = repo_id.removeprefix('s3://').split('/')
        base_dir = '/'.join(parts[1:])
        destination_file = os.path.join(self.dir, base_dir, filename)
        if os.path.exists(destination_file):
            return destination_file
        return None
