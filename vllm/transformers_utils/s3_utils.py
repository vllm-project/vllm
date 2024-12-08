from __future__ import annotations

import fnmatch
import os
import shutil
import signal
import tempfile
from pathlib import Path

import boto3


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
         allow_pattern: list[str] | None = None) -> list[str]:
    if s3 is None:
        s3 = boto3.client("s3")
    bucket_name, _, paths = list_files(s3,
                                       path=path,
                                       allow_pattern=allow_pattern)
    return [f"s3://{bucket_name}/{path}" for path in paths]


def list_files(
        s3,
        path: str,
        allow_pattern: list[str] | None = None,
        ignore_pattern: list[str] | None = None) -> tuple[str, str, list[str]]:
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


class S3Model:

    def __init__(self) -> None:
        self.s3 = boto3.client('s3')
        for sig in (signal.SIGINT, signal.SIGTERM):
            existing_handler = signal.getsignal(sig)
            signal.signal(sig, self.close_by_signal(existing_handler))
        self.dir = tempfile.mkdtemp()

    def __del__(self):
        self.close()

    def close(self) -> None:
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)

    def close_by_signal(self, existing_handler=None):

        def new_handler(signum, frame):
            self.close()
            if existing_handler:
                existing_handler(signum, frame)

        return new_handler

    def pull_files(self,
                   s3_model_path: str = "",
                   allow_pattern: list[str] | None = None,
                   ignore_pattern: list[str] | None = None) -> None:
        bucket_name, base_dir, files = list_files(self.s3, s3_model_path,
                                                  allow_pattern,
                                                  ignore_pattern)
        if len(files) == 0:
            return

        for file in files:
            destination_file = self.dir + file.removeprefix(base_dir)
            local_dir = Path(destination_file).parent
            os.makedirs(local_dir, exist_ok=True)
            self.s3.download_file(bucket_name, file, destination_file)
