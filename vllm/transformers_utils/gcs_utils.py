# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fnmatch
from typing import TYPE_CHECKING

from vllm.utils.import_utils import PlaceholderModule

if TYPE_CHECKING:
    from google.cloud.storage import Client

try:
    from google.cloud import storage
except ImportError:
    storage = PlaceholderModule("google.cloud.storage")  # type: ignore[assignment]


def _filter_allow(paths: list[str], patterns: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]


def _filter_ignore(paths: list[str], patterns: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if not any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]


def glob(
    client: "Client | None" = None,
    path: str = "",
    allow_pattern: list[str] | None = None,
) -> list[str]:
    """
    List full file names from GCS path and filter by allow pattern.

    Args:
        client: GCS client to use.
        path: The GCS path to list from.
        allow_pattern: A list of patterns of which files to pull.

    Returns:
        list[str]: List of full GCS paths allowed by the pattern
    """
    if client is None:
        client = storage.Client()
    if not path.endswith("/"):
        path = path + "/"
    bucket_name, _, paths = list_files(
        client, path=path, allow_pattern=allow_pattern
    )
    return [f"gs://{bucket_name}/{p}" for p in paths]


def list_files(
    client: "Client",
    path: str,
    allow_pattern: list[str] | None = None,
    ignore_pattern: list[str] | None = None,
) -> tuple[str, str, list[str]]:
    """
    List files from GCS path and filter by pattern.

    Args:
        client: GCS client to use.
        path: The GCS path to list from.
        allow_pattern: A list of patterns of which files to pull.
        ignore_pattern: A list of patterns of which files not to pull.

    Returns:
        tuple[str, str, list[str]]: A tuple where:
            - The first element is the bucket name
            - The second element is the prefix string
            - The third element is a list of files allowed or
              disallowed by pattern
    """
    parts = path.removeprefix("gs://").split("/")
    bucket_name = parts[0]
    prefix = "/".join(parts[1:])

    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    paths = [blob.name for blob in blobs]

    paths = _filter_ignore(paths, ["*/"])
    if allow_pattern is not None:
        paths = _filter_allow(paths, allow_pattern)

    if ignore_pattern is not None:
        paths = _filter_ignore(paths, ignore_pattern)

    return bucket_name, prefix, paths
