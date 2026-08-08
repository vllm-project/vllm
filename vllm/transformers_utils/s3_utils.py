# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fnmatch
from typing import TYPE_CHECKING, Any

from vllm.utils.import_utils import PlaceholderModule

if TYPE_CHECKING:
    from botocore.client import BaseClient


def __getattr__(name: str) -> Any:
    """
    Module-level lazy attribute access (PEP 562).

    Preserves the previous public surface where ``boto3`` was importable as
    ``vllm.transformers_utils.s3_utils.boto3`` while still avoiding the
    ~300ms of import cost on every cold-start when nobody touches it.

    When ``boto3`` is not installed, returns a ``PlaceholderModule`` so that
    attribute access surfaces the same informative ``ImportError`` shape
    callers used to see from the eager-import + placeholder fallback.
    """
    if name == "boto3":
        try:
            import boto3
        except ImportError:
            return PlaceholderModule("boto3")
        return boto3
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def _require_boto3() -> Any:
    """Lazy-import boto3 with a clear error when the dep is missing."""
    try:
        import boto3
    except ImportError as e:
        raise ImportError(
            "boto3 is required to load model weights from S3. "
            "Install it with `pip install boto3`."
        ) from e
    return boto3


def glob(
    s3: "BaseClient | None" = None,
    path: str = "",
    allow_pattern: list[str] | None = None,
) -> list[str]:
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
        # Lazy import: boto3 + botocore + s3transfer + jmespath collectively
        # add ~hundreds of ms (and 100+ modules) to every cold-start. For the
        # local-disk / HuggingFace Hub case this S3 client is never
        # constructed, so deferring the import keeps that cost off the
        # critical path.
        s3 = _require_boto3().client("s3")
    if not path.endswith("/"):
        path = path + "/"
    bucket_name, _, paths = list_files(s3, path=path, allow_pattern=allow_pattern)
    return [f"s3://{bucket_name}/{path}" for path in paths]


def list_files(
    s3: "BaseClient | None",
    path: str,
    allow_pattern: list[str] | None = None,
    ignore_pattern: list[str] | None = None,
) -> tuple[str, str, list[str]]:
    """
    List files from S3 path and filter by pattern.

    Args:
        s3: S3 client to use. If ``None``, a default client is constructed
            via ``boto3.client("s3")`` (and a clear ``ImportError`` is raised
            when boto3 is not installed).
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
    if s3 is None:
        # Defensive lazy-guard at the public-API entry point, mirroring
        # ``glob()``. Keeps callers that pass ``s3=None`` from blowing up
        # with a confusing ``AttributeError`` on ``None.list_objects_v2``.
        s3 = _require_boto3().client("s3")

    parts = path.removeprefix("s3://").split("/")
    prefix = "/".join(parts[1:])
    bucket_name = parts[0]

    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    paths = [obj["Key"] for obj in objects.get("Contents", [])]

    paths = _filter_ignore(paths, ["*/"])
    if allow_pattern is not None:
        paths = _filter_allow(paths, allow_pattern)

    if ignore_pattern is not None:
        paths = _filter_ignore(paths, ignore_pattern)

    return bucket_name, prefix, paths
