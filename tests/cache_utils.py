# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from filelock import FileLock


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _url_cache_miss_message(url: str, path: Path) -> str:
    return (
        f"Cached test asset is missing or invalid: {path}. "
        f"Refusing to download {url!r} because VLLM_TEST_CACHE_ONLY is set. "
        "Populate the persistent test cache first, or unset VLLM_TEST_CACHE_ONLY "
        "for an explicit cache-populating run."
    )


def get_vllm_test_cache_dir(namespace: str | None = None) -> Path:
    default_cache_dir = Path(tempfile.gettempdir()) / "vllm-test-cache"
    root = Path(os.environ.get("VLLM_TEST_CACHE", default_cache_dir))
    if namespace:
        root = root / namespace
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_filename_for_url(url: str) -> str:
    parsed_path = urlparse(url).path
    basename = Path(parsed_path).name or "download"
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{url_hash}-{basename}"


def _is_valid_cache_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _remove_invalid_cache_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        raise IsADirectoryError(f"Expected cached file, found directory: {path}")
    if path.exists() or path.is_symlink():
        path.unlink()


def download_url_to_file(
    url: str,
    path: str | Path,
    timeout: float = 300,
    *,
    local_only: bool | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lock = FileLock(str(path) + ".lock")
    with lock:
        if _is_valid_cache_file(path):
            return path

        _remove_invalid_cache_path(path)
        if local_only is None:
            local_only = _env_flag_enabled("VLLM_TEST_CACHE_ONLY")
        if local_only:
            raise FileNotFoundError(_url_cache_miss_message(url, path))

        tmp_file = tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        )
        tmp_path = Path(tmp_file.name)
        try:
            with tmp_file:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    shutil.copyfileobj(response, tmp_file)
            if not _is_valid_cache_file(tmp_path):
                raise ValueError(f"Downloaded empty cache file from {url}")
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)

    return path


def download_to_vllm_test_cache(
    url: str,
    namespace: str,
    filename: str | None = None,
    timeout: float = 300,
    *,
    local_only: bool | None = None,
) -> Path:
    cache_dir = get_vllm_test_cache_dir(namespace)
    cache_name = filename or _cache_filename_for_url(url)
    return download_url_to_file(
        url, cache_dir / cache_name, timeout=timeout, local_only=local_only
    )
