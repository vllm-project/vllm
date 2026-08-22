# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

TEST_CACHE_ENV = "_VLLM_TEST_CACHE"
TEST_CACHE_ONLY_ENV = "_VLLM_TEST_CACHE_ONLY"


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _url_cache_miss_message(path: Path) -> str:
    return (
        f"Cached test asset is missing or invalid: {path}. "
        f"Refusing to download it because {TEST_CACHE_ONLY_ENV} is enabled. "
        f"Populate the persistent test cache first, or unset {TEST_CACHE_ONLY_ENV}."
    )


def _relative_cache_path(value: str, *, name: str, nested: bool) -> Path:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {".", ".."} for part in path.parts)
        or (not nested and len(path.parts) != 1)
    ):
        raise ValueError(f"{name} must be a safe relative cache path: {value!r}")
    return path


def get_vllm_test_cache_dir(namespace: str | None = None) -> Path:
    default_cache_dir = Path(tempfile.gettempdir()) / "vllm-test-cache"
    root = Path(os.environ.get(TEST_CACHE_ENV, default_cache_dir))
    if namespace is not None:
        root /= _relative_cache_path(namespace, name="namespace", nested=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_filename_for_url(url: str) -> str:
    parsed_path = urlparse(url).path
    basename = Path(parsed_path).name or "download"
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{url_hash}-{basename}"


def _is_valid_cache_file(path: Path, expected_sha256: str | None = None) -> bool:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            return False
        if expected_sha256 is None:
            return True
        with path.open("rb") as file:
            digest = hashlib.sha256()
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest() == expected_sha256
    except OSError:
        return False


def _normalize_expected_sha256(expected_sha256: str | None) -> str | None:
    if expected_sha256 is None:
        return None
    expected_sha256 = expected_sha256.lower()
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal digits")
    return expected_sha256


def download_url_to_file(
    url: str,
    path: str | Path,
    timeout: float = 300,
    *,
    local_only: bool | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Download atomically, treating a nonempty regular destination as a hit.

    All callers sharing a destination must use the same ``expected_sha256``.
    Cached test assets are public, so published files are world-readable.
    """
    path = Path(path)
    expected_sha256 = _normalize_expected_sha256(expected_sha256)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _is_valid_cache_file(path, expected_sha256):
        return path
    if local_only is None:
        local_only = _env_flag_enabled(TEST_CACHE_ONLY_ENV)
    if local_only:
        raise FileNotFoundError(_url_cache_miss_message(path))
    if path.is_dir() and not path.is_symlink():
        raise IsADirectoryError(f"Expected cached file, found directory: {path}")

    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    tmp_path = Path(tmp_name)
    try:
        with (
            os.fdopen(tmp_fd, "wb") as tmp_file,
            urllib.request.urlopen(url, timeout=timeout) as response,
        ):
            shutil.copyfileobj(response, tmp_file)
        if not _is_valid_cache_file(tmp_path, expected_sha256):
            raise ValueError(f"Downloaded file failed cache validation: {path}")
        tmp_path.chmod(0o644)
        if not _is_valid_cache_file(path, expected_sha256):
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
    expected_sha256: str | None = None,
) -> Path:
    cache_dir = get_vllm_test_cache_dir(namespace)
    cache_name = (
        _relative_cache_path(filename, name="filename", nested=False)
        if filename is not None
        else Path(_cache_filename_for_url(url))
    )
    return download_url_to_file(
        url,
        cache_dir / cache_name,
        timeout=timeout,
        local_only=local_only,
        expected_sha256=expected_sha256,
    )
