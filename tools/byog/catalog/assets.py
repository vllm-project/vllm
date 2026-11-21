# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Caching AssetProvider with timeout-bounded downloads and strict SHA256 checks."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from tools.byog.executor.interfaces import AssetProvider, ResolvedAsset
from tools.byog.schemas.catalog import PinnedAsset

DEFAULT_ASSET_CACHE_DIR = Path.home() / ".cache" / "vllm-byog" / "assets"
DOWNLOAD_TIMEOUT_S = 60


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CachingAssetProvider(AssetProvider):
    """Fetches and verifies pinned catalog assets into a local content cache."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        overrides: dict[str, Path] | None = None,
        download_timeout_s: int = DOWNLOAD_TIMEOUT_S,
    ):
        env_cache = os.environ.get("BYOG_ASSET_CACHE")
        self._cache_dir = cache_dir or (
            Path(env_cache) if env_cache else DEFAULT_ASSET_CACHE_DIR
        )
        self._overrides = dict(overrides or {})
        self._download_timeout_s = download_timeout_s

    def resolve(self, asset: PinnedAsset) -> ResolvedAsset:
        if asset.id in self._overrides:
            override_path = self._overrides[asset.id]
            if not override_path.is_file():
                raise FileNotFoundError(
                    f"Override path for asset '{asset.id}' not found: "
                    f"'{override_path}'."
                )
            return ResolvedAsset(
                asset_id=asset.id,
                path=override_path,
                resolved_revision="local-override",
                sha256=_sha256_file(override_path),
                is_override=True,
            )

        url = asset.mirror_url or asset.source
        if not (url.startswith("http://") or url.startswith("https://")):
            local = Path(url)
            if not local.is_file():
                raise FileNotFoundError(
                    f"Local asset '{asset.id}' not found at '{local}'."
                )
            digest = _sha256_file(local)
            if digest != asset.sha256:
                raise ValueError(
                    f"SHA256 mismatch for local asset '{asset.id}': "
                    f"expected {asset.sha256}, got {digest}"
                )
            return ResolvedAsset(
                asset_id=asset.id,
                path=local,
                resolved_revision=asset.revision,
                sha256=digest,
                is_override=False,
            )

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        dest = self._cache_dir / f"{asset.id}-{asset.sha256[:16]}"
        if dest.is_file():
            digest = _sha256_file(dest)
            if digest == asset.sha256:
                return ResolvedAsset(
                    asset_id=asset.id,
                    path=dest,
                    resolved_revision=asset.revision,
                    sha256=digest,
                    is_override=False,
                )
            dest.unlink(missing_ok=True)

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{asset.id}-", dir=str(self._cache_dir)
        )
        tmp_path = Path(tmp_name)
        try:
            with (
                os.fdopen(fd, "wb") as out_fh,
                urllib.request.urlopen(url, timeout=self._download_timeout_s) as resp,
            ):
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out_fh.write(chunk)
            digest = _sha256_file(tmp_path)
            if digest != asset.sha256:
                raise ValueError(
                    f"SHA256 mismatch for downloaded asset '{asset.id}' from "
                    f"'{url}': expected {asset.sha256}, got {digest}"
                )
            os.replace(tmp_path, dest)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        return ResolvedAsset(
            asset_id=asset.id,
            path=dest,
            resolved_revision=asset.revision,
            sha256=digest,
            is_override=False,
        )
