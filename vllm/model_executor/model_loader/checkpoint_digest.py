# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content digest of the checkpoint files a model was loaded from."""

import hashlib
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import regex as re

from vllm.config import LoadConfig, ModelConfig
from vllm.config.utils import hash_factors

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_HASH_THREADS = 8


@dataclass
class _LoadedFiles:
    folder: str
    # SHA-256 of each file, or None when it is neither in the Hugging Face
    # cache nor hashed.
    sha256: dict[str, str | Future[str] | None] = field(default_factory=dict)


class CheckpointRecording:
    """Records the checkpoint files read while loading the model `owner`.

    Files of other models, such as a draft model loaded alongside, are not
    part of the digest.

    With `hash_local_files`, files outside the Hugging Face cache are hashed
    in the background as soon as they are recorded, while the weights load,
    so the bytes are mostly read from the page cache the load fills.
    """

    def __init__(self, owner: str, hash_local_files: bool) -> None:
        self.owner = owner
        self.hash_local_files = hash_local_files
        self._loaded: list[_LoadedFiles] = []
        self._executor: ThreadPoolExecutor | None = None

    def __enter__(self) -> "CheckpointRecording":
        global _active_recording
        _active_recording = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global _active_recording
        _active_recording = None
        if exc_type is not None and self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def record(self, owner: str, folder: str, files: list[str]) -> None:
        if owner != self.owner:
            return
        loaded = _LoadedFiles(folder)
        for path in files:
            sha256: str | Future[str] | None = _hf_cache_sha256(path)
            if sha256 is None and self.hash_local_files:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        _HASH_THREADS, thread_name_prefix="checkpoint-digest"
                    )
                sha256 = self._executor.submit(_file_sha256, path)
            loaded.sha256[path] = sha256
        self._loaded.append(loaded)

    def digest(
        self, model_config: ModelConfig, load_config: LoadConfig
    ) -> dict[str, str]:
        """Digest the recorded files.

        The digest covers each file's path relative to its checkpoint folder
        and its SHA-256, so a Hub-cached checkpoint and a local copy of the
        same files match. Without SHA-256s for every file, and for loaders that
        do not record their files, it falls back to the model path and
        revision, which cannot tell two checkpoints at the same path apart.

        Returns:
            `weights`, the digest, and `weights_source`: one of `hf_cache`,
            `hashed`, `path` or `dummy`.

        """
        try:
            return self._digest(model_config, load_config)
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)

    def _digest(
        self, model_config: ModelConfig, load_config: LoadConfig
    ) -> dict[str, str]:
        if load_config.load_format == "dummy":
            return {"weights": "dummy", "weights_source": "dummy"}
        entries: set[tuple[str, str]] = set()
        source = "hf_cache"
        for loaded in self._loaded:
            for path, sha256 in loaded.sha256.items():
                if sha256 is None:
                    return path_checkpoint_digest(model_config, load_config)
                if isinstance(sha256, Future):
                    sha256 = sha256.result()
                    source = "hashed"
                entries.add((os.path.relpath(path, loaded.folder), sha256))
        if not entries:
            return path_checkpoint_digest(model_config, load_config)
        return {
            "weights": hash_factors({"files": sorted(entries)}),
            "weights_source": source,
        }


_active_recording: CheckpointRecording | None = None


def record_checkpoint_files(owner: str, folder: str, files: list[str]) -> None:
    """Record the checkpoint files read while loading the model `owner`."""
    if _active_recording is not None:
        _active_recording.record(owner, folder, files)


def _hf_cache_sha256(path: str) -> str | None:
    """A Hugging Face cache snapshot links each LFS file to `blobs/<sha256>`.

    Only that link is read: the blob may itself link into a store that names
    files by another hash.
    """
    if not os.path.islink(path):
        return None
    target = os.readlink(path)
    name = os.path.basename(target)
    is_blob = os.path.basename(os.path.dirname(target)) == "blobs"
    return name if is_blob and _SHA256_HEX.fullmatch(name) else None


def _file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def path_checkpoint_digest(
    model_config: ModelConfig, load_config: LoadConfig
) -> dict[str, str]:
    factors: dict[str, object] = {
        "model": model_config.model,
        "model_weights": model_config.model_weights,
        "revision": model_config.revision,
        "load_format": str(load_config.load_format),
    }
    return {"weights": hash_factors(factors), "weights_source": "path"}
