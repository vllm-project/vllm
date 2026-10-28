# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit, per-model Hub selection."""

import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from filelock import FileLock
from huggingface_hub import constants
from huggingface_hub import snapshot_download as hf_snapshot_download
from huggingface_hub.utils import validate_repo_id


def parse_model_source(model: str) -> tuple[str, str] | None:
    source, sep, repo_id = model.partition("://")
    if not sep or source not in ("hf", "modelscope"):
        return None
    if repo_id.count("/") != 1:
        raise ValueError(f"Expected {source}://owner/model, got {model!r}")
    validate_repo_id(repo_id)
    return source, repo_id


def strip_model_source(model: str) -> str:
    reference = parse_model_source(model)
    return reference[1] if reference is not None else model


def resolve_model_source(
    model: str,
    *,
    revision: str | None = None,
    code_revision: str | None = None,
    download_dir: str | None = None,
    hf_token: str | bool | None = None,
    metadata_only: bool = False,
) -> str:
    """Resolve explicit Hub references; leave all other paths unchanged."""
    reference = parse_model_source(model)
    if reference is None:
        return model
    source, repo_id = reference
    kwargs: dict[str, Any] = {
        "cache_dir": download_dir,
        "local_files_only": constants.HF_HUB_OFFLINE,
        "ignore_patterns": (
            ["*.bin", "*.safetensors", "*.pt", "*.pth", "*.gguf"]
            if metadata_only
            else None
        ),
    }

    def download(repo: str, rev: str | None, code_only: bool = False) -> str:
        options = dict(kwargs)
        if rev is not None:
            options["revision"] = rev
        if code_only:
            options["allow_patterns"] = ["*.py"]
        try:
            if source == "hf":
                return hf_snapshot_download(repo, token=hf_token, **options)
            from modelscope import snapshot_download

            return snapshot_download(repo, **options)
        except Exception as exc:
            exc.add_note(f"Failed to resolve {source}://{repo} from {source}.")
            raise

    snapshot = Path(download(repo_id, revision))
    code_paths: dict[str, str] = {}

    def local_reference(value):
        if isinstance(value, list):
            return [local_reference(item) for item in value]
        if not isinstance(value, str):
            return value
        code_repo, sep, class_name = value.partition("--")
        if not sep:
            if code_revision is None:
                return value
            code_repo, class_name = repo_id, value
        validate_repo_id(code_repo)
        if code_repo not in code_paths:
            rev = code_revision or (revision if code_repo == repo_id else None)
            code_paths[code_repo] = download(code_repo, rev, code_only=True)
        namespace = "hub_code_" + hashlib.sha256(code_repo.encode()).hexdigest()[:16]
        return f"{namespace}/{class_name}"

    def localize(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "auto_map" and isinstance(item, dict):
                    value[key] = {
                        name: local_reference(ref) for name, ref in item.items()
                    }
                else:
                    localize(item)
        elif isinstance(value, list):
            for item in value:
                localize(item)

    configs = {}
    for path in snapshot.glob("*config*.json"):
        original = json.loads(path.read_text())
        updated = deepcopy(original)
        localize(updated)
        if updated != original:
            configs[path.name] = updated
    if not configs:
        return str(snapshot)

    # Keep Hub snapshots untouched; the overlay links weights and localizes code.
    from vllm import envs

    files = sorted(snapshot.iterdir())
    identity = json.dumps(
        [str(snapshot), [path.name for path in files], configs, code_paths],
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()
    root = Path(download_dir or envs.VLLM_CACHE_ROOT) / "hub_sources"
    root.mkdir(parents=True, exist_ok=True)
    overlay = root / digest / repo_id.split("/")[-1]
    with FileLock(str(root / f"{digest}.lock")):
        if not overlay.exists():
            overlay.parent.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=overlay.parent) as temporary:
                staging = Path(temporary) / overlay.name
                staging.mkdir()
                for path in files:
                    if path.name not in configs:
                        (staging / path.name).symlink_to(path.resolve())
                for name, config in configs.items():
                    (staging / name).write_text(json.dumps(config))
                for repo, path in code_paths.items():
                    namespace = (
                        "hub_code_" + hashlib.sha256(repo.encode()).hexdigest()[:16]
                    )
                    (staging / namespace).symlink_to(Path(path).resolve())
                staging.rename(overlay)
    return str(overlay)
