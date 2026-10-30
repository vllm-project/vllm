# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import huggingface_hub
import regex as re
from huggingface_hub.utils import validate_repo_id

from vllm.transformers_utils.repo_utils import hf_snapshot_download


def parse_model_source(model: str) -> tuple[str, str] | None:
    source, sep, repo = model.partition("://")
    if not sep or source not in ("hf", "modelscope"):
        return None
    if repo.count("/") != 1:
        raise ValueError(f"Expected {source}://owner/model, got {model!r}")
    validate_repo_id(repo)
    return source, repo


def resolve_model_source(
    model: str,
    *,
    revision: str | None = None,
    code_revision: str | None = None,
    download_dir: str | None = None,
    hf_token: str | bool | None = None,
    load_format: str = "auto",
    ignore_patterns: str | list[str] | None = None,
    tokenizer_only: bool = False,
    draft: bool = False,
) -> str:
    """Download an explicit Hub source before ordinary local model loading."""
    reference = parse_model_source(model)
    if reference is None:
        return model
    source, repo = reference
    if code_revision is not None and code_revision != revision:
        raise ValueError("Hub prefixes do not support a separate code_revision.")
    if not tokenizer_only and load_format.lower() not in (
        "auto",
        "hf",
        "safetensors",
        "pt",
    ):
        raise ValueError(
            "Hub prefixes support auto, hf, safetensors and pt loading only. "
            "For other loaders, download the model and use its local path."
        )
    download = hf_snapshot_download
    kwargs = dict(
        revision=revision,
        cache_dir=download_dir,
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
    )
    if source == "hf":
        kwargs["token"] = hf_token
    else:
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Install modelscope with `pip install modelscope` "
                "to use modelscope:// sources."
            ) from exc
        download = snapshot_download

    def fetch(patterns: list[str] | None = None) -> Path:
        try:
            return Path(download(repo, allow_patterns=patterns, **kwargs))
        except Exception as exc:
            exc.add_note(f"Failed to download {model} from {source}.")
            raise

    snapshot = fetch(
        [
            "*.json",
            "*.py",
            "*.model*",
            "*.txt",
            "*.tiktoken",
            "*.jinja",
            "*.bpe",
            "*.vocab",
            "*.spm",
        ]
    )
    if source == "hf" and re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
        kwargs["revision"] = snapshot.name

    def check_config(config: dict) -> dict:
        if "--" in json.dumps(config.get("auto_map", {})):
            raise ValueError(
                "Hub prefixes do not support cross-repository auto_map code; "
                "download the model and code locally first."
            )
        return config

    for path in snapshot.glob("*config*.json"):
        try:
            config = json.loads(path.read_text(), object_hook=check_config)
        except json.JSONDecodeError:
            # Leave validation of required files to the regular loader.
            continue
        if not isinstance(config, dict):
            continue
        if not (draft or tokenizer_only) and config.get("speculators_config"):
            raise ValueError(
                "With Hub prefixes, specify the target model explicitly and "
                "pass the speculator in --speculative-config."
            )
    if tokenizer_only:
        return str(snapshot)

    # Keep auxiliary files; let the existing loader select checkpoint weights.
    kwargs["ignore_patterns"] = ignore_patterns
    return str(fetch())
