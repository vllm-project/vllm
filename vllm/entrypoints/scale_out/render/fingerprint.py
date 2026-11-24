# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute the server's `render_fingerprint` (layout `rf1`) at startup."""

from argparse import Namespace
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal

import huggingface_hub

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.renderers import BaseRenderer
from vllm.renderers.chat_utils import load_chat_template
from vllm.renderers.fingerprint import (
    CONFIG_FILES,
    PROCESSOR_FILES,
    TOKENIZER_FILES,
    build_manifest,
    collect_files,
    render_fingerprint,
    renderer_tag,
)
from vllm.transformers_utils.chat_templates import (
    get_chat_template_fallback_path,
    get_chat_template_override_path,
)
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

Presence = Literal["hub", "local"]

_TOKENIZER_PATTERNS = [
    *TOKENIZER_FILES,
    "tokenizer.model.v*",
    "*.tiktoken",
    "*.jinja",
]


@dataclass(frozen=True)
class RenderFingerprint:
    render_fingerprint: str
    presence: Presence
    manifest: dict[str, Any]


def _resolve_snapshot(
    source: str,
    revision: str | None,
    patterns: list[str],
    token: bool | str | None,
) -> tuple[Path, Presence]:
    """Fetch the listed files the Hub names, then hash what the snapshot holds."""
    if Path(source).exists():
        return Path(source), "local"

    offline = huggingface_hub.constants.HF_HUB_OFFLINE
    path = huggingface_hub.snapshot_download(
        source,
        revision=revision,
        allow_patterns=patterns,
        local_files_only=offline,
        token=token,
    )
    return Path(path), "local" if offline else "hub"


def _renderer_name(model_config: ModelConfig, renderer: BaseRenderer) -> str:
    if model_config.hf_config.model_type == "gpt_oss":
        return "harmony"
    renderer_cls = type(renderer)
    if renderer_cls.__name__ == "DeepseekV41Renderer":
        return "deepseek_v41"
    return renderer_cls.__module__.rsplit(".", 1)[-1]


def _tokenizer_library(renderer: BaseRenderer) -> str:
    tokenizer = renderer.get_tokenizer()
    if is_mistral_tokenizer(tokenizer):
        return f"mistral-common/{package_version('mistral_common')}"
    if getattr(tokenizer, "is_fast", False):
        return f"tokenizers/{package_version('tokenizers')}"
    return f"transformers/{package_version('transformers')}"


def _builtin_chat_template(
    model_config: ModelConfig, renderer: BaseRenderer
) -> str | None:
    """Text of the built-in template the HF renderer uses, if it uses one."""
    from vllm.renderers.hf import HfRenderer, resolve_chat_template

    if not isinstance(renderer, HfRenderer):
        return None

    tokenizer = renderer.get_tokenizer()
    model_type = model_config.hf_config.model_type
    override = get_chat_template_override_path(model_type, tokenizer.name_or_path)
    if override is not None:
        return load_chat_template(override)

    fallback = get_chat_template_fallback_path(model_type, tokenizer.name_or_path)
    if fallback is None:
        return None
    text = load_chat_template(fallback)
    resolved = resolve_chat_template(tokenizer, None, None, model_config=model_config)
    return text if resolved == text else None


def _configured_chat_template(args: Namespace, loaded: str | None) -> str | None:
    """Text of `--chat-template` as given, without newline translation."""
    if not args.chat_template:
        return loaded
    try:
        return Path(args.chat_template).read_bytes().decode("utf-8")
    except OSError:
        # An inline template or a built-in template name.
        return loaded


def _flags(
    model_config: ModelConfig,
    args: Namespace,
    renderer: BaseRenderer,
    chat_template: str | None,
    default_chat_template_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    mm_config = model_config.multimodal_config
    language_model_only = bool(mm_config and mm_config.language_model_only)
    multimodal_active = mm_config is not None and not language_model_only
    pruning_rate = (mm_config.video_pruning_rate if mm_config else None) or None

    return {
        "tokenizer_mode": args.tokenizer_mode,
        "config_format": args.config_format,
        "trust_remote_code": args.trust_remote_code,
        "chat_template": _configured_chat_template(args, chat_template),
        "chat_template_fallback": (
            _builtin_chat_template(model_config, renderer)
            if chat_template is None
            else None
        ),
        "chat_template_content_format": args.chat_template_content_format,
        "default_chat_template_kwargs": default_chat_template_kwargs or None,
        "reasoning_parser": args.reasoning_parser or None,
        "reasoning_effort_map": args.reasoning_effort_map or None,
        "exclude_tools_when_tool_choice_none": args.exclude_tools_when_tool_choice_none,
        "hf_overrides": args.hf_overrides or None,
        "mm_processor_kwargs": args.mm_processor_kwargs or None,
        "media_io_kwargs": args.media_io_kwargs or None,
        "language_model_only": language_model_only,
        "interleave_mm_strings": bool(mm_config and mm_config.interleave_mm_strings),
        "video_pruning_rate": pruning_rate,
        "video_pruning_method": (
            mm_config.video_pruning_method if mm_config and pruning_rate else None
        ),
        "max_model_len": model_config.max_model_len if multimodal_active else None,
        "trust_request_chat_template": args.trust_request_chat_template,
    }


def compute_render_fingerprint(
    model_config: ModelConfig,
    args: Namespace,
    renderer: BaseRenderer,
    *,
    chat_template: str | None,
    default_chat_template_kwargs: dict[str, Any] | None,
) -> RenderFingerprint | None:
    """Build the manifest from the resolved sources and the render flags.

    Args:
        model_config: The server's model config.
        args: Parsed CLI arguments; flags are recorded as configured.
        renderer: The renderer that produces prompts.
        chat_template: Text of `--chat-template`, if given.
        default_chat_template_kwargs: The server's default template kwargs.

    Returns:
        The fingerprint, or None when the sources are not on the Hugging Face
        Hub or a local path.

    """
    if envs.VLLM_USE_MODELSCOPE:
        logger.info("render_fingerprint is not computed with VLLM_USE_MODELSCOPE.")
        return None

    token = model_config.hf_token
    tokenizer_dir, tokenizer_presence = _resolve_snapshot(
        model_config.tokenizer,
        model_config.tokenizer_revision,
        [*_TOKENIZER_PATTERNS, "additional_chat_templates/*.jinja"],
        token,
    )
    config_dir, config_presence = _resolve_snapshot(
        model_config.hf_config_path or model_config.model,
        model_config.hf_config_revision,
        list(CONFIG_FILES),
        token,
    )
    model_dir, model_presence = _resolve_snapshot(
        model_config.model, model_config.revision, list(PROCESSOR_FILES), token
    )

    manifest = build_manifest(
        collect_files(tokenizer_dir, config_dir, model_dir),
        _flags(
            model_config, args, renderer, chat_template, default_chat_template_kwargs
        ),
    )
    tag = renderer_tag(
        version=VLLM_VERSION,
        renderer=_renderer_name(model_config, renderer),
        tokenizer=_tokenizer_library(renderer),
    )
    hub = "hub" in (tokenizer_presence, config_presence, model_presence)
    return RenderFingerprint(
        render_fingerprint=render_fingerprint(manifest, tag),
        presence="hub" if hub else "local",
        manifest=manifest,
    )
