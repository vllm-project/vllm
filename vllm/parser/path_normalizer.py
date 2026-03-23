# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Conservative normalization for path-like model output."""

from __future__ import annotations

import json
import re
from typing import Any

_COMMON_EXTENSIONS = {
    "c",
    "cc",
    "cfg",
    "conf",
    "cpp",
    "cs",
    "css",
    "csv",
    "gd",
    "go",
    "h",
    "hpp",
    "html",
    "ini",
    "java",
    "js",
    "json",
    "jsx",
    "lua",
    "md",
    "php",
    "py",
    "rb",
    "rs",
    "scss",
    "sh",
    "sql",
    "svg",
    "swift",
    "toml",
    "ts",
    "tscn",
    "tsx",
    "txt",
    "xml",
    "yaml",
    "yml",
}

_PATH_WITH_SEP_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?(?:[^\s\"'`<>|]+[\\/])+[^\s\"'`<>|]+?)"
    r"(?P<sep>\s*\.(?:\s*\.)*\s*)"
    r"(?P<ext>[A-Za-z0-9]{1,16})(?=$|[\s\"'`),;:\]|[\\/])"
)

_STANDALONE_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9_/\\\\.-])"
    r"(?P<base>[A-Za-z0-9][A-Za-z0-9_-]{0,255})"
    r"(?P<sep>\s*\.(?:\s*\.)*\s*)"
    r"(?P<ext>[A-Za-z0-9]{1,16})"
    r"(?![A-Za-z0-9_])"
)


def _normalize_match(base: str, ext: str) -> str:
    return f"{base}.{ext}"


def _normalize_paths_with_separators(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return _normalize_match(match.group("path"), match.group("ext"))

    return _PATH_WITH_SEP_RE.sub(repl, text)


def _normalize_standalone_filenames(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        base = match.group("base")
        ext = match.group("ext")
        if (
            ext.lower() not in _COMMON_EXTENSIONS
            and "_" not in base
            and "-" not in base
            and not any(ch.isdigit() for ch in base)
        ):
            return match.group(0)
        return _normalize_match(base, ext)

    return _STANDALONE_NAME_RE.sub(repl, text)


def normalize_pathlike_text(text: str | None) -> str | None:
    if text is None or "." not in text:
        return text
    normalized = _normalize_paths_with_separators(text)
    normalized = _normalize_standalone_filenames(normalized)
    return normalized


def _normalize_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_pathlike_text(value)
    if isinstance(value, list):
        return [_normalize_jsonish(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_jsonish(item) for key, item in value.items()}
    return value


def normalize_tool_arguments_json(arguments: str | None) -> str | None:
    if arguments is None:
        return None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return normalize_pathlike_text(arguments)
    normalized = _normalize_jsonish(parsed)
    return json.dumps(normalized, ensure_ascii=False)
