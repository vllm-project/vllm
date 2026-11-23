# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-language `render_fingerprint` (layout `rf1`, RFC #65502).

`rf1:<sha256 of the canonical manifest>:<renderer tag>`. The manifest is a
fixed inventory of tokenizer/template/config file hashes and render flags, so
any implementation given the same files and flags produces the same digest.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import regex as re

LAYOUT_VERSION = 1

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "spm.model",
    "tekken.json",
    "tiktoken.model",
    "chat_template.jinja",
    "chat_template.json",
)
CONFIG_FILES = ("config.json", "params.json")
PROCESSOR_FILES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
)

_TOKENIZER_GLOBS = (
    re.compile(r"tokenizer\.model\.v\d+"),
    re.compile(r"[^/]+\.tiktoken"),
    re.compile(r"[^/]+\.jinja"),
)
_ADDITIONAL_TEMPLATES_DIR = "additional_chat_templates"
_FIXED = frozenset(TOKENIZER_FILES + CONFIG_FILES + PROCESSOR_FILES)

_MAX_SAFE_INTEGER = 2**53 - 1


def _format_number(value: float) -> str:
    """Serialize a float as ECMAScript `Number::toString` does."""
    if not math.isfinite(value):
        raise ValueError(f"Non-finite number in render manifest: {value!r}")
    if value == 0:
        return "0"

    sign, digit_tuple, exponent = Decimal(repr(value)).as_tuple()
    assert isinstance(exponent, int)
    digits = "".join(map(str, digit_tuple)).rstrip("0")
    exponent += len(digit_tuple) - len(digits)
    k = len(digits)
    n = k + exponent

    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = f"{digits[:n]}.{digits[n:]}"
    elif -6 < n <= 0:
        body = "0." + "0" * -n + digits
    else:
        mantissa = digits[0] + (f".{digits[1:]}" if k > 1 else "")
        body = f"{mantissa}e{'+' if n > 0 else '-'}{abs(n - 1)}"
    return f"-{body}" if sign else body


def _canonicalize(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError(f"Integer outside ±(2^53 - 1) in render manifest: {value}")
        return str(value)
    if isinstance(value, float):
        return _format_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Render manifest object keys must be strings")
        keys = sorted(value, key=lambda key: key.encode("utf-16-be"))
        members = (f"{_canonicalize(key)}:{_canonicalize(value[key])}" for key in keys)
        return "{" + ",".join(members) + "}"
    if isinstance(value, Sequence):
        return "[" + ",".join(map(_canonicalize, value)) + "]"
    raise TypeError(f"Unsupported type in render manifest: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Serialize `value` per RFC 8785 (JSON Canonicalization Scheme).

    Raises:
        ValueError: On a non-finite float or an integer outside ±(2^53 - 1).
        TypeError: On a value JSON cannot represent.

    """
    return _canonicalize(value).encode("utf-8")


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_files(
    tokenizer_dir: Path, config_dir: Path, model_dir: Path
) -> dict[str, str | None]:
    """Hash the fixed file list and the glob matches of layout `rf1`.

    Args:
        tokenizer_dir: Resolved snapshot of the tokenizer source.
        config_dir: Resolved snapshot of the model config source.
        model_dir: Resolved snapshot of the model, which holds the processor
            configs.

    Returns:
        File name to lowercase hex SHA-256 of its bytes, or None when absent.

    """
    files = {name: _hash_file(tokenizer_dir / name) for name in TOKENIZER_FILES}
    files.update({name: _hash_file(config_dir / name) for name in CONFIG_FILES})
    files.update({name: _hash_file(model_dir / name) for name in PROCESSOR_FILES})

    if tokenizer_dir.is_dir():
        for path in tokenizer_dir.iterdir():
            name = path.name
            if name not in files and any(p.fullmatch(name) for p in _TOKENIZER_GLOBS):
                files[name] = _hash_file(path)

    templates_dir = tokenizer_dir / _ADDITIONAL_TEMPLATES_DIR
    if templates_dir.is_dir():
        for path in templates_dir.glob("*.jinja"):
            files[f"{_ADDITIONAL_TEMPLATES_DIR}/{path.name}"] = _hash_file(path)

    return {name: digest for name, digest in files.items() if name in _FIXED or digest}


def build_manifest(
    files: Mapping[str, str | None], flags: Mapping[str, Any]
) -> dict[str, Any]:
    return {"schema": LAYOUT_VERSION, "files": dict(files), "flags": dict(flags)}


def renderer_tag(*, version: str, renderer: str, tokenizer: str) -> str:
    """Build `vllm-py/<version>;renderer=<name>;tokenizer=<lib>/<version>`."""
    tag = f"vllm-py/{version};renderer={renderer};tokenizer={tokenizer}"
    if ":" in tag or " " in tag or not tag.isascii():
        raise ValueError(f"Invalid renderer tag: {tag!r}")
    return tag


def render_fingerprint(manifest: Mapping[str, Any], tag: str) -> str:
    digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
    return f"rf{LAYOUT_VERSION}:{digest}:{tag}"
