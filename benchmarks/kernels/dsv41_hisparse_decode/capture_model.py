# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic public-code prompts and atomic benchmark receipts."""

import hashlib
import json
from pathlib import Path


def write_json(path, value):
    """Publish complete receipts so a live reader never sees partial JSON."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build_prompt(tokenizer, source_root, corpus, context_tokens, prompt_file=None):
    """Build an exact-length prompt from deterministic, public repository files."""
    suffix = (
        "\n\nUsing the material above, explain the main design choices and their "
        "tradeoffs. Give a concrete, coherent summary.\n\nSummary:\n"
    )
    start = tokenizer.encode("Repository reading task. Source material follows.\n")
    end = tokenizer.encode(suffix, add_special_tokens=False)
    budget = context_tokens - len(start) - len(end)
    if budget <= 0:
        raise ValueError("Context length is too small for the prompt framing")
    root = Path(source_root).resolve()
    if prompt_file:
        files = [Path(prompt_file).resolve()]
    else:
        directory, pattern = ("vllm", "*.py") if corpus == "code" else ("docs", "*.md")
        files = sorted((root / directory).rglob(pattern))
    pieces, manifest = [], []
    for path in files:
        if not path.is_file() or path.is_symlink():
            continue
        raw = path.read_bytes()
        if not raw or len(raw) > 512 * 1024:
            continue
        label = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
        text = f"\n--- {label} ---\n" + raw.decode("utf-8", errors="replace")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        take = min(budget, len(tokens))
        pieces.extend(tokens[:take])
        manifest.append(
            {
                "file": label,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source_bytes": len(raw),
                "tokens_used": take,
                "source_tokens": len(tokens),
            }
        )
        budget -= take
        if budget == 0:
            break
    if budget:
        raise ValueError(
            f"Corpus has {budget} too few tokens; refusing to pad by repeating text"
        )
    return start + pieces + end, manifest
