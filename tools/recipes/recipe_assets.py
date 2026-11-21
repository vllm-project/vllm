#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage approved external reasoning parser files for generated vLLM configs.

Only an explicitly allowlisted model/plugin pair is downloaded. Other recipe
arguments are never treated as executable download instructions.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

PARSER_FILENAME = "nano_v3_reasoning_parser.py"
NEMOTRON_MODELS = frozenset(
    {
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
    }
)
# NVIDIA publishes this parser alongside the BF16 checkpoint. FP8 uses it too.
PARSER_URL = (
    "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/"
    "resolve/main/nano_v3_reasoning_parser.py"
)


def prepare_assets(config_path: Path, cache_dir: Path) -> bool:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Generated vLLM config must be a YAML mapping")

    plugin = config.get("reasoning-parser-plugin")
    if not plugin:
        return False
    if not isinstance(plugin, str):
        raise ValueError("reasoning-parser-plugin must be a path string")

    requested = Path(plugin)
    # An existing absolute or relative file is a user-managed asset.
    if requested.is_file():
        config["reasoning-parser-plugin"] = str(requested.resolve())
    elif (config_path.parent / requested).is_file():
        config["reasoning-parser-plugin"] = str(
            (config_path.parent / requested).resolve()
        )
    elif (
        config.get("model") in NEMOTRON_MODELS
        and plugin == PARSER_FILENAME
        and config.get("reasoning-parser") == "nano_v3"
    ):
        cache_dir = cache_dir.resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / PARSER_FILENAME
        if not target.is_file() or target.stat().st_size == 0:
            # Never execute downloaded code here. vLLM imports it on startup.
            # A pinned HF revision/checksum should be introduced once validated.
            request = Request(
                PARSER_URL, headers={"User-Agent": "vllm-recipes-assets/1.0"}
            )
            token = os.environ.get("HF_TOKEN")
            if token:
                request.add_header("Authorization", f"Bearer {token}")
            fd, temp_name = tempfile.mkstemp(prefix=".parser-", dir=cache_dir)
            try:
                with (
                    os.fdopen(fd, "wb") as output,
                    urlopen(request, timeout=60) as response,
                ):
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                temp = Path(temp_name)
                if not temp.stat().st_size:
                    raise ValueError("Downloaded reasoning parser is empty")
                # Reject HTTP error pages before caching the file as Python code.
                with temp.open("rb") as f:
                    header = f.read(256).lstrip().lower()
                if header.startswith((b"<!doctype html", b"<html")):
                    raise ValueError("Expected Python parser, received HTML")
                temp.replace(target)
            finally:
                Path(temp_name).unlink(missing_ok=True)
        config["reasoning-parser-plugin"] = str(target)
        print(f"Recipe parser available: {target}")
    else:
        raise FileNotFoundError(
            f"Reasoning parser {plugin!r} not found. Provide the plugin file "
            "or use a supported recipe with an approved download source."
        )

    # Preserve the generated config and metadata in native vLLM YAML format.
    header = "".join(
        line
        for line in config_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if line.startswith("#")
    )
    config_path.write_text(
        header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_assets(args.config, args.cache_dir)


if __name__ == "__main__":
    main()
