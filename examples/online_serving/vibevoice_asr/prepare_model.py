# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _snapshot_download(model_id: str, model_dir: Path, hf_token: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency `huggingface_hub`. Install it in your env."
        ) from exc

    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        token=hf_token,
    )


def _run_generate_tokenizer_files(model_dir: Path) -> None:
    try:
        import vllm_plugin  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Missing `vllm_plugin` (VibeVoice). Install with:\n"
            '  python -m pip install -U --no-deps "vibevoice[vllm] @ '
            'https://github.com/microsoft/VibeVoice/archive/refs/heads/main.zip"\n'
            "Then install the remaining deps (e.g. diffusers)."
        ) from exc

    cmd = [
        sys.executable,
        "-m",
        "vllm_plugin.tools.generate_tokenizer_files",
        "--output",
        str(model_dir),
    ]
    subprocess.check_call(cmd)


def _patch_tokenizer_config(model_dir: Path) -> bool:
    cfg_path = model_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        return False

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "tokenizer_class" not in data:
        return False

    backup_path = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(cfg_path, backup_path)

    data.pop("tokenizer_class", None)
    cfg_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return True


def _print_config_summary(model_dir: Path) -> None:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return

    model_type = cfg.get("model_type")
    arch = cfg.get("architectures")
    print(
        f"[prepare_model] config.json model_type={model_type!r} architectures={arch!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare microsoft/VibeVoice-ASR for vLLM OpenAI server."
    )
    parser.add_argument(
        "--model-id",
        default="microsoft/VibeVoice-ASR",
        help="HF repo id to download (default: microsoft/VibeVoice-ASR).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local directory for the model snapshot.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HF token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN env var).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip snapshot_download even if files are missing.",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip tokenizer file generation.",
    )
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    hf_token: str | None = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or None
    )

    if not args.skip_download:
        # Treat config.json as the minimum signal that the snapshot exists.
        if not (model_dir / "config.json").exists():
            print(f"[prepare_model] Downloading {args.model_id} -> {model_dir}")
            _snapshot_download(args.model_id, model_dir, hf_token)
        else:
            print(f"[prepare_model] Model snapshot already exists: {model_dir}")

    _print_config_summary(model_dir)

    if not args.skip_tokenizer:
        tokenizer_json = model_dir / "tokenizer.json"
        if tokenizer_json.exists():
            print(
                "[prepare_model] tokenizer.json exists, skip generation:",
                tokenizer_json,
            )
        else:
            print("[prepare_model] Generating tokenizer files via vllm_plugin...")
            _run_generate_tokenizer_files(model_dir)

    if _patch_tokenizer_config(model_dir):
        print(
            "[prepare_model] Patched tokenizer_config.json (removed tokenizer_class)."
        )
    else:
        print("[prepare_model] tokenizer_config.json already OK (or missing).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
