# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Add signatures to models in config files by fetching from HuggingFace."""

import json
import sys
from pathlib import Path
from typing import Optional


def get_signature(config: dict) -> str:
    """Generate signature from config."""
    arch = config.get("architectures", ["Unknown"])[0]
    layers = config.get("num_hidden_layers", 0)
    hidden = config.get("hidden_size", 0)
    heads = config.get("num_attention_heads", 0)
    return f"{arch}_{layers}L_{hidden}H_{heads}A"


def fetch_hf_config(model_id: str) -> Optional[dict]:
    """Fetch config.json from HuggingFace."""
    try:
        import json

        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            repo_type="model",
        )

        with open(config_path) as f:
            return json.load(f)
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Install: pip install huggingface-hub")
        return None
    except Exception as e:
        print(f"Error fetching {model_id}: {e}")
        return None


def add_signatures_to_config(config_path: Path, dry_run: bool = False) -> None:
    """Add signatures to models in a config file."""
    print(f"\nProcessing {config_path.name}...")

    with open(config_path) as f:
        data = json.load(f)

    model_configs = data.get("model_configs", {})
    modified = False

    for model_id, model_data in model_configs.items():
        if "signature" in model_data:
            print(f"  {model_id}: already has signature "
                  f"({model_data['signature']})")
            continue

        if "/" not in model_id:
            print(f"  {model_id}: skipping (architecture-only entry)")
            continue

        if "recipes" not in model_data or not model_data["recipes"]:
            print(f"  {model_id}: skipping (no recipes)")
            continue

        print(f"  {model_id}: fetching from HuggingFace...")
        hf_config = fetch_hf_config(model_id)

        if not hf_config:
            print(f"  {model_id}: FAILED - could not fetch config")
            continue

        signature = get_signature(hf_config)
        print(f"  {model_id}: generated signature: {signature}")

        if not dry_run:
            model_data["signature"] = signature
            modified = True

    if modified and not dry_run:
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nUpdated {config_path.name}")
    elif dry_run:
        print("\n(Dry run - no changes made)")
    else:
        print("\n  No changes needed")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Add signatures to HuggingFace models in config files")
    parser.add_argument(
        "--config",
        type=Path,
        help="Specific config file to process (default: all in configs/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    if args.config:
        config_files = [args.config]
    else:
        configs_dir = (Path(__file__).parent.parent / "src" /
                       "vllm_rocm_autotuner_configs" / "configs")
        if not configs_dir.exists():
            print(f"Error: configs directory not found: {configs_dir}")
            return 1
        config_files = list(configs_dir.glob("rocm_config_*.json"))

    for config_file in config_files:
        if not config_file.exists():
            print(f"Error: {config_file} not found")
            continue
        add_signatures_to_config(config_file, dry_run=args.dry_run)

    print("\nDone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
