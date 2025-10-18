# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate signature for a HuggingFace model."""

import json
import sys
from pathlib import Path


def get_signature_from_local(model_path: Path) -> str:
    """Get signature from local model directory."""
    config_file = model_path / "config.json"

    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_file) as f:
        config = json.load(f)

    arch = config.get("architectures", ["Unknown"])[0]
    layers = config.get("num_hidden_layers", 0)
    hidden = config.get("hidden_size", 0)
    heads = config.get("num_attention_heads", 0)

    return f"{arch}_{layers}L_{hidden}H_{heads}A"


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_signature.py <model_path>")
        print("Example: python generate_signature.py openai/gpt-oss-120b")
        return 1

    model_path = Path(sys.argv[1])

    if not model_path.exists():
        print(f"Error: path does not exist: {model_path}")
        return 1

    try:
        signature = get_signature_from_local(model_path)
        print(f"\nModel: {model_path.name}")
        print(f"Signature: {signature}")
        print("\nAdd to your config file:")
        print('  "your-org/model-name": {')
        print(f'    "signature": "{signature}",')
        print('    "recipes": [...]')
        print("  }")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
