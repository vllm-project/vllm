#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Create a minimal synthetic LoRA adapter for a Cohere c5 (MoE) checkpoint.

The adapter uses zero-initialized weights so it has no effect on model outputs,
but it exercises the full LoRA loading/serving path inside vLLM.  Useful for
CI when a real trained adapter is not available.

Supports top-level Hugging Face configs with ``model_type`` ``cohere2moe``, and
``cohere2_vision`` wrappers whose language backbone dimensions live under
``text_config`` (same layout as ``Cohere2MoeForCausalLM`` attention LoRA targets).

Usage:
    python create_dummy_lora.py --model-dir /path/to/c5-3a30t_fp8 \\
                                --output-dir /tmp/c5_dummy_lora
    # Prints the output directory on success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REQUIRED_DIM_KEYS = (
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_hidden_layers",
)


def _language_backbone_config(cfg: dict) -> dict:
    """Return the JSON object that holds hidden/head/layer counts for LoRA dims."""
    model_type = cfg.get("model_type", "")
    if model_type == "cohere2moe":
        return cfg
    if model_type == "cohere2_vision":
        text_cfg = cfg.get("text_config")
        if not isinstance(text_cfg, dict):
            raise ValueError(
                "model_type is 'cohere2_vision' but config.json has no "
                "object 'text_config' with language backbone settings."
            )
        inner_type = text_cfg.get("model_type", "")
        if inner_type and inner_type != "cohere2moe":
            raise ValueError(
                "Expected text_config.model_type 'cohere2moe' for c5 dummy LoRA, "
                f"got '{inner_type}'."
            )
        return text_cfg
    raise ValueError(
        f"Unsupported model_type '{model_type}' for c5 dummy LoRA "
        "(expected 'cohere2moe' or 'cohere2_vision')."
    )


def create_dummy_lora(model_dir: str, output_dir: str, lora_rank: int = 8) -> str:
    """
    Create a zero-weight LoRA adapter for the c5 MoE language backbone.

    Returns the output directory path.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    lm_cfg = _language_backbone_config(cfg)
    missing = [k for k in _REQUIRED_DIM_KEYS if k not in lm_cfg]
    if missing:
        raise KeyError(
            "Language backbone config is missing required keys "
            f"{missing} (needed to size attention LoRA tensors)."
        )

    hidden = lm_cfg["hidden_size"]
    heads = lm_cfg["num_attention_heads"]
    kv_heads = lm_cfg["num_key_value_heads"]
    head_dim = hidden // heads
    num_layers = lm_cfg["num_hidden_layers"]

    # Attention projection dimensions
    target_shapes: dict[str, tuple[int, int]] = {
        "q_proj": (heads * head_dim, hidden),
        "k_proj": (kv_heads * head_dim, hidden),
        "v_proj": (kv_heads * head_dim, hidden),
        "o_proj": (hidden, heads * head_dim),
    }

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        raise ImportError(
            f"Required package not found: {e}. "
            "Install torch and safetensors to use this script."
        ) from e

    tensors: dict[str, torch.Tensor] = {}
    for layer_idx in range(num_layers):
        for mod, (out_dim, in_dim) in target_shapes.items():
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}"
            # lora_B is all-zeros so the LoRA output is zero (adapter is a no-op).
            tensors[f"{prefix}.lora_A.weight"] = torch.zeros(
                lora_rank, in_dim, dtype=torch.bfloat16
            )
            tensors[f"{prefix}.lora_B.weight"] = torch.zeros(
                out_dim, lora_rank, dtype=torch.bfloat16
            )

    os.makedirs(output_dir, exist_ok=True)
    save_file(tensors, os.path.join(output_dir, "adapter_model.safetensors"))

    adapter_config = {
        "base_model_name_or_path": model_dir,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": lora_rank * 2,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_rank,
        "revision": None,
        "target_modules": sorted(target_shapes.keys()),
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a zero-weight synthetic LoRA adapter for a cohere2moe model"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the cohere2moe (c5) model checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory where adapter_config.json and "
            "adapter_model.safetensors will be saved"
        ),
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    args = parser.parse_args()

    try:
        out = create_dummy_lora(args.model_dir, args.output_dir, args.lora_rank)
        print(out)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
