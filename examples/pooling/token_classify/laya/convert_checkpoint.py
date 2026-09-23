# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert a Laya checkpoint into a directory vLLM can serve.

Laya ships `rl_agent_config.json`, `encoder/config.json`, `tokenizer/` and
`model.safetensors`. vLLM needs a root `config.json`, so this writes the
encoder config with the `LayaForDecision` architecture and a `laya_config`
block, next to the tokenizer and (hard-linked when possible) weights.

    python convert_checkpoint.py convaiinnovations/laya ./laya-vllm
    python convert_checkpoint.py convaiinnovations/laya-multilingual \\
        ./laya-multilingual-vllm
"""

import argparse
import json
import os
import shutil

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

QTYPES = ("choice", "score", "noul")
LAYA_CONFIG_KEYS = (
    "head_layers",
    "max_len",
    "head_max_len",
    "act_costs",
    "temperature",
    "temperature_by_options",
)


def fix_tokenizer_config(path: str) -> None:
    """Same fix-ups as `laya.agent._fix_tokenizer_config`."""
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("tokenizer_class") in (None, "TokenizersBackend"):
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg.pop("backend", None)
        cfg.pop("is_local", None)
    if isinstance(extra := cfg.get("extra_special_tokens"), list):
        cfg["extra_special_tokens"] = {f"extra_{i}": t for i, t in enumerate(extra)}
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def link_or_copy(src: str, dst: str) -> None:
    src = os.path.realpath(src)
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Laya Hugging Face repo id or local directory")
    parser.add_argument("output", help="Directory to write the converted model to")
    parser.add_argument("--subfolder", default=None)
    args = parser.parse_args()

    src = args.model
    if not os.path.isdir(src):
        prefix = f"{args.subfolder}/" if args.subfolder else ""
        src = snapshot_download(
            src,
            allow_patterns=[
                prefix + p
                for p in (
                    "rl_agent_config.json",
                    "model.safetensors",
                    "tokenizer/*",
                    "encoder/*",
                )
            ],
        )
    if args.subfolder:
        src = os.path.join(src, args.subfolder)

    os.makedirs(args.output, exist_ok=True)
    for name in os.listdir(os.path.join(src, "tokenizer")):
        shutil.copyfile(
            os.path.join(src, "tokenizer", name), os.path.join(args.output, name)
        )
    fix_tokenizer_config(os.path.join(args.output, "tokenizer_config.json"))
    link_or_copy(
        os.path.join(src, "model.safetensors"),
        os.path.join(args.output, "model.safetensors"),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.output)
    # The pooler reads the question type from the first token after [CLS].
    qtype_token_ids = [
        tokenizer(f"{t} question: ", add_special_tokens=False)["input_ids"][0]
        for t in QTYPES
    ]
    if len(set(qtype_token_ids)) != len(QTYPES):
        raise ValueError(f"Question types share a first token: {qtype_token_ids}")

    with open(os.path.join(src, "rl_agent_config.json")) as f:
        agent_config = json.load(f)
    with open(os.path.join(src, "encoder", "config.json")) as f:
        config = json.load(f)
    config["architectures"] = ["LayaForDecision"]
    # Laya runs under autocast in this dtype.
    config["dtype"] = {"bf16": "bfloat16"}.get(agent_config.get("amp_dtype"), "float16")
    config["laya_config"] = {
        **{k: agent_config[k] for k in LAYA_CONFIG_KEYS if k in agent_config},
        "mask_token_id": tokenizer.mask_token_id,
        "qtype_token_ids": qtype_token_ids,
    }
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
