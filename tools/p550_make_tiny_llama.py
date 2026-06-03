#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a tiny local Llama checkpoint for P550 offline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=".p550_models/tiny-llama",
        help="Directory where the tiny model should be written.",
    )
    return parser.parse_args()


def build_tokenizer() -> PreTrainedTokenizerFast:
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]
    words = [
        "Hello",
        "from",
        "P550",
        "SiFive",
        "RISC-V",
        "CPU",
        "vLLM",
        "test",
        ".",
    ]
    vocab = {token: idx for idx, token in enumerate(special_tokens + words)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer()
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        # CPU attention dispatch on the P550 scalar path supports standard
        # Llama-style head dimensions; keep the model tiny via layer count.
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=128,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=False)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
