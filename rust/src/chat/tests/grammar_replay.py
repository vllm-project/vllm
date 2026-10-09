#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tiktoken>=0.6.0",
#   # xgrammar imports torch unconditionally; only the CPU build is needed.
#   "torch",
#   "transformers>=5.10.4",
#   # Keep in sync with requirements/common.txt.
#   "xgrammar==0.2.7",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Replay the grammar cases exported by the roundtrip tests through XGrammar.

Each file holds one model's distinct output grammars, each with the generations
of the fixture variants that build it. Every grammar is compiled against the
model tokenizer the same way as `vllm/v1/structured_output/backend_xgrammar.py`.
The generations of token-zero grammars are then replayed from the first
generated token: each token must be allowed by the bitmask and accepted by the
matcher, and the final stop token must terminate it.
"""

import json
import sys
from functools import cache
from pathlib import Path

import xgrammar as xgr
from transformers import AutoTokenizer

CASES_DIR = Path(__file__).parent / "grammar_replay"


@cache
def load_tokenizer(model_id: str, vocab_size: int):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size
    )
    return tokenizer, xgr.GrammarCompiler(tokenizer_info)


def compile_grammar(model: dict, grammar: dict) -> xgr.CompiledGrammar:
    """Compile one output grammar of a model."""
    _, compiler = load_tokenizer(model["model_id"], model["vocab_size"])
    structural_tag = {"type": "structural_tag", "format": grammar["format"]}
    return compiler.compile_structural_tag(json.dumps(structural_tag))


def replay(
    model: dict, grammar: dict, compiled: xgr.CompiledGrammar, generation: dict
) -> str:
    """Check one generation of a grammar and return a summary of what was checked."""
    tokenizer, _ = load_tokenizer(model["model_id"], model["vocab_size"])
    token_ids = generation["generation_token_ids"]

    # The recorded token IDs come from the Rust tokenizer; the runner must load the
    # same vocabulary for the grammar to mean the same thing.
    encoded = tokenizer.encode(generation["completion"], add_special_tokens=False)
    if encoded != token_ids[:-1]:
        raise AssertionError(
            "tokenizer mismatch: the completion encodes to "
            f"{encoded}, but the generation records {token_ids[:-1]}"
        )

    if grammar["coverage"] != "from_token_zero":
        return f"compiled only ({grammar['coverage']} is gated by the engine)"

    matcher = xgr.GrammarMatcher(
        compiled, override_stop_tokens=model["all_stop_token_ids"]
    )
    bitmask = xgr.allocate_token_bitmask(1, model["vocab_size"])
    for index, token_id in enumerate(token_ids):
        matcher.fill_next_token_bitmask(bitmask)
        allowed = (bitmask[0, token_id // 32].item() >> (token_id % 32)) & 1
        if not allowed or not matcher.accept_token(token_id):
            prefix = tokenizer.decode(token_ids[:index])
            raise AssertionError(
                f"token {index} ({token_id}, {tokenizer.decode([token_id])!r}) "
                f"rejected after {prefix!r}"
            )
    if not matcher.is_terminated():
        raise AssertionError("matcher not terminated after the stop token")
    return f"replayed {len(token_ids)} tokens from token zero"


def main() -> int:
    paths = sorted(CASES_DIR.glob("*.json"))
    if not paths:
        print(f"no grammar replay cases in {CASES_DIR}", file=sys.stderr)
        return 1

    failures = 0
    for path in paths:
        model = json.loads(path.read_text())
        for case in model["grammars"]:
            grammar = case["grammar"]
            coverage = grammar["coverage"]
            try:
                compiled = compile_grammar(model, grammar)
            except Exception as error:
                failures += 1
                names = ", ".join(case["generations"])
                print(
                    f"FAIL {path.name} {names} [{coverage}]: {error}", file=sys.stderr
                )
                continue
            for name, generation in case["generations"].items():
                try:
                    summary = replay(model, grammar, compiled, generation)
                except Exception as error:
                    failures += 1
                    print(
                        f"FAIL {path.name} {name} [{coverage}]: {error}",
                        file=sys.stderr,
                    )
                else:
                    print(f"PASS {path.name} {name}: {summary}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
