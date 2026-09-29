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

Every case is compiled against the model tokenizer the same way as
`vllm/v1/structured_output/backend_xgrammar.py`. Token-zero cases are then
replayed from the first generated token: each token must be allowed by the
bitmask and accepted by the matcher, and the final stop token must terminate it.
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


def replay(case: dict) -> str:
    """Check one case and return a summary of what was checked."""
    tokenizer, compiler = load_tokenizer(case["model_id"], case["vocab_size"])
    generation = case["generation_token_ids"]

    # The case token IDs come from the Rust tokenizer; the runner must load the
    # same vocabulary for the grammar to mean the same thing.
    encoded = tokenizer.encode(case["completion"], add_special_tokens=False)
    if encoded != generation[:-1]:
        raise AssertionError(
            "tokenizer mismatch: the completion encodes to "
            f"{encoded}, but the case records {generation[:-1]}"
        )

    grammar = case["grammar"]
    structural_tag = {"type": "structural_tag", "format": grammar["format"]}
    compiled = compiler.compile_structural_tag(json.dumps(structural_tag))
    if grammar["coverage"] != "from_token_zero":
        return f"compiled only ({grammar['coverage']} is gated by the engine)"

    matcher = xgr.GrammarMatcher(
        compiled, override_stop_tokens=case["all_stop_token_ids"]
    )
    bitmask = xgr.allocate_token_bitmask(1, case["vocab_size"])
    for index, token_id in enumerate(generation):
        matcher.fill_next_token_bitmask(bitmask)
        allowed = (bitmask[0, token_id // 32].item() >> (token_id % 32)) & 1
        if not allowed or not matcher.accept_token(token_id):
            prefix = tokenizer.decode(generation[:index])
            raise AssertionError(
                f"token {index} ({token_id}, {tokenizer.decode([token_id])!r}) "
                f"rejected after {prefix!r}"
            )
    if not matcher.is_terminated():
        raise AssertionError("matcher not terminated after the stop token")
    return f"replayed {len(generation)} tokens from token zero"


def main() -> int:
    paths = sorted(CASES_DIR.glob("*.json"))
    if not paths:
        print(f"no grammar replay cases in {CASES_DIR}", file=sys.stderr)
        return 1

    failures = 0
    for path in paths:
        case = json.loads(path.read_text())
        try:
            summary = replay(case)
        except Exception as error:
            failures += 1
            coverage = case["grammar"]["coverage"]
            print(f"FAIL {path.name} [{coverage}]: {error}", file=sys.stderr)
        else:
            print(f"PASS {path.name}: {summary}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
