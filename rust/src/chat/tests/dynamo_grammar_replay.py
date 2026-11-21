# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "xgrammar==0.2.9"]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Replay vLLM's Dynamo grammars without downloading checkpoint tokenizers."""

import json
import sys
from pathlib import Path

import xgrammar as xgr


def native_generation(family: str, text: str) -> str:
    # Corpus parsers accept optional framing; strict native grammars require the
    # checkpoint's newlines and continue its prompt-opened assistant header.
    if family in ("deepseek_v4", "deepseek_v41"):
        prefix = "｜DSML｜" if family == "deepseek_v4" else "｜DSML｜ "
        calls = "tool_calls" if family == "deepseek_v4" else "calls"
        text = text.replace(f"<{prefix}{calls}>", f"<{prefix}{calls}>\n")
        text = text.replace('name="get_weather">', 'name="get_weather">\n')
        for tag in ("parameter", "invoke"):
            text = text.replace(f"</{prefix}{tag}>", f"</{prefix}{tag}>\n")
    elif family == "minimax_m3":
        text = text.replace("]<]minimax[>[<tool_call>", "]<]minimax[>[<tool_call>\n")
        text = text.replace("]<]minimax[>[</tool_call>", "\n]<]minimax[>[</tool_call>")
    elif family in ("muse_spark", "gpt_oss"):
        text = text.removeprefix("<|start|>assistant")
        if family == "gpt_oss":
            text = text.replace("get_weather <|constrain|>", "get_weather<|constrain|>")
    elif family == "beam":
        text = "<|channel|>analysis<|message|>think<|end|>" + text
    return text


def main() -> None:
    root = Path(sys.argv[1])
    paths = sorted(root.glob("*.json"))
    assert len(paths) == 10, "expected every supported native structural-tag builder"
    # A raw-byte vocabulary checks actual grammar compilation and matching;
    # checkpoint tokenization and generation quality are separate tests.
    info = xgr.TokenizerInfo(
        [bytes([i]) for i in range(256)] + [b"<eos>"],
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[256],
    )
    compiler = xgr.GrammarCompiler(info)
    for path in paths:
        case = json.loads(path.read_text())
        compiled = compiler.compile_structural_tag(
            json.dumps({"type": "structural_tag", "format": case["format"]})
        )
        matcher = xgr.GrammarMatcher(compiled)
        generation = native_generation(case["family"], case["input"])
        for offset, token in enumerate(generation.encode()):
            assert matcher.accept_token(token), (
                f"{case['family']}: rejected byte {offset}"
            )
        assert matcher.accept_token(256), f"{case['family']}: rejected EOS"
        assert matcher.is_terminated(), case["family"]
        print(f"PASS {case['family']}: native generation and EOS")


if __name__ == "__main__":
    main()
