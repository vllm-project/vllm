#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os

from vllm import LLM

C5_SANITY_PROMPTS = [
    "What is the capital of France? Reply with just the city name.",
    "What sport did Michael Jordan play professionally?",
    "Write a one-line Python function `add(a, b)` that returns the sum of `a` and `b`.",
    "请用中文回答: 法国的首都是哪个城市?用一个词回答。",
]

# Expect at least one of these strings to appear in the lowercased response.
# Lists are intentionally permissive: the c5 model emits a thinking trace
# (<|end_thinking|>...<|start_text|>) followed by a final answer. Any keyword
# match across the full generation counts as a pass. The Chinese prompt
# explicitly instructs the model to answer in Chinese, so we require 巴黎
# specifically (the thinking trace stays in English).
C5_SANITY_EXPECTED = [
    ["paris"],
    ["basketball", "nba"],
    ["a + b", "a+b", "return a", "sum("],
    ["巴黎"],
]


def validate_model_path(model_path: str) -> str:
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            "Model checkpoint directory does not exist: "
            f"{model_path}. Pass the model path explicitly and ensure the "
            "checkpoint is downloaded."
        )
    return model_path


def shutdown_llm(llm: LLM) -> None:
    llm_engine = getattr(llm, "llm_engine", None)
    if llm_engine is not None and hasattr(llm_engine, "shutdown"):
        llm_engine.shutdown()
