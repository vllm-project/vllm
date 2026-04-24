#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os

from cohere.test_utils_engine_args import get_engine_kwargs_with_overrides

from vllm import LLM

C5_SANITY_PROMPTS = [
    (
        "Q: What is the capital of China?\n"
        "A: The capital of China is Beijing.\n"
        "Q: What is the capital of France?\n"
        "A: The capital of France is "
    ),
    "Michael Jordan is a ",
    "def add(a, b):\n    return",
    ("问题: 中国的首都是哪里?\n答案: 北京\n问题: 法国的首都是哪里?\n答案: "),
]

# Expect at least one of these strings to appear in the response.
C5_SANITY_EXPECTED = [["paris"], ["basketball", "nba"], ["a+b", "a + b"], ["巴黎"]]


def validate_model_path(model_path: str) -> str:
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            "Model checkpoint directory does not exist: "
            f"{model_path}. Pass the model path explicitly and ensure the "
            "checkpoint is downloaded."
        )
    return model_path


def build_c5_llm(
    model_path: str,
    tensor_parallel_size: int,
    engine_args: str | None,
) -> LLM:
    # Get effective engine kwargs with hardware profile args + test-specific
    # overrides.
    effective_kwargs = get_engine_kwargs_with_overrides(
        test_kwargs={
            "model": model_path,
            "max_model_len": 32768,  # c5 uses smaller context than default
            "tensor_parallel_size": tensor_parallel_size,
        },
        engine_args_override=engine_args,
    )
    print("Creating LLM with effective kwargs...")
    return LLM(**effective_kwargs)


def shutdown_llm(llm: LLM) -> None:
    llm_engine = getattr(llm, "llm_engine", None)
    if llm_engine is not None and hasattr(llm_engine, "shutdown"):
        llm_engine.shutdown()
