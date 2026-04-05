# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shutdown test utils"""

from vllm import LLMEngine
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.v1.engine.async_llm import AsyncLLM

SHUTDOWN_TEST_TIMEOUT_SEC = 120
SHUTDOWN_TEST_THRESHOLD_BYTES = 2 * 2**30


def get_engine_input(engine: LLMEngine | AsyncLLM, prompt: str):
    parsed_prompt = parse_model_prompt(engine.model_config, prompt)
    return engine.renderer.render_cmpl([parsed_prompt])[0]
