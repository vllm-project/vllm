# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum, auto

MODEL_NAME = "facebook/opt-125m"
DUMMY_LOGITPROC_ENTRYPOINT = "dummy_logitproc"
DUMMY_LOGITPROC_FQCN = ("vllm.test_utils:DummyLogitsProcessor")
DUMMY_LOGITPROC_ARG = "target_token"
TEMP_GREEDY = 0.0
MAX_TOKENS = 20


class LogitprocSource(Enum):
    """How to source a logitproc for testing purposes"""
    LOGITPROC_SOURCE_ENTRYPOINT = auto()
    LOGITPROC_SOURCE_FQCN = auto()
    LOGITPROC_SOURCE_CLASS = auto()


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
