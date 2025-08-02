# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum, auto

MODEL_NAME = "facebook/opt-125m"
DUMMY_LOGITPROC_ARG = "target_token"
TEMP_GREEDY = 0.0
MAX_TOKENS = 20


class CustomLogitprocSource(Enum):
    """How to source a logitproc for testing purposes"""
    LOGITPROC_SOURCE_NONE = auto()  # No custom logitproc
    LOGITPROC_SOURCE_ENTRYPOINT = auto()  # Via entrypoint
    LOGITPROC_SOURCE_FQCN = auto()  # Via fully-qualified class name (FQCN)
    LOGITPROC_SOURCE_CLASS = auto()  # Via provided class object


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
