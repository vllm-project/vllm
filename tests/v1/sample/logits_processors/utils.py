# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
MODEL_NAME = "facebook/opt-125m"
DUMMY_LOGITPROC_ENTRYPOINT = "dummy_logitproc"
DUMMY_LOGITPROC_FQN = (
    "vllm.v1.sample.logits_processor.impls:DummyLogitsProcessor")
DUMMY_LOGITPROC_ARG = "target_token"
LOGITPROC_SOURCE_ENTRYPOINT = "entrypoint"
LOGITPROC_SOURCE_FQN = "fqn"
TEMP_GREEDY = 0.0
MAX_TOKENS = 20

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]