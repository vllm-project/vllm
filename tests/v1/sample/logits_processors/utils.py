# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (BatchUpdate, LogitsProcessor,
                                             MoveDirectionality)

MODEL_NAME = "facebook/opt-125m"
DUMMY_LOGITPROC_ENTRYPOINT = "dummy_logitproc"
DUMMY_LOGITPROC_FQN = (
    "tests.v1.sample.logits_processors.utils:DummyLogitsProcessor")
DUMMY_LOGITPROC_ARG = "target_token"
LOGITPROC_SOURCE_ENTRYPOINT = "entrypoint"
LOGITPROC_SOURCE_FQCN = "fqcn"
TEMP_GREEDY = 0.0
MAX_TOKENS = 20

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class DummyLogitsProcessor(LogitsProcessor):
    """Fake logit processor to support unit testing and examples"""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        self.req_info = {}

    def is_argmax_invariant(self) -> bool:
        """Never impacts greedy sampling"""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        # Process added requests.
        for index, params, _ in batch_update.added:
            if isinstance(params, SamplingParams) and params.extra_args:
                target_token = params.extra_args.get("target_token", None)
            else:
                target_token = None
            self.req_info[index] = target_token

        if self.req_info:
            # Process removed requests.
            for index in batch_update.removed:
                self.req_info.pop(index, None)

            # Process moved requests, unidirectional (a->b) and swap (a<->b)
            for adx, bdx, direct in batch_update.moved:
                if direct == MoveDirectionality.SWAP:
                    (self.req_info[adx],
                     self.req_info[bdx]) = (self.req_info[bdx],
                                            self.req_info[adx])
                else:
                    self.req_info[bdx] = self.req_info[adx]

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for bdx in range(logits.shape[0]):
            if (target_token := self.req_info[bdx]) is not None:
                mask = torch.ones_like(logits[bdx, :], dtype=torch.bool)
                mask[target_token] = False
                logits[bdx, mask] = float('-inf')

        return logits
