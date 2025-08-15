# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum, auto
from typing import Optional

import torch

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (LOGITSPROCS_GROUP, BatchUpdate,
                                             LogitsProcessor,
                                             MoveDirectionality)

MODEL_NAME = "facebook/opt-125m"
POOLING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DUMMY_LOGITPROC_ARG = "target_token"
TEMP_GREEDY = 0.0
MAX_TOKENS = 20
DUMMY_LOGITPROC_ENTRYPOINT = "dummy_logitproc"
DUMMY_LOGITPROC_FQCN = ("tests.v1.sample.logits_processors.utils:"
                        "DummyLogitsProcessor")


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


class DummyLogitsProcessor(LogitsProcessor):
    """Fake logit processor to support unit testing and examples"""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        self.req_info: dict[int, SamplingParams] = {}

    def is_argmax_invariant(self) -> bool:
        """Never impacts greedy sampling"""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        # Process added requests.
        for index, params, _, _ in batch_update.added:
            assert params is not None
            if params.extra_args and (target_token :=
                                      params.extra_args.get("target_token")):
                self.req_info[index] = target_token

        if self.req_info:
            # Process removed requests.
            for index in batch_update.removed:
                self.req_info.pop(index, None)

            # Process moved requests, unidirectional move (a->b) and swap
            # (a<->b)
            for adx, bdx, direct in batch_update.moved:
                a_val = self.req_info.pop(adx, None)
                b_val = self.req_info.pop(bdx, None)
                if a_val is not None:
                    self.req_info[bdx] = a_val
                if direct == MoveDirectionality.SWAP and b_val is not None:
                    self.req_info[adx] = b_val

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info:
            return logits

        # Save target values before modification
        rows_list = list(self.req_info.keys())
        cols = torch.tensor([self.req_info[i] for i in rows_list],
                            dtype=torch.long,
                            device=logits.device)
        rows = torch.tensor(rows_list, dtype=torch.long, device=logits.device)
        values_to_keep = logits[rows, cols].clone()

        # Mask all but target tokens
        logits[rows] = float('-inf')
        logits[rows, cols] = values_to_keep

        return logits


class EntryPoint:
    """Dummy entrypoint class for logitsprocs testing"""

    def __init__(self):
        self.name = DUMMY_LOGITPROC_ENTRYPOINT
        self.value = DUMMY_LOGITPROC_FQCN

    def load(self):
        return DummyLogitsProcessor


class EntryPoints(list):
    """Dummy EntryPoints class for logitsprocs testing"""

    def __init__(self, group: str):
        # Emulate list-like functionality
        eps = [EntryPoint()] if group == LOGITSPROCS_GROUP else []
        super().__init__(eps)
        # Extra attributes
        self.names = [ep.name for ep in eps]


entry_points = lambda group: EntryPoints(group)
