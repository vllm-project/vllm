# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types
from enum import Enum, auto
from typing import Optional

import torch

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (LOGITSPROCS_GROUP,
                                             AdapterLogitsProcessor,
                                             BatchUpdate, LogitsProcessor,
                                             RequestLogitsProcessor)
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

MODEL_NAME = "facebook/opt-125m"
POOLING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DUMMY_LOGITPROC_ARG = "target_token"
TEMP_GREEDY = 0.0
MAX_TOKENS = 20
DUMMY_LOGITPROC_ENTRYPOINT = "dummy_logitproc"
DUMMY_LOGITPROC_MODULE = "DummyModule"
DUMMY_LOGITPROC_FQCN = f"{DUMMY_LOGITPROC_MODULE}:DummyLogitsProcessor"


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
        self.req_info: dict[int, int] = {}

    def is_argmax_invariant(self) -> bool:
        """Never impacts greedy sampling"""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        process_dict_updates(
            self.req_info,
            batch_update,
            lambda params, _, __: params.extra_args and
            (params.extra_args.get("target_token")),
        )

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


"""Dummy module with dummy logitproc class"""
dummy_module = types.ModuleType(DUMMY_LOGITPROC_MODULE)
dummy_module.DummyLogitsProcessor = DummyLogitsProcessor  # type: ignore


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


def get_req_dummy_logits_processor(
    params: SamplingParams, ) -> Optional[RequestLogitsProcessor]:
    """Fake example of a request-level logits processor implementation.

    This function returns a new request-level logits processor, customized
    to the sampling params associated with a particular request.

    Returns None if the logits processor should not be applied to the
    particular request. To use the logits processor the request must have
    a "target_token" custom argument with an integer value.

    Args:
      params: per-request sampling params

    Returns:
      `Callable` request logits processor, or None
    """
    if not params.extra_args or "target_token" not in params.extra_args:
        return None
    target_token = params.extra_args["target_token"]

    def req_dummy_logits_processor(
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """The request-level logits processor masks out all logits except the
        token id identified by target_token"""
        val_to_keep = logits[target_token]
        logits[:] = float("-inf")
        logits[target_token] = val_to_keep
        return logits

    return req_dummy_logits_processor


class DummyLogitsProcessorWrapped(AdapterLogitsProcessor):
    """Example of wrapping a fake request-level logit processor to create a
    batch-level logits processor"""

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)

    def is_argmax_invariant(self) -> bool:
        return False

    def req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        return get_req_dummy_logits_processor(params)


"""Fake version of importlib.metadata.entry_points"""
entry_points = lambda group: EntryPoints(group)
