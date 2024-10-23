from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vllm.config import ModelConfig
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalDataBuiltins,
                             MultiModalInputs, MultiModalRegistry)
from vllm.v1.processor.dist_processor import (Processor, ProcessorImpl,
                                              ProcessorInputs,
                                              ProcessorOutputs)
from vllm.v1.request import Request


@dataclass
class MMInputMapperInputs(ProcessorInputs):

    # [num_reqs]
    req_ids: List[str]
    mm_data: List[MultiModalDataBuiltins]
    mm_processor_kwargs: List[Optional[Dict[str, Any]]]

    @classmethod
    def from_requests(cls, requests: List["Request"]) -> "MMInputMapperInputs":
        req_ids = [req.request_id for req in requests]
        mm_data = [req.mm_data for req in requests]
        mm_processor_kwargs = [req.mm_processor_kwargs for req in requests]
        return cls(req_ids, mm_data, mm_processor_kwargs)


@dataclass
class MMInputMapperOutputs(ProcessorOutputs):

    # [num_reqs]
    req_ids: List[str]
    mm_inputs: List[MultiModalInputs]


class MMInputMapper(Processor):

    def __init__(self, model_config: ModelConfig):
        super().__init__(MMInputMapperImpl, MMInputMapperInputs,
                         MMInputMapperOutputs, model_config)


class MMInputMapperImpl(ProcessorImpl):

    def __init__(
        self,
        model_config: ModelConfig,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):
        self.mm_registry = mm_registry
        self.multi_modal_input_mapper = mm_registry.create_input_mapper(
            model_config)
        self.mm_registry.init_mm_limits_per_prompt(model_config)

    def process_inputs(self,
                       inputs: MMInputMapperInputs) -> MMInputMapperOutputs:
        mm_inputs: List[Optional[MultiModalInputs]] = []
        num_reqs = len(inputs.req_ids)
        for i in range(num_reqs):
            if inputs.mm_data[i] is None:
                # No multi-modal input for this request.
                mm_inputs.append(None)
                continue

            mm_input = self.multi_modal_input_mapper(
                inputs.mm_data[i],
                mm_processor_kwargs=inputs.mm_processor_kwargs[i],
            )
            mm_inputs.append(mm_input)
        return MMInputMapperOutputs(inputs.req_ids, mm_inputs)
