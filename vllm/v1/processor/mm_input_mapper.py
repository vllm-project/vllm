from typing import Dict, List

from vllm.config import ModelConfig
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalDataDict,
                             MultiModalInputs, MultiModalRegistry)
from vllm.v1.processor.dist_processor import (Processor, ProcessorImpl,
                                              ProcessorInputs,
                                              ProcessorOutputs)


class MMInputMapperInputs(ProcessorInputs):

    # [num_reqs]
    req_ids: List[str]


class MMInputMapperOutputs(ProcessorOutputs):

    # [num_reqs]
    req_ids: List[str]


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

    def process_inputs(self,
                       inputs: MMInputMapperInputs) -> MMInputMapperOutputs:
        pass
