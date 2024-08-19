from vllm.inputs.registry import InputContext
from vllm.logger import init_logger

from .base import MultiModalInputs, MultiModalPlugin

logger = init_logger(__name__)


class XLMRobertaPlugin(MultiModalPlugin):

    def get_data_key(self) -> str:
        return "xlmroberta"

    def _default_input_mapper(self, ctx: InputContext,
                              data: object) -> MultiModalInputs:

        input_ids = data['input_ids']  # type: ignore
        attention_mask = data['attention_mask']  # type: ignore

        return MultiModalInputs({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        raise NotImplementedError
