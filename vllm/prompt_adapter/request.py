from dataclasses import dataclass

from vllm.adapter_commons.request import AdapterRequest


@dataclass
class PromptAdapterRequest(AdapterRequest):
    """
    Request for a Prompt adapter.
    """

    prompt_adapter_name: str
    prompt_adapter_id: int
    prompt_adapter_local_path: str
    prompt_adapter_num_virtual_tokens: int

    def __hash__(self):
        return super().__hash__()

    @property
    def adapter_id(self):
        return self.prompt_adapter_id

    @property
    def name(self):
        return self.prompt_adapter_name

    @property
    def local_path(self):
        return self.prompt_adapter_local_path
