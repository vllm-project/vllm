
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from vllm.multimodal import MultiModalKwargs
from vllm.sampling_params import SamplingParams 
import torch
from vllm.v1.core.scheduler import  RunningRequestData
from vllm.lora.request import LoRARequest


if TYPE_CHECKING:
    from vllm.multimodal.inputs import PlaceholderRange

@dataclass
class CachedRequestState:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    mm_inputs: List[MultiModalKwargs]
    mm_positions: List["PlaceholderRange"]
    sampling_params: SamplingParams
    generator: Optional[torch.Generator]

    block_ids: List[int]
    num_computed_tokens: int
    output_token_ids: List[int]

    lora_request: Optional[LoRARequest]

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def update(self, req_data: RunningRequestData) -> None:
        self.num_computed_tokens = req_data.num_computed_tokens
        if len(req_data.new_block_ids):
            self.block_ids.extend(req_data.new_block_ids)