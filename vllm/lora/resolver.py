from abc import ABC, abstractmethod
from typing import Union
from vllm.lora.request import LoRARequest


class LoRAResolver(ABC):

    @abstractmethod
    async def resolve_lora(self, lora_name: str) -> Union[LoRARequest, None]:
        """Abstract method to resolve and fetch a LoRA model adapter.
        
        This method should implement the logic to locate and download LoRA 
        adapter based on the provided name. Implementations might fetch from 
        a blob storage or other sources.
         
        Args:
            lora_name: str - The name or identifier of the LoRA model to 
                resolve.
        
        Returns:
            Union[LoRARequest, None]: A LoRARequest object containing the 
                resolved LoRA model information, or None if the LoRA model 
                cannot be found.
        """
