import torch

from vllm.model_executor.layers.attention.base import BaseAttention
from vllm.utils import is_hip


class AttentionFactory:

    @staticmethod
    def create_attention(*args, **kwargs) -> BaseAttention:
        if not is_hip() and torch.cuda.get_device_capability()[0] >= 8:
            from vllm.model_executor.layers.attention.flash import Attention
            return Attention(*args, **kwargs)
        else:
            from vllm.model_executor.layers.attention.non_flash import Attention
            return Attention(*args, **kwargs)
