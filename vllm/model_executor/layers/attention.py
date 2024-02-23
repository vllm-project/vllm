import torch

from vllm.model_executor.layers.attention_backends.base import BaseAttention
from vllm.utils import is_hip


class AttentionFactory:

    @staticmethod
    def create_attention(*args, **kwargs) -> BaseAttention:
        if not is_hip() and torch.cuda.get_device_capability(0)[0] >= 8:
            from vllm.model_executor.layers.attention_backends.flash import Attention
            return Attention(*args, **kwargs)
        else:
            from vllm.model_executor.layers.attention_backends.non_flash import Attention
            return Attention(*args, **kwargs)
