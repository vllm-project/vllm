from typing import Union

import torch
from transformers import AutoConfig

from cacheflow.models.model_utils import get_torch_dtype

GB = 1 << 30


def compute_max_num_gpu_blocks(
    model_name: str,
    max_num_batched_tokens: int,
    block_size: int,
    dtype: Union[torch.dtype, str],
) -> int:
    torch_dtype = get_torch_dtype(dtype)
    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()

    # FIXME(woosuk)
    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size

    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory = int(0.975 * total_memory)

    param_size = (num_layers * 12 * hidden_size * hidden_size * dtype_size
                  + vocab_size * hidden_size * dtype_size)
    mha_act_size = num_heads * max_num_batched_tokens * max_num_batched_tokens * dtype_size
    ffn_act_size = 4 * hidden_size * max_num_batched_tokens * dtype_size
    # Conservative estimate of the peak activation size.
    act_size = 3 * max(mha_act_size, ffn_act_size)
    workspace_size = 1 * GB

    max_cache_size = total_memory - (param_size + act_size + workspace_size)
    max_num_blocks = max_cache_size // (num_layers * 2 * block_size * hidden_size * dtype_size)
    return max_num_blocks


def compute_max_num_cpu_blocks(
    swap_space: int,
    model_name: str,
    block_size: int,
    dtype: Union[torch.dtype, str],
) -> int:
    torch_dtype = get_torch_dtype(dtype)
    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    max_cache_size = swap_space * GB
    max_num_blocks = max_cache_size // (num_layers * 2 * block_size * hidden_size * dtype_size)
    return max_num_blocks
