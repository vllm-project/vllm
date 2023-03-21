import torch
from transformers import AutoConfig

from cacheflow.models.utils import get_cpu_memory
from cacheflow.models.utils import get_dtype_size
from cacheflow.models.utils import get_gpu_memory

_GiB = 1 << 30


class CacheFlowMemoryAnalyzer:

    def get_max_num_gpu_blocks(
        self,
        max_num_batched_tokens: int,
        memory_utilization: float,
    ) -> int:
        raise NotImplementedError()

    def get_max_num_cpu_blocks(
        self,
        memory_utilization: float,
    ) -> int:
        raise NotImplementedError()


class OPTMemoryAnalyzer(CacheFlowMemoryAnalyzer):

    def __init__(
        self,
        model_name: str,
        block_size: int,
        dtype: torch.dtype,
        tensor_parallel_size: int,
    ) -> None:
        self.model_name = model_name
        self.block_size = block_size
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size

        config = AutoConfig.from_pretrained(model_name)
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // self.num_heads
        self.ffn_size = config.ffn_dim
        self.embedding_size = config.word_embed_proj_dim
        self.vocab_size = config.vocab_size
        self.max_position = config.max_position_embeddings

    def _get_param_size(self) -> int:
        word_embedding = self.vocab_size * self.embedding_size // self.tensor_parallel_size
        if self.embedding_size != self.vocab_size:
            # Project in/out.
            word_embedding += 2 * self.embedding_size * self.vocab_size
        position_embedding = self.max_position * self.hidden_size

        ln1 = 2 * self.hidden_size
        q = self.hidden_size * self.hidden_size // self.tensor_parallel_size + self.hidden_size
        k = self.hidden_size * self.hidden_size // self.tensor_parallel_size + self.hidden_size
        v = self.hidden_size * self.hidden_size // self.tensor_parallel_size + self.hidden_size
        out = self.hidden_size * self.hidden_size // self.tensor_parallel_size + self.hidden_size
        mha = ln1 + q + k + v + out

        ln2 = 2 * self.hidden_size
        ffn1 = self.hidden_size * self.ffn_size // self.tensor_parallel_size + self.ffn_size
        ffn2 = self.ffn_size * self.hidden_size // self.tensor_parallel_size + self.hidden_size
        ffn = ln2 + ffn1 + ffn2

        total = (word_embedding + position_embedding +
                 self.num_layers * (mha + ffn))
        dtype_size = get_dtype_size(self.dtype)
        return dtype_size * total

    def _get_max_act_size(
        self,
        max_num_batched_tokens: int,
    ) -> int:
        # NOTE: We approxmiately calculate the maximum activation size by
        # estimating
        # 1) the maximum activation tensor size during inference
        # 2) the residual tensor size during inference
        # Here, we assume that FlashAttention is used and
        # thus the attention maps are never materialized in GPU DRAM.
        residual = max_num_batched_tokens * self.hidden_size
        qkv = 3 * (max_num_batched_tokens * self.hidden_size) // self.tensor_parallel_size
        ffn = max_num_batched_tokens * self.ffn_size // self.tensor_parallel_size
        # Double the activation size for input and output.
        max_act = 2 * (max(qkv, ffn) + residual)
        dtype_size = get_dtype_size(self.dtype)
        return dtype_size * max_act

    def _get_workspace_size(self) -> int:
        return 1 * _GiB

    def _get_cache_block_size(self) -> int:
        key_cache_block = self.block_size * self.num_heads * self.head_size
        value_cache_block = self.block_size * self.num_heads * self.head_size
        total = self.num_layers * (key_cache_block + value_cache_block)
        dtype_size = get_dtype_size(self.dtype)
        return dtype_size * total

    def get_max_num_gpu_blocks(
        self,
        max_num_batched_tokens: int,
        memory_utilization: float = 0.95,
    ) -> int:
        # NOTE(woosuk): This assumes that the machine has homogeneous GPUs.
        gpu_memory = get_gpu_memory()
        usable_memory = int(memory_utilization * gpu_memory)

        param_size = self._get_param_size()
        act_size = self._get_max_act_size(max_num_batched_tokens)
        workspace_size = self._get_workspace_size()

        max_cache_size = usable_memory - (param_size + act_size + workspace_size)
        max_num_blocks = max_cache_size // self._get_cache_block_size()
        return max_num_blocks

    def get_max_num_cpu_blocks(
        self,
        swap_space: int,
    ) -> int:
        swap_space = swap_space * _GiB
        cpu_memory = get_cpu_memory()
        if swap_space > 0.8 * cpu_memory:
            raise ValueError(f'The swap space ({swap_space / _GiB:.2f} GiB) '
                             'takes more than 80% of the available memory '
                             f'({cpu_memory / _GiB:.2f} GiB).'
                             'Please check the swap space size.')
        if swap_space > 0.5 * cpu_memory:
            print(f'WARNING: The swap space ({swap_space / _GiB:.2f} GiB) '
                  'takes more than 50% of the available memory '
                  f'({cpu_memory / _GiB:.2f} GiB).'
                  'This may slow the system performance.')
        max_num_blocks = swap_space // self._get_cache_block_size()
        return max_num_blocks
