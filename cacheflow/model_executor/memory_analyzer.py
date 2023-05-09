import torch
from transformers import AutoConfig

from cacheflow.logger import init_logger
from cacheflow.model_executor.utils import get_dtype_size


logger = init_logger(__name__)

_GiB = 1 << 30


class CacheFlowMemoryAnalyzer:

    def get_max_num_gpu_blocks(
        self,
        max_num_batched_tokens: int,
        memory_utilization: float,
    ) -> int:
        raise NotImplementedError()

    def get_workspace_size(self) -> int:
        return 1 * _GiB

    def get_cache_block_size(self) -> int:
        raise NotImplementedError()

    def get_max_num_cpu_blocks(
        self,
        swap_space_gib: int,
    ) -> int:
        swap_space = swap_space_gib * _GiB
        cpu_memory = self.cpu_memory
        if swap_space > 0.8 * cpu_memory:
            raise ValueError(f'The swap space ({swap_space_gib:.2f} GiB) '
                             'takes more than 80% of the available memory '
                             f'({cpu_memory / _GiB:.2f} GiB).'
                             'Please check the swap space size.')
        if swap_space > 0.5 * cpu_memory:
            logger.info(f'WARNING: The swap space ({swap_space_gib:.2f} GiB) '
                        'takes more than 50% of the available memory '
                        f'({cpu_memory / _GiB:.2f} GiB).'
                        'This may slow the system performance.')
        max_num_blocks = swap_space // self.get_cache_block_size()
        return max_num_blocks

    def get_param_size(self) -> int:
        raise NotImplementedError()

    def get_max_act_size(self, max_num_batched_tokens: int) -> int:
        raise NotImplementedError()

    def get_cache_block_size(self) -> int:
        key_cache_block = self.block_size * self.hidden_size // self.tensor_parallel_size
        value_cache_block = key_cache_block
        total = self.num_layers * (key_cache_block + value_cache_block)
        dtype_size = get_dtype_size(self.dtype)
        return dtype_size * total

    def get_max_num_gpu_blocks(
        self,
        max_num_batched_tokens: int,
        memory_utilization: float = 0.95,
    ) -> int:
        # NOTE(woosuk): This assumes that the machine has homogeneous GPUs.
        usable_memory = int(memory_utilization * self.gpu_memory)

        param_size = self.get_param_size()
        act_size = self.get_max_act_size(max_num_batched_tokens)
        workspace_size = self.get_workspace_size()

        max_cache_size = usable_memory - (param_size + act_size + workspace_size)
        if max_cache_size <= 0:
            raise RuntimeError('Not enough GPU memory.')
        max_num_blocks = max_cache_size // self.get_cache_block_size()
        return max_num_blocks


class GPT2MemoryAnalyzer(CacheFlowMemoryAnalyzer):

    def __init__(
        self,
        model_name: str,
        block_size: int,
        dtype: torch.dtype,
        gpu_memory: int,
        cpu_memory: int,
        tensor_parallel_size: int,
    ) -> None:
        self.model_name = model_name
        self.block_size = block_size
        self.dtype = dtype
        self.gpu_memory = gpu_memory
        self.cpu_memory = cpu_memory
        self.tensor_parallel_size = tensor_parallel_size

        config = AutoConfig.from_pretrained(model_name)
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // self.num_heads
        self.ffn_size = config.n_inner if config.n_inner is not None else 4 * self.hidden_size
        self.vocab_size = config.vocab_size
        self.max_position = config.max_position_embeddings

    def get_param_size(self) -> int:
        word_embedding = self.vocab_size * self.hidden_size // self.tensor_parallel_size
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

    def get_max_act_size(
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
        # Size of output logits.
        output_logits = 2 * (max_num_batched_tokens * self.vocab_size)
        max_act = max(max_act, output_logits)
        dtype_size = get_dtype_size(self.dtype)
        return dtype_size * max_act


class OPTMemoryAnalyzer(CacheFlowMemoryAnalyzer):

    def __init__(
        self,
        model_name: str,
        block_size: int,
        dtype: torch.dtype,
        gpu_memory: int,
        cpu_memory: int,
        tensor_parallel_size: int,
    ) -> None:
        self.model_name = model_name
        self.block_size = block_size
        self.dtype = dtype
        self.gpu_memory = gpu_memory
        self.cpu_memory = cpu_memory
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

    def get_param_size(self) -> int:
        word_embedding = self.vocab_size * self.embedding_size // self.tensor_parallel_size
        if self.embedding_size != self.hidden_size:
            # Project in/out.
            word_embedding += 2 * self.embedding_size * self.hidden_size
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

    def get_max_act_size(
        self,
        max_num_batched_tokens: int,
    ) -> int:
        # NOTE: We approxmiately calculate the maximum activation size by
        # estimating
        # 1) the maximum activation tensor size during inference
        # 2) the residual tensor size during inference
        # Here, we assume that we use memory-efficient attention which
        # does not materialize the attention maps in GPU DRAM.
        residual = max_num_batched_tokens * self.hidden_size
        qkv = 3 * (max_num_batched_tokens * self.hidden_size) // self.tensor_parallel_size
        ffn = max_num_batched_tokens * self.ffn_size // self.tensor_parallel_size
        # Double the activation size for input and output.
        max_act = 2 * (max(qkv, ffn) + residual)
        # Size of output logits.
        output_logits = 2 * (max_num_batched_tokens * self.vocab_size)
        max_act = max(max_act, output_logits)
        dtype_size = get_dtype_size(self.dtype)
        return dtype_size * max_act


class LlamaMemoryAnalyzer(CacheFlowMemoryAnalyzer):

    def __init__(
        self,
        model_name: str,
        block_size: int,
        dtype: torch.dtype,
        gpu_memory: int,
        cpu_memory: int,
        tensor_parallel_size: int,
    ) -> None:
        self.model_name = model_name
        self.block_size = block_size
        self.dtype = dtype
        self.gpu_memory = gpu_memory
        self.cpu_memory = cpu_memory
        self.tensor_parallel_size = tensor_parallel_size

        config = AutoConfig.from_pretrained(model_name)
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // self.num_heads
        self.ffn_size = config.intermediate_size
        self.vocab_size = config.vocab_size
        self.max_position = 8192

    def get_param_size(self) -> int:
        # NOTE: LLaMA does not tie the two embeddings.
        word_embedding = self.vocab_size * self.hidden_size // self.tensor_parallel_size
        lm_head = self.vocab_size * self.hidden_size // self.tensor_parallel_size

        # NOTE: LLaMA does not have bias terms.
        ln1 = self.hidden_size
        q = self.hidden_size * self.hidden_size // self.tensor_parallel_size
        k = self.hidden_size * self.hidden_size // self.tensor_parallel_size
        v = self.hidden_size * self.hidden_size // self.tensor_parallel_size
        out = self.hidden_size * self.hidden_size // self.tensor_parallel_size
        # Rotary embedding.
        # TODO(woosuk): Share the rotary embedding between layers.
        rot = self.max_position * self.head_size
        mha = ln1 + q + k + v + out + rot

        ln2 = self.hidden_size
        gate = self.hidden_size * self.ffn_size // self.tensor_parallel_size
        down = self.ffn_size * self.hidden_size // self.tensor_parallel_size
        up = self.hidden_size * self.ffn_size // self.tensor_parallel_size
        ffn = ln2 + gate + down + up

        total = word_embedding + self.num_layers * (mha + ffn) + lm_head
        dtype_size = get_dtype_size(self.dtype)
        return dtype_size * total

    def get_max_act_size(
        self,
        max_num_batched_tokens: int,
    ) -> int:
        # NOTE: We approxmiately calculate the maximum activation size by
        # estimating
        # 1) the maximum activation tensor size during inference
        # 2) the residual tensor size during inference
        # Here, we assume that we use memory-efficient attention which
        # does not materialize the attention maps in GPU DRAM.
        residual = max_num_batched_tokens * self.hidden_size
        qkv = 3 * (max_num_batched_tokens * self.hidden_size) // self.tensor_parallel_size
        ffn = 2 * (max_num_batched_tokens * self.ffn_size) // self.tensor_parallel_size
        # Double the activation size for input and output.
        max_act = 2 * (max(qkv, ffn) + residual)
        # Size of output logits.
        output_logits = 2 * (max_num_batched_tokens * self.vocab_size)
        max_act = max(max_act, output_logits)
        dtype_size = get_dtype_size(self.dtype)
        return dtype_size * max_act


class GPTNeoXMemoryAnalyzer(CacheFlowMemoryAnalyzer):

    def __init__(
        self,
        model_name: str,
        block_size: int,
        dtype: torch.dtype,
        gpu_memory: int,
        cpu_memory: int,
        tensor_parallel_size: int,
    ) -> None:
        self.model_name = model_name
        self.block_size = block_size
        self.dtype = dtype
        self.gpu_memory = gpu_memory
        self.cpu_memory = cpu_memory
        self.tensor_parallel_size = tensor_parallel_size

        config = AutoConfig.from_pretrained(model_name)
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // self.num_heads
        self.ffn_size = config.intermediate_size
        self.vocab_size = config.vocab_size
        self.max_position = 8192
        self.tie_word_embeddings = config.tie_word_embeddings

    def get_param_size(self) -> int:
        word_embedding = self.vocab_size * self.hidden_size // self.tensor_parallel_size
        if self.tie_word_embeddings:
            lm_head = 0
        else:
            lm_head = self.vocab_size * self.hidden_size // self.tensor_parallel_size

        ln1 = 2 * self.hidden_size
        q = self.hidden_size * self.hidden_size // self.tensor_parallel_size + self.hidden_size
        k = self.hidden_size * self.hidden_size // self.tensor_parallel_size + self.hidden_size
        v = self.hidden_size * self.hidden_size // self.tensor_parallel_size + self.hidden_size
        out = self.hidden_size * self.hidden_size // self.tensor_parallel_size + self.hidden_size
        # Rotary embedding.
        # TODO(woosuk): Share the rotary embedding between layers.
        rot = self.max_position * self.head_size
        mha = ln1 + q + k + v + out + rot

        ln2 = 2 * self.hidden_size
        ffn1 = self.hidden_size * self.ffn_size // self.tensor_parallel_size + self.ffn_size
        ffn2 = self.ffn_size * self.hidden_size // self.tensor_parallel_size + self.hidden_size
        ffn = ln2 + ffn1 + ffn2

        total = word_embedding + self.num_layers * (mha + ffn) + lm_head
        dtype_size = get_dtype_size(self.dtype)
        return dtype_size * total

    def get_max_act_size(
        self,
        max_num_batched_tokens: int,
    ) -> int:
        # NOTE: We approxmiately calculate the maximum activation size by
        # estimating
        # 1) the maximum activation tensor size during inference
        # 2) the residual tensor size during inference
        # Here, we assume that we use memory-efficient attention which
        # does not materialize the attention maps in GPU DRAM.
        residual = max_num_batched_tokens * self.hidden_size
        qkv = 3 * (max_num_batched_tokens * self.hidden_size) // self.tensor_parallel_size
        ffn = 2 * (max_num_batched_tokens * self.ffn_size) // self.tensor_parallel_size
        # Double the activation size for input and output.
        max_act = 2 * (max(qkv, ffn) + residual)
        # Size of output logits.
        output_logits = 2 * (max_num_batched_tokens * self.vocab_size)
        max_act = max(max_act, output_logits)
        dtype_size = get_dtype_size(self.dtype)
        return dtype_size * max_act
