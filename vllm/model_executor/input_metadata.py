import enum
import torch
from typing import Optional, List, Dict


class InputType(enum.Enum):
    PROMPT = enum.auto()
    DECODE = enum.auto()
    MIXED = enum.auto()

class InputMetadata:
    """Metadata for input sequences. Used in PagedAttention.

    Args:
        processed_prompt_lens: Lengths of processed parts of prompts.
        current_prompt_chunk_lens: Lengths of the current chunks (to be processed in this iteration) of prompts
        prefix_plus_current_prompt_tokens_slot_mapping: The address of the KV cache of each token in the processed part and the current chunk of prompt is stored
        current_tokens_slot_mapping: The address to write the new KV to of each token in the current chunk of prompts/decode.
        max_context_len: The maximum context length.
        context_lens: the length of attention context for each sequence.
        block_tables: The block tables. (Seq id -> list of physical block)
        kv_cache_dtype: Data type to store kv cache.
    """

    def __init__(
        self,
        input_type: InputType,
        prompt_seq_ids: List[int],
        processed_prompt_lens: List[int],
        current_prompt_chunk_lens: List[int],
        total_prompt_lens: List[int],
        prefix_plus_current_prompt_tokens_slot_mapping: torch.Tensor,
        current_tokens_slot_mapping: torch.Tensor,
        max_context_len: Optional[int],
        context_lens: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
        use_cuda_graph: bool,
        kv_cache_dtype: str,
        is_profiling_iteration: bool = False,
    ) -> None:
        self.input_type = input_type
        self.prompt_seq_ids = prompt_seq_ids
        self.processed_prompt_lens = processed_prompt_lens
        self.current_prompt_chunk_lens = current_prompt_chunk_lens
        self.total_prompt_lens = total_prompt_lens
        self.prefix_plus_current_prompt_tokens_slot_mapping = prefix_plus_current_prompt_tokens_slot_mapping
        self.current_tokens_slot_mapping = current_tokens_slot_mapping
        self.max_context_len = max_context_len
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.use_cuda_graph = use_cuda_graph
        self.kv_cache_dtype = kv_cache_dtype
        self.is_profiling_iteration = is_profiling_iteration

        assert ([x > 0 for x in current_prompt_chunk_lens].count(False) == 0)
        self.num_prompts = len(
            [x for x in self.current_prompt_chunk_lens if x > 0])
        self.num_processed_prompt_tokens = sum(self.processed_prompt_lens)
        self.num_current_prompt_tokens = sum(self.current_prompt_chunk_lens)
        self.num_generation_tokens = self.context_lens.shape[0]
        self.num_valid_tokens = current_tokens_slot_mapping.shape[0]

        # Set during the execution of the first attention op for each sequence.
        # FIXME(woosuk): This is a hack.
        self.attn_bias: Dict[str, torch.Tensor] = {}
