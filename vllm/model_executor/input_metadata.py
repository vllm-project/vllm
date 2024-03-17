from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class InputMetadata:
    """Metadata for input sequences. Used in PagedAttention.

    Args:
        prompt_lens: Lengths of prompts.
        slot_mapping: The address to write the new KV to of each token.
        max_context_len: The maximum context length.
        context_lens: the length of attention context for each sequence.
        block_tables: The block tables. (Seq id -> list of physical block)
        kv_cache_dtype: Data type to store kv cache.
    """

    is_prompt: bool
    slot_mapping: torch.Tensor
    prompt_lens: Optional[torch.Tensor]
    max_seq_len: Optional[int]
    start_loc: Optional[torch.Tensor]
    max_context_len: Optional[int]
    context_lens: Optional[torch.Tensor]
    block_tables: Optional[torch.Tensor]
    use_cuda_graph: bool
    kv_cache_dtype: str

    def __post_init__(self):
        # will not appear in the __repr__ and __init__
        self.attn_bias = None
