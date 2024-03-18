from dataclasses import dataclass
from typing import Optional, List
from xformers.ops.fmha.attn_bias import AttentionBias

import torch


@dataclass
class InputMetadata:
    """Metadata for input sequences. Used in PagedAttention.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replays. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.

    Args:
        prompt_lens: Lengths of prompts per sequence.
        slot_mapping: The indices of the token slots that input tokens will be stored into.
	        E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the three tokens
	        are stored in the 3rd slot in block 2, 2nd slot in block 0, and 1st slot in block 1,
	        respectively. 
        num_prompt_tokens: The total number of tokens in the prompts. This might
            include padding.
        num_generation_tokens: The number of tokens in the generation sequences.
            This might include padding.
        max_context_len: The maximum context length.
        context_lens: the length of attention context for each sequence.
            I.e., the number of tokens that have attended so far.
        block_tables: The block tables. (Seq id -> list of physical block)
        kv_cache_dtype: Data type to store kv cache.
    """
    is_prompt: bool
    slot_mapping: torch.Tensor
    prompt_lens: Optional[List]
    num_prompt_tokens: int
    num_generation_tokens: int
    max_seq_len: Optional[int]
    start_loc: Optional[torch.Tensor]
    max_context_len: Optional[int]
    # [batch_size]. Each index means each sequence, and the value means the length of tokens stored in the kv cache.
    # NOTE(sang): When it is prefill/decoding, the definition is different. For prefill, it means the the length of KV that are cached excluding the new KVs. In decoding, this includes a new KV.
    context_lens: Optional[torch.Tensor]
    block_tables: Optional[torch.Tensor]
    use_cuda_graph: bool
    kv_cache_dtype: str

    # Fields below are initialiezd in post init.
    prompt_lens_tensor: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None
        if self.prompt_lens is not None:
            self.prompt_lens_tensor = torch.tensor(self.prompt_lens,
                                                   dtype=torch.long,
                                                   device=self.slot_mapping.device)
