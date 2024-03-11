from typing import Optional

import torch


class InputMetadata:
    """Metadata for input sequences. Used in PagedAttention.

    Both prefill and decoding inputs are mixed up in this class,
    which can be useful to optimize performance.

    If you want metadata specific to prefill or decode,
    Use .prefill_input_metadata() or .decode_input_metadata() API.
    Some of metadata that's not needed for prefill or decode
    could be None.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.

    Args:
        prompt_lens: Lengths of prompts. Only included in prefill requests.
        num_chunked_prefill: Number of chunked prefill requests across
            sequences.
        slot_mapping: The index of each token mapped into a physical block
            in block tables. E.g., if block_size is 32, 35 means it is in
            the block number 1, 3rd index.
        num_prompt_tokens: The number of tokens in the prompts. This might
            include padding.
        num_generation_tokens: The number of tokens in the generation sequences.
            This might include padding.
        max_context_len: The maximum context length.
        context_lens: the length of attention context for each sequence.
            I.e., the number of tokens that have attended so far.
        block_tables: The block tables. (Seq id -> list of physical block)
        kv_cache_dtype: Data type to store kv cache.
        num_prompt_tokens: The number of tokens in the prompts. This might
            include padding.
        num_generation_tokens: The number of tokens in the generation sequences.
            This might include padding.
    """

    def __init__(
        self,
        slot_mapping: torch.Tensor,
        prompt_lens: Optional[torch.Tensor],
        num_chunked_prefill: int,
        num_prompt_tokens: int,
        num_generation_tokens: int,
        start_loc: Optional[torch.Tensor],
        max_context_len: Optional[int],
        context_lens: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
        use_cuda_graph: bool,
        kv_cache_dtype: str,
    ) -> None:
        # [prompt_batch_size + 1]
        # The length of prompts. If chunked prefill is enabled,
        # it is equivalent to the chunked size. 
        self.prompt_lens = prompt_lens
        # Number of prompt tokens. Include padding.
        self.num_prompt_tokens = num_prompt_tokens
        # Number of generation tokens. Include padding.
        self.num_generation_tokens = num_generation_tokens

        # The number of prefill requests.
        self.num_prompts = 0
        # The maximum sequence length from all requests.
        # None if there are only decoding requests.
        self.max_seq_len = None

        # The requests contain prefill requests.
        if self.prompt_lens is not None:
            self.num_prompts = len(prompt_lens)
            if len(prompt_lens) > 0:
                self.max_seq_len = torch.max(prompt_lens).item()

        # This tensor only contains prompts.
        # [prompt_batch_size + 1]
        # cumulative index of a prompt length.
        # E.g., [0, 3, 8] for prompt_lens [3, 5].
        self.start_loc = start_loc
        self.max_context_len = max_context_len
        # [batch_size]. A list of index that represents
        # the address in the KV cache. The value can be
        # decomposed into a block number and block offset.
        # Block number corresponds to the 0th dimension index
        # of the kv cache, and the offset corresponds to the
        # 1st dimension index of the kv cache.
        self.slot_mapping = slot_mapping
        # [batch_size]. A list containing the attention context
        # length of each request. 
        # NOTE: When it is prefill/decoding,
        # the definition is different. For prefill,
        # it means the the length of KV that are cached
        # excluding the new KVs. In decoding, this
        # includes a new KV.
        self.context_lens = context_lens
        # Location of KV inside non-contiguous KV cache (i.e.,
        # paged attention).
        # [batch_size, num_blocks].
        # Each value in num_blocks represent the block number
        # (0th dimension index in kv cache). And each block
        # can contain upto `block_size` tokens.
        # None if it is prefill request. However, if
        # chunked prefill is enabled, or there's a cached
        # prefix, it is provided even for prefill requests.
        self.block_tables = block_tables
        self.use_cuda_graph = use_cuda_graph
        self.kv_cache_dtype = kv_cache_dtype
        self.num_chunked_prefill = num_chunked_prefill

        # Set during the execution of the first attention op.
        # FIXME(woosuk): This is a hack.
        self.attn_bias = None
        # Number of valid tokens. It includes paddings.
        # See attention.py for precise definition.
        self.num_valid_tokens = slot_mapping.shape[0]

    def prefill_input_metadata(self) -> "InputMetadata":
        """Create a new InputMetadata that only contains
        metadata needed for prefill requests.
        """
        return InputMetadata(
            self.slot_mapping[:self.num_prompt_tokens],
            self.prompt_lens[:self.num_prompts],
            self.num_chunked_prefill,
            self.num_prompt_tokens,
            0,
            # start_loc only contains prompts.
            self.start_loc,
            None,
            self.context_lens[:self.num_prompts],
            self.block_tables[:self.num_prompts],
            False,
            self.kv_cache_dtype,
            self.flash_style,
        )

    def decode_input_metadata(self) -> "InputMetadata":
        """Create a new InputMetadata that only contains
        metadata needed for decoding requests.
        """
        return InputMetadata(
            self.slot_mapping[self.num_prompt_tokens:],
            None,
            0,
            0,
            self.num_generation_tokens,
            None,
            self.max_context_len,
            self.context_lens[self.num_prompts:],
            self.block_tables[self.num_prompts:],
            self.use_cuda_graph,
            self.kv_cache_dtype,
            self.flash_style,
        )

    def __repr__(self) -> str:
        return ("InputMetadata("
                f"max_context_len={self.max_context_len}, "
                f"num_generation_tokens={self.num_generation_tokens}, "
                f"num_prompt_tokens={self.num_prompt_tokens}, "
                f"slot_mapping={self.slot_mapping}, "
                f"context_lens={self.context_lens}, "
                f"block_tables={self.block_tables}, "
                f"use_cuda_graph={self.use_cuda_graph}, "
                f"kv_cache_dtype={self.kv_cache_dtype}) "
                f"num_valid_tokens={self.num_valid_tokens} "
                f"flash_style={self.flash_style}")
