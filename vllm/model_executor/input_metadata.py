from typing import Optional

import torch


class InputMetadata:
    """Metadata for input sequences. Used in PagedAttention.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replays. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.

    Args:
        prompt_lens: Lengths of prompts.
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
        is_prompt: bool,
        slot_mapping: torch.Tensor,
        prompt_lens: Optional[torch.Tensor],
        num_chunked_prefill: int,
        num_prompt_tokens: int,
        num_generation_tokens: int,
        max_seq_len: Optional[int],
        start_loc: Optional[torch.Tensor],
        max_context_len: Optional[int],
        context_lens: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
        use_cuda_graph: bool,
        kv_cache_dtype: str,
        self.flash_style: bool,
    ) -> None:
        self.is_prompt = is_prompt
        self.prompt_lens = prompt_lens
        self.num_chunked_prefill = num_chunked_prefill
        self.num_prompt_tokens = num_prompt_tokens
        self.num_generation_tokens = num_generation_tokens
        self.max_seq_len = max_seq_len
        self.start_loc = start_loc
        self.max_context_len = max_context_len
        self.slot_mapping = slot_mapping
        # Index: The batched sequence's index.
        # Value: The length of attention context.
        # NOTE(sang): When it is prefill/decoding,
        # the definition is different. For prefill,
        # it means the the length of KV that are cached
        # excluding the new KVs. In decoding, this
        # includes a new KV.
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.use_cuda_graph = use_cuda_graph
        self.kv_cache_dtype = kv_cache_dtype
        self.flash_style = flash_style

        # Set during the execution of the first attention op.
        # FIXME(woosuk): This is a hack.
        self.attn_bias = None
        # Number of valid tokens. It includes paddings.
        # See attention.py for precise definition.
        self.num_valid_tokens = slot_mapping.shape[0]

        # SANG-TODO
        # # Prompt related metadata
        # # This value might include padding if CudaGraph is enabled.
        # self.num_prompts = len(prompt_lens)
        # # This value is the source of truth.
        # self.num_prompts_tensor = torch.cuda.IntTensor([self.num_prompts])
        # # This value might include padding if CudaGraph is enabled.
        # self.num_prompt_tokens = num_prompt_tokens
        # self.prompt_lens_tensor = torch.cuda.IntTensor(self.prompt_lens)
        # self.max_prompt_len = max(prompt_lens) if prompt_lens else 0

        # # Cumulative prompt lengths for each prompt in the input
        # # tensor.
        # self.cum_prompt_query_lens = torch.zeros(
        #     self.num_prompts + 1,
        #     device=self.prompt_lens_tensor.device,
        #     dtype=torch.int32)
        # # Cumulative context lengths.
        # self.cum_prompt_context_lens = torch.zeros(
        #     self.num_prompts + 1,
        #     device=self.prompt_lens_tensor.device,
        #     dtype=torch.int32)

        # torch.cumsum(self.prompt_lens_tensor,
        #              dim=0,
        #              dtype=self.cum_prompt_query_lens.dtype,
        #              out=self.cum_prompt_query_lens[1:])
        # torch.cumsum(self.context_lens[:self.num_prompts],
        #              dim=0,
        #              dtype=self.cum_prompt_context_lens.dtype,
        #              out=self.cum_prompt_context_lens[1:])

        # # TODO: this will be different once we support chunked prefills.
        # self.cum_prompt_context_lens = self.cum_prompt_query_lens
        # self.max_context_len = max_context_len

        # # Generation related metadata
        # # This value might include padding if CudaGraph is enabled.
        # self.num_generation_tokens = num_generation_tokens
        # # This is the source of truth for the number of generation tokens.
        # self.num_generation_tokens_tensor = torch.tensor(
        #     [self.num_generation_tokens],
        #     dtype=torch.int32 if self.flash_style else torch.long,
        #     device='cuda')

    def __repr__(self) -> str:
        return ("InputMetadata("
                f"is_prompt={self.is_prompt}, "
                f"max_context_len={self.max_context_len}, "
                f"num_generation_tokens={self.num_generation_tokens}, "
                f"num_prompt_tokens={self.num_prompt_tokens}, "
                f"slot_mapping={self.slot_mapping}, "
                f"context_lens={self.context_lens}, "
                f"block_tables={self.block_tables}, "
                f"use_cuda_graph={self.use_cuda_graph}, "
                f"flash_style={self.flash_style} "
                f"kv_cache_dtype={self.kv_cache_dtype}) "
                f"num_valid_tokens={self.num_valid_tokens}")
