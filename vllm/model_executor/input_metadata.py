from dataclasses import dataclass, fields
from typing import Optional, List, Any, Dict

import torch
from xformers.ops.fmha.attn_bias import AttentionBias


# SANG-TODO Refactor prompt_lens -> seqlens
@dataclass
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
    """
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor
    # The number of chunked prefill sequences in the batch.
    num_chunked_prefill: int
    # (batch_size,). The prompt length per sequence. None if it is a decoding.
    prompt_lens: Optional[List[int]]
    # prompt_lens stored as a tensor.
    prompt_lens_tensor: Optional[torch.Tensor]
    # The number of prompt tokens. Doesn't include padding.
    num_prompt_tokens: int
    # The number of generation tokens. Doesn't include padding.
    num_generation_tokens: int
    """
    Definition of context_len, subquery_len, and seqlen.
    |---------- N-1 iteration --------|
    |---------------- N iteration ---------------------|
    |- tokenA -|......................|-- newTokens ---|
    |---------- context_len ----------|
    |-------------------- seqlen ----------------------|
                                      |- subquery_len -|

    WARNING: context_len has different definition depending on if it is
    prefill vs decoding. When it is prefill, it doesn't include new
    tokens. When it is for decoding, it includes a new token.
    """

    # Maximum subquery length in the batch.
    max_subquery_len: Optional[int]
    # Maximum context length in the batch.
    max_context_len: Optional[int]
    # FIXME: It is for flash attn.
    # Maximum sequence length in the batch.
    max_seq_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,). The length of context (tokens stored in KV cache) per
    # sequence. WARNING: When it is a prefill request, it doesn't include new
    # tokens. When it is for decoding, it includes a new token.
    context_lens: Optional[torch.Tensor]
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]
    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    use_cuda_graph: bool
    kv_cache_dtype: str

    # Fields below are lazily initialized.
    _prefill_input_metadata: Optional["InputMetadata"] = None
    _decode_input_metadata: Optional["InputMetadata"] = None

    @property
    def num_prompts(self):
        if self.prompt_lens is None:
            return 0

        return len(self.prompt_lens)

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None
        # Cuda graph is only used for decoding now.
        if self.use_cuda_graph:
            assert self.num_prompt_tokens == 0

    def prefill_input_metadata(self) -> "InputMetadata":
        """Create a new InputMetadata that only contains
        metadata needed for prefill requests.
        """
        if self._prefill_input_metadata is None:
            if self.num_prompts > 0:
                self._prefill_input_metadata = InputMetadata(
                    slot_mapping=self.slot_mapping[:self.num_prompt_tokens],
                    num_chunked_prefill=self.num_chunked_prefill,
                    prompt_lens=self.prompt_lens[:self.num_prompts],
                    prompt_lens_tensor=self.prompt_lens_tensor[:self.
                                                               num_prompts],
                    num_prompt_tokens=self.num_prompt_tokens,
                    num_generation_tokens=0,
                    max_subquery_len=self.max_subquery_len,
                    max_context_len=None,
                    max_seq_len=self.max_seq_len,
                    subquery_start_loc=self.subquery_start_loc[:self.
                                                               num_prompts],
                    seq_start_loc=self.seq_start_loc[:self.num_prompt_tokens],
                    context_lens=self.context_lens[:self.num_prompts],
                    block_tables=self.block_tables[:self.num_prompts],
                    use_cuda_graph=False,
                    kv_cache_dtype=self.kv_cache_dtype,
                )

        return self._prefill_input_metadata

    def decode_input_metadata(self) -> "InputMetadata":
        """Create a new InputMetadata that only contains
        metadata needed for decoding requests.
        """
        if self._decode_input_metadata is None:
            if self.num_generation_tokens > 0:
                self._decode_input_metadata = InputMetadata(
                    slot_mapping=self.slot_mapping[self.num_prompt_tokens:],
                    num_chunked_prefill=0,
                    prompt_lens=None,
                    prompt_lens_tensor=None,
                    num_prompt_tokens=0,
                    num_generation_tokens=self.num_generation_tokens,
                    max_subquery_len=None,
                    max_context_len=self.max_context_len,
                    max_seq_len=None,
                    subquery_start_loc=None,
                    seq_start_loc=None,
                    context_lens=self.context_lens[self.num_prompts:],
                    block_tables=self.block_tables[self.num_prompts:],
                    use_cuda_graph=self.use_cuda_graph,
                    kv_cache_dtype=self.kv_cache_dtype,
                )
        return self._decode_input_metadata

    def asdict_zerocopy(self) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }
