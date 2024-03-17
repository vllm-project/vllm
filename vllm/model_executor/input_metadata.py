from typing import Any, Dict, Optional

import torch


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

    def __init__(
        self,
        is_prompt: bool,
        slot_mapping: torch.Tensor,
        prompt_lens: Optional[torch.Tensor],
        max_seq_len: Optional[int],
        start_loc: Optional[torch.Tensor],
        max_context_len: Optional[int],
        context_lens: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
        use_cuda_graph: bool,
        kv_cache_dtype: str,
    ) -> None:
        self.is_prompt = is_prompt
        self.prompt_lens = prompt_lens
        self.max_seq_len = max_seq_len
        self.start_loc = start_loc
        self.max_context_len = max_context_len
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.use_cuda_graph = use_cuda_graph
        self.kv_cache_dtype = kv_cache_dtype

        # Set during the execution of the first attention op.
        # FIXME(woosuk): This is a hack.
        self.attn_bias = None

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> "InputMetadata":
        return cls(
            is_prompt=dict["is_prompt"],
            slot_mapping=dict["slot_mapping"],
            prompt_lens=dict["prompt_lens"],
            max_seq_len=dict["max_seq_len"],
            start_loc=dict["start_loc"],
            max_context_len=dict["max_context_len"],
            context_lens=dict["context_lens"],
            block_tables=dict["block_tables"],
            use_cuda_graph=dict["use_cuda_graph"],
            kv_cache_dtype=dict["kv_cache_dtype"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_prompt": self.is_prompt,
            "slot_mapping": self.slot_mapping,
            "prompt_lens": self.prompt_lens,
            "max_seq_len": self.max_seq_len,
            "start_loc": self.start_loc,
            "max_context_len": self.max_context_len,
            "context_lens": self.context_lens,
            "block_tables": self.block_tables,
            "use_cuda_graph": self.use_cuda_graph,
            "kv_cache_dtype": self.kv_cache_dtype,
        }

    def __repr__(self) -> str:
        return ("InputMetadata("
                f"is_prompt={self.is_prompt}, "
                f"max_context_len={self.max_context_len}, "
                f"slot_mapping={self.slot_mapping}, "
                f"context_lens={self.context_lens}, "
                f"block_tables={self.block_tables}, "
                f"use_cuda_graph={self.use_cuda_graph}, "
                f"kv_cache_dtype={self.kv_cache_dtype})")
