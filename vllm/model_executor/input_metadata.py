from typing import List, Optional

import torch


class InputMetadata:
    """Metadata for input sequences. Used in PagedAttention.

    Args:
        prompt_lens: Lengths of prompts.
        slot_mapping: The address to write the new KV to of each token.
        max_context_len: The maximum context length.
        context_lens: the length of attention context for each sequence.
        block_tables: The block tables. (Seq id -> list of physical block)
    """

    def __init__(self,
                 prompt_lens: List[int],
                 slot_mapping: torch.Tensor,
                 context_lens: torch.Tensor,
                 max_context_len: int,
                 block_tables: torch.Tensor,
                 start_loc: Optional[torch.Tensor] = None,
                 sd_len_to_gen: Optional[int] = None,
                 sd_prompt_lens: Optional[List[int]] = None) -> None:
        self.prompt_lens = prompt_lens
        self.max_context_len = max_context_len
        self.slot_mapping = slot_mapping
        self.start_loc = start_loc
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.sd_len_to_gen = sd_len_to_gen
        self.sd_prompt_lens = sd_prompt_lens

        self.is_prompt = len(prompt_lens) > 0
        # Set during the execution of the first attention op.
        # FIXME(woosuk): This is a hack.
        self.attn_bias = None

    def __repr__(self) -> str:
        return ("InputMetadata("
                f"prompt_lens={self.prompt_lens}, "
                f"max_context_len={self.max_context_len}, "
                f"slot_mapping={self.slot_mapping}, "
                f"context_lens={self.context_lens}, "
                f"block_tables={self.block_tables})")
