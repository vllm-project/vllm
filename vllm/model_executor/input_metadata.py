from typing import Optional

import torch


class InputMetadata:

    def __init__(
        self,
        is_prompt: bool,
        slot_mapping: torch.Tensor,
        context_lens: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
    ) -> None:
        self.is_prompt = is_prompt
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables
