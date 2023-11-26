from typing import List, Optional

import torch


class InputMetadata:

    def __init__(
        self,
        prompt_lens: List[int],
        slot_mapping: torch.Tensor,
        context_lens: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
    ) -> None:
        self.prompt_lens = prompt_lens
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables

        self.is_prompt = len(prompt_lens) > 0
