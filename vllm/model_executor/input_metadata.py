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

        # Set during the execution of the first attention op.
        # FIXME(woosuk): This is a hack.
        self.attn_bias = None

    def __repr__(self) -> str:
        return (f"InputMetadata("
                f"prompt_lens={self.prompt_lens}, "
                f"slot_mapping={self.slot_mapping}, "
                f"context_lens={self.context_lens}, "
                f"block_tables={self.block_tables})")
