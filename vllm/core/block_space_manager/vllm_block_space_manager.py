from vllm.sequence import Sequence
from vllm.core.block_space_manager.base_block_space_manager import BaseBlockSpaceManager


class VLLMBlockSpaceManager(BaseBlockSpaceManager):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_num_initial_blocks(self, seq: Sequence) -> int:
        return len(seq.logical_token_blocks)
