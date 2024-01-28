from math import ceil

from vllm.sequence import Sequence
from vllm.core.block_space_manager.base_block_space_manager import BaseBlockSpaceManager


class OrcaBlockSpaceManager(BaseBlockSpaceManager):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.watermark_blocks = 0
        self.request_num_blocks = ceil(self.max_model_len / self.block_size)

    def get_num_initial_blocks(self, seq: Sequence) -> int:
        return self.request_num_blocks