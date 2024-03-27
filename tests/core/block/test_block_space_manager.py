import pytest

from vllm.core.block_manager_v2 import BlockSpaceManagerV2
from vllm.core.interfaces import AllocStatus

from ..utils import create_seq_group


@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_gpu_blocks", [8, 40, 80])
@pytest.mark.parametrize("num_seqs_per_group", [1, 4])
@pytest.mark.parametrize("watermark", [0.0, 0.5])
def test_can_allocate_seq_group(block_size: int, num_seqs_per_group: int,
                                num_gpu_blocks: int, watermark: float):
    block_manager = BlockSpaceManagerV2(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=1024,
        watermark=watermark,
    )
    num_watermark_blocks = int(watermark * num_gpu_blocks)

    num_output_blocks_per_seq = 1

    # NOTE: This should be num_output_blocks_per_seq * num_seqs_per_group, but
    # the current implementation assumes all seqs are new prompts / don't have
    # different output lens.
    num_output_blocks = num_output_blocks_per_seq

    for num_prompt_blocks in range(1, num_gpu_blocks - num_output_blocks):
        seq_group = create_seq_group(
            seq_prompt_lens=block_size * num_prompt_blocks,
            seq_output_lens=[
                block_size * num_output_blocks_per_seq
                for _ in range(num_seqs_per_group)
            ],
        )

        assert num_prompt_blocks + num_output_blocks <= num_gpu_blocks

        can_allocate_result = block_manager.can_allocate(seq_group)

        num_required_blocks = num_prompt_blocks + num_output_blocks

        if num_gpu_blocks - num_required_blocks < num_watermark_blocks:
            assert can_allocate_result == AllocStatus.NEVER
        elif num_gpu_blocks >= num_required_blocks:
            assert can_allocate_result == AllocStatus.OK
        else:
            assert can_allocate_result == AllocStatus.LATER
