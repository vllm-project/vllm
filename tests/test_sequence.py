
import pytest
from vllm.sequence import SequenceData, Sequence


@pytest.fixture(name="sequence")
def create_sequence(seq_len: int, block_size: int) -> Sequence:
    return Sequence(
        seq_id=0,
        prompt="",
        prompt_token_ids=list(range(seq_len)),
        block_size=block_size,
    )


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_empty_slots", list(range(8)))
@pytest.mark.parametrize("seq_len", [0, 1, 100])
def test_ensure_num_empty_slots(block_size: int, seq_len: int,
                                num_empty_slots: int, sequence: Sequence):
    """Verify ensure_num_empty_slots correctly ensures empty slots.
    """
    sequence.ensure_num_empty_slots(num_empty_slots)

    num_total_slots = block_size * len(sequence.logical_token_blocks)
    measured_num_empty_slots = sum(block.get_num_empty_slots()
                                   for block in sequence.logical_token_blocks)
    num_full_slots = num_total_slots - measured_num_empty_slots

    assert measured_num_empty_slots >= num_empty_slots
    assert num_full_slots == seq_len


@pytest.fixture(name="sequence_with_extra_blocks")
def add_blocks_to_sequence(sequence: Sequence,
                           num_extra_blocks: int) -> Sequence:
    for _ in range(num_extra_blocks):
        sequence._append_logical_block()  # pylint: disable=protected-access
    return sequence


@pytest.mark.parametrize("num_tokens_to_append", [1, 10])
@pytest.mark.parametrize("seq_len", [0, 1, 100])
@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_extra_blocks", [0, 1, 100])
def test_append_tokens_correct_placement_in_blocks(
        num_tokens_to_append: int, sequence_with_extra_blocks: Sequence,
        block_size: int, seq_len: int):
    """Verify new tokens are appended at the end of the sequence, instead of the
    last block. This enables preallocated empty slots, which requires empty
    blocks after the sequence.
    """
    token_ids = list(range(num_tokens_to_append))
    logprobs = [{token_id: 0.0} for token_id in token_ids]
    seq_len_before_append = seq_len
    seq_len_after_append = seq_len_before_append + num_tokens_to_append

    sequence_with_extra_blocks.append_token_ids(token_ids, logprobs)

    # Assert number of full slots equal to total sequence length.
    assert sum(block_size - block.get_num_empty_slots()
               for block in sequence_with_extra_blocks.logical_token_blocks
               ) == seq_len_after_append

    # Assert each appended token is immediately after the original sequence.
    for i, token_id in enumerate(token_ids):
        index = seq_len_before_append + i
        block_token_ids = sequence_with_extra_blocks.logical_token_blocks[
            index // block_size].get_token_ids()
        assert block_token_ids[index % block_size] == token_id


@pytest.mark.parametrize("generation_or_prefill", ["generation", "prefill"])
@pytest.mark.parametrize("num_output_tokens", [0, 1, 10])
@pytest.mark.parametrize("num_prompt_tokens", [5, 50])
def test_get_unprocessed_tokens(generation_or_prefill: str,
                                num_output_tokens: int,
                                num_prompt_tokens: int):
    """Verify sequence data correctly tracks the number of processed tokens.
    """
    is_generation = generation_or_prefill == "generation"

    if is_generation:
        generated_token_id = 1337

    prompt_token_ids = list(range(num_prompt_tokens))
    output_token_ids = list(range(num_output_tokens))
    data = SequenceData(
        token_ids=prompt_token_ids[:] + output_token_ids[:],
        num_prompt_tokens=len(prompt_token_ids[:]),
    )

    if is_generation:
        data.append_token_ids([generated_token_id], logprobs=[0.0])

    unprocessed_token_ids = data.get_unprocessed_token_ids()
    unprocessed_token_positions = data.get_unprocessed_token_positions()

    if is_generation:
        assert unprocessed_token_ids == [generated_token_id]
        assert unprocessed_token_positions == [
            num_prompt_tokens + num_output_tokens
        ]
    else:
        assert unprocessed_token_ids == prompt_token_ids + output_token_ids
        assert unprocessed_token_positions == list(
            range(num_prompt_tokens + num_output_tokens))

    # Reset processed tokens. Everything should behave like a prompt run now.
    data.reset_processed_tokens()

    unprocessed_token_ids = data.get_unprocessed_token_ids()
    unprocessed_token_positions = data.get_unprocessed_token_positions()

    if is_generation:
        assert unprocessed_token_ids == (prompt_token_ids + output_token_ids +
                                         [generated_token_id])
        assert unprocessed_token_positions == list(
            range(num_prompt_tokens + num_output_tokens + 1))
    if not is_generation:
        assert unprocessed_token_ids == prompt_token_ids + output_token_ids
        assert unprocessed_token_positions == list(
            range(num_prompt_tokens + num_output_tokens))


def test_sequence_data_prefill():
    seq_data = SequenceData(prompt_token_ids=[1, 2, 3, 4], output_token_ids=[])
    assert seq_data.get_prefill_range() == (0, 0)
    assert seq_data.get_num_unprefilled() == 4

    # advance by 2
    assert seq_data.advance_prefill_range(2) == 2
    assert seq_data.get_num_unprefilled() == 2
    assert seq_data.get_prefill_range() == (0, 2)

    # advance range by 3 even though there are only 2 unprefilled tokens
    assert seq_data.advance_prefill_range(3) == 2
    assert seq_data.get_num_unprefilled() == 0
    assert seq_data.get_prefill_range() == (2, 4)

    # following advances should not change anything
    assert seq_data.advance_prefill_range(2) == 0
    assert seq_data.get_num_unprefilled() == 0
    assert seq_data.get_prefill_range() == (4, 4)

    # append tokens and reset, simulating recompute
    seq_data.append_token_ids([1], logprobs=[0.0])
    seq_data.reset_processed_tokens()

    # after reset, the prefill range should be reset to 0
    # but the num_unprefilled should include.
    # output tokens
    assert seq_data.get_prefill_range() == (0, 0)
    assert seq_data.get_num_unprefilled() == 5

    # advance by 2
    assert seq_data.advance_prefill_range(2) == 2
    assert seq_data.get_num_unprefilled() == 3
    assert seq_data.get_prefill_range() == (0, 2)

    # advance range by 3 even though there are only 2 unprefilled tokens
    assert seq_data.advance_prefill_range(3) == 3
    assert seq_data.get_num_unprefilled() == 0
    assert seq_data.get_prefill_range() == (2, 5)
