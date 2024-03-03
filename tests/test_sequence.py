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


# TODO(sang): Upstream more tests.
def test_sequence_data_prefill():
    seq_data = SequenceData(prompt_token_ids=[1, 2, 3, 4])
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
    seq_data.append_token_id(1, logprob=0.0)
