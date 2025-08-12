from unittest.mock import MagicMock

import pytest

from vllm.sequence import SequenceGroupMetadata
from vllm.spec_decode.util import get_all_seq_ids, split_batch_by_proposal_len


def test_get_all_seq_ids():
    """Verify get_all_seq_ids extracts all seq ids.
    """
    expected_seq_ids = list(range(10)) + list(range(100, 110))

    seq_group_metadata_list = [
        SequenceGroupMetadata(
            request_id=str(seq_id),
            is_prompt=True,
            seq_data={
                seq_id: MagicMock(),
            },
            sampling_params=MagicMock(),
            block_tables={
                seq_id: MagicMock(),
            },
            lora_request=None,
        ) for seq_id in expected_seq_ids
    ]

    actual_seq_ids = get_all_seq_ids(seq_group_metadata_list)
    assert actual_seq_ids == expected_seq_ids


@pytest.fixture
def fake_sequence_group_metadata():
    seq_ids = list(range(3))
    return [
        SequenceGroupMetadata(
            request_id=str(i),
            is_prompt=True,
            seq_data={
                i: MagicMock(),
            },
            sampling_params=MagicMock(),
            block_tables={
                i: MagicMock(),
            },
            lora_request=None,
        ) for i in seq_ids
    ]


def test_filter_zero_length_proposals(fake_sequence_group_metadata):
    proposal_lens = [0, 1, 0]
    filtered_groups, indices = split_batch_by_proposal_len(
        fake_sequence_group_metadata,
        proposal_lens,
        select_proposal_len_zero=True)

    expected_groups = [
        fake_sequence_group_metadata[0], fake_sequence_group_metadata[2]
    ]
    expected_indices = [0, 2]

    assert filtered_groups == expected_groups
    assert indices == expected_indices


def test_filter_non_zero_length_proposals(fake_sequence_group_metadata):
    proposal_lens = [0, 1, 2]
    filtered_groups, indices = split_batch_by_proposal_len(
        fake_sequence_group_metadata,
        proposal_lens,
        select_proposal_len_zero=False)

    expected_groups = [
        fake_sequence_group_metadata[1], fake_sequence_group_metadata[2]
    ]
    expected_indices = [1, 2]

    assert filtered_groups == expected_groups
    assert indices == expected_indices


def test_empty_inputs():
    filtered_groups, indices = split_batch_by_proposal_len(
        [], [], select_proposal_len_zero=True)

    assert filtered_groups == []
    assert indices == []


def test_all_zero_with_non_zero_filter(fake_sequence_group_metadata):
    proposal_lens = [0, 0, 0]
    filtered_groups, indices = split_batch_by_proposal_len(
        fake_sequence_group_metadata,
        proposal_lens,
        select_proposal_len_zero=False)

    assert filtered_groups == []
    assert indices == []


def test_all_non_zero_with_zero_filter(fake_sequence_group_metadata):
    proposal_lens = [1, 1, 1]
    filtered_groups, indices = split_batch_by_proposal_len(
        fake_sequence_group_metadata,
        proposal_lens,
        select_proposal_len_zero=True)

    assert filtered_groups == []
    assert indices == []
