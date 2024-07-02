from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.layers.typical_acceptance_sampler import (
    TypicalAcceptanceSampler)
from vllm.sequence import SequenceGroupMetadata, get_all_seq_ids
from vllm.spec_decode.util import split_batch_by_proposal_len


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


def mock_spec_decode_sampler(acceptance_sampler_method):
    """
    Returns either a RejectionSampler or TypicalAcceptanceSampler
    object depending on whether acceptance_sampler_method is 
    'rejection_sampler' or 'typical_acceptance_sampler' respectively.
    """
    if acceptance_sampler_method == "rejection_sampler":
        sampler = MagicMock(spec=RejectionSampler)
        sampler.token_id_dtype = torch.int64
        return sampler
    elif acceptance_sampler_method == "typical_acceptance_sampler":
        sampler = MagicMock(spec=TypicalAcceptanceSampler)
        sampler.token_id_dtype = torch.int64
        return sampler
    else:
        raise ValueError(f"Invalid sampler name {acceptance_sampler_method}")
