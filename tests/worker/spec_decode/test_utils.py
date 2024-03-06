import torch
import random
import pytest

from vllm.worker.spec_decode.util import SpeculativeProposals, get_all_seq_ids
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import SequenceGroupMetadata

from .utils import mock_worker, create_batch, ExecuteModelData, create_seq_group_metadata_from_prompts, create_sampler_output_list

from unittest.mock import MagicMock


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
