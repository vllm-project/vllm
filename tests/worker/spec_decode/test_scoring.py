import torch
import random
import pytest

from vllm.worker.spec_decode.scoring import BatchExpansionTop1Scorer
from vllm.worker.spec_decode.util import SpeculativeProposals
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import SequenceGroupMetadata

from .utils import mock_worker, create_batch, ExecuteModelData, create_seq_group_metadata_from_prompts, create_sampler_output_list

from unittest.mock import MagicMock


@pytest.mark.parametrize('num_target_seq_ids', [100])
def test_create_target_seq_id_iterator(num_target_seq_ids: int):
    """Assert all target seq ids are greater than input seq ids.
    """
    scorer = BatchExpansionTop1Scorer(mock_worker(), 'cuda:0', 32_000)

    all_seq_ids = [
        [1, 3, 5, 7],
        list(range(100)) + [0],
        [100],
    ]

    for seq_ids in all_seq_ids:
        max_seq_id = max(seq_ids)
        iterator = scorer._create_target_seq_id_iterator(seq_ids)  # pylint: disable=protected-access
        for _ in range(num_target_seq_ids):
            assert next(iterator) > max_seq_id


@pytest.mark.parametrize('k', [1, 2, 6])
def test_get_token_ids_to_score(k: int):
    """Verify BatchExpansionTop1Scorer correctly determines which token ids need
    to be scored.
    """
    proposal_token_ids = torch.tensor(
        list(range(k)),
        dtype=torch.int64,
        device='cuda',
    )

    expected_output = [
        [],
    ]
    for i in range(proposal_token_ids.shape[0]):
        expected_output.append(proposal_token_ids[:i + 1].tolist())

    scorer = BatchExpansionTop1Scorer(mock_worker(), 'cuda:0', 32_000)
    actual_output = scorer._get_token_ids_to_score(proposal_token_ids)  # pylint: disable=protected-access

    actual_output = [
        x.tolist() if isinstance(x, torch.Tensor) else x for x in actual_output
    ]

    assert actual_output == expected_output


@pytest.mark.parametrize('k', [1, 2, 6])
def test_create_single_target_seq_group_metadata(k: int):
    """Verify correct creation of a target seq group metadata.
    """

    prompt_tokens = [1, 2, 3]
    prev_output_tokens = [4, 5, 6]

    token_ids = list(range(k))

    num_tokens_processed = len(prompt_tokens) + len(prev_output_tokens) - 1

    final_seq_len = len(prompt_tokens) + len(prev_output_tokens) + len(
        token_ids)

    block_size = 32
    input_seq_group_metadata = create_seq_group_metadata_from_prompts(
        [prompt_tokens], 2048 // block_size, block_size, [final_seq_len],
        [prev_output_tokens], [num_tokens_processed])[0]

    input_seq_id = list(input_seq_group_metadata.seq_data.keys())[0]
    target_seq_id = 100

    scorer = BatchExpansionTop1Scorer(mock_worker(), 'cuda:0', 32_000)
    output = scorer._create_single_target_seq_group_metadata(  # pylint: disable=protected-access
        input_seq_group_metadata,
        input_seq_id,
        target_seq_id,
        token_ids,
    )

    assert output.request_id == input_seq_group_metadata.request_id
    assert len(output.seq_data) == 1
    assert output.seq_data[target_seq_id].get_prompt_token_ids(
    ) == prompt_tokens
    assert output.seq_data[target_seq_id].get_output_token_ids(
    ) == prev_output_tokens + token_ids

    assert len(output.block_tables) == 1
    assert output.block_tables[
        target_seq_id] == input_seq_group_metadata.block_tables[input_seq_id]
