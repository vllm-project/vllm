from contextlib import contextmanager
from itertools import chain
from typing import List, Tuple

import torch

from vllm.sequence import SamplerOutput, SequenceGroupMetadata

SeqId = int


def get_all_seq_ids(
        seq_group_metadata_list: List[SequenceGroupMetadata]) -> List[SeqId]:
    """Given a list of SequenceGroupMetadata, create a list of all
    sequence ids.
    """
    return list(
        chain.from_iterable([
            seq_group_metadata.seq_data.keys()
            for seq_group_metadata in seq_group_metadata_list
        ]))


def split_batch_by_proposal_len(
    seq_group_metadata_list: List[SequenceGroupMetadata],
    proposal_lens: List[int], select_proposal_len_zero: bool
) -> Tuple[List[SequenceGroupMetadata], List[int]]:
    """Utility function that splits a batch based on whether the proposal len is
    zero or not. We should remove this once vLLM supports per-sequence proposal
    lens in a batch.
    """

    if select_proposal_len_zero:
        predicate = lambda proposal_len: proposal_len == 0
    else:
        predicate = lambda proposal_len: proposal_len != 0

    indices = [
        i for i, (_, proposal_len
                  ) in enumerate(zip(seq_group_metadata_list, proposal_lens))
        if predicate(proposal_len)
    ]
    seq_groups = [
        seq_group for seq_group, proposal_len in zip(
            seq_group_metadata_list, proposal_lens) if predicate(proposal_len)
    ]

    return seq_groups, indices


def sampler_output_to_torch(
    sampler_output_list: List[SamplerOutput],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Utility function which converts a list of SamplerOutput to tensors.

        Returns:
            sampled_token_ids: torch.Tensor
                shape: [batch_size, len(sampler_output_list)]

            sampled_token_probs: torch.Tensor
                shape: [batch_size, len(sampler_output_list), vocab_size]
        """

    # shape: [batch_size, num_sampler_output, vocab_size]
    sampled_token_probs = torch.stack(
        [
            sampler_output.sampled_token_probs
            for sampler_output in sampler_output_list
        ],
        dim=0,
    ).transpose(0, 1)

    # shape: [batch_size, num_sampler_output]
    sampled_token_ids = torch.stack(
        [
            sampler_output.sampled_token_ids.flatten()
            for sampler_output in sampler_output_list
        ],
        dim=0,
    ).transpose(0, 1)

    return sampled_token_ids, sampled_token_probs


def maybe_mock_device_tensors(sampler_output: SamplerOutput, batch_size: int,
                              vocab_size: int, device: str) -> None:
    """Helper method which mocks out the GPU tensors in SamplerOutput with dummy
    values. This will be removed in PR 7/9.
    https://docs.google.com/document/d/1rE4pr3IdspRw97XbImY4fS9IWYuJJ3HGtL7AdIKGrw8/edit#heading=h.qijw1sdidrer
    """
    values = [
        sampler_output.sampled_token_probs, sampler_output.sampled_token_ids
    ]
    assert all(v is None for v in values) or not any(v is None for v in values)
    if not any(v is None for v in values):
        # Do nothing if the tensors are already created (usually in unit tests).
        return

    # Softmax to ensure valid probs.
    sampler_output.sampled_token_probs = torch.nn.functional.softmax(
        torch.rand(batch_size, vocab_size, dtype=torch.float32, device=device),
        dim=-1)

    sampler_output.sampled_token_ids = torch.randint(low=10,
                                                     high=100,
                                                     size=(batch_size, ),
                                                     dtype=torch.long,
                                                     device=device)


@contextmanager
def nvtx_range(msg, *args, **kwargs):
    """ 
    Context manager / decorator that pushes an NVTX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    If running with cuda graphs, you must enable nsys cuda graph profiling.

    Arguments:
        msg (string): message to associate with the range
    """
    torch.cuda.nvtx.range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
