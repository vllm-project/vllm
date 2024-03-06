import torch
from typing import List, Tuple
from dataclasses import dataclass
from vllm.sequence import SequenceGroupMetadata, SamplerOutput
from contextlib import contextmanager
from itertools import chain

SeqId = int


def get_all_seq_ids(seq_group_metadata_list: List[SequenceGroupMetadata]) -> List[SeqId]:
    """Given a list of SequenceGroupMetadata, create a list of all
    sequence ids.
    """
    return list(
        chain.from_iterable([
            seq_group_metadata.seq_data.keys()
            for seq_group_metadata in seq_group_metadata_list
        ]))

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
        [sampler_output.sampled_token_probs for sampler_output in sampler_output_list],
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
