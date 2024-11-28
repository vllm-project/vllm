import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                           PromptLogprobs, SequenceGroupMetadata,
                           SequenceOutput)

SeqId = int


def get_all_num_logprobs(
        seq_group_metadata_list: List[SequenceGroupMetadata]) -> List[int]:
    """Given a list of SequenceGroupMetadata, create a list of all num_logprobs.

    If the sampling params do not call for any logprobs, return 0 for that
    sequence.
    """

    all_num_logprobs: List[int] = []
    for seq_group_metadata in seq_group_metadata_list:
        num_logprobs = seq_group_metadata.sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = 0
        all_num_logprobs.append(num_logprobs)

    return all_num_logprobs


def get_sampled_token_logprobs(
        # shape [num_steps, batch_size, vocab_size]
        logprob_tensor: torch.Tensor,
        sampled_token_ids: torch.Tensor,  # shape [num_steps, batch_size]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the logprobs for the sampled tokens. Returns the ranks and logprobs.
    """
    num_steps, batch_size, vocab_size = logprob_tensor.shape

    selected_logprobs = logprob_tensor[torch.arange(num_steps).unsqueeze(1),
                                       torch.arange(batch_size),
                                       sampled_token_ids, ]
    expanded_selected_logprobs = selected_logprobs.unsqueeze(-1).expand(
        -1, -1, vocab_size)
    sampled_token_ids_ranks = (logprob_tensor >
                               expanded_selected_logprobs).sum(-1).add_(1)

    return sampled_token_ids_ranks, selected_logprobs


def create_logprobs_output(
    token_id: int,
    token_id_logprob_rank: int,
    token_id_logprob: float,
    topk_token_ids: List[Optional[int]],
    topk_logprobs: List[Optional[float]],
) -> Dict[int, Logprob]:
    """Create a Logprob Dict for a token given the sampling results.

    Args:
        token_id (int): The sampled token for the sequence.
        token_id_logprob_rank (int): The logprob rank of the sampled token.
        token_id_logprob (float): The logprob value of the sampled token.
        topk_token_ids (List[Optional[int]]): The list of top-k token ids.
        topk_logprobs (List[Optional[float]]): The list of top-k logprobs.
    """
    # vLLM logprobs always include the sampled token. In addition, the user may
    # request topk-logprobs (where top-k varies per user up to max_logprobs).
    logprobs: Dict[int, Logprob] = {
        token_id: Logprob(
            logprob=token_id_logprob,
            rank=token_id_logprob_rank,
        ),
    }
    logprobs.update({
        topk_token_id: Logprob(
            logprob=topk_logprob if topk_logprob is not None else 0.0,
            rank=topk_index + 1,
        )
        for topk_index, (topk_token_id, topk_logprob) \
            in enumerate(zip(topk_token_ids, topk_logprobs)) \
        if topk_token_id is not None
    })

    return logprobs


def create_sequence_group_output(
    token_id: int,
    token_id_logprob_rank: int,
    token_id_logprob: float,
    seq_id: SeqId,
    topk_token_ids: List[Optional[int]],
    topk_logprobs: List[Optional[float]],
    prompt_logprobs: Optional[PromptLogprobs] = None,
) -> CompletionSequenceGroupOutput:
    """Create a SequenceGroupOutput given the sampling results.

    Args:
        token_id (int): The sampled token for the sequence.
        token_id_logprob_rank (int): The logprob rank of the sampled token.
        token_id_logprob (float): The logprob value of the sampled token.
        seq_id (int): The sequence id.
        topk_token_ids (List[Optional[int]]): The list of top-k token ids.
        topk_logprobs (List[Optional[float]]): The list of top-k logprobs.
    """

    logprobs = create_logprobs_output(
        token_id,
        token_id_logprob_rank,
        token_id_logprob,
        topk_token_ids,
        topk_logprobs,
    )

    return CompletionSequenceGroupOutput(
        samples=[
            SequenceOutput(parent_seq_id=seq_id,
                           output_token=token_id,
                           logprobs=logprobs)
        ],
        prompt_logprobs=prompt_logprobs,
    )


def split_batch_by_proposal_len(
    seq_group_metadata_list: List[SequenceGroupMetadata],
    proposal_lens: List[int],
) -> Tuple[Tuple[List[SequenceGroupMetadata], List[int]], Tuple[
        List[SequenceGroupMetadata], List[int]]]:
    """Utility function that splits a batch based on whether the proposal len is
    zero or not. We should remove this once vLLM supports per-sequence proposal
    lens in a batch.
    """

    nonzero_lists: Tuple[List[SequenceGroupMetadata], List[int]] = ([], [])
    zero_lists: Tuple[List[SequenceGroupMetadata], List[int]] = ([], [])
    for i, (seq_group, proposal_len) in enumerate(
            zip(seq_group_metadata_list, proposal_lens)):
        seq_groups, indices = nonzero_lists if proposal_len else zero_lists
        seq_groups.append(seq_group)
        indices.append(i)
    return nonzero_lists, zero_lists


def sampler_output_to_torch(
    sampler_output_list: Sequence[SamplerOutput], sampler_transposed: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Utility function which converts a list of SamplerOutput to tensors.

        sampler_transposed here is used as the indicator for whether
        we need do additional tensor transpose logic here.

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
    )

    # shape: [batch_size, num_sampler_output, vocab_size]
    sampled_token_logprobs = torch.stack(
        [sampler_output.logprobs for sampler_output in sampler_output_list],
        dim=0,
    )

    # shape: [batch_size, num_sampler_output]
    sampled_token_ids = torch.stack(
        [
            sampler_output.sampled_token_ids.flatten()
            for sampler_output in sampler_output_list
        ],
        dim=0,
    )

    if sampler_transposed:
        sampled_token_probs = sampled_token_probs.transpose(0, 1)
        sampled_token_logprobs = sampled_token_logprobs.transpose(0, 1)
        sampled_token_ids = sampled_token_ids.transpose(0, 1)

    if sampler_output_list[0].hidden_states is not None:
        # shape: [batch_size, num_sampler_output, hidden_dim]
        sampled_hidden_states = torch.stack(
            [
                sampler_output.hidden_states
                for sampler_output in sampler_output_list
            ],
            dim=0,
        )

        if sampler_transposed:
            sampled_hidden_states = sampled_hidden_states.transpose(0, 1)
    else:
        sampled_hidden_states = None

    return (sampled_token_ids, sampled_token_probs, sampled_token_logprobs,
            sampled_hidden_states)


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
    if current_platform.is_cuda_alike():
        torch.cuda.nvtx.range_push(msg.format(*args, **kwargs))
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


class Timer:
    """Basic timer context manager for measuring CPU time.
    """

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time_s = self.end_time - self.start_time
        self.elapsed_time_ms = self.elapsed_time_s * 1000
