"""A layer that samples the next tokens from the model's outputs."""
import itertools
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.model_executor.layers.ops.sample import sample as sample_triton
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
                                                   SamplingTensors)
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (Logprob, PromptLogprobs, SampleLogprobs,
                           SamplerOutput, SequenceData, SequenceGroupOutput,
                           SequenceOutput)


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence, frequency and repetition penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        assert logits is not None
        _, vocab_size = logits.shape

        # Apply min_tokens penalty which sets stop tokens to -inf if min_tokens
        # have not been generated yet
        logits = _apply_min_tokens_penalty(logits, sampling_metadata)

        # Prepare sampling tensors with pinned memory to avoid blocking.
        (sampling_tensors, do_penalties, do_top_p_top_k,
         do_min_p) = SamplingTensors.from_sampling_metadata(
             sampling_metadata, vocab_size, logits.device, logits.dtype)

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = _apply_penalties(logits, sampling_tensors.prompt_tokens,
                                      sampling_tensors.output_tokens,
                                      sampling_tensors.presence_penalties,
                                      sampling_tensors.frequency_penalties,
                                      sampling_tensors.repetition_penalties)

        # Apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits.div_(sampling_tensors.temperatures.unsqueeze_(dim=1))

        if do_top_p_top_k:
            logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps,
                                        sampling_tensors.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        sample_results = _sample(probs, logprobs, sampling_metadata,
                                 sampling_tensors)
        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results)
        return _build_sampler_output(sample_results, sampling_metadata,
                                     prompt_logprobs, sample_logprobs)


def _get_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def _apply_min_tokens_penalty(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    # list of indices in logits that will be set to -inf
    logits_to_penalize = []
    start_idx = 0
    for seq_ids, sampling_params in sampling_metadata.seq_groups:
        min_tokens = sampling_params.min_tokens
        if min_tokens > 0:
            seqs_to_penalize = []
            for i, seq_id in enumerate(seq_ids):
                seq_data = sampling_metadata.seq_data[seq_id]
                if len(seq_data.output_token_ids) < min_tokens:
                    seqs_to_penalize.append(i)

            if seqs_to_penalize:
                # convert to the index into logits
                seqs_to_penalize = [start_idx + i for i in seqs_to_penalize]
                # use set() to remove any duplicates
                token_ids_to_penalize = set(sampling_params.stop_token_ids +
                                            [sampling_params.eos_token_id])
                # itertools.product pairs each seq index with every token id
                logits_to_penalize.extend(
                    itertools.product(seqs_to_penalize, token_ids_to_penalize))

        start_idx += len(seq_ids)

    if logits_to_penalize:
        # use zip and * to group indices along each dimension
        # eg. [ (1,2), (1,3), (5,6) ] -> ( (1,1,5), (2,3,6) )
        logits[tuple(zip(*logits_to_penalize))] = -float("inf")

    return logits


def _apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                     output_tokens_tensor: torch.Tensor,
                     presence_penalties: torch.Tensor,
                     frequency_penalties: torch.Tensor,
                     repetition_penalties: torch.Tensor) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = _get_bin_counts_and_mask(prompt_tokens_tensor, vocab_size,
                                              num_seqs)
    output_bin_counts, output_mask = _get_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)

    repetition_penalties = repetition_penalties[:, None].repeat(1, vocab_size)
    repetition_penalties[~(prompt_mask | output_mask)] = 1.0
    logits = torch.where(logits > 0, logits / repetition_penalties,
                         logits * repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze_(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze_(dim=1) * output_mask
    return logits


def _apply_top_k_top_p(
    logits: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    # Apply top-k.
    top_k_mask = logits_sort.size(1) - k.to(torch.long)
    # Get all the top_k values.
    top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
    top_k_mask = logits_sort < top_k_mask
    logits_sort.masked_fill_(top_k_mask, -float("inf"))

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    # at least one
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    src = torch.arange(logits_idx.shape[-1],
                       device=logits_idx.device).expand_as(logits_idx)
    logits_idx_inv = torch.empty_like(logits_idx).scatter_(dim=-1,
                                                           index=logits_idx,
                                                           src=src)
    logits = torch.gather(logits_sort, dim=-1, index=logits_idx_inv)
    return logits


def _apply_min_p(
    logits: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    """
    Adapted from
    https://github.com/oobabooga/text-generation-webui/blob/3146124ec01f02c8fb1650a6517cf1b60b537aaf/modules/sampler_hijack.py#L16C17-L16C17
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    scaled_min_p = min_p.unsqueeze_(dim=1) * top_probs
    tokens_to_remove = probs < scaled_min_p
    logits = logits.masked_fill_(tokens_to_remove, -float("inf"))

    return logits


def _greedy_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    samples: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    samples = samples.tolist()
    sample_idx = 0
    results = []
    for seq_group in selected_seq_groups:
        seq_ids, _ = seq_group
        num_parent_seqs = len(seq_ids)
        assert num_parent_seqs == 1, (
            "Greedy sampling should have only one seq.")
        parent_ids = list(range(num_parent_seqs))
        next_token_ids = [samples[sample_idx]]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def _random_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    is_prompts: List[bool],
    random_samples: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    # Find the maximum best_of value of the prompt phase requests.
    random_samples = random_samples.cpu()
    sample_idx = 0
    results = []
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        seq_ids, sampling_params = seq_group
        num_parent_seqs = len(seq_ids)
        if is_prompt:
            # Prompt phase.
            parent_ids = [0] * sampling_params.best_of
            next_token_ids = random_samples[
                sample_idx, :sampling_params.best_of].tolist()
        else:
            # Generation phase.
            parent_ids = list(range(num_parent_seqs))
            next_token_ids = random_samples[sample_idx:sample_idx +
                                            num_parent_seqs, 0].tolist()
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def _beam_search_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    is_prompts: List[bool],
    seq_data: Dict[int, SequenceData],
    logprobs: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    # We sample 2 * beam_width candidates to make sure that with high
    # probability we can get `beam_width` candidates in addition to
    # the finished sequences for the next iteration. See
    # https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/utils/beam_search.py#L557-L563
    # for details. See also HF reference:
    # https://github.com/huggingface/transformers/blob/a4dd53d88e4852f023332d284ff07a01afcd5681/src/transformers/generation/utils.py#L3063-L3065
    #
    # NOTE: Beam search is not vectorized, so its speed can be slower than
    # other sampling methods.
    sample_idx = 0
    results = []
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        seq_ids, sampling_params = seq_group
        num_parent_seqs = len(seq_ids)
        beam_width = sampling_params.best_of
        seq_group_logprobs = logprobs[sample_idx:sample_idx + num_parent_seqs]
        if is_prompt:
            # Prompt phase.
            assert num_parent_seqs == 1, (
                "Prompt input should have only one seq.")
            parent_ids = [0] * (2 * beam_width)
            _, next_token_ids = torch.topk(seq_group_logprobs[0],
                                           2 * beam_width)
            next_token_ids = next_token_ids.tolist()
        else:
            # Generation phase.
            cumulative_logprobs = [
                seq_data[seq_id].cumulative_logprob for seq_id in seq_ids
            ]
            cumulative_logprobs = torch.tensor(
                cumulative_logprobs,
                dtype=torch.float,
                device=seq_group_logprobs.device)
            seq_group_logprobs = (seq_group_logprobs +
                                  cumulative_logprobs.unsqueeze(dim=1))
            _, topk_ids = torch.topk(seq_group_logprobs.flatten(),
                                     2 * beam_width)
            topk_ids = topk_ids.tolist()
            vocab_size = seq_group_logprobs.size(-1)
            parent_ids = [i // vocab_size for i in topk_ids]
            next_token_ids = [i % vocab_size for i in topk_ids]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    assert sample_idx == logprobs.size(0)
    return results


# torch.multinomial forces a GPU<->CPU sync.
# Therefore, we use an optimized implementation instead.
# Note that we always sample with replacement.
# probs will be modified in place, but this is fine, as we pass
# in a copy already.
def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
    seq_groups: Optional[List[Tuple[List[int], SamplingParams]]] = None,
    generators: Optional[List[torch.Generator]] = None,
) -> torch.Tensor:
    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        # This allows us to do sampling with replacement by creating
        # num_samples copies of each row in the tensor, and then
        # batch sampling the resulting tensor.
        probs = probs[:, None, :].expand(probs.shape[0], num_samples,
                                         probs.shape[1]).contiguous().view(
                                             -1, probs.shape[1])
    q = torch.empty_like(probs)
    if seq_groups is None:
        q.exponential_()
    else:
        sample_idx = 0
        for (seq_ids, _), generator in zip(seq_groups, generators):
            next_sample_idx = sample_idx + len(seq_ids) * num_samples
            q[sample_idx:next_sample_idx].exponential_(generator=generator)
            sample_idx = next_sample_idx
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)


def _sample_with_torch(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> List[Tuple[List[int], List[int]]]:
    categorized_seq_group_ids = {t: [] for t in SamplingType}
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        _, sampling_params = seq_group
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

    sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    sample_metadata = {}
    multinomial_samples = {}

    # Counterintiutively, having two loops here is actually faster.
    # The first loop can run without waiting on GPU<->CPU sync.
    for sampling_type in SamplingType:
        sample_indices = categorized_sample_indices[sampling_type][:, 0]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue
        seq_group_ids = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_ids]
        is_prompts = [i < sampling_metadata.num_prompts for i in seq_group_ids]
        sample_metadata[sampling_type] = (seq_group_ids, seq_groups,
                                          is_prompts, sample_indices)
        if sampling_type == SamplingType.GREEDY:
            greedy_samples = torch.argmax(logprobs[sample_indices.long()],
                                          dim=-1)
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            max_best_of_in_batch = 1
            for seq_group, is_prompt in zip(seq_groups, is_prompts):
                if is_prompt:
                    _, sampling_params = seq_group
                    max_best_of_in_batch = max(max_best_of_in_batch,
                                               sampling_params.best_of)
            seeded_args = {} if sampling_type == SamplingType.RANDOM else {
                "seq_groups": seq_groups,
                "generators": sampling_metadata.generators,
            }
            multinomial_samples[sampling_type] = _multinomial(
                probs[sample_indices.long()], max_best_of_in_batch,
                **seeded_args)
        elif sampling_type == SamplingType.BEAM:
            beam_search_logprobs = logprobs[sample_indices]
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

    # GPU<->CPU sync happens in the loop below.

    for sampling_type in SamplingType:
        if sampling_type not in sample_metadata:
            continue
        seq_group_ids, seq_groups, is_prompts, sample_indices = sample_metadata[
            sampling_type]
        if sampling_type == SamplingType.GREEDY:
            sample_results = _greedy_sample(seq_groups, greedy_samples)
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            sample_results = _random_sample(seq_groups, is_prompts,
                                            multinomial_samples[sampling_type])
        elif sampling_type == SamplingType.BEAM:
            sample_results = _beam_search_sample(seq_groups, is_prompts,
                                                 sampling_metadata.seq_data,
                                                 beam_search_logprobs)
        sample_results_dict.update(zip(seq_group_ids, sample_results))

    sample_results = [
        sample_results_dict[i]
        for i in range(len(sampling_metadata.seq_groups))
    ]
    return sample_results


def _sample_with_triton_kernel(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sampling_tensors: SamplingTensors,
) -> List[Tuple[List[int], List[int]]]:
    categorized_seq_group_ids = {t: [] for t in SamplingType}
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        _, sampling_params = seq_group
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

    sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    sample_metadata = {}
    max_best_of_in_batch = 1

    # Counterintiutively, having two loops here is actually faster.
    # The first loop can run without waiting on GPU<->CPU sync.
    for sampling_type in SamplingType:
        sample_indices = categorized_sample_indices[sampling_type][:, 0]
        sampled_token_indices = categorized_sample_indices[sampling_type][:, 1]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue
        seq_group_ids = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_ids]
        is_prompts = [i < sampling_metadata.num_prompts for i in seq_group_ids]
        sample_metadata[sampling_type] = (seq_group_ids, seq_groups,
                                          is_prompts, sample_indices,
                                          sampled_token_indices)
        if sampling_type in (SamplingType.GREEDY, SamplingType.RANDOM,
                             SamplingType.RANDOM_SEED):
            for seq_group, is_prompt in zip(seq_groups, is_prompts):
                if is_prompt:
                    _, sampling_params = seq_group
                    max_best_of_in_batch = max(max_best_of_in_batch,
                                               sampling_params.best_of)
        elif sampling_type == SamplingType.BEAM:
            beam_search_logprobs = logprobs[sample_indices]
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

    sampled_tokens, _, _ = sample_triton(
        probs=probs,
        seeds=sampling_tensors.sampling_seeds,
        max_best_of=max_best_of_in_batch,
        sample_indices=sampling_tensors.sample_indices,
        logprobs=logprobs,
        # don't save logprobs because we have logic for that below
        # TODO: use this instead of the CPU-based logic below
        save_logprobs=False,
    )

    # GPU<->CPU sync happens in the loop below.

    for sampling_type in SamplingType:
        if sampling_type not in sample_metadata:
            continue
        (seq_group_ids, seq_groups, is_prompts, sample_indices,
         sampled_token_indices) = sample_metadata[sampling_type]
        if sampling_type == SamplingType.GREEDY:
            sample_results = _greedy_sample(
                seq_groups, sampled_tokens[sampled_token_indices][:, 0])
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            sample_results = _random_sample(
                seq_groups, is_prompts, sampled_tokens[sampled_token_indices])
        elif sampling_type == SamplingType.BEAM:
            sample_results = _beam_search_sample(seq_groups, is_prompts,
                                                 sampling_metadata.seq_data,
                                                 beam_search_logprobs)
        sample_results_dict.update(zip(seq_group_ids, sample_results))

    sample_results = [
        sample_results_dict[i]
        for i in range(len(sampling_metadata.seq_groups))
    ]
    return sample_results


def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sampling_tensors: SamplingTensors,
) -> List[Tuple[List[int], List[int]]]:
    return _sample_with_torch(probs, logprobs, sampling_metadata)

    # TODO: Enable once Triton kernel & associated code is faster.
    # return _sample_with_triton_kernel(probs, logprobs, sampling_metadata,
    #                                   sampling_tensors)


def _get_ranks(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    This function calculates the ranks of the chosen tokens in a logprob tensor.

    Args:
        x (torch.Tensor): 2D logprob tensor of shape (N, M)
                        where N is the no. of tokens and M is the vocab dim.
        indices (torch.Tensor): List of chosen token indices.

    Returns:
        torch.Tensor: 1D tensor of shape (N,) where N is the no. of tokens.
                    Each element in the returned tensor represents the rank 
                    of the chosen token in the input logprob tensor.
    """
    vals = x[torch.arange(0, len(x), device=x.device, dtype=indices.dtype),
             indices]
    return (x > vals[:, None]).long().sum(1).add_(1)


def _get_logprobs(
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sample_results: List[Tuple[List[int], List[int]]],
) -> Tuple[List[Optional[List[Optional[Dict[int, float]]]]], List[List[Dict[
        int, float]]]]:
    # Prepare query indices
    batched_logprobs_query_seq_indices: List[int] = []
    batched_logprobs_query_token_indices: List[int] = []
    largest_num_logprobs = 0
    sample_idx = 0
    for i, (seq_group, sample_result) in enumerate(
            zip(sampling_metadata.seq_groups, sample_results)):
        seq_ids, sampling_params = seq_group
        next_token_ids, parent_ids = sample_result
        num_parent_seqs = len(seq_ids)
        if (i < sampling_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            largest_num_logprobs = max(largest_num_logprobs,
                                       sampling_params.prompt_logprobs)
            prompt_len = sampling_metadata.prompt_lens[i]
            prompt_tokens = sampling_metadata.seq_data[
                seq_ids[0]].prompt_token_ids
            batched_logprobs_query_seq_indices.extend(
                sample_idx + j for j in range(prompt_len - 1))
            batched_logprobs_query_token_indices.extend(
                token_id for token_id in prompt_tokens[1:])
            sample_idx += prompt_len - 1
        batched_logprobs_query_seq_indices.extend(
            [sample_idx + parent_id for parent_id in parent_ids])
        batched_logprobs_query_token_indices.extend(next_token_ids)
        if sampling_params.logprobs is not None:
            largest_num_logprobs = max(largest_num_logprobs,
                                       sampling_params.logprobs)
        sample_idx += num_parent_seqs
    assert sample_idx == logprobs.size(0)

    batched_logprobs_query_seq_indices_gpu = torch.tensor(
        batched_logprobs_query_seq_indices, device=logprobs.device)
    batched_logprobs_query_token_indices_gpu = torch.tensor(
        batched_logprobs_query_token_indices, device=logprobs.device)

    # Batched query for logprobs of selected token
    batched_logprobs_query_result = logprobs[[
        batched_logprobs_query_seq_indices_gpu,
        batched_logprobs_query_token_indices_gpu
    ]]

    batched_ranks_query_result = _get_ranks(
        logprobs[batched_logprobs_query_seq_indices_gpu],
        batched_logprobs_query_token_indices_gpu)

    # Batched query for logprobs of topk tokens
    if largest_num_logprobs > 0:
        top_logprobs, top_token_ids = torch.topk(logprobs,
                                                 largest_num_logprobs,
                                                 dim=-1)
        top_logprobs = top_logprobs.cpu()
        top_token_ids = top_token_ids.cpu()
    else:
        top_logprobs, top_token_ids = None, None

    batched_logprobs_query_result = batched_logprobs_query_result.cpu()
    batched_ranks_query_result = batched_ranks_query_result.cpu()

    # Gather results
    result_prompt_logprobs: List[Optional[PromptLogprobs]] = []
    result_sample_logprobs: List[SampleLogprobs] = []
    sample_idx = 0
    query_result_idx = 0
    for i, (seq_group, sample_result) in enumerate(
            zip(sampling_metadata.seq_groups, sample_results)):
        seq_ids, sampling_params = seq_group
        next_token_ids, parent_ids = sample_result

        # Prompt logprobs
        if (i < sampling_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            num_logprobs = sampling_params.prompt_logprobs
            prompt_tokens = sampling_metadata.seq_data[
                seq_ids[0]].prompt_token_ids
            group_prompt_logprobs: PromptLogprobs = [None]
            for token_id in prompt_tokens[1:]:
                prompt_logprobs_dict = {
                    token_id:
                    (batched_logprobs_query_result[query_result_idx].item(),
                     batched_ranks_query_result[query_result_idx].item())
                }
                if num_logprobs > 0:
                    prompt_logprobs_dict.update(
                        zip(
                            top_token_ids[sample_idx, :num_logprobs].tolist(),
                            zip(
                                top_logprobs[
                                    sample_idx, :num_logprobs].tolist(),
                                range(1, num_logprobs + 1))))
                group_prompt_logprobs.append({
                    token_id: Logprob(*logprob_rank)
                    for token_id, logprob_rank in prompt_logprobs_dict.items()
                })
                sample_idx += 1
                query_result_idx += 1
            result_prompt_logprobs.append(group_prompt_logprobs)
        else:
            result_prompt_logprobs.append(None)

        # Sample logprobs
        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = 0
        group_sample_logprobs: SampleLogprobs = []
        for next_token_id, parent_id in zip(next_token_ids, parent_ids):
            sample_logprobs_dict = {
                next_token_id:
                (batched_logprobs_query_result[query_result_idx].item(),
                 batched_ranks_query_result[query_result_idx].item())
            }
            query_result_idx += 1
            if num_logprobs > 0:
                sample_logprobs_dict.update(
                    zip(
                        top_token_ids[sample_idx +
                                      parent_id, :num_logprobs].tolist(),
                        zip(
                            top_logprobs[sample_idx +
                                         parent_id, :num_logprobs].tolist(),
                            range(1, num_logprobs + 1))))
            group_sample_logprobs.append({
                token_id: Logprob(*logprob_rank)
                for token_id, logprob_rank in sample_logprobs_dict.items()
            })
        result_sample_logprobs.append(group_sample_logprobs)
        sample_idx += len(seq_ids)

    return result_prompt_logprobs, result_sample_logprobs


def _build_sampler_output(
    sample_results: List[Tuple[List[int], List[int]]],
    sampling_metadata: SamplingMetadata,
    prompt_logprobs: List[Optional[PromptLogprobs]],
    sample_logprobs: List[SampleLogprobs],
) -> SamplerOutput:
    sampler_output = []
    for (seq_group, sample_result, group_prompt_logprobs,
         group_sample_logprobs) in zip(sampling_metadata.seq_groups,
                                       sample_results, prompt_logprobs,
                                       sample_logprobs):
        seq_ids, _ = seq_group
        next_token_ids, parent_ids = sample_result
        seq_outputs = []
        for parent_id, next_token_id, logprobs in zip(parent_ids,
                                                      next_token_ids,
                                                      group_sample_logprobs):
            seq_outputs.append(
                SequenceOutput(seq_ids[parent_id], next_token_id, logprobs))
        sampler_output.append(
            SequenceGroupOutput(seq_outputs, group_prompt_logprobs))
    return SamplerOutput(outputs=sampler_output)
