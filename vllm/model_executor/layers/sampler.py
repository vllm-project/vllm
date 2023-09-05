"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.parallel_utils.tensor_parallel import (
    gather_from_tensor_model_parallel_region)
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceOutputs

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence and frequency penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def _get_logits(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, input_metadata)

        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        logits = gather_from_tensor_model_parallel_region(logits)
        # Remove paddings in vocab (if any).
        logits = logits[:, :self.vocab_size]
        return logits

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> SamplerOutput:
        logits = self._get_logits(embedding, hidden_states, input_metadata,
                                  embedding_bias)

        # Apply presence and frequency penalties.
        output_tokens = _get_output_tokens(input_metadata)
        assert len(output_tokens) == logits.shape[0]
        presence_penalties, frequency_penalties = _get_penalties(
            input_metadata)
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        logits = _apply_penalties(logits, output_tokens, presence_penalties,
                                  frequency_penalties)

        # Sort logits by sampling type.
        sampling_type_indices = input_metadata.sampling_type_indices
        sampling_type_indices_values, sampling_type_indices_index = (
            sampling_type_indices.sort())
        # We use the bincount+cumsum to compute the offsets.
        sampling_type_offsets = torch.bincount(sampling_type_indices_values,
                                               minlength=3).cumsum(0)[:-1]
        # The logit_to_id_map lets us reverse the sort we did above when we match
        # outputs to sequences.
        logit_to_id_map = sampling_type_indices_index.argsort()
        logits = logits[sampling_type_indices_index]

        # If there are only greedy requests (and therefore, no requests
        # we can apply temperature & top_p/top_k to), we can return early.
        # Otherwise, we will slice the logits and modify only the non-greedy
        # subset.
        if sampling_type_indices.max() > SamplingType.GREEDY:
            non_greedy_offset = sampling_type_offsets[0]
            # Apply temperature scaling.
            temperatures = _get_temperatures(input_metadata)
            assert len(temperatures) == logits.shape[0]
            temperatures = temperatures[non_greedy_offset:]
            if any(t != 1.0 for t in temperatures):
                t = torch.tensor(temperatures,
                                 dtype=logits.dtype,
                                 device=logits.device)
                # Use in-place division to avoid creating a new tensor.
                logits[non_greedy_offset:].div_(t.unsqueeze(dim=1))

            # Apply top-p and top-k truncation.
            top_ps, top_ks = _get_top_p_top_k(input_metadata, self.vocab_size)
            assert len(top_ps) == len(top_ks) == logits.shape[0]
            top_ps = top_ps[non_greedy_offset:]
            top_ks = top_ks[non_greedy_offset:]
            do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
            do_top_k = any(k != self.vocab_size for k in top_ks)
            if do_top_p or do_top_k:
                p = torch.tensor(top_ps,
                                 dtype=logits.dtype,
                                 device=logits.device)
                k = torch.tensor(top_ks, dtype=torch.int, device=logits.device)
                _apply_top_p_top_k_in_place(logits[non_greedy_offset:], p, k)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities (before applying top-p and top-k).
        logprobs = torch.log(probs)

        # Sample the next tokens.
        return _sample(probs, logprobs, input_metadata, sampling_type_offsets,
                       logit_to_id_map)


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    input_metadata: InputMetadata,
) -> torch.Tensor:
    start_idx = 0
    last_token_indicies: List[int] = []
    for prompt_len in input_metadata.prompt_lens:
        last_token_indicies.append(start_idx + prompt_len - 1)
        start_idx += prompt_len
    last_token_indicies.extend(
        range(start_idx, start_idx + input_metadata.num_generation_tokens))
    return hidden_states.index_select(
        0, torch.tensor(last_token_indicies, device=hidden_states.device))


def _get_penalties(
        input_metadata: InputMetadata) -> Tuple[List[float], List[float]]:
    # Collect the presence and frequency penalties.
    presence_penalties: List[float] = []
    frequency_penalties: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        p = sampling_params.presence_penalty
        f = sampling_params.frequency_penalty
        if i < input_metadata.num_prompts:
            # A prompt input.
            presence_penalties.append(p)
            frequency_penalties.append(f)
        else:
            # A generation token.
            presence_penalties += [p] * len(seq_ids)
            frequency_penalties += [f] * len(seq_ids)
    return presence_penalties, frequency_penalties


def _get_output_tokens(input_metadata: InputMetadata) -> List[List[int]]:
    output_tokens: List[List[int]] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, _ = seq_group
        if i < input_metadata.num_prompts:
            # A prompt input.
            # NOTE: While the prompt input usually has no output tokens,
            # it may have output tokens in the case of recomputation.
            seq_id = seq_ids[0]
            seq_data = input_metadata.seq_data[seq_id]
            output_tokens.append(seq_data.output_token_ids)
        else:
            # A generation token.
            for seq_id in seq_ids:
                seq_data = input_metadata.seq_data[seq_id]
                output_tokens.append(seq_data.output_token_ids)
    return output_tokens


def _batched_bincount(x: torch.Tensor, dim: int,
                      max_value: int) -> torch.Tensor:
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def _pad_to_len(x: List[int], length: int) -> List[int]:
    if len(x) >= length:
        return x
    return x.copy() + [0] * (length - len(x))


def _apply_penalties(
    logits: torch.Tensor,
    output_tokens: List[List[int]],
    presence_penalties: List[float],
    frequency_penalties: List[float],
) -> torch.Tensor:
    num_seqs = logits.shape[0]
    # Collect the indices of sequences that have non-zero penalties.
    indices = []
    max_len = 0
    for i in range(num_seqs):
        if not output_tokens[i]:
            continue
        if len(output_tokens[i]) > max_len:
            max_len = len(output_tokens[i])
        p = presence_penalties[i]
        f = frequency_penalties[i]
        if abs(p) < _SAMPLING_EPS and abs(f) < _SAMPLING_EPS:
            continue
        indices.append(i)

    # Return early if all sequences have zero penalties.
    if not indices:
        return logits

    frequency_penalties = torch.tensor(frequency_penalties,
                                       dtype=logits.dtype,
                                       device=logits.device).unsqueeze(dim=1)
    presence_penalties = torch.tensor(presence_penalties,
                                      dtype=logits.dtype,
                                      device=logits.device).unsqueeze(dim=1)
    output_tokens = [_pad_to_len(x, max_len) for x in output_tokens]
    input_ids = torch.tensor(output_tokens,
                             dtype=torch.long,
                             device=logits.device)
    occurences = _batched_bincount(input_ids, 1, logits.shape[1])

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    frequency_penalty = occurences * frequency_penalties
    presence_penalty = (occurences > 0) * presence_penalties

    logits.sub_(frequency_penalty).sub_(presence_penalty)
    return logits


def _get_temperatures(input_metadata: InputMetadata) -> List[float]:
    # Collect the temperatures for the logits.
    temperatures: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        temperature = sampling_params.temperature
        if temperature < _SAMPLING_EPS:
            # NOTE: Zero temperature means deterministic sampling
            # (i.e., greedy sampling or beam search).
            # Set the temperature to 1 to avoid division by zero.
            temperature = 1.0

        if i < input_metadata.num_prompts:
            # A prompt input.
            temperatures.append(temperature)
        else:
            # A generation token.
            temperatures += [temperature] * len(seq_ids)
    return temperatures


def _get_top_p_top_k(
    input_metadata: InputMetadata,
    vocab_size: int,
) -> Tuple[List[float], List[int]]:
    top_ps: List[float] = []
    top_ks: List[int] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        top_p = sampling_params.top_p
        # k should not be greater than the vocab size.
        top_k = min(sampling_params.top_k, vocab_size)
        # k=-1 means no truncation.
        top_k = vocab_size if top_k == -1 else top_k
        if i < input_metadata.num_prompts:
            # A prompt input.
            top_ps.append(top_p)
            top_ks.append(top_k)
        else:
            # A generation token.
            top_ps += [top_p] * len(seq_ids)
            top_ks += [top_k] * len(seq_ids)
    return top_ps, top_ks


def _apply_top_p_top_k_in_place(
    logits: torch.Tensor,
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
) -> None:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
    logits_sort[top_p_mask] = -float("inf")

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= top_ks.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Re-sort the probabilities.
    torch.gather(logits_sort,
                 dim=-1,
                 index=torch.argsort(logits_idx, dim=-1),
                 out=logits)


def _get_topk_logprobs(
    logprobs: torch.Tensor,
    num_logprobs: Optional[int],
) -> Dict[int, float]:
    if num_logprobs is None or num_logprobs == 0:
        return {}

    topk_logprobs, topk_ids = torch.topk(logprobs, num_logprobs)
    if num_logprobs == 1:
        topk_logprobs = [topk_logprobs.item()]
        topk_ids = [topk_ids.item()]
    else:
        topk_logprobs = topk_logprobs.tolist()
        topk_ids = topk_ids.tolist()

    token_to_logprob: Dict[int, float] = {}
    for token_id, logprob in zip(topk_ids, topk_logprobs):
        token_to_logprob[token_id] = logprob
    return token_to_logprob


def _sample_from_prompt(
    prob: torch.Tensor,
    sampling_params: SamplingParams,
) -> List[int]:
    if sampling_params.sampling_type == SamplingType.BEAM:
        # Beam search.
        beam_width = sampling_params.best_of
        # Sample 2 * beam_width candidates to make sure that with high
        # probability we can get `beam_width` candidates in addition to
        # the finished sequences for the next iteration. See
        # https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/utils/beam_search.py#L557-L563
        # for details. See also HF reference:
        # https://github.com/huggingface/transformers/blob/a4dd53d88e4852f023332d284ff07a01afcd5681/src/transformers/generation/utils.py#L3063-L3065
        _, next_token_ids = torch.topk(prob, 2 * beam_width)
        next_token_ids = next_token_ids.tolist()
    elif sampling_params.sampling_type == SamplingType.GREEDY:
        # Greedy sampling.
        assert sampling_params.best_of == 1
        next_token_id = torch.argmax(prob)
        next_token_ids = [next_token_id.item()]
    else:
        # Random sampling.
        # Sample `best_of` tokens for the prompt.
        num_seqs = sampling_params.best_of
        next_token_ids = torch.multinomial(prob,
                                           num_samples=num_seqs,
                                           replacement=True)
        next_token_ids = next_token_ids.tolist()
    return next_token_ids


def _sample_from_generation_tokens(
    seq_ids: List[int],
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    seq_logprobs: List[float],
    sampling_params: SamplingParams,
) -> Tuple[List[int], List[int]]:
    # NOTE(woosuk): sampling_params.best_of can be greater than
    # len(seq_ids) because some sequences in the group might have
    # been already terminated.
    if sampling_params.sampling_type == SamplingType.BEAM:
        # Beam search.
        # Add cumulative logprobs for the sequences in the group.
        seq_logprobs = torch.tensor(seq_logprobs,
                                    dtype=torch.float,
                                    device=logprobs.device)
        logprobs = logprobs + seq_logprobs.unsqueeze(dim=1)

        vocab_size = logprobs.size(-1)
        beam_width = len(seq_ids)
        _, topk_ids = torch.topk(logprobs.flatten(), 2 * beam_width)
        topk_ids = topk_ids.tolist()
        seq_idx = [i // vocab_size for i in topk_ids]
        parent_seq_ids = [seq_ids[i] for i in seq_idx]
        next_token_ids = [i % vocab_size for i in topk_ids]
    elif sampling_params.sampling_type == SamplingType.GREEDY:
        # Greedy sampling.
        assert len(seq_ids) == 1
        next_token_id = torch.argmax(probs, dim=-1)
        next_token_ids = [int(next_token_id.item())]
        parent_seq_ids = seq_ids
    else:
        # Random sampling.
        # Sample 1 token for each sequence in the group.
        next_token_ids = torch.multinomial(probs,
                                           num_samples=1,
                                           replacement=True)
        next_token_ids = next_token_ids.squeeze(dim=-1).tolist()
        parent_seq_ids = seq_ids
    return parent_seq_ids, next_token_ids


def _batched_sample(gen_probs: torch.Tensor, gen_logprobs: torch.Tensor,
                    sampling_type_offsets: torch.Tensor,
                    max_best_of: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Split the probs into multiple tensors, one for each sampling type.
    # tensor_split will create views of the probs tensor, meaning
    # there will be no copying involved (this should be very fast).
    sampling_type_tensors = torch.tensor_split(gen_probs,
                                               sampling_type_offsets)

    # Create a tensor to store the tokens sampled by greedy or multinomial
    # sampling. Note that beam search tokens are not stored here.
    batch_chosen_tokens = torch.empty(
        (sum(t.shape[0] for t in sampling_type_tensors[:2]), max_best_of),
        dtype=torch.long,
        device=gen_probs.device)
    # Because batch_chosen_tokens_tensors are views of
    # batch_chosen_tokens_tensor, any changes to
    # batch_chosen_tokens_tensors will be reflected in
    # batch_chosen_tokens_tensor.
    batch_chosen_tokens_tensors = torch.tensor_split(batch_chosen_tokens,
                                                     sampling_type_offsets[:2])

    # Do vectorized sampling.
    if len(sampling_type_tensors) > 0 and sampling_type_tensors[0].numel() > 0:
        # If we have best_of, use topk.
        if max_best_of > 1:
            dummy_out = torch.empty_like(sampling_type_tensors[0])
            torch.topk(sampling_type_tensors[0],
                       max_best_of,
                       dim=-1,
                       sorted=False,
                       out=(dummy_out, batch_chosen_tokens_tensors[0]))
        # Otherwise, do argmax which is faster
        else:
            torch.argmax(sampling_type_tensors[0],
                         dim=-1,
                         keepdim=True,
                         out=batch_chosen_tokens_tensors[0])
    if len(sampling_type_tensors) > 1 and sampling_type_tensors[1].numel() > 0:
        torch.multinomial(sampling_type_tensors[1],
                          num_samples=max_best_of,
                          replacement=True,
                          out=batch_chosen_tokens_tensors[1])
    # Everything that is not greedy or multinomial (currently, beam search)
    # (sampling_type_tensors[2:]) is handled iteratively below.

    batch_chosen_logprobs = torch.gather(gen_logprobs,
                                         dim=-1,
                                         index=batch_chosen_tokens)
    return batch_chosen_tokens, batch_chosen_logprobs


def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    input_metadata: InputMetadata,
    sampling_type_offsets: torch.Tensor,
    logit_to_id_map: torch.Tensor,
) -> SamplerOutput:
    seq_outputs: SamplerOutput = []

    # TODO(woosuk): Optimize.
    idx = 0

    # Find the maximum best_of value to use as a shape for
    # topk/multinomial tensors.
    max_best_of = 1
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        if (i < input_metadata.num_prompts
                and sampling_params.best_of > max_best_of):
            max_best_of = sampling_params.best_of

    gen_probs = probs
    gen_logprobs = logprobs

    batch_chosen_tokens_tensor, batch_chosen_logprobs_tensor = _batched_sample(
        gen_probs, gen_logprobs, sampling_type_offsets, max_best_of)

    batch_chosen_logprobs_tensor = batch_chosen_logprobs_tensor[
        logit_to_id_map]
    batch_chosen_tokens_tensor = batch_chosen_tokens_tensor[logit_to_id_map]

    batch_chosen_logprobs_list = batch_chosen_logprobs_tensor.tolist()
    batch_chosen_tokens_list = batch_chosen_tokens_tensor.tolist()

    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_group_outputs: List[SequenceOutputs] = []
        seq_ids, sampling_params = seq_group
        if i < input_metadata.num_prompts:
            # Generate the next tokens for a prompt input.
            assert len(seq_ids) == 1, "Prompt input should have only one seq."
            parent_seq_id = seq_ids[0]
            logprob = logprobs[idx]

            # Sample the next tokens.
            if sampling_params.sampling_type in (SamplingType.GREEDY,
                                                 SamplingType.RANDOM):
                next_token_ids = batch_chosen_tokens_list[idx][:sampling_params
                                                               .best_of]
            else:
                prob = gen_probs[idx]
                next_token_ids = _sample_from_prompt(prob, sampling_params)

            # Get top-k log probabilities for the next tokens.
            next_logprobs = _get_topk_logprobs(logprob,
                                               sampling_params.logprobs)

            # Build the output.
            j = 0
            for next_token_id in next_token_ids:
                output_logprobs = next_logprobs.copy()
                if sampling_params.sampling_type in (SamplingType.GREEDY,
                                                     SamplingType.RANDOM):
                    output_logprobs[
                        next_token_id] = batch_chosen_logprobs_list[idx][j]
                else:
                    output_logprobs[next_token_id] = logprob[
                        next_token_id].item()
                seq_group_outputs.append(
                    SequenceOutputs(parent_seq_id, next_token_id,
                                    output_logprobs))
                j += 1

            idx += 1
        else:
            # Generate the next tokens for generation tokens.
            num_parent_seqs = len(seq_ids)
            logprob = logprobs[idx:idx + num_parent_seqs]

            # Sample the next tokens.
            parent_seq_ids = seq_ids

            if sampling_params.sampling_type in (SamplingType.GREEDY,
                                                 SamplingType.RANDOM):
                next_token_ids = batch_chosen_tokens_list[idx:idx +
                                                          num_parent_seqs]
                next_token_ids = [
                    item for sublist in next_token_ids for item in sublist
                ]
            else:
                prob = gen_probs[idx]
                seq_logprobs = [
                    input_metadata.seq_data[seq_id].cumulative_logprob
                    for seq_id in seq_ids
                ]
                parent_seq_ids, next_token_ids = _sample_from_generation_tokens(
                    seq_ids, prob, logprob, seq_logprobs, sampling_params)

            # Get top-k log probabilities for the next tokens.
            next_logprobs: Dict[int, Dict[int, float]] = {}
            for j, seq_id in enumerate(seq_ids):
                next_logprobs[seq_id] = _get_topk_logprobs(
                    logprob[j], sampling_params.logprobs)

            # Build the output.
            for parent_seq_id, next_token_id in zip(parent_seq_ids,
                                                    next_token_ids):
                j = seq_ids.index(parent_seq_id)
                output_logprobs = next_logprobs[parent_seq_id].copy()
                if sampling_params.sampling_type in (SamplingType.GREEDY,
                                                     SamplingType.RANDOM):
                    output_logprobs[
                        next_token_id] = batch_chosen_logprobs_list[idx + j][0]
                else:
                    output_logprobs[next_token_id] = logprob[
                        j, next_token_id].item()
                seq_group_outputs.append(
                    SequenceOutputs(parent_seq_id, next_token_id,
                                    output_logprobs))
            idx += num_parent_seqs
        seq_outputs.append(seq_group_outputs)

    return seq_outputs
