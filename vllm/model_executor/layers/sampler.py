"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.parallel_utils.tensor_parallel import (
    gather_from_tensor_model_parallel_region)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceOutputs

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

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Dict[int, SequenceOutputs]:
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, input_metadata)

        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        logits = gather_from_tensor_model_parallel_region(logits)
        # Remove paddings in vocab (if any).
        logits = logits[:, :self.vocab_size]

        # Apply presence and frequency penalties.
        output_tokens = _get_output_tokens(input_metadata)
        assert len(output_tokens) == logits.shape[0]
        presence_penalties, frequency_penalties = _get_penalties(
            input_metadata)
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        logits = _apply_penalties(logits, output_tokens, presence_penalties,
                                  frequency_penalties, self.vocab_size)

        # Apply temperature scaling.
        temperatures = _get_temperatures(input_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures,
                             dtype=logits.dtype,
                             device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # Apply top-p and top-k truncation.
        top_ps, top_ks = _get_top_p_top_k(input_metadata, self.vocab_size)
        assert len(top_ps) == len(top_ks) == logits.shape[0]
        do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
        do_top_k = any(k != self.vocab_size for k in top_ks)
        if do_top_p or do_top_k:
            logits = _apply_top_p_top_k(logits, top_ps, top_ks)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities (before applying top-p and top-k).
        logprobs = torch.log(probs)

        # Sample the next tokens.
        return _sample(probs, logprobs, input_metadata)


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
    return hidden_states[last_token_indicies]


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

@torch.jit.script
def batched_bincount(
    x: torch.Tensor, dim: int, max_value: int
) -> torch.Tensor:
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
    vocab_size: int,
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
        if p < _SAMPLING_EPS and f < _SAMPLING_EPS:
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
    output_tokens = [
        _pad_to_len(x, max_len) for x in output_tokens
    ]
    input_ids = torch.tensor(output_tokens,dtype=torch.long,device=logits.device)
    occurences = batched_bincount(input_ids, 1, logits.shape[1])
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


def _apply_top_p_top_k(
    logits: torch.Tensor,
    top_ps: List[float],
    top_ks: List[int],
) -> torch.Tensor:
    p = torch.tensor(top_ps, dtype=logits.dtype, device=logits.device)
    k = torch.tensor(top_ks, dtype=torch.int, device=logits.device)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    logits_sort[top_p_mask] = -float("inf")

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Re-sort the probabilities.
    logits = torch.gather(logits_sort,
                          dim=-1,
                          index=torch.argsort(logits_idx, dim=-1))
    return logits


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
    if sampling_params.use_beam_search:
        # Beam search.
        beam_width = sampling_params.best_of
        _, next_token_ids = torch.topk(prob, beam_width)
        next_token_ids = next_token_ids.tolist()
    elif sampling_params.temperature < _SAMPLING_EPS:
        # Greedy sampling.
        assert sampling_params.best_of == 1
        next_token_id = torch.argmax(prob)
        next_token_ids = [next_token_id.item()]
    else:
        # Random sampling.
        # Sample `best_of` tokens for the prompt.
        num_seqs = sampling_params.best_of
        next_token_ids = torch.multinomial(prob, num_samples=num_seqs)
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
    if sampling_params.use_beam_search:
        # Beam search.
        # Add cumulative logprobs for the sequences in the group.
        seq_logprobs = torch.tensor(seq_logprobs,
                                    dtype=torch.float,
                                    device=logprobs.device)
        logprobs = logprobs + seq_logprobs.unsqueeze(dim=1)

        vocab_size = logprobs.size(-1)
        beam_width = len(seq_ids)
        _, topk_ids = torch.topk(logprobs.flatten(), beam_width)
        topk_ids = topk_ids.tolist()
        seq_idx = [i // vocab_size for i in topk_ids]
        beam_seq_ids = [seq_ids[i] for i in seq_idx]
        token_ids = [i % vocab_size for i in topk_ids]

        beam_outputs: Dict[int, Tuple[int, int]] = {}
        outstanding_beams: List[Tuple[int, int]] = []
        # If a beam survives, continue with it.
        for seq_id, token_id in zip(beam_seq_ids, token_ids):
            if seq_id not in beam_outputs:
                beam_outputs[seq_id] = (seq_id, token_id)
            else:
                outstanding_beams.append((seq_id, token_id))

        # If a beam is discarded, fork another beam.
        for seq_id in seq_ids:
            if seq_id not in beam_outputs:
                beam_outputs[seq_id] = outstanding_beams.pop()
        assert not outstanding_beams

        parent_seq_ids = [beam_outputs[seq_id][0] for seq_id in seq_ids]
        next_token_ids = [beam_outputs[seq_id][1] for seq_id in seq_ids]
    elif sampling_params.temperature < _SAMPLING_EPS:
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


def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    input_metadata: InputMetadata,
) -> Dict[int, SequenceOutputs]:
    seq_outputs: Dict[int, SequenceOutputs] = {}

    # TODO(woosuk): Optimize.
    idx = 0
    max_best_of = 1

    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        if i < input_metadata.num_prompts and sampling_params.best_of > max_best_of:
            max_best_of = sampling_params.best_of

    gen_probs = probs
    # gen_logprobs = logprobs

    sampling_offsets = input_metadata.sampling_offsets
    sampling_tensors = torch.tensor_split(gen_probs, sampling_offsets)
    greedy_tensor = None
    sampling_tensor = None

    if len(sampling_offsets) < 1 or sampling_offsets[0] > 0:
        if max_best_of > 1:
            _, greedy_tensor = torch.topk(sampling_tensors[0], max_best_of, dim=-1, sorted=False)
        else:
            greedy_tensor = torch.argmax(sampling_tensors[0], dim=-1).unsqueeze(1)
    if len(sampling_tensors) > 1:
        sampling_tensor = torch.multinomial(sampling_tensors[1], num_samples=max_best_of, replacement=False)

    if not sampling_offsets:
        sampling_offsets.append(0)

    if greedy_tensor is not None:
        greedy = greedy_tensor.tolist()
    if sampling_tensor is not None:
        sampling = sampling_tensor.tolist()

    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        if i < input_metadata.num_prompts:
            # Generate the next tokens for a prompt input.
            assert len(seq_ids) == sampling_params.best_of
            logprob = logprobs[idx]

            # Sample the next tokens.
            next_token_ids = greedy[idx][:sampling_params.best_of] if sampling_params.temperature < _SAMPLING_EPS else sampling[idx-sampling_offsets[0]][:sampling_params.best_of]
            idx += 1

            # Get top-k log probabilities for the next tokens.
            next_logprobs = _get_topk_logprobs(logprob,
                                               sampling_params.logprobs)

            # Build the output.
            for seq_id, next_token_id in zip(seq_ids, next_token_ids):
                output_logprobs = next_logprobs.copy()
                output_logprobs[next_token_id] = logprob[next_token_id].item()
                seq_outputs[seq_id] = SequenceOutputs(seq_id, seq_id,
                                                      next_token_id,
                                                      output_logprobs)
        else:
            # Generate the next tokens for generation tokens.
            logprob = logprobs[idx:idx + len(seq_ids)]

            # Sample the next tokens.
            parent_seq_ids = seq_ids
            offset_idx = idx - sampling_offsets[0]
            next_token_ids = greedy[idx:idx + len(seq_ids)] if sampling_params.temperature < _SAMPLING_EPS else sampling[offset_idx:offset_idx + len(seq_ids)]
            next_token_ids = [item for sublist in next_token_ids for item in sublist]

            idx += len(seq_ids)

            # Get top-k log probabilities for the next tokens.
            next_logprobs: Dict[int, Dict[int, float]] = {}
            for j, seq_id in enumerate(seq_ids):
                next_logprobs[seq_id] = _get_topk_logprobs(
                    logprob[j], sampling_params.logprobs)

            # Build the output.
            for seq_id, parent_seq_id, next_token_id in zip(
                    seq_ids, parent_seq_ids, next_token_ids):
                j = seq_ids.index(parent_seq_id)
                output_logprobs = next_logprobs[parent_seq_id].copy()
                output_logprobs[next_token_id] = logprob[j,
                                                         next_token_id].item()
                seq_outputs[seq_id] = SequenceOutputs(
                    seq_id,
                    parent_seq_id,
                    next_token_id,
                    output_logprobs,
                )

    return seq_outputs
