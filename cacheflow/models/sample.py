from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from cacheflow.models import InputMetadata
from cacheflow.sampling_params import SamplingParams
from cacheflow.sequence import SequenceOutputs


class Sampler(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> Dict[int, SequenceOutputs]:
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, input_metadata)

        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())

        # Apply temperature scaling.
        temperatures = _get_temperatures(input_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(
                temperatures, dtype=logits.dtype, device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities (before applying top-p).
        logprobs = torch.log(probs)

        # Apply top-p truncation.
        top_ps = _get_top_ps(input_metadata)
        assert len(top_ps) == probs.shape[0]
        if any(p < 1.0 for p in top_ps):
            p = torch.tensor(top_ps, dtype=probs.dtype, device=probs.device)
            probs = _apply_top_p(probs, p)

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


def _get_temperatures(
    input_metadata: InputMetadata,
) -> List[float]:
    # Collect the temperatures for the logits.
    temperatures: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        temperature = sampling_params.temperature
        if temperature == 0.0:
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


def _get_top_ps(
    input_metadata: InputMetadata,
) -> List[float]:
    top_ps: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        if i < input_metadata.num_prompts:
            # A prompt input.
            top_ps.append(sampling_params.top_p)
        else:
            # A generation token.
            top_ps += [sampling_params.top_p] * len(seq_ids)
    return top_ps


def _apply_top_p(
    probs: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    # TODO(woosuk): Optimize.
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    probs = torch.gather(
        probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
    return probs


def _get_topk_logprobs(
    logprobs: torch.Tensor,
    num_logprobs: int,
) -> Dict[int, float]:
    if num_logprobs == 0:
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
        beam_width = sampling_params.n
        _, next_token_ids = torch.topk(prob, beam_width)
        next_token_ids = next_token_ids.tolist()
    elif sampling_params.temperature == 0.0:
        # Greedy sampling.
        assert sampling_params.n == 1
        next_token_id = torch.argmax(prob)
        next_token_ids = [next_token_id.item()]
    else:
        # Neucleus sampling.
        # Sample n tokens for the prompt.
        n = sampling_params.n
        next_token_ids = torch.multinomial(
            prob, num_samples=n, replacement=True)
        next_token_ids = next_token_ids.tolist()
    return next_token_ids


def _sample_from_generation_tokens(
    seq_ids: List[int],
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    seq_logprobs: List[float],
    sampling_params: SamplingParams,
) -> Tuple[List[int], List[int]]:
    # NOTE(woosuk): sampling_params.n can be greater than
    # len(seq_ids) because some sequences in the group might have
    # been already terminated.
    if sampling_params.use_beam_search:
        # Beam search.
        # Add cumulative logprobs for the sequences in the group.
        seq_logprobs = torch.tensor(
            seq_logprobs, dtype=torch.float, device=logprobs.device)
        logprobs = logprobs + seq_logprobs.unsqueeze(dim=1)

        vocab_size = logprobs.size(-1)
        beam_width = len(seq_ids)
        _, topk_ids = torch.topk(logprobs.flatten(), beam_width)
        seq_idx = torch.div(topk_ids, vocab_size, rounding_mode='floor').tolist()
        beam_seq_ids = [seq_ids[i] for i in seq_idx]
        token_ids = (topk_ids % vocab_size).tolist()

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
    elif sampling_params.temperature == 0.0:
        # Greedy sampling.
        assert len(seq_ids) == 1
        next_token_id = torch.argmax(probs, dim=-1)
        next_token_ids = [next_token_id.item()]
        parent_seq_ids = seq_ids
    else:
        # Neucleus sampling.
        # Sample 1 token for each sequence in the group.
        next_token_ids = torch.multinomial(
            probs, num_samples=1, replacement=True)
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
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        if i < input_metadata.num_prompts:
            # Generate the next tokens for a prompt input.
            assert len(seq_ids) == sampling_params.n
            prob = probs[idx]
            logprob = logprobs[idx]
            idx += 1

            # Sample the next tokens.
            next_token_ids = _sample_from_prompt(prob, sampling_params)
            # Get top-k log probabilities for the next tokens.
            next_logprobs = _get_topk_logprobs(
                logprob, sampling_params.num_logprobs)

            # Build the output.
            for seq_id, next_token_id in zip(seq_ids, next_token_ids):
                output_logprobs = next_logprobs.copy()
                output_logprobs[next_token_id] = logprob[next_token_id].item()
                seq_outputs[seq_id] = SequenceOutputs(
                    seq_id, seq_id, next_token_id, output_logprobs)
        else:
            # Generate the next tokens for generation tokens.
            prob = probs[idx:idx + len(seq_ids)]
            logprob = logprobs[idx:idx + len(seq_ids)]
            idx += len(seq_ids)

            # Sample the next tokens.
            seq_logprobs = [
                input_metadata.seq_logprobs[seq_id] for seq_id in seq_ids]
            parent_seq_ids, next_token_ids = _sample_from_generation_tokens(
                seq_ids, prob, logprob, seq_logprobs, sampling_params)

            # Get top-k log probabilities for the next tokens.
            next_logprobs: Dict[int, Dict[int, float]] = {}
            for i, seq_id in enumerate(seq_ids):
                next_logprobs[seq_id] = _get_topk_logprobs(
                    logprob[i], sampling_params.num_logprobs)

            # Build the output.
            for seq_id, parent_seq_id, next_token_id in zip(
                seq_ids, parent_seq_ids, next_token_ids):
                i = seq_ids.index(parent_seq_id)
                output_logprobs = next_logprobs[parent_seq_id].copy()
                output_logprobs[next_token_id] = logprob[i, next_token_id].item()
                seq_outputs[seq_id] = SequenceOutputs(
                    seq_id,
                    parent_seq_id,
                    next_token_id,
                    output_logprobs,
                )

    return seq_outputs
