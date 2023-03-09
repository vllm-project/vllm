from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from cacheflow.models import InputMetadata


class Sampler(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def _compute_prob(
        self,
        logits: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # Collect the temperatures for the logits.
        temperatures: List[float] = []
        for i, seq_group in enumerate(input_metadata.seq_groups):
            _, seq_ids, sampling_params = seq_group
            temperature = sampling_params.temperature
            if temperature == 0.0:
                # NOTE: Zero temperature means deterministic sampling
                # (i.e., greedy sampling or beam search).
                # Set the temperature to 1 to avoid division by zero.
                temperature = 1.0

            if i < input_metadata.num_prompts:
                # Logits for a prompt input.
                temperatures.append(temperature)
            else:
                # Logits for a generation token.
                temperatures += [temperature] * len(seq_ids)
        assert len(temperatures) == logits.shape[0]

        if all(t == 1.0 for t in temperatures):
            return torch.softmax(logits, dim=-1)

        t = torch.tensor(temperatures, device=logits.device)
        probs = torch.softmax(logits / t.unsqueeze(dim=1), dim=-1)
        return probs

    def _apply_top_p(
        self,
        probs: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # Collect the top-p thresholds for the logits.
        top_ps: List[float] = []
        for i, seq_group in enumerate(input_metadata.seq_groups):
            _, seq_ids, sampling_params = seq_group
            if i < input_metadata.num_prompts:
                # Logits for a prompt input.
                top_ps.append(sampling_params.top_p)
            else:
                # Logits for a generation token.
                top_ps += [sampling_params.top_p] * len(seq_ids)
        assert len(top_ps) == probs.shape[0]

        if all(p == 1.0 for p in top_ps):
            return probs
        p = torch.tensor(top_ps, device=probs.device)

        # Apply top-p.
        # TODO(woosuk): Optimize the following with a custom op.
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(
            probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
        return probs

    def _sample(
        self,
        probs: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> Dict[int, Tuple[int, int]]:
        seq_outputs: Dict[int, Tuple[int, int]] = {}

        # TODO(woosuk): Optimize the following with a custom op.
        idx = 0
        for i, seq_group in enumerate(input_metadata.seq_groups):
            _, seq_ids, sampling_params = seq_group
            if i < input_metadata.num_prompts:
                # Generate the next tokens for a prompt input.
                assert len(seq_ids) == sampling_params.n
                prob = probs[idx]
                idx += 1

                if sampling_params.use_beam_search:
                    # Beam search.
                    beam_width = len(seq_ids)
                    _, next_token_ids = torch.topk(prob, beam_width)
                    next_token_ids = next_token_ids.tolist()
                elif sampling_params.temperature == 0.0:
                    # Greedy sampling.
                    assert len(seq_ids) == 1
                    next_token_id = torch.argmax(prob)
                    next_token_ids = [next_token_id.item()]
                else:
                    # Neucleus sampling.
                    n = len(seq_ids)
                    next_token_ids = torch.multinomial(prob, num_samples=n)
                    next_token_ids = next_token_ids.tolist()

                # Build the output.
                for seq_id, next_token_id in zip(seq_ids, next_token_ids):
                    seq_outputs[seq_id] = (seq_id, next_token_id)
            else:
                # Generate the next tokens for generation tokens.
                prob = probs[idx:idx + len(seq_ids)]
                idx += len(seq_ids)

                # NOTE(woosuk): sampling_params.n can be greater than
                # len(seq_ids) because some sequences may have been terminated.
                if sampling_params.use_beam_search:
                    # Beam search.
                    beam_width = len(seq_ids)
                    # FIXME
                    raise NotImplementedError()
                else:
                    if sampling_params.temperature == 0.0:
                        # Greedy sampling.
                        assert len(seq_ids) == 1
                        next_token_id = torch.argmax(prob, dim=-1)
                        next_token_ids = [next_token_id.item()]
                    else:
                        # Neucleus sampling.
                        # Sample 1 token for each sequence.
                        next_tokens = torch.multinomial(prob, num_samples=1)
                        next_token_ids = next_tokens.squeeze(dim=-1).tolist()

                    # Build the output.
                    for seq_id, next_token_id in zip(seq_ids, next_token_ids):
                        seq_outputs[seq_id] = (seq_id, next_token_id)

        return seq_outputs

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> Dict[int, Tuple[int, int]]:
        # Get the hidden states of the last tokens.
        start_idx = 0
        last_token_indicies: List[int] = []
        for prompt_len in input_metadata.prompt_lens:
            last_token_indicies.append(start_idx + prompt_len - 1)
            start_idx += prompt_len
        last_token_indicies.extend(
            range(start_idx, start_idx + input_metadata.num_generation_tokens))
        hidden_states = hidden_states[last_token_indicies]

        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())

        # Sample the next tokens.
        probs = self._compute_prob(logits, input_metadata)
        probs = self._apply_top_p(probs, input_metadata)
        output = self._sample(probs, input_metadata)
        return output
