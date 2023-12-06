from vllm.model_executor.layers.sampler import (
    _prune_hidden_states, 
    SamplingMetadata, 
    SamplerOutput, 
    _get_logits, 
    _apply_logits_processors, 
    _get_penalties,
    _get_temperatures,
    _get_top_p_top_k_min_p,
    _SAMPLING_EPS,
    _apply_top_p_top_k,
    _apply_min_p,
    _get_logprobs,
    _build_sampler_output,
    _apply_penalties,
    SamplingType,
    _random_sample,
    _beam_search_sample,
    _greedy_sample)
from typing import Optional, List, Tuple, Dict
import torch
from torch import nn

def ilql_sample(
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
    for sampling_type in SamplingType:
        seq_group_ids = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_ids]
        is_prompts = [i < sampling_metadata.num_prompts for i in seq_group_ids]
        sample_indices = categorized_sample_indices[sampling_type]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue
        if sampling_type == SamplingType.GREEDY:
            category_logprobs = logprobs[sample_indices]
            sample_results = _greedy_sample(seq_groups, category_logprobs)
        elif sampling_type == SamplingType.RANDOM:
            category_probs = probs[sample_indices]
            sample_results = _random_sample(seq_groups, is_prompts,
                                            category_probs)
        elif sampling_type == SamplingType.BEAM:
            category_logprobs = logprobs[sample_indices]
            sample_results = _beam_search_sample(seq_groups, is_prompts,
                                                 sampling_metadata.seq_data,
                                                 category_logprobs)
        else:
            raise ValueError("Only SamplingType.GREEDY is supported for LlamaIlqlForCausalLM")
        sample_results_dict.update(zip(seq_group_ids, sample_results))

    sample_results = [
        sample_results_dict[i]
        for i in range(len(sampling_metadata.seq_groups))
    ]
    return sample_results

class IlqlSampler(nn.Module):
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

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        embedding: torch.Tensor,
        logprob_bias: torch.Tensor, # beta * advantage term from π(a|h) ∝ πβ (a|h)exp[β(Q(h,a)−V (h))]
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> SamplerOutput:
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, sampling_metadata)

        # Get the logits for the next tokens.
        logits = _get_logits(hidden_states, embedding, embedding_bias,
                             self.vocab_size)

        # Apply logits processors (if any).
        logits = _apply_logits_processors(logits, sampling_metadata)
        logprob_bias = _apply_logits_processors(logprob_bias, sampling_metadata)
        # Apply presence and frequency penalties.
        presence_penalties, frequency_penalties, repetition_penalties = (
            _get_penalties(sampling_metadata))
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        assert len(repetition_penalties) == logits.shape[0]
        logits = _apply_penalties(logits, sampling_metadata,
                                  presence_penalties, frequency_penalties,
                                  repetition_penalties)
        logprob_bias = _apply_penalties(logprob_bias, sampling_metadata,
                                  presence_penalties, frequency_penalties,
                                  repetition_penalties)

        # Apply temperature scaling.
        temperatures = _get_temperatures(sampling_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures,
                             dtype=logits.dtype,
                             device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # Apply top-p and top-k truncation.
        top_ps, top_ks, min_ps = _get_top_p_top_k_min_p(
            sampling_metadata, self.vocab_size)
        assert len(top_ps) == len(top_ks) == logits.shape[0]
        do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
        do_top_k = any(k != self.vocab_size for k in top_ks)
        if do_top_p or do_top_k:
            logits = _apply_top_p_top_k(logits, top_ps, top_ks)
            logprob_bias = _apply_top_p_top_k(logprob_bias, top_ps, top_ks)

        do_min_p = any(mp > _SAMPLING_EPS for mp in min_ps)
        if do_min_p:
            logits = _apply_min_p(logits, min_ps)
            logprob_bias = _apply_min_p(logprob_bias, min_ps)

        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        logprobs.add_(logprob_bias)
        probs = torch.exp(logprobs)

        # Sample the next tokens.
        sample_results = ilql_sample(probs, logprobs, sampling_metadata)
        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results)
        return _build_sampler_output(sample_results, sampling_metadata,
                                     prompt_logprobs, sample_logprobs)
