from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData
from vllm.utils import in_wsl, is_neuron

_SAMPLING_EPS = 1e-5


class SamplingMetadata:
    """Metadata for input sequences. Used in sampler.

    Args:
        seq_groups: List of (seq_ids, sampling_params).
        seq_data: Seq_id -> SequenceData.
        prompt_lens: Lengths of prompts.
        selected_token_indices: Token indices selected for sampling.
        categorized_sample_indices: SamplingType -> token indices to sample.
        generators: List of torch.Generators to use for seeded sampling
        perform_sampling: Whether to perform sampling. This option is used to
            make the sampling only happens in the driver worker, and disable
            sampling in other worker processes.
    """

    def __init__(
        self,
        seq_groups: Optional[List[Tuple[List[int], SamplingParams]]],
        seq_data: Optional[Dict[int, SequenceData]],
        prompt_lens: Optional[List[int]],
        selected_token_indices: torch.Tensor,
        categorized_sample_indices: Optional[Dict[SamplingType, torch.Tensor]],
        generators: Optional[List[torch.Generator]] = None,
        perform_sampling: bool = True,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.selected_token_indices = selected_token_indices
        self.categorized_sample_indices = categorized_sample_indices
        self.generators = generators
        self.perform_sampling = perform_sampling

        self.num_prompts = len(prompt_lens) if prompt_lens is not None else 0

    def __repr__(self) -> str:
        return (
            "SamplingMetadata("
            f"seq_groups={self.seq_groups}, "
            f"seq_data={self.seq_data}, "
            f"prompt_lens={self.prompt_lens}, "
            f"selected_token_indices={self.selected_token_indices}, "
            f"categorized_sample_indices={self.categorized_sample_indices}), "
            f"perform_sampling={self.perform_sampling})")


@dataclass
class SamplingTensors:
    """Tensors for sampling."""

    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor
    presence_penalties: torch.Tensor
    frequency_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    prompt_tokens: torch.Tensor
    output_tokens: torch.Tensor

    @classmethod
    def from_sampling_metadata(
            cls, sampling_metadata: "SamplingMetadata", vocab_size: int,
            device: torch.device,
            dtype: torch.dtype) -> Tuple["SamplingTensors", bool, bool, bool]:
        prompt_tokens: List[List[int]] = []
        output_tokens: List[List[int]] = []
        top_ks: List[int] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        min_ps: List[float] = []
        presence_penalties: List[float] = []
        frequency_penalties: List[float] = []
        repetition_penalties: List[float] = []
        do_penalties = False
        do_top_p_top_k = False
        do_min_p = False
        for i, seq_group in enumerate(sampling_metadata.seq_groups):
            seq_ids, sampling_params = seq_group
            temperature = sampling_params.temperature
            p = sampling_params.presence_penalty
            f = sampling_params.frequency_penalty
            r = sampling_params.repetition_penalty
            top_p = sampling_params.top_p
            min_p = sampling_params.min_p
            # k should not be greater than the vocab size.
            top_k = min(sampling_params.top_k, vocab_size)
            top_k = vocab_size if top_k == -1 else top_k
            if temperature < _SAMPLING_EPS:
                # NOTE: Zero temperature means deterministic sampling
                # (i.e., greedy sampling or beam search).
                # Set the temperature to 1 to avoid division by zero.
                temperature = 1.0
            if not do_top_p_top_k and (top_p < 1.0 - _SAMPLING_EPS
                                       or top_k != vocab_size):
                do_top_p_top_k = True
            if not do_min_p and min_p > _SAMPLING_EPS:
                do_min_p = True
            if not do_penalties and (abs(p) >= _SAMPLING_EPS
                                     or abs(f) >= _SAMPLING_EPS
                                     or abs(r - 1.0) >= _SAMPLING_EPS):
                do_penalties = True
            if (i < sampling_metadata.num_prompts
                    and sampling_params.prompt_logprobs is not None):
                # For tokens in the prompt that we only need to get their logprobs
                prompt_len = sampling_metadata.prompt_lens[i]
                temperatures += [temperature] * (prompt_len - 1)
                top_ps += [top_p] * (prompt_len - 1)
                top_ks += [top_k] * (prompt_len - 1)
                min_ps += [min_p] * (prompt_len - 1)
                presence_penalties += [0] * (prompt_len - 1)
                frequency_penalties += [0] * (prompt_len - 1)
                repetition_penalties += [1] * (prompt_len - 1)
                prompt_tokens.extend([] for _ in range(prompt_len - 1))
                output_tokens.extend([] for _ in range(prompt_len - 1))
            for seq_id in seq_ids:
                seq_data = sampling_metadata.seq_data[seq_id]
                prompt_tokens.append(seq_data.prompt_token_ids)
                output_tokens.append(seq_data.output_token_ids)
            temperatures += [temperature] * len(seq_ids)
            top_ps += [top_p] * len(seq_ids)
            top_ks += [top_k] * len(seq_ids)
            min_ps += [min_p] * len(seq_ids)
            presence_penalties += [p] * len(seq_ids)
            frequency_penalties += [f] * len(seq_ids)
            repetition_penalties += [r] * len(seq_ids)

        sampling_tensors = SamplingTensors.from_lists(
            temperatures, top_ps, top_ks, min_ps, presence_penalties,
            frequency_penalties, repetition_penalties, prompt_tokens,
            output_tokens, vocab_size, device, dtype)
        return (sampling_tensors, do_penalties, do_top_p_top_k, do_min_p)

    @classmethod
    def from_lists(cls, temperatures: List[float], top_ps: List[float],
                   top_ks: List[int], min_ps: List[float],
                   presence_penalties: List[float],
                   frequency_penalties: List[float],
                   repetition_penalties: List[float],
                   prompt_tokens: List[List[int]],
                   output_tokens: List[List[int]], vocab_size: int,
                   device: torch.device,
                   dtype: torch.dtype) -> "SamplingTensors":
        # Note that the performance will be very bad without
        # pinned memory.
        pin_memory = not in_wsl() and not is_neuron()
        prompt_max_len = max(len(tokens) for tokens in prompt_tokens)
        prompt_padded_tokens = [
            tokens + [vocab_size] * (prompt_max_len - len(tokens))
            for tokens in prompt_tokens
        ]
        output_max_len = max(len(tokens) for tokens in output_tokens)
        output_padded_tokens = [
            tokens + [vocab_size] * (output_max_len - len(tokens))
            for tokens in output_tokens
        ]

        temperatures_t = torch.tensor(
            temperatures,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        top_ps_t = torch.tensor(
            top_ps,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        min_ps_t = torch.tensor(
            min_ps,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        presence_penalties_t = torch.tensor(
            presence_penalties,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        frequency_penalties_t = torch.tensor(
            frequency_penalties,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        repetition_penalties_t = torch.tensor(
            repetition_penalties,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        top_ks_t = torch.tensor(
            top_ks,
            device="cpu",
            dtype=torch.int,
            pin_memory=pin_memory,
        )
        prompt_tensor = torch.tensor(
            prompt_padded_tokens,
            device="cpu",
            dtype=torch.long,
            pin_memory=pin_memory,
        )
        output_tensor = torch.tensor(
            output_padded_tokens,
            device="cpu",
            dtype=torch.long,
            pin_memory=pin_memory,
        )
        # Because the memory is pinned, we can do non-blocking
        # transfer to device.
        return cls(
            temperatures=temperatures_t.to(device=device, non_blocking=True),
            top_ps=top_ps_t.to(device=device, non_blocking=True),
            top_ks=top_ks_t.to(device=device, non_blocking=True),
            min_ps=min_ps_t.to(device=device, non_blocking=True),
            presence_penalties=presence_penalties_t.to(device=device,
                                                       non_blocking=True),
            frequency_penalties=frequency_penalties_t.to(device=device,
                                                         non_blocking=True),
            repetition_penalties=repetition_penalties_t.to(device=device,
                                                           non_blocking=True),
            prompt_tokens=prompt_tensor.to(device=device, non_blocking=True),
            output_tokens=output_tensor.to(device=device, non_blocking=True),
        )
