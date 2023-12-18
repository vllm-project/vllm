"""A layer that samples the next tokens from the model's outputs."""
import math
import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import msgspec
import ray
import numpy as np
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import torch
import torch.jit
import torch.nn as nn

from vllm.anyscale.shm.msgspec_shm import RayEvent, SharedMsgspecBufferWithEvent, SharedMemoryManager
from vllm.anyscale.shm.numpy import numpy_encode_hook, numpy_ext_hook
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.sampler_ops.penalty_triton import apply_penalty
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_gather)
from vllm.sampling_params import SamplingType
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SamplerOutput,
                           SequenceGroupOutputs, SequenceOutputs)
from vllm.utils import in_wsl

_SAMPLING_EPS = 1e-5
SHARED_MEMORY_BUFFER_SIZE = int(2e+7)  # 20 MB
logger = logging.getLogger(__name__)


@dataclass
class SamplingTokenTensors:
    """Datastructure used to encode sampling inputs in GPU tensors.

    This enables a non-blocking sampler.
    """
    unique_output_token_ids: torch.Tensor
    output_token_counts: torch.Tensor
    sample_indices: torch.Tensor
    prompt_indices: torch.Tensor
    categorized_sample_indices: Tuple[torch.Tensor, torch.Tensor]
    cumsum_penalties_seq_lens: torch.Tensor
    max_penalties_seq_len: int

    @classmethod
    def from_lists(
        cls,
        unique_output_token_ids: List[int],
        output_token_counts: List[int],
        penalties_seq_lens: List[int],
        sample_indices: List[int],
        prompt_indices: List[int],
        categorized_sample_indices: Dict[SamplingType, List[int]],
        device: torch.device,
        vocab_size: int  # pylint: disable=unused-argument
    ) -> "SamplingTokenTensors":
        # WSL doesn't support pinned memory.
        # Note that the performance will be very bad without
        # pinned memory.
        pin_memory = not in_wsl()

        max_penalties_seq_len = max(penalties_seq_lens)
        # Must have length of batch_size+1 for cumsum used by triton
        # penalty kernel
        # Represents the number of unique token ids in output for
        # each sequence
        penalties_seq_lens = [0] + penalties_seq_lens

        sampling_categories = (SamplingType.GREEDY, SamplingType.RANDOM)
        indicies_list = sample_indices + prompt_indices
        offset = len(indicies_list)

        for indicies in categorized_sample_indices.values():
            indicies_list.extend(indicies)

        output_count_int_tensor = torch.tensor(
            [unique_output_token_ids, output_token_counts],
            device="cpu",
            dtype=torch.int,
            pin_memory=pin_memory,
        )
        penalties_seq_len_tensor = torch.tensor(
            penalties_seq_lens,
            device="cpu",
            dtype=torch.int,
            pin_memory=pin_memory,
        )
        indices_tensor = torch.tensor(
            indicies_list,
            device="cpu",
            dtype=torch.long,
            pin_memory=pin_memory,
        )
        output_count_int_tensor_gpu = output_count_int_tensor.to(
            device=device, non_blocking=True)
        penalties_seq_len_tensor_gpu = penalties_seq_len_tensor.to(
            device=device, non_blocking=True)
        indices_tensor_gpu = indices_tensor.to(device=device,
                                               non_blocking=True)
        categorized_sample_indices_tensors = [None] * len(sampling_categories)
        for category in sampling_categories:
            sample_indices_len = len(
                categorized_sample_indices.get(category, []))
            categorized_sample_indices_tensors[category] = indices_tensor_gpu[
                offset:offset + sample_indices_len]
            offset += sample_indices_len

        return cls(
            sample_indices=indices_tensor_gpu[:len(sample_indices)],
            prompt_indices=indices_tensor_gpu[len(sample_indices
                                                  ):len(sample_indices) +
                                              len(prompt_indices)],
            categorized_sample_indices=tuple(
                categorized_sample_indices_tensors),
            unique_output_token_ids=output_count_int_tensor_gpu[0],
            output_token_counts=output_count_int_tensor_gpu[1],
            cumsum_penalties_seq_lens=penalties_seq_len_tensor_gpu.cumsum(0),
            max_penalties_seq_len=max_penalties_seq_len)

    @classmethod
    def from_input_metadata(cls, input_metadata: InputMetadata,
                            vocab_size: int, device: torch.device):
        unique_output_token_ids: List[int] = []
        output_token_counts: List[int] = []
        penalties_seq_lens: List[int] = []
        sample_indices: List[int] = []
        prompt_indices: List[int] = []
        categorized_sample_indices: Dict[SamplingType,
                                         List[int]] = defaultdict(list)

        sample_indices_start_idx = 0
        categorized_indices_start_idx = 0

        for i, seq_group in enumerate(input_metadata.seq_groups):
            seq_ids, sampling_params = seq_group

            is_prompt = i < input_metadata.num_prompts
            if is_prompt:
                prompt_len = input_metadata.prompt_lens[i]

                if sampling_params.prompt_logprobs is not None:
                    prompt_indices.extend(
                        range(sample_indices_start_idx,
                              sample_indices_start_idx + prompt_len - 1))
                    # NOTE: prompt token positions do not need sample, skip
                    sample_indices_start_idx += prompt_len - 1

                categorized_sample_indices[
                    sampling_params.sampling_type].append(
                        categorized_indices_start_idx)
                sample_indices.append(sample_indices_start_idx)
                sample_indices_start_idx += 1
                categorized_indices_start_idx += 1

            for seq_id in seq_ids:
                if sampling_params.has_penalties:
                    seq_data = input_metadata.seq_data[seq_id]
                    id_to_counts = seq_data.output_token_id_count
                    for token_id, count in id_to_counts.items():
                        unique_output_token_ids.append(token_id)
                        output_token_counts.append(count)
                    penalties_seq_lens.append(len(id_to_counts))
                else:
                    penalties_seq_lens.append(0)

                if not is_prompt:
                    categorized_sample_indices[
                        sampling_params.sampling_type].append(
                            categorized_indices_start_idx)
                    sample_indices.append(sample_indices_start_idx)
                    sample_indices_start_idx += 1
                    categorized_indices_start_idx += 1

        return cls.from_lists(unique_output_token_ids, output_token_counts,
                              penalties_seq_lens, sample_indices,
                              prompt_indices, categorized_sample_indices,
                              device, vocab_size)


@dataclass
class SamplingParametersTensors:
    """Datastructure used to encode sampling inputs in GPU tensors.

    This enables a non-blocking sampler.
    """
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor
    presence_penalties: torch.Tensor
    frequency_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    max_top_k: int
    max_prompt_best_of: int
    do_penalties: bool
    do_top_p_top_k: bool
    do_min_p: bool
    largest_num_logprobs: int

    @classmethod
    def from_lists(cls, temperatures: List[float], top_ps: List[float],
                   top_ks: List[int], min_ps: List[float],
                   presence_penalties: List[float],
                   frequency_penalties: List[float],
                   repetition_penalties: List[float],
                   prompt_best_of: List[int], do_penalties: bool,
                   do_top_p_top_k: bool, do_min_p: bool,
                   largest_num_logprobs: int, device: torch.device,
                   dtype: torch.dtype) -> "SamplingParametersTensors":

        # WSL doesn't support pinned memory.
        # Note that the performance will be very bad without
        # pinned memory.
        pin_memory = not in_wsl()

        max_top_k = max(top_ks)
        max_prompt_best_of = max(prompt_best_of) if prompt_best_of else 1

        float_tensor = torch.tensor(
            [
                temperatures, top_ps, min_ps, presence_penalties,
                frequency_penalties, repetition_penalties
            ],
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        int_tensor = torch.tensor(
            [top_ks],
            device="cpu",
            dtype=torch.int,
            pin_memory=pin_memory,
        )
        float_tensor_gpu = float_tensor.to(device=device, non_blocking=True)
        int_tensor_gpu = int_tensor.to(device=device, non_blocking=True)

        return cls(temperatures=float_tensor_gpu[0],
                   top_ps=float_tensor_gpu[1],
                   top_ks=int_tensor_gpu[0],
                   min_ps=float_tensor_gpu[2],
                   presence_penalties=float_tensor_gpu[3],
                   frequency_penalties=float_tensor_gpu[4],
                   repetition_penalties=float_tensor_gpu[5],
                   max_top_k=max_top_k,
                   max_prompt_best_of=max_prompt_best_of,
                   do_penalties=do_penalties,
                   do_top_p_top_k=do_top_p_top_k,
                   do_min_p=do_min_p,
                   largest_num_logprobs=largest_num_logprobs)

    @classmethod
    def from_input_metadata(cls, input_metadata: InputMetadata,
                            vocab_size: int, device: torch.device,
                            dtype: torch.dtype):
        top_ks: List[int] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        min_ps: List[float] = []
        presence_penalties: List[float] = []
        frequency_penalties: List[float] = []
        repetition_penalties: List[float] = []
        prompt_best_of: List[int] = []

        do_penalties = False
        do_top_p_top_k = False
        do_min_p = False
        largest_num_logprobs = 0

        for i, seq_group in enumerate(input_metadata.seq_groups):
            seq_ids, sampling_params = seq_group
            is_prompt = i < input_metadata.num_prompts

            temperature = sampling_params.temperature
            # k should not be greater than the vocab size.
            top_k = min(sampling_params.top_k, vocab_size)
            top_k = vocab_size if top_k == -1 else top_k

            if temperature < _SAMPLING_EPS:
                # NOTE: Zero temperature means deterministic sampling
                # (i.e., greedy sampling or beam search).
                # Set the temperature to 1 to avoid division by zero.
                temperature = 1.0
            if not do_top_p_top_k and (
                    sampling_params.top_p < 1.0 - _SAMPLING_EPS
                    or top_k != vocab_size):
                do_top_p_top_k = True
            if not do_min_p and sampling_params.min_p > _SAMPLING_EPS:
                do_min_p = True
            if not do_penalties and sampling_params.has_penalties:
                do_penalties = True

            if is_prompt:
                prompt_best_of.append(sampling_params.actual_best_of)
                if sampling_params.prompt_logprobs is not None:
                    prompt_len = input_metadata.prompt_lens[i]
                    temperatures += [temperature] * (prompt_len - 1)
                    top_ps += [sampling_params.top_p] * (prompt_len - 1)
                    top_ks += [top_k] * (prompt_len - 1)
                    min_ps += [sampling_params.min_p] * (prompt_len - 1)
                    presence_penalties += [0] * (prompt_len - 1)
                    frequency_penalties += [0] * (prompt_len - 1)
                    repetition_penalties += [1] * (prompt_len - 1)

            top_ks += [top_k] * len(seq_ids)
            temperatures += [temperature] * len(seq_ids)
            top_ps += [sampling_params.top_p] * len(seq_ids)
            min_ps += [sampling_params.min_p] * len(seq_ids)
            presence_penalties += [sampling_params.presence_penalty
                                   ] * len(seq_ids)
            frequency_penalties += [sampling_params.frequency_penalty
                                    ] * len(seq_ids)
            repetition_penalties += [sampling_params.repetition_penalty
                                     ] * len(seq_ids)
            if sampling_params.logprobs:
                largest_num_logprobs = max(largest_num_logprobs,
                                           sampling_params.logprobs)

        return cls.from_lists(temperatures, top_ps, top_ks, min_ps,
                              presence_penalties, frequency_penalties,
                              repetition_penalties, prompt_best_of,
                              do_penalties, do_top_p_top_k, do_min_p,
                              largest_num_logprobs, device, dtype)


@dataclass
class RawSamplerOutput:
    """Class containing sampler output stored in torch tensors.
    """
    sampled_tokens: torch.Tensor
    sampled_logprobs: torch.Tensor
    prompt_logprobs: torch.Tensor
    probs: torch.Tensor
    sampling_parameters_tensors: "SamplingParametersTensors"
    sampling_token_tensors: "SamplingTokenTensors"
    top_logprobs: Optional[torch.Tensor]
    top_token_ids: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]


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

    _copy_stream: Optional[torch.cuda.Stream] = None

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: Optional[int] = None,
        include_gpu_probs_tensor: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.org_vocab_size = org_vocab_size or vocab_size
        self.include_gpu_probs_tensor = include_gpu_probs_tensor

    def __del__(self):
        if getattr(self, "_shared_mem_manager", None) is not None:
            self._shared_mem_manager.shutdown()

    def _get_logits(
            self,
            embedding: torch.Tensor,
            hidden_states: torch.Tensor,
            embedding_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Get the logits for the next tokens.
        logits = torch.nn.functional.linear(hidden_states, embedding,
                                            embedding_bias)
        logits = tensor_model_parallel_all_gather(logits)
        # Remove paddings in vocab (if any).
        logits = logits[:, :self.org_vocab_size]
        return logits

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
        sampling_parameters_tensors: Optional[
            SamplingParametersTensors] = None,
        sampling_token_tensors: Optional[SamplingTokenTensors] = None,
    ) -> RawSamplerOutput:
        # Get logits for entire sequence before pruning hidden states
        # for model quality evaluation.
        batched_seq_logits = None
        if input_metadata.return_logits:
            batched_seq_logits = self._get_logits(
                embedding, hidden_states, embedding_bias).to(torch.float)

        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, input_metadata)

        # Get the logits for the next tokens.
        logits = self._get_logits(embedding, hidden_states, embedding_bias)

        _, vocab_size = logits.shape

        # Prepare sampling tensors in another stream to overlap
        # CPU<->GPU data transfer with GPU computation in forward pass.
        if not sampling_parameters_tensors or not sampling_token_tensors:
            if Sampler._copy_stream is None:
                # Initialize stream here once to make sure it uses the
                # correct device.
                Sampler._copy_stream = torch.cuda.Stream()
            with torch.cuda.stream(Sampler._copy_stream):
                if not sampling_parameters_tensors:
                    sampling_parameters_tensors = (
                        SamplingParametersTensors.from_input_metadata(
                            input_metadata, vocab_size, logits.device,
                            logits.dtype))
                if not sampling_token_tensors:
                    sampling_token_tensors = (
                        SamplingTokenTensors.from_input_metadata(
                            input_metadata, vocab_size, logits.device))

            torch.cuda.current_stream().wait_stream(Sampler._copy_stream)

        # Apply presence and frequency penalties.
        if sampling_parameters_tensors.do_penalties:
            logits = _apply_penalties_triton(
                logits, sampling_token_tensors.unique_output_token_ids,
                sampling_token_tensors.output_token_counts,
                sampling_token_tensors.cumsum_penalties_seq_lens,
                sampling_token_tensors.max_penalties_seq_len,
                sampling_parameters_tensors.presence_penalties,
                sampling_parameters_tensors.frequency_penalties,
                sampling_parameters_tensors.repetition_penalties)

        # Use in-place division to avoid creating a new tensor.
        logits.div_(sampling_parameters_tensors.temperatures.unsqueeze(dim=1))

        # Apply top-p and top-k truncation.
        if sampling_parameters_tensors.do_top_p_top_k:
            logits = _apply_top_p_top_k(logits,
                                        sampling_parameters_tensors.top_ps,
                                        sampling_parameters_tensors.top_ks)

        if sampling_parameters_tensors.do_min_p:
            logits = _apply_min_p(logits, sampling_parameters_tensors.min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        if sampling_parameters_tensors.largest_num_logprobs > 0:
            top_logprobs, top_token_ids = torch.topk(
                logprobs,
                sampling_parameters_tensors.largest_num_logprobs,
                dim=-1)
        else:
            top_logprobs, top_token_ids = None, None

        # Sample the next tokens.
        sampled_tokens, sampled_logprobs, prompt_logprobs = _sample(
            probs=probs,
            logprobs=logprobs,
            prompt_token_indices=sampling_token_tensors.prompt_indices,
            sample_indices=sampling_token_tensors.sample_indices,
            categorized_sample_indices=sampling_token_tensors.
            categorized_sample_indices,
            max_best_of=sampling_parameters_tensors.max_prompt_best_of,
            modify_greedy_probs=self.include_gpu_probs_tensor,
        )

        return RawSamplerOutput(sampled_tokens, sampled_logprobs,
                                prompt_logprobs, probs,
                                sampling_parameters_tensors,
                                sampling_token_tensors, top_logprobs,
                                top_token_ids, batched_seq_logits)


def _flatten_list(lst):
    return [item for sublist in lst for item in sublist]


def pythonize_sampler_output(raw_sampler_output: RawSamplerOutput,
                             input_metadata: InputMetadata) -> SamplerOutput:
    """Convert sampling output stored in PyTorch tensors to sampling output
    stored in Python datastructures.

    This blocks the CPU until the GPU catches up, so should only be used when
    necessary.
    """
    # GPU<->CPU sync happens below.

    samples = raw_sampler_output.sampled_tokens.tolist()
    logprobs = raw_sampler_output.sampled_logprobs.tolist()
    prompt_logprobs = raw_sampler_output.prompt_logprobs.tolist()
    if raw_sampler_output.top_logprobs is not None:
        top_logprobs = raw_sampler_output.top_logprobs.tolist()
        top_token_ids = raw_sampler_output.top_token_ids.tolist()
    sample_idx = 0
    prompt_logprobs_idx = 0
    top_logprob_idx = 0
    sampler_output = []

    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        is_prompt = i < input_metadata.num_prompts
        num_parent_seqs = len(seq_ids)
        if sampling_params.sampling_type == SamplingType.GREEDY:
            assert num_parent_seqs == 1, (
                "Greedy sampling should have only one seq.")
            parent_ids = list(range(num_parent_seqs))
            token_ids = samples[sample_idx][0:1]
            seq_logprobs = logprobs[sample_idx][0:1]
            offset = 1
        elif is_prompt:
            actual_best_of = sampling_params.actual_best_of
            parent_ids = [0] * actual_best_of
            token_ids = samples[sample_idx][:actual_best_of]
            seq_logprobs = logprobs[sample_idx][:actual_best_of]
            offset = 1
        else:
            parent_ids = list(range(num_parent_seqs))
            token_ids = _flatten_list(samples[sample_idx:sample_idx +
                                              num_parent_seqs])
            seq_logprobs = _flatten_list(logprobs[sample_idx:sample_idx +
                                                  num_parent_seqs])
            offset = num_parent_seqs

        if is_prompt and sampling_params.prompt_logprobs is not None:
            group_prompt_logprobs: PromptLogprobs = [None]
            prompt_tokens = input_metadata.seq_data[
                seq_ids[0]].get_prompt_token_ids()
            for token_id in prompt_tokens[1:]:
                prompt_logprobs_dict = {
                    token_id: prompt_logprobs[prompt_logprobs_idx][token_id]
                }
                if sampling_params.prompt_logprobs > 0:
                    prompt_logprobs_dict.update(
                        zip(
                            top_token_ids[top_logprob_idx]
                            [:sampling_params.prompt_logprobs],
                            top_logprobs[top_logprob_idx]
                            [:sampling_params.prompt_logprobs]))
                group_prompt_logprobs.append(prompt_logprobs_dict)
                top_logprob_idx += 1
                prompt_logprobs_idx += 1
        else:
            group_prompt_logprobs = None

        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = 0
        group_sample_logprobs: SampleLogprobs = []
        for next_token_id, logprob, parent_id in zip(token_ids, seq_logprobs,
                                                     parent_ids):
            sample_logprobs_dict = {next_token_id: logprob}
            if num_logprobs > 0:
                sample_logprobs_dict.update(
                    zip(
                        top_token_ids[top_logprob_idx +
                                      parent_id][:num_logprobs],
                        top_logprobs[top_logprob_idx +
                                     parent_id][:num_logprobs]))
            group_sample_logprobs.append(sample_logprobs_dict)

        sample_idx += offset
        top_logprob_idx += offset
        sampler_output.append(
            SequenceGroupOutputs([
                SequenceOutputs(seq_ids[parent_id], token_id, seq_logprobs)
                for parent_id, token_id, seq_logprobs in zip(
                    parent_ids, token_ids, group_sample_logprobs)
            ], group_prompt_logprobs))

    return SamplerOutput(sampler_output)


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    input_metadata: InputMetadata,
) -> torch.Tensor:
    return hidden_states.index_select(0, input_metadata.selected_token_indices)


@torch.jit.script
def _get_bin_counts_and_mask(
    tokens_tensor: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=tokens_tensor.device)
    bin_counts.scatter_add_(1, tokens_tensor, torch.ones_like(tokens_tensor))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


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
    repetition_penalties[prompt_mask.logical_or_(
        output_mask).logical_not_()] = 1.0
    logits = torch.where(logits > 0, logits / repetition_penalties,
                         logits * repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def _apply_penalties_triton(
        logits: torch.Tensor, unique_output_token_ids: torch.Tensor,
        output_token_counts: torch.Tensor,
        cumsum_penalties_seq_lens: torch.Tensor, max_penalties_seq_len: int,
        presence_penalties: torch.Tensor, frequency_penalties: torch.Tensor,
        repetition_penalties: torch.Tensor) -> torch.Tensor:
    apply_penalty(logits, presence_penalties, frequency_penalties,
                  repetition_penalties, unique_output_token_ids,
                  output_token_counts, cumsum_penalties_seq_lens,
                  max_penalties_seq_len)
    return logits


@torch.jit.script
def _apply_top_p_top_k(
    logits: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)

    p = p.unsqueeze(dim=1)
    k = k.unsqueeze(dim=1)
    # Final mask.
    mask = ((probs_sum - probs_sort) > p) | (top_k_mask >= k)
    logits_sort.masked_fill_(mask, -float("inf"))

    # Re-sort the probabilities.
    src = torch.arange(logits_idx.shape[-1],
                       device=logits_idx.device).expand_as(logits_idx)
    logits_idx_inv = torch.empty_like(logits_idx).scatter_(dim=-1,
                                                           index=logits_idx,
                                                           src=src)
    logits = torch.gather(logits_sort, dim=-1, index=logits_idx_inv)
    return logits


@torch.jit.script
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
    min_p = min_p.unsqueeze(dim=1)
    tokens_to_remove = probs < (min_p * top_probs)
    logits = logits.masked_fill_(tokens_to_remove, -float("inf"))

    return logits


# def _beam_search_sample(
#     selected_seq_groups: List[Tuple[List[int], SamplingParams]],
#     is_prompts: List[bool],
#     seq_data: Dict[int, SequenceData],
#     logprobs: torch.Tensor,
# ) -> List[Tuple[List[int], List[int]]]:
#     # We sample 2 * beam_width candidates to make sure that with high
#     # probability we can get `beam_width` candidates in addition to
#     # the finished sequences for the next iteration. See
#     # https://github.com/tensorflow/tensor2tensor/blob/bafdc1
# b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/utils/beam_search.py
# #L557-L563
#     # for details. See also HF reference:
#     # https://github.com/huggingface/transformers/blob/a4dd53d8
# 8e4852f023332d284ff07a01afcd5681/src/transformers/generation/utils.py
# #L3063-L3065
#     #
#     # NOTE: Beam search is not vectorized, so its speed can be slower than
#     # other sampling methods.
#     sample_idx = 0
#     results = []
#     for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
#         seq_ids, sampling_params = seq_group
#         num_parent_seqs = len(seq_ids)
#         beam_width = sampling_params.actual_best_of
#         seq_group_logprobs = logprobs[sample_idx:sample_idx + num_parent_seqs]
#         if is_prompt:
#             # Prompt phase.
#             assert num_parent_seqs == 1, (
#                 "Prompt input should have only one seq.")
#             parent_ids = [0] * (2 * beam_width)
#             _, next_token_ids = torch.topk(seq_group_logprobs[0],
#                                            2 * beam_width)
#             next_token_ids = next_token_ids.tolist()
#         else:
#             # Generation phase.
#             cumulative_logprobs = [
#                 seq_data[seq_id].cumulative_logprob for seq_id in seq_ids
#             ]
#             cumulative_logprobs = torch.tensor(
#                 cumulative_logprobs,
#                 dtype=torch.float,
#                 device=seq_group_logprobs.device)
#             seq_group_logprobs = (seq_group_logprobs +
#                                   cumulative_logprobs.unsqueeze(dim=1))
#             _, topk_ids = torch.topk(seq_group_logprobs.flatten(),
#                                      2 * beam_width)
#             topk_ids = topk_ids.tolist()
#             vocab_size = seq_group_logprobs.size(-1)
#             parent_ids = [i // vocab_size for i in topk_ids]
#             next_token_ids = [i % vocab_size for i in topk_ids]
#         results.append((next_token_ids, parent_ids))
#         sample_idx += num_parent_seqs
#     assert sample_idx == logprobs.size(0)
#     return results


@torch.jit.script
def _modify_greedy_probs(probs: torch.Tensor, sample_indices: torch.Tensor,
                         sampled_tokens: torch.Tensor) -> None:
    """Set the probability of the sampled token to 1, all other tokens to zero.
    This is used in speculative decoding where the sampling method must be
    encoded within the sampled probability distributions.
    """
    sample_indices = sample_indices.to(torch.long)
    probs.index_fill_(0, sample_indices, 0)
    probs.flatten().index_fill_(0, (sample_indices * probs.stride()[0]) +
                                sampled_tokens, 1)


@torch.jit.script
def _sample_tensor(
    probs: torch.Tensor,
    num_samples: int,
    random_sample_indices: torch.Tensor,
) -> torch.Tensor:
    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        probs = probs[:, None, :].expand(probs.shape[0], num_samples,
                                         probs.shape[1]).contiguous()
    else:
        probs = probs.view(probs.shape[0], num_samples, probs.shape[1])

    has_random_sample_indices = bool(random_sample_indices.numel())
    if has_random_sample_indices:
        random_sample_probs = probs[random_sample_indices]
        random_sample_probs_v = random_sample_probs.view(
            random_sample_probs.shape[0] * num_samples,
            random_sample_probs.shape[-1])
        q = torch.empty_like(random_sample_probs_v).exponential_(1.0).pow_(-1)
        probs.index_reduce_(0, random_sample_indices,
                            q.view_as(random_sample_probs), "prod")
    return probs.view(probs.shape[0] * num_samples,
                      -1).argmax(dim=1).view(-1, num_samples)


@torch.jit.script
def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    prompt_token_indices: torch.Tensor,
    sample_indices: torch.Tensor,
    categorized_sample_indices: Tuple[torch.Tensor, torch.Tensor],
    max_best_of: int,
    modify_greedy_probs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    prompt_logprobs = logprobs[prompt_token_indices]
    sample_probs = probs[sample_indices]

    sampled_tokens = _sample_tensor(sample_probs, max_best_of,
                                    categorized_sample_indices[1])

    has_greedy_indices = bool(categorized_sample_indices[0].numel())
    if modify_greedy_probs and has_greedy_indices:
        # Note: in greedy sampling, there only one sample per sequence
        # group.
        greedy_sample_indices = sample_indices[categorized_sample_indices[0]]
        _modify_greedy_probs(
            probs, greedy_sample_indices,
            sampled_tokens[categorized_sample_indices[0]][:, 0])

    sampled_logprobs = torch.gather(logprobs[sample_indices], 1,
                                    sampled_tokens)
    return sampled_tokens, sampled_logprobs, prompt_logprobs
