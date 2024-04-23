import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from vllm.model_executor.layers.ops.sample import get_num_triton_sampler_splits
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData
from vllm.utils import is_pin_memory_available, async_tensor_h2d, maybe_expand_dim
from vllm.sequence import (SequenceData, SequenceGroupMetadata)

_SAMPLING_EPS = 1e-5
_SEED_0_REPLACEMENT = 3403598558


@dataclass
class SequenceGroupToSample:
    seq_ids: List[int]
    sampling_params: SamplingParams
    seq_data: Dict[int, SequenceData]
    prompt_len: Optional[int]
    subquery_len: Optional[int]
    generator: Optional[torch.Generator]
    is_prompt: bool
    # Prefill query token indices from a batched input. Empty if prompt logprob
    # is not required.
    prefill_indices: List[int]
    # Sample token indices from a bathced input. Empty if sampling is not
    # required.
    sample_indices: List[int]


class SamplingMetadata:
    """Metadata for input sequences. Used in sampler.

    Metadata is used for sampling (for the new token) and computing logprobs
    (both prompt and sampled logprob).

    The usage is as follow;
    ```
    logits = execute_model(...)
    # prune logits
    logits[sampling_metadata.selected_token_indices]
    sample(logits)
    ```

    categorized_sample_indices is index after the pruning.

    Args:
        seq_groups: List of (seq_ids, sampling_params). In vLLM, each seq
            group can have its own sampling parameters. Sorted by prefill
            -> decode requests.
        seq_data: Seq_id -> SequenceData. Should be equivalent to the seq_data
            queued in a scheduler
        prompt_lens: (num_prefill_seq_groups,) Lengths of entire prompt for
            the prefill request. None if requests only contain decoding. The
            length is equivalent to the number of prefill requests batched.
        selected_token_indices: (num_query_tokens_to_logprob), Query token indices to process for
            sampling or logprob calculation.
        categorized_sample_indices: SamplingType -> token indices to sample.
            Each token indices is 2D tensor of (num_indices, num_indices) where
            the first item means the sample index within the returned logit,
            and the second item means the sample index within selected tokens.
            For example, if the returned logit is [1, 2, 3], and we select
            [1, 2] for sampling, the pruned logit will be [2, 3]. The first
            tuple is [1, 2] (sampled index within original logit), and the
            second tuple is [0, 1] (sampled index within pruned logit).
        perform_sampling: Whether to perform sampling. This option is used to
            make the sampling only happens in the driver worker, and disable
            sampling in other worker processes. Setting this True is equivalent
            to choose no indices for categorized_sample_indices.
        subquery_lens: (num_prefill_seq_groups). Length of query tokens to
            compute attention. None if all batched requests are decode.
            The length is equivalent to the number of prefill requests batched.
    """

    def __init__(
        self,
        seq_groups: List[SequenceGroupToSample],
        selected_token_indices: torch.Tensor,
        categorized_sample_indices: Optional[Dict[SamplingType, torch.Tensor]],
        num_prompts: int,
        perform_sampling: bool = True,
    ) -> None:
        self.seq_groups = seq_groups
        self.selected_token_indices = selected_token_indices
        self.categorized_sample_indices = categorized_sample_indices
        self.perform_sampling = perform_sampling
        self.num_prompts = num_prompts

        # self.num_prompts = len(prompt_lens) if prompt_lens is not None else 0

    @staticmethod
    def prepare(
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
        subquery_lens: Optional[List[int]],
        device: str,
        pin_memory: bool,
    ) -> "SamplingMetadata":
        # Batched sequence groups for the current model forward stsep.
        seq_groups: List[SequenceGroupToSample] = []
        # A list of token indices to sample/compute logprob. It is used to
        # prune the outcome logits from the model for the performance.
        selected_token_indices: List[int] = []
        # Used for selected_token_indices.
        selected_token_start_idx = 0

        # Sampling type -> (
        # indices to sample within pruned output logits,
        # indices to sample after running a triton sample kernel)
        categorized_sample_indices: Dict[SamplingType,
                                         List[Tuple[int, int]]] = {
                                             t: []
                                             for t in SamplingType
                                         }
        # Used for categorized_sample_indices.
        categorized_sample_indices_start_idx = 0
        categorized_sampled_token_indices_start_idx = 0

        num_prompts = 0
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            # seq_groups.append((seq_ids, sampling_params))
            is_prompt = seq_group_metadata.is_prompt
            generator: Optional[torch.Generator] = None
            # If the current seq group is in decode stage, it is None.
            prompt_len: Optional[int] = None
            subquery_len: Optional[int] = None
            # prefill indices for this particular seq group.
            prefill_indices = []
            # sample indices for this particular seq group.
            sample_indices = []
            do_sample = seq_group_metadata.do_sample

            if seq_group_metadata.is_prompt:
                num_prompts += 1
                assert len(seq_ids) == 1
                assert subquery_lens is not None
                assert prompt_lens is not None
                subquery_len = subquery_lens[i]
                prompt_len = prompt_lens[i]
                num_samples = len(seq_ids)

                # First, let's update what indices we want to post-process.
                # The output logits will be pruned by selected_token_indices
                # chosen from this logic.
                if sampling_params.prompt_logprobs is not None:
                    selected_token_end_idx = selected_token_start_idx + subquery_len
                    # If we need sampling, the last num_samples indexes are for sampling.
                    if do_sample:
                        selected_token_end_idx -= num_samples

                    # Select prompt indices for prompt logprob.
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_end_idx))

                # Add sample indices for sampling and sample logprob if we need sampling.
                if do_sample:
                    prefill_sample_indice = selected_token_start_idx + subquery_len - num_samples
                    selected_token_indices.append(prefill_sample_indice)
                    selected_token_start_idx += subquery_len

                # Second, find indices to sample. The index here is applied after
                # logits are pruned by selected_token_start_idx.
                if sampling_params.prompt_logprobs is not None:
                    # Update prefill indices for this seq_group if prompt logprob is required.
                    prefill_indices = list(
                        range(
                            categorized_sample_indices_start_idx,
                            categorized_sample_indices_start_idx +
                            subquery_len - num_samples))
                    # NOTE: prompt token positions do not need sample, skip
                    categorized_sample_indices_start_idx += subquery_len
                    if do_sample:
                        categorized_sample_indices_start_idx -= num_samples

                if do_sample:
                    # Update sample indices for this seq_group.
                    sample_indices = list(
                        range(
                            categorized_sample_indices_start_idx,
                            categorized_sample_indices_start_idx +
                            num_samples))
                    categorized_sample_indices[
                        sampling_params.sampling_type].append(
                            (categorized_sample_indices_start_idx,
                             categorized_sampled_token_indices_start_idx))
                    categorized_sample_indices_start_idx += num_samples
                    categorized_sampled_token_indices_start_idx += num_samples

                if sampling_params.seed is not None:
                    seq_group_metadata.state.generator = torch.Generator(
                        device=device).manual_seed(sampling_params.seed)
            else:
                if do_sample:
                    num_seqs = len(seq_ids)
                    sample_indices = range(selected_token_start_idx,
                                           selected_token_start_idx + num_seqs)
                    selected_token_indices.extend(sample_indices)
                    selected_token_start_idx += num_seqs

                    categorized_sample_indices[sampling_params.sampling_type].extend(
                        list(
                            zip(
                                range(
                                    categorized_sample_indices_start_idx,
                                    categorized_sample_indices_start_idx +
                                    num_seqs),
                                range(
                                    categorized_sampled_token_indices_start_idx,
                                    categorized_sampled_token_indices_start_idx
                                    + num_seqs))))
                    categorized_sample_indices_start_idx += num_seqs
                    categorized_sampled_token_indices_start_idx += num_seqs

            if sampling_params.seed is not None:
                generator = seq_group_metadata.state.generator
                # generators.append(seq_group_metadata.state.generator)

            seq_groups.append(
                SequenceGroupToSample(seq_ids=seq_ids,
                                      sampling_params=sampling_params,
                                      seq_data=seq_group_metadata.seq_data,
                                      prompt_len=prompt_len,
                                      subquery_len=subquery_len,
                                      generator=generator,
                                      is_prompt=is_prompt,
                                      prefill_indices=list(prefill_indices),
                                      sample_indices=list(sample_indices)))

        selected_token_indices = async_tensor_h2d(selected_token_indices,
                                                  dtype=torch.long,
                                                  target_device=device,
                                                  pin_memory=pin_memory)

        categorized_sample_indices = {
            t: maybe_expand_dim(
                async_tensor_h2d(seq_ids,
                                 dtype=torch.int,
                                 target_device=device,
                                 pin_memory=pin_memory), 2, 2)
            for t, seq_ids in categorized_sample_indices.items()
        }

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            # seq_data=seq_data,
            # prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            # generators=generators,
            # subquery_lens=subquery_lens,
            num_prompts=num_prompts,
        )
        return sampling_metadata

    def __repr__(self) -> str:
        return (
            "SamplingMetadata("
            f"seq_groups={self.seq_groups}, "
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
    sampling_seeds: torch.Tensor
    sample_indices: torch.Tensor
    extra_seeds: Optional[torch.Tensor]
    prompt_tokens: torch.Tensor
    output_tokens: torch.Tensor

    @classmethod
    def from_sampling_metadata(
        cls,
        sampling_metadata: "SamplingMetadata",
        vocab_size: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        extra_seeds_to_generate: int = 0,
        extra_entropy: Optional[Tuple[int, ...]] = None
    ) -> Tuple["SamplingTensors", bool, bool, bool]:
        """
        extra_seeds_to_generate: extra seeds to generate using the
            user-defined seed for each sequence.
        extra_entropy: extra entropy to use when generating seeds.
        """
        prompt_tokens: List[List[int]] = []
        output_tokens: List[List[int]] = []
        top_ks: List[int] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        min_ps: List[float] = []
        presence_penalties: List[float] = []
        frequency_penalties: List[float] = []
        repetition_penalties: List[float] = []
        sampling_seeds: List[int] = []
        sample_indices: List[int] = []
        prompt_best_of: List[int] = []
        do_penalties = False
        do_top_p_top_k = False
        do_min_p = False

        # We need one base seed per Triton slice.
        seeds_to_generate = (extra_seeds_to_generate +
                             get_num_triton_sampler_splits(vocab_size))

        sample_indices_start_idx = 0
        assert sampling_metadata.seq_groups is not None
        for seq_group in sampling_metadata.seq_groups:
            seq_ids = seq_group.seq_ids
            sampling_params = seq_group.sampling_params
            temperature = sampling_params.temperature
            p = sampling_params.presence_penalty
            f = sampling_params.frequency_penalty
            r = sampling_params.repetition_penalty
            top_p = sampling_params.top_p
            min_p = sampling_params.min_p
            seed = sampling_params.seed

            is_greedy = sampling_params.sampling_type == SamplingType.GREEDY

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

            is_prompt = seq_group.is_prompt
            if (seq_group.is_prompt
                    and sampling_params.prompt_logprobs is not None):
                # For tokens in the prompt that we only need to get
                # their logprobs
                subquery_len = seq_group.subquery_len
                assert subquery_len is not None
                temperatures += [temperature] * (subquery_len - 1)
                top_ps += [top_p] * (subquery_len - 1)
                top_ks += [top_k] * (subquery_len - 1)
                min_ps += [min_p] * (subquery_len - 1)
                presence_penalties += [0] * (subquery_len - 1)
                frequency_penalties += [0] * (subquery_len - 1)
                repetition_penalties += [1] * (subquery_len - 1)
                prompt_tokens.extend([] for _ in range(subquery_len - 1))
                output_tokens.extend([] for _ in range(subquery_len - 1))
                # assert sampling_metadata.subquery_lens is not None
                # prompt_len = sampling_metadata.subquery_lens[i]
                # temperatures += [temperature] * (prompt_len - 1)
                # top_ps += [top_p] * (prompt_len - 1)
                # top_ks += [top_k] * (prompt_len - 1)
                # min_ps += [min_p] * (prompt_len - 1)
                # presence_penalties += [0] * (prompt_len - 1)
                # frequency_penalties += [0] * (prompt_len - 1)
                # repetition_penalties += [1] * (prompt_len - 1)
                # prompt_tokens.extend([] for _ in range(prompt_len - 1))
                # output_tokens.extend([] for _ in range(prompt_len - 1))
            for seq_id in seq_ids:
                seq_data = seq_group.seq_data[seq_id]
                prompt_tokens.append(seq_data.prompt_token_ids)
                output_tokens.append(seq_data.output_token_ids)
            temperatures += [temperature] * len(seq_ids)
            top_ps += [top_p] * len(seq_ids)
            top_ks += [top_k] * len(seq_ids)
            min_ps += [min_p] * len(seq_ids)
            presence_penalties += [p] * len(seq_ids)
            frequency_penalties += [f] * len(seq_ids)
            repetition_penalties += [r] * len(seq_ids)

            if is_prompt:
                prompt_best_of.append(sampling_params.best_of)
                subquery_len = seq_group.subquery_len
                assert subquery_len is not None

                if sampling_params.prompt_logprobs is not None:
                    # NOTE: the sampling position is the last token
                    # in the prompt
                    sample_indices_start_idx += subquery_len - 1
                # assert sampling_metadata.subquery_lens is not None
                # prompt_len = sampling_metadata.subquery_lens[i]

                # if sampling_params.prompt_logprobs is not None:
                #     # NOTE: the sampling position is the last token
                #     # in the prompt
                #     sample_indices_start_idx += prompt_len - 1
            for seq_id in seq_ids:
                seq_data = seq_group.seq_data[seq_id]
                extra_entropy = extra_entropy or ()
                seq_seeds = cls._get_sequence_seeds(
                    seed,
                    seq_data.get_len(),
                    *extra_entropy,
                    seq_id,
                    seeds_to_generate=seeds_to_generate,
                    is_greedy=is_greedy)
                sampling_seeds.append(seq_seeds)
                sample_indices.append(sample_indices_start_idx)
                sample_indices_start_idx += 1

        sampling_tensors = SamplingTensors.from_lists(
            temperatures, top_ps, top_ks, min_ps, presence_penalties,
            frequency_penalties, repetition_penalties, sampling_seeds,
            sample_indices, prompt_tokens, output_tokens, vocab_size,
            extra_seeds_to_generate, device, dtype)
        return (sampling_tensors, do_penalties, do_top_p_top_k, do_min_p)

    @classmethod
    def from_lists(cls, temperatures: List[float], top_ps: List[float],
                   top_ks: List[int], min_ps: List[float],
                   presence_penalties: List[float],
                   frequency_penalties: List[float],
                   repetition_penalties: List[float],
                   sampling_seeds: List[int], sample_indices: List[int],
                   prompt_tokens: List[List[int]],
                   output_tokens: List[List[int]], vocab_size: int,
                   extra_seeds_to_generate: int, device: torch.device,
                   dtype: torch.dtype) -> "SamplingTensors":
        # Note that the performance will be very bad without
        # pinned memory.
        pin_memory = is_pin_memory_available()
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
        sample_indices_t = torch.tensor(
            sample_indices,
            device="cpu",
            dtype=torch.long,
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
        # need to transpose and make contiguous to
        # copy the tensor correctly.
        # [batch_size, n_seeds] -> [n_seeds, batch_size]
        sampling_seeds_t = torch.tensor(
            sampling_seeds,
            device="cpu",
            dtype=torch.long,
            pin_memory=pin_memory,
        ).T.contiguous()

        # Because the memory is pinned, we can do non-blocking
        # transfer to device.

        # How many seeds the sample operation itself will need.
        num_base_seeds = sampling_seeds_t.shape[0] - extra_seeds_to_generate
        sampling_seeds_gpu = sampling_seeds_t.to(device=device,
                                                 non_blocking=True)
        extra_seeds_gpu = sampling_seeds_gpu[num_base_seeds:]
        if not extra_seeds_gpu.numel():
            extra_seeds_gpu = None
        sampling_seeds_gpu = sampling_seeds_gpu[:num_base_seeds]

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
            sampling_seeds=sampling_seeds_gpu,
            sample_indices=sample_indices_t.to(device=device,
                                               non_blocking=True),
            extra_seeds=extra_seeds_gpu,
        )

    @staticmethod
    def _get_sequence_seeds(
        seed: int,
        *extra_entropy: int,
        seeds_to_generate: int,
        is_greedy: bool,
    ):
        """Get `seeds_to_generate` child seeds from `seed` and extra entropy."""
        if not is_greedy:
            if seed is None:
                randint_fn = random.randint
            else:
                generator = random.Random(str((seed, ) + extra_entropy))
                randint_fn = generator.randint
            lo, hi = torch.iinfo(torch.long).min, torch.iinfo(torch.long).max
            # If the user/random sets seed = 0 but request should
            # have sampling, we need to change it to something
            # else. We use a constant in that case.
            # This way we don't need to create and load a bool
            # matrix in the sampling kernel, which reduces CPU
            # overhead and latency.
            seq_seeds = [
                randint_fn(lo, hi) or _SEED_0_REPLACEMENT
                for _ in range(seeds_to_generate)
            ]
        else:
            # For the kernel, seed == 0 means greedy decoding.
            seq_seeds = [0] * seeds_to_generate
        return seq_seeds
