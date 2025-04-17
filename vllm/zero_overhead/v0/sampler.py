# SPDX-License-Identifier: Apache-2.0

from importlib.util import find_spec
from typing import Optional

import torch

from vllm import envs
from vllm.model_executor.layers.sampler import (
    MultinomialSamplesType, SampleMetadataType, Sampler, SampleResultArgsType,
    SampleResultsDictType, SampleResultType, SampleReturnType, SamplerOutput,
    _apply_min_p, _apply_min_tokens_penalty, _apply_top_k_top_p,
    _build_sampler_output, _modify_greedy_probs_inplace, _multinomial,
    _top_k_top_p_multinomial_with_flashinfer, get_logprobs)
from vllm.model_executor.layers.utils import apply_penalties
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
                                                   SamplingTensors,
                                                   SequenceGroupToSample)
from vllm.sampling_params import SamplingType
from vllm.sequence import VLLM_INVALID_TOKEN_ID

if envs.VLLM_USE_FLASHINFER_SAMPLER and find_spec("flashinfer"):
    # yapf: disable
    from flashinfer.sampling import (
        top_k_top_p_sampling_from_probs as flashinfer_top_k_top_p_sampling)

    # yapf: enable
else:
    flashinfer_top_k_top_p_sampling = None


class SampleRecorder:

    def __init__(self):
        self.seq_id: torch.Tensor = None
        self.sampled_token_ids_tensor: torch.Tensor = None


last_sampler = None


def get_last_sampler():
    return last_sampler


class ZeroOverheadSampler(Sampler):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """
        Single-step scheduling:
        * Perform GPU-side sampling computation & compute
          GPU-side logprobs tensor
        * Pythonize sampling result & logprobs tensor

        Multi-step scheduling:
        * Perform GPU-side sampling computation & compute
          GPU-side logprobs tensor
        * Defer Pythonization of sampling result & logprobs
          tensor
        * Encapsulate arguments required for deferred Pythonization
          in the :class:`SamplerOutput` structure

        Args:
            logits: (num_tokens, vocab_size).
            sampling_metadata: Metadata for sampling.
        """
        global last_sampler
        last_sampler = SampleRecorder()
        assert logits is not None
        _, vocab_size = logits.shape

        # Prepare sampling tensors with pinned memory to avoid blocking.
        if not sampling_metadata.reuse_sampling_tensors:
            self._init_sampling_tensors(logits, sampling_metadata)
        elif self._do_penalties:
            # In this case, the sampling tensors logic depends on
            # "output_tokens" of a sequence. As a result, we cannot
            # reuse sampling tensors, since "output_tokens" changes
            # between decode runs.
            self._init_sampling_tensors(logits, sampling_metadata)

        assert self._sampling_tensors is not None
        sampling_tensors = self._sampling_tensors
        do_penalties = self._do_penalties
        do_top_p_top_k = self._do_top_p_top_k
        do_min_p = self._do_min_p

        logits = _apply_min_tokens_penalty(logits, sampling_metadata)

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = apply_penalties(logits, sampling_tensors.prompt_tokens,
                                     sampling_tensors.output_tokens,
                                     sampling_tensors.presence_penalties,
                                     sampling_tensors.frequency_penalties,
                                     sampling_tensors.repetition_penalties)

        # Use float32 to apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits = logits.to(torch.float)
        logits.div_(sampling_tensors.temperatures.unsqueeze(dim=1))

        if do_top_p_top_k and flashinfer_top_k_top_p_sampling is None:
            logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps,
                                        sampling_tensors.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        maybe_deferred_sample_results, maybe_sampled_tokens_tensor = _sample(
            probs,
            logprobs,
            sampling_metadata,
            sampling_tensors,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
            modify_greedy_probs=self._should_modify_greedy_probs_inplace,
        )

        if self.include_gpu_probs_tensor:
            # Since we will defer sampler result Pythonization,
            # preserve GPU-side tensors in support of later
            # deferred pythonization of logprobs
            assert maybe_sampled_tokens_tensor is not None
            on_device_tensors = (probs, logprobs, maybe_sampled_tokens_tensor)
        else:
            # Since Pythonization has already happened, don't preserve
            # GPU-side tensors.
            on_device_tensors = None

        # Get the logprobs query results.
        prompt_logprobs = None
        sample_logprobs = None
        if not sampling_metadata.skip_sampler_cpu_output:
            # Pythonize logprobs now (GPU -> CPU); do not defer.
            assert not isinstance(maybe_deferred_sample_results,
                                  SampleResultArgsType)
            prompt_logprobs, sample_logprobs = get_logprobs(
                logprobs, sampling_metadata, maybe_deferred_sample_results)

        return _build_sampler_output(
            maybe_deferred_sample_results,
            sampling_metadata,
            prompt_logprobs,
            sample_logprobs,
            on_device_tensors=on_device_tensors,
            skip_sampler_cpu_output=sampling_metadata.skip_sampler_cpu_output)


def _greedy_sample(
    selected_seq_groups: list[SequenceGroupToSample],
    samples: torch.Tensor,
) -> SampleResultType:
    """Run greedy sampling on a given samples.

    Args:
        selected_seq_groups: A list of sequence groups batched.
        samples: (num_selected_samples,) A tensor of samples. The length of
            samples could be smaller than selected_seq_groups if
            seq_group.do_sample is False.
    Returns:
        Tuple of (next_token_ids, parent_ids). The length of returned list is
        same as the length of selected_seq_groups. If the corresponding
        seq_group has do_sample=False, tuple contains ([], [])
    """
    sample_idx = 0
    results: SampleResultType = []
    for seq_group in selected_seq_groups:
        if not seq_group.do_sample:
            results.append(([], []))
            continue

        seq_ids = seq_group.seq_ids
        num_parent_seqs = len(seq_ids)
        assert num_parent_seqs == 1, (
            "Greedy sampling should have only one seq.")
        parent_ids = list(range(num_parent_seqs))
        # not support multi sequences in sequence group
        assert num_parent_seqs == 1
        next_token_ids = [0]  #place holder token id
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def _random_sample(
    selected_seq_groups: list[SequenceGroupToSample],
    random_samples: torch.Tensor,
) -> SampleResultType:
    """Run random sampling on a given samples.

    Args:
        selected_seq_groups: A list of sequence groups batched.
        random_samples: (num_selected_samples,) A tensor of samples. The
            length of samples could be smaller than selected_seq_groups if
            seq_group.do_sample is False.
    Returns:
        Tuple of (next_token_ids, parent_ids). The length of returned list is
        same as the length of selected_seq_groups. If the corresponding
        seq_group has do_sample=False, tuple contains ([], [])
    """
    # Find the maximum n value of the prompt phase requests.
    sample_idx = 0
    results: SampleResultType = []
    for seq_group in selected_seq_groups:
        if not seq_group.do_sample:
            results.append(([], []))
            continue

        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        is_prompt = seq_group.is_prompt
        num_parent_seqs = len(seq_ids)
        if is_prompt:
            # Prompt phase.
            parent_ids = [0] * sampling_params.n
            # not support multi sequences in sequence group
            assert num_parent_seqs == 1
            #place holder token id
            next_token_ids = [0] * sampling_params.n
        else:
            # Generation phase.
            parent_ids = list(range(num_parent_seqs))
            # not support multi sequences in sequence group
            assert num_parent_seqs == 1
            #place holder token id
            next_token_ids = [0] * num_parent_seqs
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sampling_tensors: SamplingTensors,
    include_gpu_probs_tensor: bool,
    modify_greedy_probs: bool,
) -> SampleReturnType:
    """
    Args:
        probs: (num_query_tokens_in_batch, num_vocab)
        logprobs: (num_query_tokens_in_batch, num_vocab)
        sampling_metadata: The metadata for a batch for sampling.
        sampling_tensors: Tensors that include sampling related metadata.

    Returns:
        (next_token_ids, parent_seq_ids) for each seq group in a batch.
            If sampling is skipped, it returns ([], [])
        sampled_token_ids_tensor: A tensor of sampled token ids.
    """
    return _sample_with_torch(
        probs,
        logprobs,
        sampling_metadata,
        sampling_tensors,
        include_gpu_probs_tensor=include_gpu_probs_tensor,
        modify_greedy_probs=modify_greedy_probs,
    )


def _sample_with_torch(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sampling_tensors: SamplingTensors,
    include_gpu_probs_tensor: bool,
    modify_greedy_probs: bool,
) -> SampleReturnType:
    '''Torch-oriented _sample() implementation.

    Single-step scheduling:
    * Perform GPU-side sampling computation
    * Immediately Pythonize sampling result

    Multi-step scheduling:
    * Perform GPU-side sampling computation
    * Defer Pythonization & preserve GPU-side
      tensors required for Pythonization
    '''

    categorized_seq_group_ids: dict[SamplingType, list[int]] = {
        t: []
        for t in SamplingType
    }
    last_sampler.seq_id = torch.zeros(len(sampling_metadata.seq_groups),
                                      dtype=torch.int32)
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        last_sampler.seq_id[i] = seq_group.seq_ids[0]
        sampling_params = seq_group.sampling_params
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)
    sample_results_dict: SampleResultsDictType = {}
    sample_metadata: SampleMetadataType = {}
    multinomial_samples: MultinomialSamplesType = {}
    greedy_samples: Optional[torch.Tensor] = None

    # Create output tensor for sampled token ids.
    if include_gpu_probs_tensor:
        sampled_token_ids_tensor = torch.full((logprobs.shape[0], 1),
                                              VLLM_INVALID_TOKEN_ID,
                                              dtype=torch.long,
                                              device=logprobs.device)
    else:
        sampled_token_ids_tensor = None

    # Counterintiutively, having two loops here is actually faster.
    # The first loop can run without waiting on GPU<->CPU sync.
    for sampling_type in SamplingType:
        sample_indices = categorized_sample_indices[sampling_type]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue

        seq_group_id = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_id]
        sample_metadata[sampling_type] = (seq_group_id, seq_groups)
        long_sample_indices = sample_indices.long()
        if sampling_type == SamplingType.GREEDY:
            greedy_samples = torch.argmax(logprobs[long_sample_indices],
                                          dim=-1)

            last_sampler.sampled_token_ids_tensor = greedy_samples.unsqueeze(
                -1)

            if sampled_token_ids_tensor is not None:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[
                    long_sample_indices] = greedy_samples.unsqueeze(-1)

            if modify_greedy_probs:
                # If required, modify the probabilities such that sampling from
                # the modified distribution would always sample the argmax
                # token id.
                _modify_greedy_probs_inplace(logprobs, probs,
                                             long_sample_indices,
                                             greedy_samples)

        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            max_n_in_batch = 1
            for seq_group in seq_groups:
                if seq_group.is_prompt:
                    sampling_params = seq_group.sampling_params
                    max_n_in_batch = max(max_n_in_batch, sampling_params.n)
            seq_groups_arg = (None if sampling_type == SamplingType.RANDOM else
                              seq_groups)

            if flashinfer_top_k_top_p_sampling is not None:
                multinomial_samples[
                    sampling_type] = _top_k_top_p_multinomial_with_flashinfer(
                        probs[long_sample_indices],
                        sampling_tensors.top_ks[long_sample_indices],
                        sampling_tensors.top_ps[long_sample_indices],
                        max_n_in_batch,
                        seq_groups_arg,
                    )
            else:
                multinomial_samples[sampling_type] = _multinomial(
                    probs[long_sample_indices],
                    max_n_in_batch,
                    seq_groups=seq_groups_arg)

            last_sampler.sampled_token_ids_tensor = \
                multinomial_samples[sampling_type].to(torch.long)

            if sampled_token_ids_tensor is not None:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[long_sample_indices] = \
                    multinomial_samples[sampling_type].to(torch.long)

    # Encapsulate arguments for computing Pythonized sampler
    # results, whether deferred or otherwise.
    maybe_deferred_args = SampleResultArgsType(
        sampling_metadata=sampling_metadata,
        sample_metadata=sample_metadata,
        multinomial_samples=multinomial_samples,
        greedy_samples=greedy_samples,
        sample_results_dict=sample_results_dict)

    if not sampling_metadata.skip_sampler_cpu_output:
        # GPU<->CPU sync happens here.
        # This also converts the sampler output to a Python object.
        # Return Pythonized sampler result & sampled token ids
        return get_pythonized_sample_results(
            maybe_deferred_args), sampled_token_ids_tensor
    else:
        # Defer sampler result Pythonization; return deferred
        # Pythonization args & sampled token ids
        return (
            maybe_deferred_args,
            sampled_token_ids_tensor,
        )


def get_pythonized_sample_results(
        sample_result_args: SampleResultArgsType) -> SampleResultType:
    '''This function consumes GPU-side sampler results and computes
    Pythonized CPU-side sampler results (GPU -> CPU sync.)

    Single-step scheduling: this function is invoked at sampling-time
    for immediate Pythonization.

    Multi-step scheduling: Pythonization is deferred until after multiple
    GPU-side steps have been completed.

    Args:
      sample_result_args: GPU-side inputs to the Pythonization process

    Returns:
      Pythonized sampler results
    '''

    (
        sample_metadata,
        sampling_metadata,
        greedy_samples,
        multinomial_samples,
        sample_results_dict,
    ) = (
        sample_result_args.sample_metadata,
        sample_result_args.sampling_metadata,
        sample_result_args.greedy_samples,
        sample_result_args.multinomial_samples,
        sample_result_args.sample_results_dict,
    )

    for sampling_type in SamplingType:
        if sampling_type not in sample_metadata:
            continue
        (seq_group_id, seq_groups) = sample_metadata[sampling_type]
        if sampling_type == SamplingType.GREEDY:
            sample_results = _greedy_sample(seq_groups, greedy_samples)
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            sample_results = _random_sample(seq_groups,
                                            multinomial_samples[sampling_type])
        sample_results_dict.update(zip(seq_group_id, sample_results))

    return [
        sample_results_dict.get(i, ([], []))
        for i in range(len(sampling_metadata.seq_groups))
    ]
