# SPDX-License-Identifier: Apache-2.0
"""A layer that samples the next tokens from the model's outputs."""
import itertools
import warnings
from dataclasses import dataclass
from importlib.util import find_spec
from math import inf
from typing import Dict, Iterator, List, Optional, Tuple, Union

import msgspec
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.model_executor.layers.utils import apply_penalties
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
                                                   SamplingTensors,
                                                   SequenceGroupToSample)
from vllm.sampling_params import SamplingType
from vllm.sequence import (VLLM_INVALID_TOKEN_ID,
                           CompletionSequenceGroupOutput, Logprob,
                           PromptLogprobs, SampleLogprobs, SequenceOutput)
from vllm.spec_decode.metrics import SpecDecodeWorkerMetrics

if envs.VLLM_USE_FLASHINFER_SAMPLER and find_spec("flashinfer"):
    import flashinfer.sampling
    # yapf: disable
    from flashinfer.sampling import (
        top_k_top_p_sampling_from_probs as flashinfer_top_k_top_p_sampling)

    # yapf: enable
else:
    flashinfer_top_k_top_p_sampling = None


def get_sampler() -> torch.nn.Module:
    if envs.VLLM_USE_V1:
        # Lazy import: the v1 package isn't distributed
        from vllm.v1.sample.sampler import Sampler as V1Sampler
        return V1Sampler()
    return Sampler()


# (num_token_ids, num_parent_ids) per sequence group.
SampleResultType = List[Tuple[List[int], List[int]]]

# Types of temporary data structures used for
# computing sample_result
SampleMetadataType = Dict[SamplingType, Tuple[List[int],
                                              List[SequenceGroupToSample]]]
MultinomialSamplesType = Dict[SamplingType, torch.Tensor]
SampleResultsDictType = Dict[int, Tuple[List[int], List[int]]]


# Encapsulates temporary data structures for computing
# sample_result.
#
# * For multi-step scheduling: must be returned
#   by `Sampler.forward()` and used later to compute the pythonized
#   sample_result
#
# * For single-step scheduling: consumed immediately
#   inside `Sampler.forward()` to compute pythonized sample_result.
@dataclass
class SampleResultArgsType:
    sample_metadata: SampleMetadataType
    multinomial_samples: MultinomialSamplesType
    sample_results_dict: SampleResultsDictType
    sampling_metadata: SamplingMetadata
    greedy_samples: Optional[torch.Tensor]


# Union of non-deferred (single-step scheduling)
# vs deferred (multi-step scheduling)
# sample result types
MaybeDeferredSampleResultType = Union[SampleResultType, SampleResultArgsType]

# Abbreviation of the _sample() return type
SampleReturnType = Tuple[MaybeDeferredSampleResultType, Optional[torch.Tensor]]


class SamplerOutput(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """For each sequence group, we generate a list of SequenceOutput object,
    each of which contains one possible candidate for the next token.

    This data structure implements methods, so it can be used like a list, but
    also has optional fields for device tensors.
    """

    outputs: List[CompletionSequenceGroupOutput]

    # On-device tensor containing probabilities of each token.
    sampled_token_probs: Optional[torch.Tensor] = None

    # On-device tensor containing the logprobs of each token.
    logprobs: Optional["torch.Tensor"] = None

    # Holds either (1) the pythonized sampler result (single-step scheduling)
    # or (2) what will be arguments for later deferred pythonization of the
    # sampler result (muliti-step scheduling)
    deferred_sample_results_args: Optional[SampleResultArgsType] = None

    # On-device tensor containing the sampled token ids.
    sampled_token_ids: Optional[torch.Tensor] = None
    # CPU tensor containing the sampled token ids. Used during multi-step to
    # return the sampled token ids from last rank to AsyncLLMEngine to be
    # 'broadcasted' to all other PP ranks for next step.
    sampled_token_ids_cpu: Optional[torch.Tensor] = None

    # On-device tensor containing the sampled token embeddings (embeddings
    # corresponding to the sampled token ids). Used when prompt embeddings are
    # specified in lieu of prompt token ids or text.
    sampled_token_embeds: Optional[torch.Tensor] = None

    # Spec decode metrics populated by workers.
    spec_decode_worker_metrics: Optional[SpecDecodeWorkerMetrics] = None

    # Optional last hidden states from the model.
    hidden_states: Optional[torch.Tensor] = None

    # Optional prefill hidden states from the model
    # (used for models like EAGLE).
    prefill_hidden_states: Optional[torch.Tensor] = None

    # Time taken in the forward pass for this across all workers
    model_forward_time: Optional[float] = None

    # Time taken in the model execute function. This will include model forward,
    # block/sync across workers, cpu-gpu sync time and sampling time.
    model_execute_time: Optional[float] = None

    def __getitem__(self, idx: int) -> CompletionSequenceGroupOutput:
        return self.outputs[idx]

    def __setitem__(self, idx: int, value):
        self.outputs[idx] = value

    def __iter__(self) -> Iterator[CompletionSequenceGroupOutput]:
        return iter(self.outputs)

    def __len__(self):
        return len(self.outputs)

    def __eq__(self, other: object):
        return isinstance(other,
                          self.__class__) and self.outputs == other.outputs

    def __repr__(self) -> str:
        """Show the shape of a tensor instead of its values to reduce noise.
        """
        sampled_token_probs_repr = ("None" if self.sampled_token_probs is None
                                    else self.sampled_token_probs.shape)
        sampled_token_ids_repr = ("None" if self.sampled_token_ids is None else
                                  self.sampled_token_ids.shape)
        return (
            f"SamplerOutput(outputs={self.outputs}, "
            f"sampled_token_probs={sampled_token_probs_repr}, "
            f"sampled_token_ids={sampled_token_ids_repr}, "
            f"spec_decode_worker_metrics={self.spec_decode_worker_metrics})")


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

    The structure of the logits tensor is coupled with the seq_groups in
    sampling_metadata. Typically, each sequence in each seq_group has one row in
    logits for the next token to be sampled; however, for a seq_group with a
    prompt request with the prompt_logprobs sampling parameter, there are rows
    in logits for each token in the input prompt.
    """

    def __init__(self):
        super().__init__()

        # Whether or not the SamplerOutput should have on-device tensors
        # containing the sampled token ids and probabilities. This is used by
        # speculative decoding and when prompt embeddings are specified.
        self.include_gpu_probs_tensor = False
        self.should_modify_greedy_probs_inplace = False

    def _init_sampling_tensors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ):
        """The goal here is to reuse sampling tensors between similar decode
        runs. This is possible because sampling logic does not change between
        decodes of the same sequences.
        """
        _, vocab_size = logits.shape

        # First free any existing stored sampling tensors.
        # This is necessary because some sampling tensors may
        # have pinned memory.
        self._sampling_tensors = None

        # Initialize new sampling tensors
        (sampling_tensors, do_penalties, do_top_p_top_k,
         do_min_p) = SamplingTensors.from_sampling_metadata(
             sampling_metadata, vocab_size, logits.device, logits.dtype)

        self._sampling_tensors = sampling_tensors
        self._do_penalties = do_penalties
        self._do_top_p_top_k = do_top_p_top_k
        self._do_min_p = do_min_p

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

    @property
    def _should_modify_greedy_probs_inplace(self) -> bool:
        """Whether or not the sampler should modify the probability distribution
        of greedily-sampled tokens such that multinomial sampling would sample
        the greedily-sampled token.

        In other words, if True then we set the probability of the greedily-
        sampled token to 1.

        This is used by speculative decoding, which requires that the sampling
        method be encoded into the probability distribution.
        """
        return self.should_modify_greedy_probs_inplace


def _apply_min_tokens_penalty(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Apply min_tokens penalty which sets stop tokens to -inf if min_tokens
        have not been generated yet
    """
    # list of indices in logits that will be set to -inf
    logits_to_penalize: List[Tuple[int, int]] = []
    logits_applied = 0
    for seq_group in sampling_metadata.seq_groups:
        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params

        sample_indices = seq_group.sample_indices
        logits_applied += len(sample_indices) + len(
            seq_group.prompt_logprob_indices)
        if not seq_group.do_sample:
            continue

        start_idx = sample_indices[0]
        min_tokens = sampling_params.min_tokens
        token_ids_to_penalize = sampling_params.all_stop_token_ids
        if min_tokens > 0 and token_ids_to_penalize:
            seqs_to_penalize: List[int] = []
            for j, seq_id in enumerate(seq_ids):
                seq_data = seq_group.seq_data[seq_id]
                if len(seq_data.output_token_ids_array) < min_tokens:
                    seqs_to_penalize.append(j)

            if seqs_to_penalize:
                # convert to the index into logits
                seqs_to_penalize = [start_idx + j for j in seqs_to_penalize]
                # itertools.product pairs each seq index with every token id
                logits_to_penalize.extend(
                    itertools.product(seqs_to_penalize, token_ids_to_penalize))

    if logits_to_penalize:
        # use zip and * to group indices along each dimension
        # eg. [ (1,2), (1,3), (5,6) ] -> ( (1,1,5), (2,3,6) )
        logits[tuple(zip(*logits_to_penalize))] = -float("inf")

    # verifies that no rows in logits were missed unexpectedly
    assert logits_applied == logits.shape[0]
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
    logits = torch.empty_like(logits_sort).scatter_(dim=-1,
                                                    index=logits_idx,
                                                    src=logits_sort)
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
    selected_seq_groups: List[SequenceGroupToSample],
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
    samples_lst = samples.tolist()
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
        next_token_ids = [samples_lst[sample_idx]]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def _random_sample(
    selected_seq_groups: List[SequenceGroupToSample],
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
    random_samples = random_samples.cpu()
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
            next_token_ids = random_samples[
                sample_idx, :sampling_params.n].tolist()
        else:
            # Generation phase.
            parent_ids = list(range(num_parent_seqs))
            next_token_ids = random_samples[sample_idx:sample_idx +
                                            num_parent_seqs, 0].tolist()
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


# torch.multinomial forces a GPU<->CPU sync.
# Therefore, we use an optimized implementation instead.
# Note that we always sample with replacement.
# probs will be modified in place, but this is fine, as we pass
# in a copy already.
def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
    seq_groups: Optional[List[SequenceGroupToSample]] = None,
) -> torch.Tensor:
    if num_samples > 1:
        probs = probs.repeat_interleave(num_samples, dim=0)
    q = torch.empty_like(probs)
    if seq_groups is None:
        q.exponential_()
    else:
        sample_idx = 0
        for seq_group in seq_groups:
            seq_ids = seq_group.seq_ids
            stride = len(seq_ids) * num_samples
            assert seq_group.generator is not None
            q[sample_idx:sample_idx +
              stride].exponential_(generator=seq_group.generator)
            sample_idx += stride
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)


def _top_k_top_p_multinomial_with_flashinfer(
        probs: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor,
        num_samples: int, seq_groups: Optional[List[SequenceGroupToSample]]):
    max_top_k_round = 32
    if num_samples > 1:
        probs = probs.repeat_interleave(num_samples, dim=0)
        top_ks = top_ks.repeat_interleave(num_samples)
        top_ps = top_ps.repeat_interleave(num_samples)
    batch_size = probs.shape[0]
    uniform_samples = torch.empty((max_top_k_round, batch_size),
                                  device=probs.device)
    if seq_groups is None:
        uniform_samples.uniform_()
    else:
        sample_idx = 0
        for seq_group in seq_groups:
            seq_ids = seq_group.seq_ids
            stride = len(seq_ids) * num_samples
            assert seq_group.generator is not None
            uniform_samples[:, sample_idx:sample_idx +
                            stride].uniform_(generator=seq_group.generator)
            sample_idx += stride
    batch_next_token_ids, success = flashinfer_top_k_top_p_sampling(
        probs,
        uniform_samples,
        top_ks,
        top_ps,
    )
    if not success.all():
        warnings.warn("FlashInfer rejection sampling failed, fallback.",
                      stacklevel=1)
        probs = flashinfer.sampling.top_k_renorm_prob(probs, top_ks)
        probs = flashinfer.sampling.top_p_renorm_prob(probs, top_ps)
        batch_next_token_ids = flashinfer.sampling.sampling_from_probs(
            probs, uniform_samples[0])
    return batch_next_token_ids.view(-1, num_samples)


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

    categorized_seq_group_ids: Dict[SamplingType, List[int]] = {
        t: []
        for t in SamplingType
    }
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
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

            if sampled_token_ids_tensor is not None:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[long_sample_indices] = \
                    multinomial_samples[sampling_type].to(torch.long)

        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

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
    result = (x > vals[:, None])
    del vals
    return result.sum(1).add_(1)


def get_logprobs(
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sample_results: SampleResultType,
) -> Tuple[List[Optional[PromptLogprobs]], List[SampleLogprobs]]:
    """Return sample logprobs and prompt logprobs.

    The logic consists of 3 parts.
    - Select indices to compute logprob from, ranks of token ids, and
        the top k token ids from logprobs.
    - Compute prompt logprobs if required.
    - Compute sample logprobs if required.

    Args:
        logprobs: (num_query_tokens_across_batch, num_vocab). Each query token's
            logprob per vocab. Sequence groups' query tokens are batched in a
            single flattened tensor. For example, assuming there are N
            seq groups, it is sorted by prefill tokens for seq_group_1 (if
            prompt logprob is enabled), decode tokens for seq_group_1 (if
            sampling is required), prefill tokens for seq_group_2, ...
        sampling_metadata: The sampling metadata.
        sample_results: (num_seq_groups) The tuple of (next_token_ids,
            parent_ids) for each sequence group. When beam search is enabled,
            sample_results can contain different number of seq_ids from
            sampling_metadata.seq_groups. It is because beam search creates
            2 * BEAM_WIDTH number of samples (whereas there are only up to
            BEAM_WIDTH number of seq_ids).

    Returns:
        A tuple of prompt and sample logprobs per sequence group in a batch.
    """
    # The index of query token to calculate logprobs. It includes both
    # prompt and sample logprob indices.
    query_indices: List[int] = []
    # The next token ids to get the logprob value from.
    next_token_ids: List[int] = []
    # The largest requested number of logprobs. We find logprobs as many as the
    # largest num logprobs in this API. If every logprobs is None, it will be
    # set to -1.
    largest_num_logprobs = -1

    # Select indices to compute logprob from, ranks of token ids, and the top
    # k token ids from logprobs.
    for (seq_group, sample_result) in zip(sampling_metadata.seq_groups,
                                          sample_results):
        sampling_params = seq_group.sampling_params

        # Update indices and tokens for prompt logprobs.
        if (seq_group.is_prompt
                and sampling_params.prompt_logprobs is not None):
            largest_num_logprobs = max(largest_num_logprobs,
                                       sampling_params.prompt_logprobs)
            next_prompt_tokens = _get_next_prompt_tokens(seq_group)
            query_indices.extend(seq_group.prompt_logprob_indices)
            next_token_ids.extend(next_prompt_tokens)

        # Update indices and next tokenes for sample logprob.
        if seq_group.do_sample:
            token_ids, parent_seq_ids = sample_result
            # NOTE: We cannot directly use sample_indices because
            # sample_indices only contain parent seq_ids of a previous step.
            # The current step may have different number of seq_ids, and
            # we can obtain it from `sample_result[1]`.
            query_idx = seq_group.sample_indices[0]
            query_indices.extend(
                [query_idx + parent_id for parent_id in parent_seq_ids])
            next_token_ids.extend(token_ids)

            if sampling_params.logprobs is not None:
                largest_num_logprobs = max(largest_num_logprobs,
                                           sampling_params.logprobs)

        assert len(next_token_ids) == len(query_indices)

    if len(query_indices) == 0:
        empty_sampled_logprob: SampleLogprobs = []
        empty_prompt_logprob: Optional[PromptLogprobs] = None
        num_seq_groups = len(sampling_metadata.seq_groups)
        return [empty_prompt_logprob
                ] * num_seq_groups, [empty_sampled_logprob] * num_seq_groups

    selected_logprobs, ranks = None, None
    top_logprobs, top_token_ids = None, None

    # If largest_num_logprobs == -1, i.e. no logprobs are requested, we can
    # skip the whole logprob calculation.
    if largest_num_logprobs >= 0:
        query_indices_gpu = torch.tensor(query_indices, device=logprobs.device)
        next_token_ids_gpu = torch.tensor(next_token_ids,
                                          device=logprobs.device)

        # (num_selected_query_tokens, num_logprobs). Note that query_indices can
        # contain duplicates if beam search is enabled.
        selected_logprobs = logprobs[[
            query_indices_gpu,
            next_token_ids_gpu,
        ]]
        ranks = _get_ranks(
            logprobs[query_indices_gpu],
            next_token_ids_gpu,
        )
        assert selected_logprobs.shape[0] == ranks.shape[0]

        # We need to compute top k only if there exists logprobs > 0.
        if largest_num_logprobs > 0:
            # Logprobs of topk tokens for a batch of sequence groups.
            # (num_query_tokens_across_batch).
            top_logprobs, top_token_ids = torch.topk(logprobs,
                                                     largest_num_logprobs,
                                                     dim=-1)
            top_logprobs = top_logprobs.to('cpu')
            top_token_ids = top_token_ids.to('cpu')

        selected_logprobs = selected_logprobs.to('cpu')
        ranks = ranks.to('cpu')

    # Find prompt/sample logprobs.
    prompt_logprobs_per_seq_group: List[Optional[PromptLogprobs]] = []
    sample_logprobs_per_seq_group: List[SampleLogprobs] = []
    top_logprob_idx = 0
    selected_logprobs_idx = 0

    for seq_group, sample_result in zip(sampling_metadata.seq_groups,
                                        sample_results):
        (prompt_logprobs, top_logprob_idx,
         selected_logprobs_idx) = _get_prompt_logprob_if_needed(
             seq_group, selected_logprobs, ranks, top_token_ids, top_logprobs,
             selected_logprobs_idx, top_logprob_idx)
        prompt_logprobs_per_seq_group.append(prompt_logprobs)

        (sampled_logprobs, top_logprob_idx,
         selected_logprobs_idx) = _get_sampled_logprob_if_needed(
             seq_group, sample_result, selected_logprobs, ranks, top_token_ids,
             top_logprobs, selected_logprobs_idx, top_logprob_idx)
        sample_logprobs_per_seq_group.append(sampled_logprobs)

    return prompt_logprobs_per_seq_group, sample_logprobs_per_seq_group


def _get_prompt_logprob_if_needed(
    seq_group: SequenceGroupToSample,
    selected_logprobs: torch.Tensor,
    ranks: torch.Tensor,
    top_token_ids: torch.Tensor,
    top_logprobs: torch.Tensor,
    selected_logprobs_idx: int,
    top_logprob_idx: int,
):
    """Compute the prompt logprob from a sequence group if needed."""
    sampling_params = seq_group.sampling_params
    is_prompt = seq_group.is_prompt

    # Find prompt logprobs
    prompt_logprobs: Optional[PromptLogprobs] = None
    if is_prompt and sampling_params.prompt_logprobs is not None:
        prompt_logprobs = []
        num_logprobs = sampling_params.prompt_logprobs
        next_prompt_tokens = _get_next_prompt_tokens(seq_group)
        # Pre-select indexes and create a list. It is faster than calling .item
        # repetitively.
        selected_logprob_items = selected_logprobs[
            selected_logprobs_idx:selected_logprobs_idx +
            len(next_prompt_tokens)].tolist()
        rank_items = ranks[selected_logprobs_idx:selected_logprobs_idx +
                           len(next_prompt_tokens)].tolist()

        for idx, token_id in enumerate(next_prompt_tokens):
            # Calculate the prompt logprob of the real prompt tokens.
            # {token_id: (logprob, rank_from_vocab)}
            prompt_logprobs_dict: Dict[int, Tuple[float, int]] = {
                token_id: (selected_logprob_items[idx], rank_items[idx])
            }

            # Add top K prompt logprobs along with its rank.
            if num_logprobs > 0:
                top_ids = top_token_ids[
                    top_logprob_idx, :num_logprobs].tolist()
                top_probs = top_logprobs[
                    top_logprob_idx, :num_logprobs].tolist()
                # Top K is already sorted by rank, so we can use 1 ~
                # num_logprobs + 1 for rank.
                top_ranks = range(1, num_logprobs + 1)
                prompt_logprobs_dict.update({
                    top_id: (top_prob, rank)
                    for top_id, top_prob, rank in zip(top_ids, top_probs,
                                                      top_ranks)
                })
            prompt_logprobs.append({
                token_id: Logprob(*logprob_and_rank)
                for token_id, logprob_and_rank in prompt_logprobs_dict.items()
            })
            # + 1 to go to the next prompt token.
            top_logprob_idx += 1

        # + len(next_prompt_tokens) to go to the next prompt.
        selected_logprobs_idx += len(next_prompt_tokens)
    return prompt_logprobs, top_logprob_idx, selected_logprobs_idx


def _get_sampled_logprob_if_needed(
    seq_group: SequenceGroupToSample,
    sample_result: Tuple[List[int], List[int]],
    selected_logprobs: torch.Tensor,
    ranks: torch.Tensor,
    top_token_ids: torch.Tensor,
    top_logprobs: torch.Tensor,
    selected_logprobs_idx: int,
    top_logprob_idx: int,
):
    """Compute the sample logprob if needed."""
    seq_ids = seq_group.seq_ids
    num_logprobs = seq_group.sampling_params.logprobs
    sampled_logprobs: SampleLogprobs = []
    next_token_ids, parent_seq_ids = sample_result

    if seq_group.do_sample:
        assert len(next_token_ids) > 0
        if num_logprobs is None:
            for next_token_id in next_token_ids:
                # Use a dummy logprob
                sampled_logprobs.append({next_token_id: Logprob(inf)})
        else:
            # Pre-select items from tensor. tolist() is faster than repetitive
            # `.item()` calls.
            selected_logprob_items = selected_logprobs[
                selected_logprobs_idx:selected_logprobs_idx +
                len(next_token_ids)].tolist()
            rank_items = ranks[selected_logprobs_idx:selected_logprobs_idx +
                               len(next_token_ids)].tolist()
            for idx, (next_token_id, parent_id) in enumerate(
                    zip(next_token_ids, parent_seq_ids)):
                # Get the logprob of a sampled token.
                sampled_logprobs_dict = {
                    next_token_id:
                    (selected_logprob_items[idx], rank_items[idx])
                }
                if num_logprobs is not None and num_logprobs > 0:
                    # Get top K logprobs.
                    top_ids = top_token_ids[top_logprob_idx +
                                            parent_id, :num_logprobs].tolist()
                    top_probs = top_logprobs[
                        top_logprob_idx + parent_id, :num_logprobs].tolist()
                    # Top K is already sorted by rank, so we can use 1 ~
                    # num_logprobs + 1 for rank.
                    top_ranks = range(1, num_logprobs + 1)
                    sampled_logprobs_dict.update({
                        top_id: (top_prob, rank)
                        for top_id, top_prob, rank in zip(
                            top_ids, top_probs, top_ranks)
                    })

                sampled_logprobs.append({
                    token_id: Logprob(*logprob_and_rank)
                    for token_id, logprob_and_rank in
                    sampled_logprobs_dict.items()
                })

        # NOTE: This part of code is not intuitive. `selected_logprobs` include
        # logprobs for the current step, which has len(next_token_ids) tokens
        # per sequence group. `logprobs` includes logprobs from the previous
        # steps, which has len(seq_ids) tokens per sequence group.

        # Iterate to the next sequence group in a batch.
        selected_logprobs_idx += len(next_token_ids)
        # Iterate to the next sequence group in a batch.
        top_logprob_idx += len(seq_ids)
    return sampled_logprobs, top_logprob_idx, selected_logprobs_idx


def _modify_greedy_probs_inplace(logprobs: torch.Tensor, probs: torch.Tensor,
                                 sample_indices: torch.Tensor,
                                 greedy_samples: torch.Tensor) -> None:
    """Modify the probability distributions of the greedily-sampled tokens such
    that each sampled token has a "probability" of 1.0. This is required by
    speculative decoding, which depends on the sampling method being encoded
    within the probability distribution for correctness.

    # Why do we only need to do this for greedy sampling?

    vLLM's sampler performs the following steps for greedy or multinomial
    (random) sampling:
        1. Get logits from model.
        2. Modify logits according to per-sequence sampling parameters.
            - Multiply by temperature, top-k and top-p masking, penalize tokens
                according to their frequency, etc.
        3. Sample a token.
            - Random sampling simply samples from the modified probability
                distribution.
            - Greedy sampling performs `argmax` to obtain the token with the
                highest likelihood.

    Ignoring greedy sampling for a moment, we find that the computed probability
    distribution has the following property: we can sample from it independently
    and find that the token sampled by the Sampler has a frequency corresponding
    to how often we see it in our sampling. In other words, for tokens sampled
    with vLLM's random SamplingType, the computed probability distribution
    encodes the sampling methodology completely.

    Greedy sampling does not normally have this property. vLLM modifies logits
    according to sampling params, then performs `argmax`, then returns the
    sampled token and the computed probability distribution. If we sample from
    the distribution, we'll find the likelihood of the greedily-sampled token
    is not always 1.0.

    Since lossless speculative decoding requires that the sampling methodology
    be encoded within the probability distribution, we are motivated to modify
    the probability distribution such that the sampled token has probability 1
    when speculative decoding is used.

    NOTE: Alternatively, we could use an extremely low temperature to achieve
    greedy sampling using multinomial computation and unite the codepaths. This
    has implications on the overall design of the sampler, e.g. how to record
    accurate logprobs for the user, so this improvement is deferred to later.
    """
    # NOTE: logprobs are not modified so they can be returned to the user.
    probs[sample_indices, :] = 0
    probs[sample_indices, greedy_samples] = 1.0


def _build_sampler_output(
    maybe_deferred_sample_results: MaybeDeferredSampleResultType,
    sampling_metadata: SamplingMetadata,
    prompt_logprobs: Optional[List[Optional[PromptLogprobs]]],
    sample_logprobs: Optional[List[SampleLogprobs]],
    on_device_tensors: Optional[Tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor]],
    skip_sampler_cpu_output: bool = False,
) -> SamplerOutput:
    """Construct Python objects with the output of sampling.

    Args:
        on_device_tensors: Tuple containing on-device tensors with the
            probabilities used in sampling and the sampled token ids. This
            allows post-processing without copies to CPU/serialization, e.g. in
            speculative decoding rejection sampling.
    """
    sampler_output: List[CompletionSequenceGroupOutput] = []

    if skip_sampler_cpu_output:
        assert isinstance(maybe_deferred_sample_results, SampleResultArgsType)
        deferred_sample_results_args = maybe_deferred_sample_results
    else:
        assert prompt_logprobs is not None
        assert sample_logprobs is not None
        assert not isinstance(maybe_deferred_sample_results,
                              SampleResultArgsType)
        assert len(sampling_metadata.seq_groups) \
            == len(maybe_deferred_sample_results) \
            == len(prompt_logprobs) \
            == len(sample_logprobs)
        deferred_sample_results_args = None

        for (seq_group, sample_result, group_prompt_logprobs,
             group_sample_logprobs) in zip(sampling_metadata.seq_groups,
                                           maybe_deferred_sample_results,
                                           prompt_logprobs, sample_logprobs):
            seq_ids = seq_group.seq_ids
            next_token_ids, parent_ids = sample_result
            seq_outputs: List[SequenceOutput] = []
            for parent_id, next_token_id, logprobs in zip(
                    parent_ids, next_token_ids, group_sample_logprobs):
                seq_outputs.append(
                    SequenceOutput(seq_ids[parent_id], next_token_id,
                                   logprobs))
            sampler_output.append(
                CompletionSequenceGroupOutput(seq_outputs,
                                              group_prompt_logprobs))

    # If not specified, store None values in SamplerOutput.
    if on_device_tensors is not None:
        (sampled_token_probs, logprobs_tensor,
         sampled_token_ids) = on_device_tensors
    else:
        sampled_token_probs, logprobs_tensor, sampled_token_ids = (None, None,
                                                                   None)

    return SamplerOutput(
        outputs=sampler_output,
        sampled_token_probs=sampled_token_probs,
        sampled_token_ids=sampled_token_ids,
        logprobs=logprobs_tensor,
        deferred_sample_results_args=deferred_sample_results_args)


def _get_next_prompt_tokens(
        seq_group: SequenceGroupToSample) -> tuple[int, ...]:
    """Get a list of next prompt tokens to compute logprob from a
        given sequence group.

    It is used to compute prompt logprob. Imagine you have logprob for each
    query token. Query token needs to know the next prompt token id to compute
    prompt logprob. This is a helper to obtain next prompt token ids.

    This API has to be used only when the caller knows seq_group is in prefill
    stage.

    Returns:
        A list of next prompt tokens to compute logprob.
    """
    assert seq_group.is_prompt, (
        "Caller should ensure the sequence group is in a prefill stage.")
    seq_ids = seq_group.seq_ids
    query_len = seq_group.query_len
    assert query_len is not None
    # prompt has only 1 seq id.
    assert len(seq_ids) == 1
    seq_data = seq_group.seq_data[seq_ids[0]]
    computed_len = seq_data.get_num_computed_tokens()
    prompt_tokens = seq_data.prompt_token_ids
    # +1 because we are looking for a next prompt token.
    next_token_index_start = computed_len + 1
    next_token_index_end = min(computed_len + query_len + 1,
                               len(prompt_tokens))
    next_prompt_tokens = prompt_tokens[
        next_token_index_start:next_token_index_end]
    return next_prompt_tokens
