# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Optional, cast

import torch
import torch_xla.core.xla_model as xm

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch


# TODO make it return a TPUSamplingMetadata and refactor
def make_sampling_metadata(input_batch: InputBatch,
                           padded_num_reqs: int) -> SamplingMetadata:
    """
    `InputBatch._make_sampling_metadata` causes recompilation on XLA as it 
    slices dynamic shapes on device tensors. This impl moves the dynamic ops
    to CPU and produces tensors of fixed `padded_num_reqs` size.
    """
    num_reqs = input_batch.num_reqs

    def pad_and_slice(cpu_tensor: torch.Tensor) -> torch.Tensor:
        # Zero-out padding explicitly on CPU and slice.
        # TODO use actual default values
        cpu_tensor[num_reqs:padded_num_reqs] = 0
        return cpu_tensor[:padded_num_reqs]

    if not input_batch.all_greedy:
        # Make temperature non-optional
        temperature = input_batch.temperature_cpu_tensor[:padded_num_reqs]
        temperature[num_reqs:] = -1.0
    else:
        temperature = input_batch.temperature_cpu_tensor[:padded_num_reqs]
        temperature[:] = -1.0

    # TODO (NickLucche) all the pre-allocated on-device tensors are wasted here
    # if not input_batch.no_penalties:
    # Since syncing these tensors is expensive only copy them
    # if necessary i.e. if there are requests which require
    # penalties to be applied during sampling.
    # copy_slice(input_batch.frequency_penalties_cpu_tensor,
    #             input_batch.frequency_penalties, num_reqs, padded_num_reqs)
    # copy_slice(input_batch.presence_penalties_cpu_tensor,
    #             input_batch.presence_penalties, num_reqs, padded_num_reqs)
    # copy_slice(input_batch.repetition_penalties_cpu_tensor,
    #             input_batch.repetition_penalties, num_reqs, padded_num_reqs)

    # The prompt tokens are used only for applying penalties during
    # the sampling process. Hence copy these tensors only when
    # there are requests which need penalties to be applied.
    # prompt_token_ids = input_batch._make_prompt_token_ids_tensor()
    prompt_token_ids = None

    allowed_token_ids_mask: Optional[torch.Tensor] = None
    # if not input_batch.no_allowed_token_ids:
    #     assert input_batch.allowed_token_ids_mask is not None
    #     copy_slice(input_batch.allowed_token_ids_mask_cpu_tensor,
    #                 input_batch.allowed_token_ids_mask, num_reqs,
    #                   padded_num_reqs)
    #     allowed_token_ids_mask = input_batch.\
    #           allowed_token_ids_mask[:padded_num_reqs]

    # Every slice is done on CPU now
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=input_batch.all_greedy,
        all_random=input_batch.all_random,
        # TODO do not allow Nones here
        top_p=None if input_batch.no_top_p else pad_and_slice(
            input_batch.top_p_cpu_tensor),
        top_k=None if input_batch.no_top_k else pad_and_slice(
            input_batch.top_k_cpu_tensor),
        min_p=pad_and_slice(input_batch.min_p_cpu_tensor),
        generators=input_batch.generators,
        max_num_logprobs=input_batch.max_num_logprobs,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=
        None,  #pad_and_slice(input_batch.frequency_penalties_cpu_tensor),
        presence_penalties=
        None,  #pad_and_slice(input_batch.presence_penalties_cpu_tensor),
        repetition_penalties=
        None,  #pad_and_slice(input_batch.repetition_penalties_cpu_tensor),
        output_token_ids=cast(list[list[int]],
                              input_batch.req_output_token_ids),
        min_tokens=input_batch.min_tokens,
        no_penalties=input_batch.no_penalties,
        logit_bias=input_batch.logit_bias[:padded_num_reqs],
        allowed_token_ids_mask=allowed_token_ids_mask,
        bad_words_token_ids=input_batch.bad_words_token_ids,
    )


@dataclass
class TPUSupportedSamplingMetadata:
    # This class exposes a more xla-friendly interface than SamplingMetadata
    # on TPU, in particular all arguments should be traceable and no optionals
    # are allowed, to avoid graph recompilation on Nones.
    temperature: torch.Tensor

    min_p: torch.Tensor
    # Still too slow on forward_native!
    top_k: torch.Tensor = None
    top_p: torch.Tensor = None

    # XLA-unfriendly control flow in Sampler
    all_greedy: bool = False
    all_random: bool = False
    # Greedy sampling flag for compiling single xla graph.
    do_argmax: torch.Tensor = None

    # speculation not supported
    spec_token_ids = None

    # Generator not supported by xla
    generators: dict[int,
                     torch.Generator] = field(default_factory=lambda: dict())

    # unsupported, you need to return an extra tensor of static size BxV
    max_num_logprobs = None

    # TODO No penalties for now
    no_penalties: bool = True
    prompt_token_ids = None
    frequency_penalties = None
    presence_penalties = None
    repetition_penalties = None
    # should use tensor
    output_token_ids: list[list[int]] = field(default_factory=lambda: list())

    min_tokens = None  # impl is not vectorized

    logit_bias: list[Optional[dict[int, float]]] = field(
        default_factory=lambda: list())

    allowed_token_ids_mask = None
    bad_words_token_ids = None
    indices_do_sample: torch.Tensor = None

    def __post_init__(self):
        temp = self.temperature
        if self.indices_do_sample is None:
            self.indices_do_sample = torch.zeros(temp.shape[0],
                                                 device=temp.device,
                                                 dtype=torch.int32)
        if self.do_argmax is None:
            self.do_argmax = torch.tensor(0,
                                          dtype=torch.bool,
                                          device=temp.device)

    @classmethod
    def from_sampling_metadata(
            cls, metadata: SamplingMetadata,
            padded_do_sample_indices: torch.Tensor,
            device: torch.device) -> "TPUSupportedSamplingMetadata":
        """
        Create an XLA-frienly SamplingMetadata structure. Move sampling params
        tensors from host to device.
        
        In order to handle different sizes for the params that range from 1 up 
        to `max_num_seqs`, pad tensors to the closest pre-compiled shape.
        Same thing for `padded_do_sample_indices`, which contains the indices 
        to be fed to the Sampler, padded to the closest pre-compiled shape.

        Eg. pad to 4 temperature: [0.7, 0.2]=>[0.7, 0.2, 0.0, 0.0]
            do_sample_indices: [4, 10]=>padded_do_sample_indices: [4, 10, 0, 0]
        """
        do_argmax = torch.tensor(metadata.all_greedy,
                                 dtype=torch.bool,
                                 device=device)
        supported_params = \
            TPUSupportedSamplingMetadata._get_default_params_values()

        # All tensors we get here are already padded, just move them to TPU.
        kwargs = dict()
        for p_name in supported_params:
            old_val = getattr(metadata, p_name)
            if isinstance(old_val, torch.Tensor):
                kwargs[p_name] = old_val.to(device)
        xm.mark_step()
        xm.wait_device_ops()
        return cls(**kwargs,
                   indices_do_sample=padded_do_sample_indices,
                   do_argmax=do_argmax)

    @classmethod
    def get_default_sampling_params(
            cls,
            num_samples: int,
            device: torch.device,
            indices_do_sample=None,
            do_argmax=None) -> "TPUSupportedSamplingMetadata":
        # As sampling happens on a single traced graph, options
        # are "disabled" by having them evaluate to an Identity op.
        # Note that initialization is dependent on num_samples.
        sampling_metadata_disable_value = \
            TPUSupportedSamplingMetadata._get_default_params_values()
        init_kwargs = dict()
        for p_name, (default_val,
                     dtype) in sampling_metadata_disable_value.items():
            default_tensor = torch.full((num_samples, ),
                                        default_val,
                                        dtype=dtype,
                                        device=device)
            init_kwargs[p_name] = default_tensor

        return cls(**init_kwargs,
                   indices_do_sample=indices_do_sample,
                   do_argmax=do_argmax)

    @staticmethod
    def _validate_sampling_metadata(
            sampling_metadata: SamplingMetadata) -> SamplingMetadata:
        if sampling_metadata.all_greedy:
            # Set to None since #13587. Make sure default isn't overruled.
            assert sampling_metadata.temperature is None
        return sampling_metadata

    @staticmethod
    def _get_default_params_values():
        return dict(
            # Since #13587 greedy sampling requires branching off which leads
            # to separate graphs. We set temp to noop and handle argmax here.
            temperature=(1.0, torch.float32),
            min_p=(0.0, torch.float32),
            # strictly disabled for now
            # top_k=(-1, torch.int32),
            # top_p=(0.0, torch.float32),
            # frequency_penalties=(0.0, torch.float32),
            # presence_penalties=(0.0, torch.float32),
            # repetition_penalties=(0.0, torch.float32),
        )
