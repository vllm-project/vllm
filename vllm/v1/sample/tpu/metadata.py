# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch_xla.core.xla_model as xm

from vllm.v1.sample.metadata import SamplingMetadata


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
            padded_do_sample_indices: torch.Tensor, num_do_sample: int,
            device: torch.device) -> "TPUSupportedSamplingMetadata":
        """
        Create an XLA-frienly SamplingMetadata structure. Do so by first 
        instantiating an object with fixed-sized tensors and then writing the
        values in input `metadata`. Do that only for non-None values so that 
        recompilation is not triggered for optional values (None/torch.Tensor).
        
        In order to handle different sizes for the params that range from 1 up 
        to `max_num_seqs`, pad tensors to the closest pre-compiled shape.
        Same thing for `padded_do_sample_indices`, which contains the indices 
        to be fed to the Sampler, padded to the closest pre-compiled shape.

        Eg. pad to 4 temperature: [0.7, 0.2]=>[0.7, 0.2, 0.0, 0.0]
            do_sample_indices: [4, 10]=>padded_do_sample_indices: [4, 10, 0, 0]
        """
        metadata = cls._validate_sampling_metadata(metadata)
        # NOTE we have to initialize default tensor-based params first and
        # skip None values altogether to produce the same xla graph.
        num_samples = len(padded_do_sample_indices)
        do_argmax = torch.tensor(metadata.all_greedy,
                                 dtype=torch.bool,
                                 device=device)
        new_metadata = cls.get_default_sampling_params(num_samples, device,
                                                    indices_do_sample=\
                                                    padded_do_sample_indices,
                                                    do_argmax=do_argmax
                                                    )
        supported_params = \
            TPUSupportedSamplingMetadata._get_default_params_values()
        # Copy input non-None values into `new_metadata` fixed-sized tensors.
        for p_name in supported_params:
            old_val = getattr(metadata, p_name)
            new_val = getattr(new_metadata, p_name)
            if isinstance(old_val, torch.Tensor):
                new_val[:num_do_sample] = old_val
            setattr(new_metadata, p_name, new_val)

        xm.mark_step()
        xm.wait_device_ops()
        return new_metadata

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