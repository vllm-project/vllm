# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Optional

import torch

from vllm.v1.worker.gpu_input_batch import InputBatch

DEFAULT_SAMPLING_PARAMS = dict(
    temperature=-1.0,
    min_p=0.0,
    # strictly disabled for now
    top_k=0,
    top_p=1.0,
    # frequency_penalties=0.0,
    # presence_penalties=0.0,
    # repetition_penalties=0.0,
)


@dataclass
class TPUSupportedSamplingMetadata:
    # This class exposes a more xla-friendly interface than SamplingMetadata
    # on TPU, in particular all arguments should be traceable and no optionals
    # are allowed, to avoid graph recompilation on Nones.
    temperature: torch.Tensor = None

    min_p: torch.Tensor = None
    top_k: torch.Tensor = None
    top_p: torch.Tensor = None

    all_greedy: bool = True

    # Whether logprobs are to be gathered in this batch of request. To balance
    # out compile time and runtime, a fixed `max_number_logprobs` value is used
    # when gathering logprobs, regardless of the values specified in the batch.
    logprobs: bool = False

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

    # Generator not supported by xla
    _generators: dict[int,
                      torch.Generator] = field(default_factory=lambda: dict())

    @property
    def generators(self) -> dict[int, torch.Generator]:
        # Generator not supported by torch/xla. This field must be immutable.
        return self._generators

    @classmethod
    def from_input_batch(
        cls,
        input_batch: InputBatch,
        padded_num_reqs: int,
        xla_device: torch.device,
        generate_params_if_all_greedy: bool = False
    ) -> "TPUSupportedSamplingMetadata":
        """
        Copy sampling tensors slices from `input_batch` to on device tensors.

        `InputBatch._make_sampling_metadata` causes recompilation on XLA as it 
        slices dynamic shapes on device tensors. This impl moves the dynamic 
        ops to CPU and produces tensors of fixed `padded_num_reqs` size.

        Args:
            input_batch: The input batch containing sampling parameters.
            padded_num_reqs: The padded number of requests.
            xla_device: The XLA device.
            generate_params_if_all_greedy: If True, generate sampling parameters
                even if all requests are greedy. this is useful for cases where
                we want to pre-compile a graph with sampling parameters, even if
                they are not strictly needed for greedy decoding.
        """
        needs_logprobs = input_batch.max_num_logprobs>0 if \
            input_batch.max_num_logprobs else False
        # Early return to avoid unnecessary cpu to tpu copy
        if (input_batch.all_greedy is True
                and generate_params_if_all_greedy is False):
            return cls(all_greedy=True, logprobs=needs_logprobs)

        num_reqs = input_batch.num_reqs

        def fill_slice(cpu_tensor: torch.Tensor, fill_val) -> torch.Tensor:
            # Pad value is the default one.
            cpu_tensor[num_reqs:padded_num_reqs] = fill_val

        fill_slice(input_batch.temperature_cpu_tensor,
                   DEFAULT_SAMPLING_PARAMS["temperature"])
        fill_slice(input_batch.min_p_cpu_tensor,
                   DEFAULT_SAMPLING_PARAMS["min_p"])
        fill_slice(input_batch.top_k_cpu_tensor,
                   DEFAULT_SAMPLING_PARAMS["top_k"])
        fill_slice(input_batch.top_p_cpu_tensor,
                   DEFAULT_SAMPLING_PARAMS["top_p"])

        # Slice persistent device tensors to a fixed pre-compiled padded shape.
        return cls(
            temperature=input_batch.temperature_cpu_tensor[:padded_num_reqs].
            to(xla_device),
            all_greedy=input_batch.all_greedy,
            # TODO enable more and avoid returning None values
            top_p=input_batch.top_p_cpu_tensor[:padded_num_reqs].to(
                xla_device),
            top_k=input_batch.top_k_cpu_tensor[:padded_num_reqs].to(
                xla_device),
            min_p=input_batch.min_p_cpu_tensor[:padded_num_reqs].to(
                xla_device),
            logprobs=needs_logprobs)
