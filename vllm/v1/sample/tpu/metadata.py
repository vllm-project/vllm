# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch_xla.core.xla_model as xm

from vllm.v1.worker.gpu_input_batch import InputBatch

DEFAULT_SAMPLING_PARAMS = dict(
    temperature=-1.0,
    min_p=0.0,
    # strictly disabled for now
    # top_k=-1,
    # top_p=0.0,
    # frequency_penalties=0.0,
    # presence_penalties=0.0,
    # repetition_penalties=0.0,
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

    # Greedy sampling flag for compiling single xla graph.
    all_greedy: torch.Tensor = None

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

    @classmethod
    def from_input_batch(
            cls, input_batch: InputBatch,
            indices_do_sample: torch.Tensor) -> "TPUSupportedSamplingMetadata":
        """
        Copy sampling tensors slices from `input_batch` to on device tensors.

        `InputBatch._make_sampling_metadata` causes recompilation on XLA as it 
        slices dynamic shapes on device tensors. This impl moves the dynamic 
        ops to CPU and produces tensors of fixed `padded_num_reqs` size. It 
        also reuses the on-device persistent tensors managed in `input_batch`
        to reduce waste. 

        `indices_do_sample` contains the indices to be fed to the  Sampler, 
        normally one per request, here padded to the closest pre-compiled shape
        We expect sampling params tensors to be padded to the same fixed shape.

        Eg. 3 requests, tensors padded to 4 
            temperature: [0.7, 0.2, 0.9]=>[0.7, 0.2, 0.9, 0.0]
            sample indices: [4, 10, 11]=>indices_do_sample: [4, 10, 11, 0]
        """
        num_reqs = input_batch.num_reqs
        padded_num_reqs = len(indices_do_sample)

        def copy_slice(cpu_tensor: torch.Tensor, tpu_tensor: torch.Tensor,
                       fill_val) -> torch.Tensor:
            # Copy slice from CPU to corresponding TPU pre-allocated tensor.
            # Pad value is the default one.
            cpu_tensor[num_reqs:padded_num_reqs] = fill_val
            # Subtle compilation: len(tpu_tensor) must be >= `padded_num_reqs`
            tpu_tensor[:padded_num_reqs] = cpu_tensor[:padded_num_reqs]

        # NOTE NickLucche The sync CPU-TPU graph we produce here must be
        # consistent. We can't have flags to skip copies or we'll end up
        # recompiling.
        copy_slice(input_batch.temperature_cpu_tensor, input_batch.temperature,
                   DEFAULT_SAMPLING_PARAMS["temperature"])
        # TODO Temporarily disabled until sampling options are enabled
        # copy_slice(input_batch.top_p_cpu_tensor, input_batch.top_p)
        # copy_slice(input_batch.top_k_cpu_tensor, input_batch.top_k)
        copy_slice(input_batch.min_p_cpu_tensor, input_batch.min_p,
                   DEFAULT_SAMPLING_PARAMS["min_p"])

        xm.mark_step()
        xm.wait_device_ops()

        # Slice persistent device tensors to a fixed pre-compiled padded shape.
        return cls(
            temperature=input_batch.temperature[:padded_num_reqs],
            # Scalar tensor for xla-friendly tracing.
            all_greedy=torch.tensor(input_batch.all_greedy,
                                    dtype=torch.bool,
                                    device=input_batch.device),
            # TODO enable more and avoid returning None values
            top_p=None,  # input_batch.top_p[:padded_num_reqs],
            top_k=None,  # input_batch.top_k[:padded_num_reqs],
            min_p=input_batch.min_p[:padded_num_reqs],
            generators=input_batch.generators,
            indices_do_sample=indices_do_sample)
