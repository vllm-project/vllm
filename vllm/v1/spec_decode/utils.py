# SPDX-License-Identifier: Apache-2.0
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import random_sample  # noqa


def is_spec_decode_supported(req_idx: int, sampling_metadata: SamplingMetadata,
                             vocab_size: int) -> bool:
    if ((sampling_metadata.top_p and sampling_metadata.top_p[req_idx] < 1.0)
            or (sampling_metadata.top_k
                and 0 < sampling_metadata.top_k[req_idx] < vocab_size)):
        # Spec decode doesn't support top_p/top_k sampling.
        return False
    elif sampling_metadata.min_p and sampling_metadata.min_p[req_idx] > 0.0:
        # Spec decode doesn't support min_p sampling.
        return False
    elif (sampling_metadata.frequency_penalties[req_idx] != 0.0
          or sampling_metadata.presence_penalties[req_idx] != 0.0
          or sampling_metadata.repetition_penalties[req_idx] != 1.0):
        # Spec decode doesn't support penalties.
        return False
    elif sampling_metadata.max_num_logprobs is not None:
        # Spec decode doesn't support logprobs.
        return False

    return True
