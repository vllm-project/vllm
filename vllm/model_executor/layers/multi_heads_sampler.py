"""A layer that samples the next tokens from the model's outputs."""
import itertools
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.model_executor.layers.ops.sample import sample as sample_triton
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
                                                   SamplingTensors,
                                                   SequenceGroupToSample)
from vllm.sampling_params import SamplingType
from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                           PromptLogprobs, SampleLogprobs, SamplerOutput,
                           SequenceOutput)
from vllm.model_executor.layers.sampler import Sampler, _apply_top_k_top_p, _sample, _get_logprobs, _build_sampler_output

# (num_token_ids, num_parent_ids) per sequence group.
SampleResultType = List[Tuple[List[int], List[int]]]


class MultiheadsSampler(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()

        # Whether or not the SamplerOutput should have on-device tensors
        # containing the sampled token ids and probabilities. This is used by
        # speculative decoding.
        self.num_heads = num_heads
        self.include_gpu_probs_tensor = False
        self.heads = nn.ModuleList([Sampler() for _ in range(num_heads)])

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        # Sample from each head
        head_logits = logits.permute(1, 0, 2)
        output0 = self.heads[0](head_logits[0], sampling_metadata)
        for i in range(self.num_heads - 1):
            output = self.heads[i + 1](head_logits[i], sampling_metadata)
            self.merge_sample_results(output0, output)

        return output0

    def merge_sample_results(
        self,
        source: SamplerOutput,
        target: SamplerOutput,
    ):
        for o_a, o_b in zip(source.outputs, target.outputs):
            for s_a, s_b in zip(o_a.samples, o_b.samples):
                s_a.output_tokens.append(s_b.output_token)