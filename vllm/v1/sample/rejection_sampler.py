import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

logger = init_logger(__name__)


class RejectionSampler(nn.Module):

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> SamplerOutput:
        # num_reqs x [num_specuated_tokens]
        spec_token_ids = sampling_metadata.spec_token_ids
        # only argmax is supported for now
        output_token_ids_cpu = logits.argmax(dim=-1).view(-1).tolist()

        sampled_token_ids = []
        # Stop at the first mismatch place.
        # spec_tokens:    [1, 2, 3]
        # output_tokens:  [1, 2, 4, 5]
        # sampled_tokens: [1, 2, 4]
        output_token_start_idx = 0
        for spec_tokens in spec_token_ids:
            num_spec_tokens = len(spec_tokens)
            output_tokens = output_token_ids_cpu[
                output_token_start_idx:output_token_start_idx + 1 +
                num_spec_tokens]
            i = 0
            while i < len(spec_tokens):
                if spec_tokens[i] != output_tokens[i]:
                    break
                i += 1
            # +1 to include the bonus token.
            i += 1
            output_tokens = output_tokens[:i]
            sampled_token_ids.append(output_tokens)
            output_token_start_idx += num_spec_tokens + 1
        assert output_token_start_idx == len(output_token_ids_cpu)

        return SamplerOutput(sampled_token_ids=sampled_token_ids,
                             logprob_token_ids=None,
                             logprobs=None,
                             prompt_logprob_token_ids=None,
                             prompt_logprobs=None)
