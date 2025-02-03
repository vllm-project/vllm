import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

try:
    import flashinfer.sampling as fs
    is_flashinfer_available = True
except ImportError:
    is_flashinfer_available = False

logger = init_logger(__name__)
INVALID_TOKEN_ID = -1


class RejectionSampler(nn.Module):

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> SamplerOutput:
        if not sampling_metadata.all_greedy:
            raise NotImplementedError(
                "Only greedy sampling is supported for now.")

        if is_flashinfer_available:
            return self.flashinfer_sample(logits, sampling_metadata)
        else:
            return self.greedy_sample_ref(logits, sampling_metadata)

    def flashinfer_sample(
            self, logits: torch.Tensor,
            sampling_metadata: SamplingMetadata) -> SamplerOutput:
        spec_token_ids = sampling_metadata.spec_token_ids
        spec_lengths = torch.tensor([len(s) for s in spec_token_ids],
                                    device="cpu")
        max_spec_len = torch.max(spec_lengths).item()
        batch_size = len(spec_lengths)
        draft_token_ids = torch.full((batch_size, max_spec_len),
                                     INVALID_TOKEN_ID,
                                     device="cpu",
                                     dtype=torch.long)

        target_token_ids = torch.full((batch_size, max_spec_len + 1),
                                      fill_value=INVALID_TOKEN_ID,
                                      device=logits.device,
                                      dtype=torch.long)

        start_loc = 0
        for i in range(batch_size):
            num_spec_tokens = len(spec_token_ids[i])
            draft_token_ids[i, :num_spec_tokens] = torch.tensor(
                spec_token_ids[i], device="cpu", dtype=torch.long)
            end_loc = start_loc + num_spec_tokens + 1
            # Assume greedy sampling here
            target_token_ids[i, :num_spec_tokens + 1] = torch.argmax(
                logits[start_loc:end_loc], dim=-1)
            start_loc = end_loc

        vocab_size = logits.size(-1)
        draft_token_ids = draft_token_ids.to(logits.device)
        draft_probs = self._create_greedy_token_probs(draft_token_ids,
                                                      vocab_size,
                                                      logits.device)
        target_probs = self._create_greedy_token_probs(target_token_ids,
                                                       vocab_size,
                                                       logits.device)
        uniform_samples = torch.zeros(batch_size,
                                      max_spec_len + 1,
                                      device=logits.device)

        sampled_token_ids, _, _ = fs.chain_speculative_sampling(
            draft_probs,
            draft_token_ids,
            uniform_samples,
            target_probs,
        )
        return SamplerOutput(sampled_token_ids=sampled_token_ids,
                             logprob_token_ids=None,
                             logprobs=None,
                             prompt_logprob_token_ids=None,
                             prompt_logprobs=None)

    def greedy_sample_ref(
            self, logits: torch.Tensor,
            sampling_metadata: SamplingMetadata) -> SamplerOutput:
        # num_reqs x [num_speculated_tokens]
        spec_token_ids = sampling_metadata.spec_token_ids
        # only argmax is supported for now
        output_token_ids_cpu = logits.argmax(dim=-1).view(-1).tolist()

        sampled_token_ids = []
        # Stop at the first mismatch place.
        # spec_tokens:    [1, 2, 3]
        # output_tokens:  [1, 2, 4, 5]
        # sampled_tokens: [1, 2, 4]
        output_token_start_idx = 0
        max_spec_len = -1
        for spec_tokens in spec_token_ids:
            num_spec_tokens = len(spec_tokens)
            max_spec_len = max(max_spec_len, num_spec_tokens)
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

        sampled_token_ids = [
            x + [INVALID_TOKEN_ID] * (max_spec_len + 1 - len(x))
            for x in sampled_token_ids
        ]
        sampled_token_ids = torch.tensor(sampled_token_ids,
                                         device=logits.device,
                                         dtype=torch.int)

        assert output_token_start_idx == len(output_token_ids_cpu)

        return SamplerOutput(sampled_token_ids=sampled_token_ids,
                             logprob_token_ids=None,
                             logprobs=None,
                             prompt_logprob_token_ids=None,
                             prompt_logprobs=None)

    def _create_greedy_token_probs(self, token_ids: torch.Tensor,
                                   vocab_size: int,
                                   out_device: torch.device) -> torch.Tensor:
        batch_size, num_tokens = token_ids.shape

        token_probs = torch.zeros(batch_size,
                                  num_tokens,
                                  vocab_size,
                                  dtype=torch.float,
                                  device=out_device)

        # Ignore INVALID_TOKEN_ID
        valid_mask = (token_ids != INVALID_TOKEN_ID)
        valid_indices = token_ids.clone()
        valid_indices[~valid_mask] = 0

        token_probs.scatter_(dim=2,
                             index=valid_indices.unsqueeze(-1),
                             src=valid_mask.unsqueeze(-1).float())

        return token_probs
