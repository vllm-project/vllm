# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
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

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda:
            if is_flashinfer_available:
                if envs.VLLM_USE_FLASHINFER_SAMPLER is not False:
                    # NOTE(woosuk): The V0 sampler doesn't use FlashInfer for
                    # sampling unless VLLM_USE_FLASHINFER_SAMPLER=1 (i.e., by
                    # default it is unused). For backward compatibility, we set
                    # `VLLM_USE_FLASHINFER_SAMPLER` as None by default and
                    # interpret it differently in V0 and V1 samplers: In V0,
                    # None means False, while in V1, None means True. This is
                    # why we use the condition
                    # `envs.VLLM_USE_FLASHINFER_SAMPLER is not False` here.
                    logger.info("Using FlashInfer for rejection sampling.")
                    self.forward_method = self.flashinfer_sample
                else:
                    logger.warning(
                        "FlashInfer is available, but it is not enabled. "
                        "Falling back to the PyTorch-native implementation of "
                        "rejection sampling. For the best performance, "
                        "please set VLLM_USE_FLASHINFER_SAMPLER=1.")
                    self.forward_method = self.forward_native
            else:
                logger.warning(
                    "FlashInfer is not available. Falling back to the PyTorch-"
                    "native implementation of rejection sampling. For the "
                    "best performance, please install FlashInfer.")
                self.forward_method = self.forward_native
        else:
            self.forward_method = self.forward_native

    def forward(self, draft_token_ids: list[list[int]],
                target_probs: torch.Tensor,
                sampling_metadata: SamplingMetadata) -> SamplerOutput:
        if not sampling_metadata.all_greedy:
            raise NotImplementedError(
                "Currently, only greedy sampling is supported by "
                "rejection sampler.")
        return self.forward_method(draft_token_ids, target_probs,
                                   sampling_metadata)

    def flashinfer_sample(
        self,
        draft_token_ids: list[list[int]],
        target_probs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        # NOTE: The following input preparationg can be moved
        # to the model runner with a persistent manner for better
        # performance.
        sample_lens = [len(x) + 1 for x in draft_token_ids]
        # Convert draft token IDs to a tensor, split by sample_lens, then pad.
        draft_token_ids = [
            torch.tensor(x, dtype=int, device='cpu') for x in draft_token_ids
        ]
        draft_token_ids_tensor = pad_sequence(draft_token_ids,
                                              batch_first=True,
                                              padding_value=INVALID_TOKEN_ID)

        if sampling_metadata.all_greedy:
            target_token_ids = target_probs.argmax(dim=-1).view(-1)
            target_token_ids = target_token_ids.split(sample_lens)
            target_token_ids = pad_sequence(target_token_ids,
                                            batch_first=True,
                                            padding_value=INVALID_TOKEN_ID)

            vocab_size = target_probs.size(-1)
            # NOTE: CPU <-> GPU synchronization happens here.
            draft_token_ids_tensor = draft_token_ids_tensor.to(
                target_probs.device)
            draft_probs = _create_greedy_token_probs(draft_token_ids_tensor,
                                                     vocab_size,
                                                     target_probs.device)
            target_probs = _create_greedy_token_probs(target_token_ids,
                                                      vocab_size,
                                                      target_probs.device)
            uniform_samples = torch.zeros(draft_token_ids_tensor.size(0),
                                          draft_token_ids_tensor.size(1) + 1,
                                          device=target_probs.device)
        else:
            raise NotImplementedError(
                "Currently, only greedy sampling is supported by "
                "rejection sampler.")

        sampled_token_ids, _, _ = fs.chain_speculative_sampling(
            draft_probs,
            draft_token_ids_tensor,
            uniform_samples,
            target_probs,
        )
        return SamplerOutput(sampled_token_ids=sampled_token_ids,
                             logprobs_tensors=None)

    # TODO: The following method can be optimized for better performance.
    def forward_native(
        self,
        draft_token_ids: list[list[int]],
        target_probs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        sample_lens = [len(x) + 1 for x in draft_token_ids]
        # Convert draft token IDs to a tensor, split by sample_lens, then pad.
        draft_token_ids = [
            torch.tensor(x, dtype=int, device='cpu') for x in draft_token_ids
        ]
        draft_token_ids_tensor = pad_sequence(draft_token_ids,
                                              batch_first=True,
                                              padding_value=INVALID_TOKEN_ID)
        draft_token_ids_tensor = draft_token_ids_tensor.to(target_probs.device)
        # Add 1 to include the 'bonus' token.
        if sampling_metadata.all_greedy:
            output_token_ids = target_probs.argmax(dim=-1).view(-1)
            output_token_ids = output_token_ids.split(sample_lens)
            output_token_ids = pad_sequence(output_token_ids,
                                            batch_first=True,
                                            padding_value=INVALID_TOKEN_ID)
            # Produce a mask that remains 1 (True) until the first
            # mismatch (cumprod turns 0 after a mismatch).
            accept_mask = (
                output_token_ids[:, :-1] == draft_token_ids_tensor).cumprod(
                    dim=1)
        else:
            raise NotImplementedError(
                "Currently, only greedy sampling is supported by "
                "rejection sampler.")
        # Identify valid positions (non-padding).
        valid_mask = output_token_ids != INVALID_TOKEN_ID
        # Generate mask with bonus token.
        generate_mask = torch.cat([
            accept_mask,
            torch.zeros(accept_mask.size(0), 1, device=accept_mask.device)
        ],
                                  dim=1).to(torch.bool) & valid_mask
        zeros_mask = (generate_mask == 0)
        first_zero_idx = zeros_mask.float().argmax(dim=1)
        # Figure out which rows actually contain at least one zero.
        rows_with_zero = zeros_mask.any(dim=1)
        # Use indexing to set the first zero in each of those rows to 1.
        generate_mask[rows_with_zero, first_zero_idx[rows_with_zero]] = 1

        output_token_ids[~generate_mask] = INVALID_TOKEN_ID
        return SamplerOutput(sampled_token_ids=output_token_ids,
                             logprobs_tensors=None)


def _create_greedy_token_probs(
    token_ids: torch.Tensor,
    vocab_size: int,
    out_device: torch.device,
) -> torch.Tensor:
    batch_size, num_tokens = token_ids.shape

    token_probs = torch.zeros(batch_size,
                              num_tokens,
                              vocab_size,
                              dtype=torch.float,
                              device=out_device)

    # Ignore INVALID_TOKEN_ID.
    valid_mask = (token_ids != INVALID_TOKEN_ID)
    valid_indices = token_ids.clone()
    valid_indices[~valid_mask] = 0

    token_probs.scatter_(dim=2,
                         index=valid_indices.unsqueeze(-1),
                         src=valid_mask.unsqueeze(-1).float())

    return token_probs
