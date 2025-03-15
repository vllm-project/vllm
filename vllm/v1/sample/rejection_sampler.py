# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm.utils import is_pin_memory_available
from vllm.v1.sample.metadata import SamplingMetadata

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
GREEDY_TEMPERATURE: tl.constexpr = -1
TINY: tl.constexpr = 1.1754943508222875e-38  # torch.finfo(torch.float32).tiny


class RejectionSampler(nn.Module):

    def __init__(
        self,
        max_batch_size: int = 8 * 1024,
        max_num_draft_tokens: int = 32 * 1024,
    ):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_num_draft_tokens = max_num_draft_tokens

        self.cu_num_tokens_buffer = torch.empty(
            max_batch_size,
            dtype=torch.int32,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )
        self.cu_num_tokens_buffer_np = self.cu_num_tokens_buffer.numpy()

        self.token_ids_buffer = torch.empty(
            max_num_draft_tokens,
            dtype=torch.int64,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )
        self.token_ids_buffer_np = self.token_ids_buffer.numpy()

    def forward(
        self,
        # batch_size x [0, max_spec_len)
        draft_token_ids: list[list[int]],
        # [num_tokens, vocab_size]
        draft_probs: Optional[torch.Tensor],
        # [num_tokens, vocab_size]
        target_logits: torch.Tensor,
        # [batch_size]
        bonus_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        token_ids, cu_num_draft_tokens = self._async_copy_to_device(
            draft_token_ids,
            target_logits.device,
        )

        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        max_spec_len = max(num_draft_tokens)
        # [num_tokens, vocab_size]
        target_probs = compute_probs(
            target_logits,
            sampling_metadata.temperature,
            cu_num_draft_tokens,
            max_spec_len,
        )

        output_token_ids = rejection_sample(
            token_ids,
            num_draft_tokens,
            cu_num_draft_tokens,
            draft_probs,
            target_probs,
            bonus_token_ids,
            sampling_metadata,
        )
        return output_token_ids

    @staticmethod
    def parse_output(output_token_ids: torch.Tensor) -> list[list[int]]:
        output_token_ids = output_token_ids.tolist()
        # Preallocate outputs.
        outputs: list[list[int]] = [[] for _ in output_token_ids]
        for i, token_ids in enumerate(output_token_ids):
            for token_id in token_ids:
                if token_id == PLACEHOLDER_TOKEN_ID:
                    break
                outputs[i].append(token_id)
        return outputs

    def _async_copy_to_device(
        self,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flattened_token_ids: list[int] = []
        cu_num_tokens: list[int] = []
        for token_ids in draft_token_ids:
            flattened_token_ids.extend(token_ids)
            cu_num_tokens.append(len(token_ids))

        num_draft_tokens = len(flattened_token_ids)
        assert num_draft_tokens <= self.max_num_draft_tokens
        self.token_ids_buffer_np[:num_draft_tokens] = flattened_token_ids
        draft_token_ids_cpu = self.token_ids_buffer[:num_draft_tokens]
        draft_token_ids_gpu = draft_token_ids_cpu.to(device=device,
                                                     non_blocking=True)

        batch_size = len(cu_num_tokens)
        self.cu_num_tokens_buffer_np[:batch_size] = cu_num_tokens
        cu_num_draft_tokens_cpu = self.cu_num_tokens_buffer[:batch_size]
        cu_num_draft_tokens_gpu = cu_num_draft_tokens_cpu.to(device=device,
                                                             non_blocking=True)
        return draft_token_ids_gpu, cu_num_draft_tokens_gpu


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    batch_size = len(num_draft_tokens)
    max_spec_len = max(num_draft_tokens)
    num_tokens = sum(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()

    # Create output buffer.
    output_token_ids = torch.empty(
        (batch_size, max_spec_len + 1),
        dtype=torch.int64,
        device=device,
    )
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

    is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size, )](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata,
        device,
    )

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Rejection sampling for random sampling requests.
    rejection_random_sample_kernel[(batch_size, )](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        IS_NGRAM=draft_probs is None,
    )
    return output_token_ids


def compute_probs(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    temperature: torch.Tensor,  # [batch_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    max_spec_len: int,
) -> torch.Tensor:
    output_prob = torch.empty_like(logits, dtype=torch.float32)
    batch_size = temperature.shape[0]
    vocab_size = logits.shape[-1]
    compute_probs_kernel[(batch_size, max_spec_len)](
        output_prob,
        logits,
        temperature,
        cu_num_draft_tokens,
        vocab_size,
        triton.next_power_of_2(vocab_size),
    )
    return output_prob


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    uniform_probs = torch.rand(
        (num_tokens, ),
        dtype=torch.float32,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = sampling_metadata.generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    recovered_token_ids = torch.empty_like(draft_token_ids)
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        triton.next_power_of_2(vocab_size),
        IS_NGRAM=draft_probs is None,
    )
    return recovered_token_ids


@triton.jit
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                     target_argmax_id)
            if draft_token_id != target_argmax_id:
                # Reject.
                rejected = True

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) +
            num_draft_tokens, bonus_token_id)


@triton.jit
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    IS_NGRAM: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            if IS_NGRAM:
                draft_prob = 1
            else:
                draft_prob = tl.load(draft_probs_ptr +
                                     (start_idx + pos) * vocab_size +
                                     draft_token_id)
            target_prob = tl.load(target_probs_ptr +
                                  (start_idx + pos) * vocab_size +
                                  draft_token_id)
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            if draft_prob == 0 or (target_prob / draft_prob >= uniform_prob):
                # Accept.
                token_id = draft_token_id
            else:
                # Reject. Use recovered token.
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                     token_id)

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) +
            num_draft_tokens, bonus_token_id)


@triton.jit
def compute_probs_kernel(
    output_prob_ptr,  # [num_tokens, vocab_size]
    logits_ptr,  # [num_tokens, vocab_size]
    temperature_ptr,  # [batch_size]
    cu_num_draft_tokens_ptr,  # [batch_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= end_idx - start_idx:
        return

    vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)
    logits = tl.load(logits_ptr + (start_idx + pos) * vocab_size +
                     vocab_offset,
                     mask=vocab_offset < vocab_size)
    logits = logits.to(dtype=tl.float32)
    temperature = tl.load(temperature_ptr + req_idx)
    if temperature == GREEDY_TEMPERATURE:
        # Greedy sampling. Just return the logits.
        output_prob = logits
    else:
        # Random sampling.
        output_prob = tl.softmax(logits / temperature)

    tl.store(output_prob_ptr + (start_idx + pos) * vocab_size + vocab_offset,
             output_prob,
             mask=vocab_offset < vocab_size)


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    IS_NGRAM: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)
    if IS_NGRAM:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        orig_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size +
                            draft_token_id)
        # Temporarily zero out the probability of the draft token.
        tl.store(
            target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id,
            0)
        prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size +
                       vocab_offset,
                       mask=vocab_offset < vocab_size)
    else:
        draft_prob = tl.load(draft_probs_ptr + (start_idx + pos) * vocab_size +
                             vocab_offset,
                             mask=vocab_offset < vocab_size)
        target_prob = tl.load(target_probs_ptr +
                              (start_idx + pos) * vocab_size + vocab_offset,
                              mask=vocab_offset < vocab_size)
        prob = target_prob - draft_prob
        prob = tl.maximum(prob, TINY)
    prob = prob / tl.sum(prob, axis=-1, keep_dims=True)

    q = tl.load(q_ptr + req_idx * vocab_size + vocab_offset,
                mask=vocab_offset < vocab_size)
    recovered_id = tl.argmax(prob / q, axis=-1)
    tl.store(output_token_ids_ptr + start_idx + pos, recovered_id)
    if IS_NGRAM:
        tl.store(
            target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id,
            orig_prob)
