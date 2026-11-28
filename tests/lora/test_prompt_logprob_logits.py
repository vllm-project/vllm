# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt-logprob logits must use the token-row LoRA ids, not the sampler rows."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.layers import LoRAMapping
from vllm.lora.layers.logits_processor import compute_logits_for_prompt_rows
from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.prompt_logprob import (
    CHUNK_SIZE,
    compute_prompt_logprobs_with_chunking,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="lm_head LoRA shrink/expand runs on CUDA"
)


@pytest.fixture(autouse=True)
def cleanup_fixture():
    """Override conftest's cleanup_fixture — not needed for this file."""
    yield


@pytest.fixture(autouse=True, scope="module")
def cleanup_after_module():
    yield
    cleanup_dist_env_and_memory()


def _reference_logits(
    hidden: torch.Tensor,
    row_slots: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> torch.Tensor:
    """y[i] = hidden[i] @ A[slot]^T @ B[slot]^T, or 0 when slot < 0."""
    vocab = lora_b.shape[-2]
    out = torch.zeros(hidden.shape[0], vocab, dtype=torch.float32, device=hidden.device)
    for row, slot in enumerate(row_slots.tolist()):
        if slot < 0:
            continue
        shrink = hidden[row].float() @ lora_a[slot, 0].float().T
        out[row] = shrink @ lora_b[slot, 0].float().T
    return out


class _LogitsModel(torch.nn.Module):
    def __init__(self, wrapper, lora_a, lora_b):
        super().__init__()
        self.logits_processor = torch.nn.Identity()
        self.logits_processor.punica_wrapper = wrapper
        self.lora_a = lora_a
        self.lora_b = lora_b

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            hidden_states.shape[0],
            self.lora_b.shape[-2],
            dtype=torch.float32,
            device=hidden_states.device,
        )
        self.logits_processor.punica_wrapper.add_lora_logits(
            logits, hidden_states, self.lora_a, self.lora_b, 1.0
        )
        return logits


def test_prompt_row_logits_follow_token_lora_ids() -> None:
    """A prompt-logprob slice uses that slice's token LoRA ids.

    Sampler metadata stays the metadata of the sampling rows. Applying it to a
    shorter prompt slice both indexes the wrong rows and can run past the slice.
    """
    device = torch.device(f"{current_platform.device_type}:0")
    torch.accelerator.set_device_index(device)
    max_loras, rank, hidden_size, vocab = 1, 4, 8, 6
    # Token rows: two base, two adapter, one base, one adapter.
    # Sampler rows: four adapter requests. The lengths differ on purpose.
    token_lora_ids = (-1, -1, 1, 1, -1, 1)
    sampler_lora_ids = (1, 1, 1, 1)
    wrapper = PunicaWrapperGPU(
        max_num_batched_tokens=16,
        max_batches=8,
        device=device,
        lora_config=SimpleNamespace(max_loras=max_loras, specialize_active_lora=False),
    )
    wrapper.update_metadata(
        LoRAMapping(token_lora_ids, sampler_lora_ids, is_prefill=True),
        lora_index_to_id=[1],
        max_loras=max_loras,
        vocab_size=vocab,
    )

    lora_a = torch.zeros(
        max_loras, 1, rank, hidden_size, dtype=torch.bfloat16, device=device
    )
    lora_b = torch.zeros(max_loras, 1, vocab, rank, dtype=torch.bfloat16, device=device)
    lora_a[0, 0, :, :rank] = torch.eye(rank, dtype=torch.bfloat16, device=device)
    lora_b[0, 0, :rank, :] = torch.eye(rank, dtype=torch.bfloat16, device=device)

    hidden = torch.arange(
        len(token_lora_ids) * hidden_size, dtype=torch.bfloat16, device=device
    ).reshape(len(token_lora_ids), hidden_size)
    # Rows [2, 5): slots 0, 0, -1. Shorter than the four sampler rows.
    row_start, row_end = 2, 5
    logits = compute_logits_for_prompt_rows(
        _LogitsModel(wrapper, lora_a, lora_b),
        hidden[row_start:row_end],
        row_start,
    )
    assert logits is not None
    expected = _reference_logits(
        hidden[row_start:row_end],
        wrapper.token_lora_indices[row_start:row_end],
        lora_a,
        lora_b,
    )
    torch.testing.assert_close(logits, expected, atol=1e-2, rtol=1e-2)
    assert logits[2].abs().sum().item() == 0

    # The sampler mapping is intact for a later logits call of sampler width.
    sampler_hidden = torch.ones(
        len(sampler_lora_ids), hidden_size, dtype=torch.bfloat16, device=device
    )
    sampler_logits = torch.zeros(
        len(sampler_lora_ids), vocab, dtype=torch.float32, device=device
    )
    wrapper.add_lora_logits(sampler_logits, sampler_hidden, lora_a, lora_b, 1.0)
    torch.testing.assert_close(
        sampler_logits,
        _reference_logits(sampler_hidden, wrapper.sampler_indices, lora_a, lora_b),
        atol=1e-2,
        rtol=1e-2,
    )


def test_prompt_logprob_chunks_report_their_row_offset() -> None:
    device = torch.device(f"{current_platform.device_type}:0")
    num_tokens = CHUNK_SIZE + 10
    seen: list[int] = []

    def logits_fn(hidden_states: torch.Tensor, row_start: int = 0) -> torch.Tensor:
        seen.append(row_start)
        return torch.zeros(
            hidden_states.shape[0], 4, dtype=torch.float32, device=hidden_states.device
        )

    compute_prompt_logprobs_with_chunking(
        torch.zeros(num_tokens, dtype=torch.int64, device=device),
        torch.zeros(num_tokens, 4, dtype=torch.bfloat16, device=device),
        logits_fn,
        num_prompt_logprobs=0,
    )
    assert seen == [0, CHUNK_SIZE]
