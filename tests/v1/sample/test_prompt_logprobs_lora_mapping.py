# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample import prompt_logprob


def test_prompt_logprobs_lora_mapping_is_restored(monkeypatch) -> None:
    token_lora_indices = torch.tensor([0, 1] * 512 + [0], dtype=torch.int32)
    sampler_indices = torch.tensor([2, 3], dtype=torch.int32)
    wrapper = SimpleNamespace(
        token_lora_indices=token_lora_indices,
        sampler_indices=sampler_indices,
        prompt_mapping_meta=LoRAKernelMeta.make(4, 1025, "cpu"),
    )
    wrapper.prompt_mapping_meta.prepare_tensors(sampler_indices)
    original_mapping = wrapper.prompt_mapping_meta.token_lora_mapping[:2].clone()
    seen_mappings = []

    def logits_fn(hidden_states: torch.Tensor) -> torch.Tensor:
        seen_mappings.append(
            wrapper.prompt_mapping_meta.token_lora_mapping[
                : hidden_states.shape[0]
            ].clone()
        )
        return torch.zeros((hidden_states.shape[0], 8))

    def fake_compute_topk_scores(
        logits: torch.Tensor,
        num_logprobs: int,
        target_token_ids: torch.Tensor,
        **kwargs,
    ) -> LogprobsTensors:
        return LogprobsTensors(
            logprob_token_ids=target_token_ids[:, None],
            logprobs=torch.zeros((logits.shape[0], 1)),
            selected_token_ranks=torch.ones(logits.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", fake_compute_topk_scores)
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.arange(1025),
        torch.zeros((1025, 2)),
        logits_fn,
        0,
        lora_wrapper=wrapper,
    )

    assert [mapping.numel() for mapping in seen_mappings] == [1024, 1]
    assert torch.equal(seen_mappings[0], token_lora_indices[:1024])
    assert torch.equal(seen_mappings[1], token_lora_indices[1024:])
    assert torch.equal(
        wrapper.prompt_mapping_meta.token_lora_mapping[:2], original_mapping
    )
