# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that M-RoPE position initialization handles prompt_embeds-only inputs.

Regression test for GHSA-33cg-gxv8-3p8g: sending /v1/completions with
prompt_embeds and no prompt_token_ids on M-RoPE models crashed the
EngineCore via an assertion failure.
"""

import pytest
import torch
from transformers import Qwen2VLConfig

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.request import Request
from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

NUM_DIMS = 3


class FakeMRoPEModel(torch.nn.Module, SupportsMRoPE):
    """Minimal model that passes supports_mrope() check."""

    def get_mrope_input_positions(self, input_tokens, mm_features):
        seq_len = len(input_tokens)
        positions = torch.arange(seq_len).unsqueeze(0).expand(NUM_DIMS, -1)
        return positions.clone(), 0


def _add_request(
    tmp_path, prompt_token_ids: list[int] | None, prompt_embeds: torch.Tensor | None
) -> RopeState:
    """Schedule a new request the way the scheduler does and add it to a
    model state with M-RoPE; return the initialized RoPE state."""
    request = Request(
        request_id="req",
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        prompt_embeds=prompt_embeds,
    )
    new_req_data = NewRequestData.from_request(
        request, ([0],), request.prefill_token_ids, uses_mrope=True
    )

    Qwen2VLConfig(
        architectures=["Qwen2VLForConditionalGeneration"],
        text_config={
            "hidden_size": 896,
            "num_attention_heads": 7,
            "num_key_value_heads": 1,
            "num_hidden_layers": 1,
            "intermediate_size": 128,
            "rope_parameters": {"rope_type": "default", "mrope_section": [16, 24, 24]},
        },
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            max_model_len=32,
            enable_prompt_embeds=True,
        ),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_num_seqs=1,
            max_num_batched_tokens=32,
            max_model_len=32,
        ),
    )
    model_state = DefaultModelState(
        config, FakeMRoPEModel(), None, torch.device("cuda", 0)
    )
    model_state.add_request(0, new_req_data)
    model_state.apply_staged_writes()
    assert model_state.rope_state is not None
    return model_state.rope_state


@pytest.mark.parametrize(
    ("prompt_token_ids", "prompt_embeds", "num_tokens"),
    [
        pytest.param(None, torch.randn(15, 896), 15, id="prompt-embeds-only"),
        pytest.param([1, 2, 3, 4, 5], None, 5, id="token-ids"),
    ],
)
def test_mrope_positions_initialized(
    tmp_path, prompt_token_ids, prompt_embeds, num_tokens
):
    rope_state = _add_request(tmp_path, prompt_token_ids, prompt_embeds)
    positions = rope_state.read_prefill_positions(0, num_tokens).cpu()
    expected = torch.arange(num_tokens, dtype=torch.int32).expand(NUM_DIMS, -1)
    torch.testing.assert_close(positions, expected)
    assert rope_state.get_prefill_delta(0) == 0
