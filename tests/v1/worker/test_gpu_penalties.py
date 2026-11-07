# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 penalties tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.model_states.prompt_embeds import PromptEmbedsState
from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcRequestState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128
HIDDEN_SIZE = 8


def _new_req_data(
    req_id: str,
    embeds_len: int = 0,
    prompt_is_token_ids: list[bool] | None = None,
) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=None,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=(torch.zeros(embeds_len, HIDDEN_SIZE) if embeds_len else None),
        prompt_is_token_ids=prompt_is_token_ids,
    )


def _prompt_bin_tokens(penalties: PenaltiesState, req_idx: int) -> set[int]:
    words = penalties.prompt_bin_mask[req_idx].cpu().numpy().astype(np.uint32)
    return {t for t in range(VOCAB_SIZE) if words[t // 32] & (1 << (t % 32))}


def test_bincount_ignores_prompt_embeds_positions():
    """Prompt-embedding positions must not contribute to the prompt bin mask.

    `all_token_ids` holds placeholder zeros at prompt-embedding positions, so
    without masking the bincount would count token 0 as a prompt token and the
    repetition penalty would wrongly scale its logit.
    """
    req_states = RequestState(
        max_num_reqs=3,
        max_model_len=64,
        max_num_batched_tokens=64,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    pe_state = PromptEmbedsState(3, HIDDEN_SIZE, torch.bfloat16, DEVICE)

    # Ordinary token prompt.
    req_states.add_request("tok", 3, [5, 6, 7], 0, 16)
    # Pure prompt-embeds prompt: all positions are placeholder zeros.
    req_states.add_request("embeds", 3, [0, 0, 0], 0, 16)
    # Mixed prompt: position 1 is an embedding (placeholder zero).
    req_states.add_request("mixed", 3, [9, 0, 10], 0, 16)
    pe_state.add_request(req_states.req_id_to_index["tok"], _new_req_data("tok"))
    pe_state.add_request(
        req_states.req_id_to_index["embeds"], _new_req_data("embeds", embeds_len=3)
    )
    pe_state.add_request(
        req_states.req_id_to_index["mixed"],
        _new_req_data("mixed", embeds_len=3, prompt_is_token_ids=[True, False, True]),
    )
    req_states.apply_staged_writes()
    pe_state.apply_staged_writes()

    penalties = PenaltiesState(
        None, LogitsProcRequestState.from_request_state(req_states), pe_state
    )
    params = SamplingParams(repetition_penalty=1.1)
    for req_idx in range(3):
        penalties.add_request(req_idx, params)
    penalties.apply_staged_writes()

    idx = req_states.req_id_to_index
    assert _prompt_bin_tokens(penalties, idx["tok"]) == {5, 6, 7}
    assert _prompt_bin_tokens(penalties, idx["embeds"]) == set()
    assert _prompt_bin_tokens(penalties, idx["mixed"]) == {9, 10}
