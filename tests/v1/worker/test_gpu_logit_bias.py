# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytest.importorskip("triton")
if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
    pytest.skip(
        "CUDA required for Model Runner V2 logit bias tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.sample.penalties import bincount
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device(current_platform.device_type)
VOCAB_SIZE = 128
STOP_TOKEN = 3
OTHER_TOKEN = 4
PROMPT_LEN = 2
MIN_TOKENS = 10
# min_len is PROMPT_LEN + MIN_TOKENS, so this position is below the minimum
# and the kernel masks stop tokens.
POS = 5


def _params(structured: bool) -> SamplingParams:
    return SamplingParams(
        min_tokens=MIN_TOKENS,
        stop_token_ids=[STOP_TOKEN],
        structured_outputs=(
            StructuredOutputsParams(json={"type": "boolean"}) if structured else None
        ),
    )


def _only_stop_token_left(num_rows: int) -> torch.Tensor:
    """Logits as a terminal grammar leaves them: every token masked but one."""
    logits = torch.full((num_rows, VOCAB_SIZE), -float("inf"), device=DEVICE)
    logits[:, STOP_TOKEN] = 1.0
    return logits


def _apply(
    logits: torch.Tensor,
    structured: list[bool],
    params: list[SamplingParams] | None = None,
) -> torch.Tensor:
    """Run the fused bias kernel with one logits row per request."""
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    # add_request() reads prompt_len per slot; every slot here uses the same one.
    req_states.prompt_len.np[:] = PROMPT_LEN

    state = LogitBiasState(None, LogitsProcRequestState.from_request_state(req_states))
    for req_idx, is_structured in enumerate(structured):
        state.add_request(
            req_idx, params[req_idx] if params else _params(is_structured)
        )
    state.apply_staged_writes()

    num_reqs = len(structured)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    state.apply(
        logits,
        LogitsContext(
            expanded_idx_mapping=idx_mapping,
            idx_mapping=idx_mapping,
            idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
            expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
            input_ids=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
            pos=torch.full((num_reqs,), POS, dtype=torch.int32, device=DEVICE),
            seq_lens_upper_bound_np=np.full(num_reqs, POS + 1, dtype=np.int64),
        ),
    )
    return logits.cpu()


def test_v2_min_tokens_restores_stop_token_when_row_would_be_empty():
    """Structured output plus min_tokens must not leave an all -inf row."""
    out = _apply(_only_stop_token_left(1), structured=[True])
    assert out[0, STOP_TOKEN] == 1.0


def test_v2_min_tokens_keeps_empty_row_without_structured_output():
    """Plain min_tokens requests keep the pre-existing masking behavior."""
    out = _apply(_only_stop_token_left(1), structured=[False])
    assert torch.isneginf(out).all()


def test_v2_min_tokens_keeps_masking_when_other_candidates_remain():
    """The fallback must not weaken min_tokens while a non-stop token is legal."""
    logits = _only_stop_token_left(1)
    logits[0, OTHER_TOKEN] = 2.0

    out = _apply(logits, structured=[True])

    assert torch.isneginf(out[0, STOP_TOKEN])
    assert out[0, OTHER_TOKEN] == 2.0


def test_v2_min_tokens_mixed_batch_gates_restore_per_request():
    """CHECK_ALL_MASKED_ROWS is a batch-wide constexpr, so a mixed batch must
    still consult the per-request flag."""
    out = _apply(_only_stop_token_left(2), structured=[True, False])

    assert out[0, STOP_TOKEN] == 1.0
    assert torch.isneginf(out[1]).all()


@pytest.mark.parametrize(
    "parameter", ["allowed_token_ids", "logit_bias", "stop_token_ids"]
)
@pytest.mark.parametrize("structured", [False, True])
def test_v2_token_bounds_preserve_neighboring_logits(parameter, structured):
    """An ID outside the vocabulary must not read or write a neighboring row."""
    ids = [-1, STOP_TOKEN, VOCAB_SIZE]
    values = {token_id: 2.0 for token_id in ids} if parameter == "logit_bias" else ids
    params = SamplingParams(
        **{parameter: values},
        min_tokens=MIN_TOKENS,
        structured_outputs=StructuredOutputsParams(json={"type": "boolean"})
        if structured
        else None,
    )
    logits = torch.arange(3 * VOCAB_SIZE, device=DEVICE, dtype=torch.float32).reshape(
        3, VOCAB_SIZE
    )
    expected = logits.cpu().clone()
    if parameter == "allowed_token_ids":
        expected[1] = -float("inf")
        expected[1, STOP_TOKEN] = logits[1, STOP_TOKEN].item()
    elif parameter == "logit_bias":
        expected[1, STOP_TOKEN] += 2.0
    else:
        expected[1, STOP_TOKEN] = -float("inf")
    out = _apply(
        logits, [False, structured, False], [SamplingParams(), params, SamplingParams()]
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_v2_penalty_counts_ignore_ids_outside_model_vocab():
    """Tokenizer-only prompt/output IDs must leave adjacent request counts intact."""
    tokens = torch.tensor(
        [[0] * 6, [-1, 5, VOCAB_SIZE, -1, 6, VOCAB_SIZE], [0] * 6],
        dtype=torch.int32,
        device=DEVICE,
    )
    prompt_mask = torch.zeros((3, VOCAB_SIZE // 32), dtype=torch.int32, device=DEVICE)
    output_counts = torch.zeros((3, VOCAB_SIZE), dtype=torch.int32, device=DEVICE)
    bincount(
        torch.tensor([1], dtype=torch.int32, device=DEVICE),
        tokens,
        torch.tensor([0, 3, 0], dtype=torch.int32, device=DEVICE),
        torch.tensor([0, 6, 0], dtype=torch.int32, device=DEVICE),
        prompt_mask,
        output_counts,
        max_prefill_len=6,
    )
    expected_mask = torch.zeros_like(prompt_mask)
    expected_mask[1, 0] = 1 << 5
    expected_counts = torch.zeros_like(output_counts)
    expected_counts[1, 6] = 1
    torch.testing.assert_close(prompt_mask, expected_mask, rtol=0, atol=0)
    torch.testing.assert_close(output_counts, expected_counts, rtol=0, atol=0)
