# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 logit bias tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState

DEVICE = torch.device("cuda")
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


def _apply(logits: torch.Tensor, structured: list[bool]) -> torch.Tensor:
    """Run the fused bias kernel with one logits row per request."""
    state = LogitBiasState(max_num_reqs=4, device=DEVICE)
    for req_idx, is_structured in enumerate(structured):
        state.add_request(req_idx, PROMPT_LEN, _params(is_structured))
    state.apply_staged_writes()

    num_reqs = len(structured)
    state.apply_logit_bias(
        logits,
        torch.arange(num_reqs, dtype=torch.int32, device=DEVICE),
        np.arange(num_reqs, dtype=np.intp),
        torch.full((num_reqs,), POS, dtype=torch.int32, device=DEVICE),
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
