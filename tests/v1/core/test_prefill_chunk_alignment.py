# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler, mamba_layer

pytestmark = pytest.mark.cpu_test


def _scheduler(alignment, max_num_batched_tokens=200):
    spec = MambaSpec(block_size=16, shapes=((1, 1),), dtypes=(torch.float32,))
    return create_scheduler(
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=2048,
        mamba_layer_config=mamba_layer(spec, prefill_chunk_alignment=alignment),
    )


@pytest.mark.parametrize("alignment", [None, 64])
def test_prefill_chunks_end_on_the_alignment(alignment):
    # "b" starts with the 50 tokens "a" leaves, then runs beside a's decode.
    scheduler = _scheduler(alignment)
    (a,) = create_requests(num_requests=1, num_tokens=150, req_ids=["a"])
    (b,) = create_requests(num_requests=1, num_tokens=500, req_ids=["b"])
    scheduler.add_request(a)
    scheduler.add_request(b)
    ends = []
    while b.num_computed_tokens < b.num_prompt_tokens:
        out = scheduler.schedule()
        ids = list(out.num_scheduled_tokens)
        sampled = [
            [0]
            if scheduler.requests[r].num_computed_tokens
            >= scheduler.requests[r].num_tokens
            else []
            for r in ids
        ]
        scheduler.update_from_output(
            out, ModelRunnerOutput(req_ids=ids, sampled_token_ids=sampled)
        )
        if "b" in out.num_scheduled_tokens:
            ends.append(min(b.num_computed_tokens, b.num_prompt_tokens))
    aligned = all(end % 64 == 0 for end in ends[:-1])
    assert ends[-1] == 500 and aligned == (alignment is not None), ends


def test_budget_below_the_alignment_is_rejected():
    with pytest.raises(ValueError, match="prefill chunk alignment"):
        _scheduler(64, max_num_batched_tokens=32)
