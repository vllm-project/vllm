# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""update_draft_inputs shifts each draft window left by one, in place.

Without a barrier between the load and the store, a warp could overwrite a row
that another warp had not loaded yet, so the window was shifted by two.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytest.importorskip("triton")
if not current_platform.is_cuda_alike():
    pytest.skip("CUDA required for MTP draft-input kernels", allow_module_level=True)

from vllm.v1.worker.gpu.input_batch import InputBuffers  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (  # noqa: E402
    update_draft_inputs,
)

DEVICE = torch.device(current_platform.device_type)
HIDDEN_SIZE = 4096


def _setup(query_lens):
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    num_reqs = len(query_lens)
    qsl = [0]
    for q in query_lens:
        qsl.append(qsl[-1] + q)
    num_tokens = qsl[-1] + 64
    buffers = InputBuffers(num_reqs, num_tokens, DEVICE)
    buffers.query_start_loc[:] = torch.tensor(qsl, dtype=torch.int32)
    input_ids = torch.randint(
        0, 1 << 30, (num_tokens,), generator=gen, device=DEVICE
    ).to(torch.int32)
    last_token_indices = buffers.query_start_loc[1:].to(torch.int64) - 1
    idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
    draft_tokens = torch.randint(0, 1 << 30, (num_reqs,), generator=gen, device=DEVICE)
    return qsl, buffers, input_ids, last_token_indices, idx_mapping, draft_tokens, gen


def _shift(x, draft, qsl):
    ref = x.clone()
    for r in range(len(qsl) - 1):
        start, last = qsl[r], qsl[r + 1] - 1
        ref[start:last] = x[start + 1 : last + 1]
        ref[last] = draft[r]
    return ref


@pytest.mark.parametrize("query_lens", [[16384, 3, 8192]])
def test_shift_input_ids(query_lens):
    qsl, buffers, input_ids, last, idx_mapping, draft_tokens, _ = _setup(query_lens)
    ref = _shift(input_ids, draft_tokens.to(torch.int32), qsl)
    for _ in range(100):
        buffers.input_ids.copy_(input_ids)
        update_draft_inputs(
            draft_tokens, None, buffers, None, last, idx_mapping, len(query_lens)
        )
        torch.testing.assert_close(buffers.input_ids, ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "query_lens",
    [
        pytest.param([4] * 256, id="decode"),
        pytest.param([4] * 16 + [2048] + [3] * 16, id="prefill_chunk"),
    ],
)
def test_shift_input_embeds(query_lens):
    qsl, buffers, input_ids, last, idx_mapping, draft_tokens, gen = _setup(query_lens)
    num_tokens = input_ids.numel()
    embeds = torch.randn(num_tokens, HIDDEN_SIZE, generator=gen, device=DEVICE).to(
        torch.bfloat16
    )
    draft_embeds = torch.randn(
        len(query_lens), HIDDEN_SIZE, generator=gen, device=DEVICE
    ).to(torch.bfloat16)
    ref = _shift(embeds, draft_embeds, qsl)
    for _ in range(20):
        out = embeds.clone()
        update_draft_inputs(
            draft_tokens,
            draft_embeds,
            buffers,
            out,
            last,
            idx_mapping,
            len(query_lens),
        )
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
