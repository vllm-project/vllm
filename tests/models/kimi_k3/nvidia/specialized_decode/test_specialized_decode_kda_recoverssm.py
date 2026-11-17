# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's KdaSplit verify under RecoverSSM: it leaves each request's
checkpoint, and the stock commit of its records reproduces the per-token states the
plain verify writes."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip("The specialized decode needs SM10x", allow_module_level=True)
pytest.importorskip("cutlass")

from vllm.models.kimi_k3.nvidia.ops.recoverssm import (  # noqa: E402
    KDARecoverSSMCommitContext,
    KDARecoverSSMRecords,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.prepared import (  # noqa: E402
    PreparedKda,
)

D, HEADS, Q = 128, 12, 8  # TP8 Kimi-K3 heads, DSpark K=7 verify
DIM = HEADS * D
LOWER_BOUND = -5.0
DEVICE = torch.device("cuda")


def _attention(records):
    bf = torch.bfloat16
    linear = lambda rows, cols, s: SimpleNamespace(
        weight=(torch.randn(rows, cols, device=DEVICE) * s).to(bf)
    )
    return SimpleNamespace(
        conv1d=SimpleNamespace(
            weight=(torch.randn(3 * DIM, 1, 4, device=DEVICE) * 0.5).to(bf), bias=None
        ),
        dt_bias=torch.randn(DIM, device=DEVICE) * 0.5,
        A_log=torch.randn(HEADS, device=DEVICE) * 0.5,
        f_b_proj=linear(DIM, D, D**-0.5),
        o_norm=SimpleNamespace(weight=torch.rand(D, device=DEVICE) + 0.5, eps=1e-6),
        gate_lower_bound=LOWER_BOUND,
        recoverssm_records=records,
    )


@pytest.mark.parametrize("requests", [1, 4])
@torch.inference_mode()
def test_recoverssm_verify_and_commit_match_plain_verify(requests, monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
        lambda: "DS",
    )
    torch.manual_seed(requests)
    records = KDARecoverSSMRecords.allocate(
        requests, HEADS, Q, D, torch.bfloat16, DEVICE
    )
    plain = _attention(None)
    recover = SimpleNamespace(**{**vars(plain), "recoverssm_records": records})

    slots = 1 + requests * Q
    conv = torch.randn(slots, 3 * DIM, Q + 2, device=DEVICE).to(torch.bfloat16)
    state = torch.randn(slots, HEADS, D, D, device=DEVICE) * 0.1
    pitch = 4 * DIM + D + HEADS
    projected = torch.randn(requests * Q, pitch, device=DEVICE).to(torch.bfloat16)
    accepted = torch.ones(requests, dtype=torch.int32, device=DEVICE)
    # Plain verify: request b's token t state goes to slot 1 + b * Q + t.
    table = torch.arange(1, slots, dtype=torch.int32, device=DEVICE).view(-1, Q)
    checkpoints = table[:, :1].contiguous()

    outs, caches = [], []
    for attention, rows in ((plain, table), (recover, checkpoints)):
        cache = (conv.clone(), state.clone())
        kda = PreparedKda(attention, {(requests, Q): (pitch, 1)}, True)
        out = torch.empty(requests * Q, DIM, device=DEVICE, dtype=torch.bfloat16)
        kda.prepare_caches(cache).run(
            projected, out, caches=cache, slots=rows, accepted=accepted, q=Q
        )
        outs.append(out)
        caches.append(cache)
    (plain_conv, plain_state), (conv_after, checkpoint) = caches

    # The same outputs and convolution window; the checkpoints stay.
    torch.testing.assert_close(outs[1], outs[0], atol=0, rtol=0)
    torch.testing.assert_close(conv_after, plain_conv, atol=0, rtol=0)
    torch.testing.assert_close(checkpoint, state, atol=0, rtol=0)

    # The stock commit of n sampled tokens folds tokens 0..n-1 into the checkpoint
    # and shifts the convolution window to them.
    sampled = torch.randint(1, Q + 1, (requests,), dtype=torch.int32, device=DEVICE)
    layer = SimpleNamespace(
        kv_cache=(conv_after, checkpoint),
        recoverssm_records=records,
        A_log=recover.A_log,
        dt_bias=recover.dt_bias,
        local_num_heads=HEADS,
        head_dim=D,
        gate_lower_bound=LOWER_BOUND,
    )
    context = KDARecoverSSMCommitContext.create(
        [layer], spec_query_len=Q, max_num_reqs=requests, backend="triton"
    )
    query_start = torch.arange(0, (requests + 1) * Q, Q, device=DEVICE).int()
    context.commit(sampled, checkpoints[:, 0], query_start)
    for b in range(requests):
        n, slot = int(sampled[b]), int(checkpoints[b, 0])
        expected = plain_state[int(table[b, n - 1])]
        torch.testing.assert_close(checkpoint[slot], expected, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(
            conv_after[slot, :, :3], plain_conv[slot, :, n - 1 : n + 2], atol=0, rtol=0
        )
