# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused q/kv-a RMSNorm must equal the two separate RMSNorm launches.

The Kimi-K3 AMD MLA front-end replaces ``q_a_layernorm(q_c)`` +
``kv_a_layernorm(kv_c)`` with a single ``fused_q_kv_rmsnorm`` call. A wrong
weight binding or eps still runs and still produces plausible text, so this
pins the arithmetic against two independent RMSNorm modules.
"""

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.common.ops.fused_qk_rmsnorm import fused_q_kv_rmsnorm
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3 AMD fused q/kv RMSNorm is only wired on ROCm",
)

# K3 q-LoRA rank and kv-LoRA rank; the two norms operate on different widths.
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
EPS = 1e-6
DTYPE = torch.bfloat16


def _build_norms(device: torch.device) -> tuple[RMSNorm, RMSNorm]:
    torch.manual_seed(0)
    q_norm = RMSNorm(Q_LORA_RANK, eps=EPS).to(device=device, dtype=DTYPE)
    kv_norm = RMSNorm(KV_LORA_RANK, eps=EPS).to(device=device, dtype=DTYPE)
    q_norm.weight.data.copy_(1 + 0.1 * torch.randn_like(q_norm.weight))
    kv_norm.weight.data.copy_(1 + 0.1 * torch.randn_like(kv_norm.weight))
    return q_norm, kv_norm


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 512, 4096])
def test_fused_matches_separate_norms(num_tokens: int) -> None:
    device = torch.device("cuda")
    q_norm, kv_norm = _build_norms(device)

    q_c = torch.randn(num_tokens, Q_LORA_RANK, device=device, dtype=DTYPE)
    kv_c = torch.randn(num_tokens, KV_LORA_RANK, device=device, dtype=DTYPE)

    expected_q = q_norm(q_c.clone())
    expected_kv = kv_norm(kv_c.clone())

    fused_q, fused_kv = fused_q_kv_rmsnorm(
        q_c, kv_c, q_norm.weight, kv_norm.weight, EPS
    )

    torch.testing.assert_close(fused_q, expected_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fused_kv, expected_kv, atol=2e-2, rtol=2e-2)


def test_empty_batch_is_a_noop() -> None:
    device = torch.device("cuda")
    q_norm, kv_norm = _build_norms(device)
    q_c = torch.empty(0, Q_LORA_RANK, device=device, dtype=DTYPE)
    kv_c = torch.empty(0, KV_LORA_RANK, device=device, dtype=DTYPE)

    fused_q, fused_kv = fused_q_kv_rmsnorm(
        q_c, kv_c, q_norm.weight, kv_norm.weight, EPS
    )
    assert fused_q.shape == (0, Q_LORA_RANK)
    assert fused_kv.shape == (0, KV_LORA_RANK)
