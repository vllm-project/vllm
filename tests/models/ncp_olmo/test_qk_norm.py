# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Full-hidden Q/K normalization tests for token and HLM attention."""

from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

if torch is not None:
    from vllm.model_executor.models.ncp_olmo.hlm import (
        ConceptLMHLMIncrementalAttention,
    )
    from vllm.model_executor.models.ncp_olmo.token_tower import (
        ConceptLMOlmo3Attention,
    )


if nn is not None:

    class ReferenceRMSNorm(nn.Module):
        """Small CPU reference that follows vLLM RMSNorm's last-dim rule."""

        def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            variance = value.float().pow(2).mean(dim=-1, keepdim=True)
            normalized = value.float() * torch.rsqrt(variance + self.eps)
            return (normalized * self.weight.float()).to(value.dtype)


@unittest.skipIf(torch is None, "torch is not installed in the host-only environment")
class TestQKNorm(unittest.TestCase):
    @staticmethod
    def _make_attention(cls: type[nn.Module]) -> nn.Module:
        module = cls.__new__(cls)  # type: ignore[call-overload]
        nn.Module.__init__(module)
        module.num_heads = 4
        module.num_kv_heads = 4
        module.head_dim = 3
        module.tp_size = 1
        module.tp_rank = 0
        module.q_layernorm = ReferenceRMSNorm(12)
        module.k_layernorm = ReferenceRMSNorm(12)
        return module

    def _check_full_hidden(self, cls: type[nn.Module]) -> None:
        torch.manual_seed(42)
        query = torch.randn(2, 12)
        key = torch.randn(2, 12)

        full_hidden = self._make_attention(cls)
        actual_q, actual_k = full_hidden._apply_qk_norm(query, key)
        torch.testing.assert_close(actual_q, full_hidden.q_layernorm(query))
        torch.testing.assert_close(actual_k, full_hidden.k_layernorm(key))
        self.assertEqual(tuple(full_hidden.q_layernorm.weight.shape), (12,))
        self.assertEqual(tuple(full_hidden.k_layernorm.weight.shape), (12,))

    def test_token_tower_qk_norm(self) -> None:
        self._check_full_hidden(ConceptLMOlmo3Attention)

    def test_hlm_qk_norm(self) -> None:
        self._check_full_hidden(ConceptLMHLMIncrementalAttention)


if __name__ == "__main__":
    unittest.main()
