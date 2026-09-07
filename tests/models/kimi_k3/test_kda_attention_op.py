# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.compilation import CompilationConfig
from vllm.models.kimi_k3.common import kda_attention_op  # noqa: F401


def test_kimi_k3_kda_attention_is_registered_and_split() -> None:
    assert hasattr(torch.ops.vllm, "kimi_k3_kda_attention")
    assert "vllm::kimi_k3_kda_attention" in CompilationConfig._attention_ops
