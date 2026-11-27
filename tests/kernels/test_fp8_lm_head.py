# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.logits_processor import LogitsProcessor


class _Head:
    tp_size = 1

    def __init__(self, w):
        self.weight = torch.nn.Parameter(w, requires_grad=False)


@pytest.mark.parametrize("m", [1, 3, 17, 128, 255, 256])
def test_fp8_head_close_to_bf16(m):
    torch.manual_seed(0)
    v, k = 4096, 2560
    w = (torch.randn(v, k, device="cuda") * 0.05).to(torch.bfloat16)
    x = (torch.randn(m, k, device="cuda") * 2).to(torch.bfloat16)
    with set_current_vllm_config(VllmConfig()):
        lp = LogitsProcessor(v, soft_cap=30.0)
    lp.fp8_head = True
    head = _Head(w)
    out = lp._apply_head(head, x, None)
    ref = torch.nn.functional.linear(x, w)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel < 0.06
    assert hasattr(head, "_fp8_head")


def test_fp8_head_off_by_default():
    with set_current_vllm_config(VllmConfig()):
        lp = LogitsProcessor(4096)
    assert not lp.fp8_head
