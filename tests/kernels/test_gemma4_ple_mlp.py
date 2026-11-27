# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.model_executor.layers.fusion.gemma4_ple_mlp import gate_act, proj

HIDDEN, PLE, LAYERS = 2560, 256, 42


@pytest.mark.parametrize("tokens", [1, 3, 17, 100, 256, 300, 777, 2048, 5000, 16384])
def test_ple_mlp(tokens):
    torch.manual_seed(0)
    x = (torch.randn(tokens, HIDDEN, device="cuda") * 2).bfloat16()
    wg = (torch.randn(PLE, HIDDEN, device="cuda") * 0.03).bfloat16()
    wp = (torch.randn(HIDDEN, PLE, device="cuda") * 0.06).bfloat16()
    pli = (torch.randn(tokens, LAYERS, PLE, device="cuda") * 0.5).bfloat16()[:, 5, :]

    g = (x.float() @ wg.float().T).bfloat16().float()
    ref_h = (torch.nn.functional.gelu(g, approximate="tanh") * pli.float()).bfloat16()
    h = gate_act(x, wg, pli)
    torch.testing.assert_close(h, ref_h, atol=0.03, rtol=1e-2)

    out = proj(ref_h, wp)
    ref = (ref_h.float() @ wp.float().T).bfloat16()
    torch.testing.assert_close(out, ref, atol=0.07, rtol=1e-2)
