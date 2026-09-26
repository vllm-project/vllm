# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU KDA reference ops vs the vendor Triton kernels, on a real GPU.

The pure-torch reference (``glm5next/cpu/ops/kda.py``) is what GLM-5.3-Flash
executes on CPU; these tests pin it to the CUDA kernels on hardware, which
is the strongest ground truth available. fp32 inputs keep the comparison
clean of bf16 rounding; the tolerances absorb fp32 reduction-order
differences between the kernel and the torch formulation (and, for the
chunk op, the chunked algorithm's reordered arithmetic vs the recurrent
form the reference uses).
"""

import pytest
import torch

from vllm.models.glm5next.cpu.ops.kda import (
    chunk_kda_with_fused_gate as ref_chunk,
)
from vllm.models.glm5next.cpu.ops.kda import (
    fused_recurrent_kda as ref_recurrent,
)
from vllm.models.glm5next.nvidia.ops.third_party.kda import (
    chunk_kda_with_fused_gate as vendor_chunk,
)
from vllm.models.glm5next.nvidia.ops.third_party.kda import (
    fused_recurrent_kda as vendor_recurrent,
)
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("needs a GPU to run the vendor kernels", allow_module_level=True)

torch.manual_seed(12345)

H, K, V = 4, 64, 64
LOWER_BOUND = -5.0
SEQ_LENS = [32, 40, 24]  # varlen batch, B == 1 flattened
SLOTS = [1, 3, 5]
T = sum(SEQ_LENS)
CU = torch.tensor([0, 32, 72, T], dtype=torch.int32, device="cuda")


def _inputs(device):
    return (
        torch.randn(1, T, H, K, device=device),
        torch.randn(1, T, H, K, device=device),
        torch.randn(1, T, H, V, device=device),
        torch.randn(1, T, H, K, device=device) * 0.5,
        torch.randn(1, T, H, device=device) * 0.5,
        torch.randn(H, device=device) * 0.3 - 1.0,
        torch.randn(H, K, device=device) * 0.1,
    )


def _check(name, a, b, atol):
    d = (a.float().cpu() - b.float().cpu()).abs().max().item()
    assert d < atol, f"{name}: maxdiff={d:.3e} (atol {atol:.0e})"


def test_recurrent_reference_matches_vendor():
    q, k, v, g, beta, a_log, g_bias = _inputs("cuda")
    cache_gpu = torch.randn(8, H, V, K, device="cuda") * 0.3
    cache_cpu = cache_gpu.cpu()
    slots = torch.tensor(SLOTS, dtype=torch.int32, device="cuda")

    o_vendor, _ = vendor_recurrent(
        q,
        k,
        v,
        g,
        beta,
        initial_state=cache_gpu,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=CU,
        ssm_state_indices=slots,
        sigmoid_beta=True,
        a_log=a_log,
        g_bias=g_bias,
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )
    o_ref, _ = ref_recurrent(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        g.cpu(),
        beta.cpu(),
        initial_state=cache_cpu,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=CU.cpu(),
        ssm_state_indices=slots.cpu(),
        sigmoid_beta=True,
        a_log=a_log.cpu(),
        g_bias=g_bias.cpu(),
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )
    # vendor output carries a leading NK axis
    _check("recurrent output", o_vendor[0, 0], o_ref[0], atol=1e-3)
    for s in SLOTS:
        _check(f"recurrent slot {s} state", cache_gpu[s], cache_cpu[s], atol=1e-3)


@pytest.mark.parametrize("safe_gate", [True, False])
def test_chunk_reference_matches_vendor(safe_gate):
    q, k, v, g, beta, a_log, g_bias = _inputs("cuda")
    beta_sig = torch.sigmoid(beta)  # chunk contract: pre-sigmoided fp32
    init_gpu = torch.randn(len(SLOTS), H, V, K, device="cuda") * 0.3

    o_vendor, fs_vendor = vendor_chunk(
        q,
        k,
        v,
        g,
        beta_sig,
        A_log=a_log,
        g_bias=g_bias,
        initial_state=init_gpu.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=CU,
        safe_gate=safe_gate,
        lower_bound=LOWER_BOUND,
    )
    o_ref, fs_ref = ref_chunk(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        g.cpu(),
        beta_sig.cpu(),
        a_log.cpu(),
        g_bias.cpu(),
        initial_state=init_gpu.cpu(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=CU.cpu(),
        safe_gate=safe_gate,
        lower_bound=LOWER_BOUND if safe_gate else None,
    )
    _check(f"chunk output (safe={safe_gate})", o_vendor[0], o_ref[0], atol=1e-2)
    _check(f"chunk final state (safe={safe_gate})", fs_vendor, fs_ref, atol=1e-2)
