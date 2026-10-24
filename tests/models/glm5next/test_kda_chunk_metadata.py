# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA chunk (prefill) kernel with precomputed chunk metadata.

``prepare_chunk_indices`` reads the sequence lengths back to the host, a GPU
sync on every prefill step. The GDN metadata builder already computes the same
chunk indices/offsets on the host, so the KDA layer passes them in. With them
the kernel must not derive its own, and its output must be bit-identical.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import chunk_delta_h, solve_tril
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.utils.torch_utils import async_tensor_h2d

if current_platform.is_rocm():
    from vllm.models.glm5next.amd.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )
    from vllm.models.glm5next.amd.ops.third_party.kda import kernels as kda_kernels
else:
    from vllm.models.glm5next.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )
    from vllm.models.glm5next.nvidia.ops.third_party.kda import (
        kernels as kda_kernels,
    )

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm"
)

H, D = 4, 128


def _run(seq_lens: list[int], device: torch.device, **chunk_metadata):
    torch.manual_seed(0)
    T = sum(seq_lens)
    q, k, v = (
        torch.randn(1, T, H, D, device=device, dtype=torch.bfloat16) for _ in range(3)
    )
    raw_g = torch.randn(1, T, H, D, device=device, dtype=torch.bfloat16)
    beta = torch.rand(1, T, H, device=device, dtype=torch.float32)
    a_log = torch.randn(H, device=device, dtype=torch.float32)
    g_bias = torch.randn(H * D, device=device, dtype=torch.float32)
    h0 = torch.randn(len(seq_lens), H, D, D, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(seq_lens).cumsum(0).tolist(),
        device=device,
        dtype=torch.int32,
    )
    return chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        beta=beta,
        A_log=a_log,
        g_bias=g_bias,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        safe_gate=True,
        lower_bound=-5.0,
        **chunk_metadata,
    )


@pytest.mark.parametrize("seq_lens", [[1], [64], [7, 130, 1, 64, 300]])
def test_precomputed_chunk_metadata(seq_lens, monkeypatch):
    device = torch.device(current_platform.device_type)
    ref_o, ref_state = _run(seq_lens, device)

    # Built the way GDNAttentionMetadataBuilder._build_chunk_metadata does.
    cu_seqlens_cpu = torch.tensor(
        [0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int32
    )
    chunk_indices = async_tensor_h2d(
        prepare_chunk_indices(cu_seqlens_cpu, FLA_CHUNK_SIZE), device=device
    )
    chunk_offsets = async_tensor_h2d(
        prepare_chunk_offsets(cu_seqlens_cpu, FLA_CHUNK_SIZE), device=device
    )

    def _no_derive(*args, **kwargs):
        raise AssertionError("chunk metadata derived despite being passed in")

    for module in (kda_kernels, solve_tril, chunk_delta_h):
        monkeypatch.setattr(module, "prepare_chunk_indices", _no_derive)
    monkeypatch.setattr(chunk_delta_h, "prepare_chunk_offsets", _no_derive)

    o, state = _run(
        seq_lens, device, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets
    )
    torch.testing.assert_close(o, ref_o, rtol=0, atol=0)
    torch.testing.assert_close(state, ref_state, rtol=0, atol=0)
