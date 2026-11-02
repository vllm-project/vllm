# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2: Qwen3.5 AR + Gemma norm + FP8/NVFP4 quant vs AR + norm + the unfused
quant (scaled_fp8_quant / scaled_fp4_quant).

All cases run in one 2-rank spawn; spawning per case re-imports vLLM and
recreates the FlashInfer workspaces, which costs far more than the kernels.
"""

import pytest
import torch
from torch.multiprocessing import spawn

from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="Qwen3.5 fused norm + FP8/NVFP4 quant targets SM100+",
)

WORLD_SIZE = 2


@ensure_current_vllm_config()
def _worker(local_rank, world_size, port):
    import vllm.models.qwen3_5.nvidia.ops.fused_allreduce_norm_quant as ar_quant
    from tests.models.qwen3_5.test_gemma_add_rms_norm_quant import (
        _fp4_codes,
        _fp8_codes,
        _swizzled_sf_rows,
    )
    from vllm import _custom_ops as ops
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        tensor_model_parallel_all_reduce,
    )
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
        fused_allreduce_gemma_rms_norm,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticTensorSym,
        kNvfp4Dynamic,
    )
    from vllm.models.qwen3_5.nvidia.model import Qwen3_5RMSNorm

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )
    get_workspace = ar_quant._get_workspace

    def run_one(nvfp4, num_tokens, force_fallback):
        hidden_size = 5120
        eps = 1e-6
        dtype = torch.bfloat16
        key = kNvfp4Dynamic if nvfp4 else kFp8StaticTensorSym
        ar_quant._get_workspace = (lambda *a: None) if force_fallback else get_workspace

        set_random_seed(42)
        norm = Qwen3_5RMSNorm(hidden_size, eps=eps).cuda()
        with torch.no_grad():
            norm.weight.normal_(mean=0.0, std=0.1)
            norm.weight.data = norm.weight.data.to(dtype)

        # The residual is replicated; each rank holds its own partial sum.
        torch.manual_seed(42 + 7)
        residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        torch.manual_seed(42 + 1000 + local_rank)
        partial = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

        scale = torch.tensor(
            [740.0 if nvfp4 else 0.11], dtype=torch.float32, device=device
        )
        linear = torch.nn.Module()
        if nvfp4:
            linear.input_global_scale_inv = scale
        else:
            linear.input_scale = scale

        # Reference: the chain each arm replaces, then the quant op. The fused
        # kernel replaces FlashInfer AR + norm; the fallback replaces explicit
        # AR + norm (FlashInfer's norm reads the fp32 sum, not the bf16 one).
        if force_fallback:
            reduced = tensor_model_parallel_all_reduce(partial.clone())
            ref_out, ref_res = norm(reduced, residual.clone())
        else:
            ref_out, ref_res = fused_allreduce_gemma_rms_norm(
                partial.clone(), residual.clone(), norm
            )

        if not force_fallback:
            # The fused kernel must actually run, not the fallback.
            assert get_workspace(partial, world_size) is not None
        qa, y, res = norm.forward_allreduce_quant(
            partial.clone(), residual.clone(), linear, key
        )
        torch.accelerator.synchronize()

        assert qa.quant_key == key
        torch.testing.assert_close(res, ref_res, atol=0, rtol=0)
        if force_fallback:
            # The Triton norm-quant reduces the sum of squares in a different
            # order than the FlashInfer norm.
            assert (y != ref_out).float().mean().item() <= 1e-4
            torch.testing.assert_close(y, ref_out, rtol=2**-7, atol=0)
        else:
            torch.testing.assert_close(y, ref_out, atol=0, rtol=0)

        if nvfp4:
            ref_q, ref_sf = ops.scaled_fp4_quant(
                ref_out, scale, True, backend="flashinfer-cutedsl"
            )
            dist = (_fp4_codes(qa.data) - _fp4_codes(ref_q.view(torch.uint8))).abs()
            sf = _swizzled_sf_rows(qa.scale, num_tokens, hidden_size)
            ref_sf = _swizzled_sf_rows(ref_sf, num_tokens, hidden_size)
            sf_mismatch = (sf != ref_sf).float().mean().item()
        else:
            ref_q, _ = ops.scaled_fp8_quant(ref_out, scale, group_shape=(-1, -1))
            dist = (_fp8_codes(qa.data) - _fp8_codes(ref_q)).abs()
            sf_mismatch = 0.0
        if force_fallback:
            # Off-by-one codes where the normed values differ in rounding.
            assert dist.max().item() <= 1
            assert (dist > 0).float().mean().item() <= 1e-4
            assert sf_mismatch <= 1e-4
        else:
            assert dist.max().item() == 0
            assert sf_mismatch == 0.0

    for nvfp4 in (False, True):
        for force_fallback in (False, True):
            for num_tokens in (1, 17, 333):
                run_one(nvfp4, num_tokens, force_fallback)
    ar_quant._get_workspace = get_workspace
    cleanup_dist_env_and_memory()


def test_fused_allreduce_gemma_rms_norm_quant():
    if current_platform.device_count() < WORLD_SIZE:
        pytest.skip(f"Need >= {WORLD_SIZE} GPUs")
    spawn(
        _worker,
        args=(WORLD_SIZE, str(get_open_port())),
        nprocs=WORLD_SIZE,
        join=True,
    )
