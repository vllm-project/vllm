# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the manual AllReduce + GemmaRMSNorm fusion used by MiniMax M3
and Qwen3.5.

``fused_allreduce_gemma_rms_norm`` must match the unfused model path, i.e.
``GemmaRMSNorm(all_reduce(partial), residual)``, both on the flashinfer fast
path (TP>1 with flashinfer + NVSwitch) and on the eager fallback (TP==1, or when
flashinfer is unavailable / the GPU has no NVSwitch).
"""

import pytest
import torch
from torch.multiprocessing import spawn

from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    fused_allreduce_gemma_rms_norm,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed


@ensure_current_vllm_config()
def _worker_fused_ar_norm(local_rank, world_size, port):
    """Per-rank worker: compare the fused helper vs all_reduce + GemmaRMSNorm.

    All shapes run in one process; spawning per case costs far more than the
    kernels themselves.
    """
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )

    for num_tokens in (1, 128, 333):
        for hidden_size in (2048, 4096):
            dtype = torch.bfloat16
            seed = 42
            eps = 1e-6
            # Norm weights are identical across ranks (replicated GemmaRMSNorm).
            set_random_seed(seed)
            norm = GemmaRMSNorm(hidden_size, eps=eps).cuda().to(dtype)
            with torch.no_grad():
                norm.weight.normal_(mean=0.0, std=0.1)

            # Residual is shared across ranks; the partial o_proj output
            # differs per rank (each rank holds a partial sum that all_reduce
            # combines).
            torch.manual_seed(seed + 7)
            residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            torch.manual_seed(seed + 1000 + local_rank)
            partial = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

            # Reference: the unfused model path.
            reduced = tensor_model_parallel_all_reduce(partial.clone())
            ref_out, ref_res = norm(reduced, residual.clone())

            # Fused helper (flashinfer fast path when available, else fallback).
            out, res = fused_allreduce_gemma_rms_norm(
                partial.clone(), residual.clone(), norm
            )
            torch.accelerator.synchronize()

            torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(res, ref_res, atol=2e-2, rtol=2e-2)

    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA required",
)
# world_size=1 exercises the TP==1 identity branch on a single GPU; >1 exercises
# the all_reduce + GemmaRMSNorm equivalence (flashinfer kernel or fallback).
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_fused_allreduce_gemma_rms_norm(world_size):
    num_gpus = current_platform.device_count()
    if num_gpus < world_size:
        pytest.skip(f"Need >= {world_size} GPUs, have {num_gpus}")
    spawn(
        _worker_fused_ar_norm,
        args=(world_size, str(get_open_port())),
        nprocs=world_size,
        join=True,
    )


@ensure_current_vllm_config()
def _worker_fused_ar_norm_quant(local_rank, world_size, port):
    """Per-rank worker: run all quant-mode cases in one process.

    Workspaces are created once and shared across modes, mirroring serving.
    """
    import vllm.model_executor.layers.fused_allreduce_gemma_rms_norm as ar_norm_mod
    from tests.models.qwen3_5.test_gemma_add_rms_norm_quant import (
        _fp4_codes,
        _fp8_codes,
        _swizzled_sf_rows,
    )
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticTensorSym,
        kNvfp4Dynamic,
    )
    from vllm.models.qwen3_5.nvidia.model import Qwen3_5RMSNorm

    keys = {
        "static_fp8": kFp8StaticTensorSym,
        "nvfp4": kNvfp4Dynamic,
    }

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )
    sm100 = current_platform.is_device_capability_family(100)
    orig_can_use = ar_norm_mod._can_use_flashinfer

    def run_one(mode, num_tokens, force_fallback):
        hidden_size = 5120
        eps = 1e-6
        dtype = torch.bfloat16
        key = keys[mode]
        ar_norm_mod._can_use_flashinfer = (
            (lambda *a, **k: (False, 0)) if force_fallback else orig_can_use
        )

        set_random_seed(42)
        norm = Qwen3_5RMSNorm(hidden_size, eps=eps).cuda()
        with torch.no_grad():
            norm.weight.normal_(mean=0.0, std=0.1)
            norm.weight.data = norm.weight.data.to(dtype)

        torch.manual_seed(42 + 7)
        residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        torch.manual_seed(42 + 1000 + local_rank)
        partial = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

        scale = torch.tensor(
            [740.0 if mode == "nvfp4" else 0.11],
            dtype=torch.float32,
            device=device,
        )
        linear = torch.nn.Module()
        if mode == "nvfp4":
            linear.input_global_scale_inv = scale
        elif mode == "static_fp8":
            linear.input_scale = scale

        # Reference: today's chain, AR + norm.
        ref_out, ref_res = fused_allreduce_gemma_rms_norm(
            partial.clone(), residual.clone(), norm
        )

        qa, y, res = ar_norm_mod.fused_allreduce_gemma_rms_norm_quant(
            partial.clone(), residual.clone(), norm, linear, key
        )
        torch.accelerator.synchronize()

        assert qa.quant_key == key
        torch.testing.assert_close(res, ref_res, atol=0, rtol=0)
        if force_fallback:
            # The Triton norm-quant reduces the sum of squares in a
            # different order than the flashinfer norm.
            assert (y != ref_out).float().mean().item() <= 1e-4
            torch.testing.assert_close(y, ref_out, rtol=2**-7, atol=0)
        else:
            torch.testing.assert_close(y, ref_out, atol=0, rtol=0)

        if mode == "nvfp4":
            ref_q, ref_sf = ops.scaled_fp4_quant(
                ref_out, scale, True, backend="flashinfer-cutedsl"
            )
            dist = (_fp4_codes(qa.data) - _fp4_codes(ref_q.view(torch.uint8))).abs()
            sf = _swizzled_sf_rows(qa.scale, num_tokens, hidden_size)
            ref_sf = _swizzled_sf_rows(ref_sf, num_tokens, hidden_size)
            assert (sf != ref_sf).float().mean().item() <= 1e-4
        else:
            ref_q, _ = ops.scaled_fp8_quant(ref_out, scale, group_shape=(-1, -1))
            dist = (_fp8_codes(qa.data) - _fp8_codes(ref_q)).abs()
        # Bitwise at small M on B300; allow one quantization step in a small
        # fraction of entries elsewhere.
        assert dist.max().item() <= 1
        assert (dist > 0).float().mean().item() <= 1e-4

    for mode in ("static_fp8", "nvfp4"):
        if mode == "nvfp4" and not sm100:
            continue
        for force_fallback in (False, True):
            for num_tokens in (1, 17, 333):
                run_one(mode, num_tokens, force_fallback)
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA required",
)
def test_fused_allreduce_gemma_rms_norm_quant():
    """TP2: the AR+norm+quant variants vs AR + norm + the unfused quant."""
    world_size = 2
    if current_platform.device_count() < world_size:
        pytest.skip(f"Need >= {world_size} GPUs")
    spawn(
        _worker_fused_ar_norm_quant,
        args=(world_size, str(get_open_port())),
        nprocs=world_size,
        join=True,
    )
