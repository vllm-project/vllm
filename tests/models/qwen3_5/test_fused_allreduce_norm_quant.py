# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2: ``Qwen3_5RMSNorm.forward_fused`` vs the unfused chain it replaces,
(all-reduce +) Gemma norm, then scaled_fp8_quant / scaled_fp4_quant, for real
ModelOpt static FP8 and NVFP4 consumers.

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
HIDDEN_SIZE = 5120
# Above the TP2 FlashInfer workspace budget (64 MiB of bf16 at H=5120):
# the all-reduce + norm-quant falls back to explicit all-reduce + Triton.
FALLBACK_TOKENS = 8192


@ensure_current_vllm_config()
def _worker(local_rank, world_size, port):
    from tests.models.qwen3_5.test_gemma_add_rms_norm_quant import (
        _fp4_codes,
        _fp8_codes,
        _swizzled_sf_rows,
    )
    from vllm import _custom_ops as ops
    from vllm.config import get_current_vllm_config
    from vllm.config.model import ModelConfig
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
        fused_allreduce_gemma_rms_norm,
    )
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptFp8Config,
        ModelOptNvFp4Config,
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
    get_current_vllm_config().model_config = ModelConfig()
    dtype = torch.bfloat16

    def make_linear(nvfp4):
        """A loaded ModelOpt linear whose input quant the norm can fuse."""
        if nvfp4:
            quant_config = ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                exclude_modules=[],
            )
            input_scale = 1 / 740.0
        else:
            quant_config = ModelOptFp8Config(
                quant_method="FP8",
                is_checkpoint_fp8_serialized=True,
                kv_cache_quant_method=None,
                exclude_modules=[],
            )
            input_scale = 0.11
        with torch.device(device):
            linear = ReplicatedLinear(
                HIDDEN_SIZE,
                256,
                bias=False,
                quant_config=quant_config,
                params_dtype=dtype,
            )
            for param in linear.parameters():
                if param.dtype == torch.uint8:
                    param.data.copy_(torch.randint(0, 256, param.shape))
                else:
                    param.data.copy_(torch.rand(param.shape) + 0.5)
            linear.input_scale.data.fill_(input_scale)
            linear.quant_method.process_weights_after_loading(linear)
        return linear

    linears = {nvfp4: make_linear(nvfp4) for nvfp4 in (False, True)}

    def run_one(nvfp4, allreduce, num_tokens):
        linear = None if nvfp4 is None else linears[nvfp4]
        set_random_seed(42)
        norm = Qwen3_5RMSNorm(HIDDEN_SIZE, eps=1e-6).cuda()
        with torch.no_grad():
            norm.weight.normal_(mean=0.0, std=0.1)
            norm.weight.data = norm.weight.data.to(dtype)

        # The residual is replicated; each rank holds its own partial sum.
        torch.manual_seed(42 + 7)
        residual = torch.randn(num_tokens, HIDDEN_SIZE, dtype=dtype, device=device)
        torch.manual_seed(42 + 1000 + local_rank)
        x = torch.randn(num_tokens, HIDDEN_SIZE, dtype=dtype, device=device)

        # Reference: the chain forward_fused replaces, then the quant op.
        if allreduce:
            ref_y, ref_res = fused_allreduce_gemma_rms_norm(
                x.clone(), residual.clone(), norm
            )
        else:
            ref_y, ref_res = norm(x.clone(), residual.clone())
        qa, y, res = norm.forward_fused(
            x.clone(), residual.clone(), linear, allreduce=allreduce, write_y=True
        )
        torch.accelerator.synchronize()

        torch.testing.assert_close(res, ref_res, atol=0, rtol=0)
        if linear is None:
            assert qa is None
            torch.testing.assert_close(y, ref_y, atol=0, rtol=0)
            return

        assert qa is not None and y is not None
        # FlashInfer's AR+norm+quant kernel matches its AR+norm bitwise; the
        # Triton norm-quant (no all-reduce, or above the workspace budget)
        # reduces the sum of squares in a different order than FlashInfer's
        # norm, so off-by-one codes appear where the normed values differ.
        exact = allreduce and num_tokens < FALLBACK_TOKENS
        if exact:
            torch.testing.assert_close(y, ref_y, atol=0, rtol=0)
        else:
            assert (y != ref_y).float().mean().item() <= 1e-4
            torch.testing.assert_close(y, ref_y, rtol=2**-7, atol=0)

        if nvfp4:
            assert qa.quant_key == kNvfp4Dynamic
            ref_q, ref_sf = ops.scaled_fp4_quant(
                ref_y, linear.input_global_scale_inv, True, backend="flashinfer-cutedsl"
            )
            dist = (_fp4_codes(qa.data) - _fp4_codes(ref_q.view(torch.uint8))).abs()
            sf = _swizzled_sf_rows(qa.scale, num_tokens, HIDDEN_SIZE)
            ref_sf = _swizzled_sf_rows(ref_sf, num_tokens, HIDDEN_SIZE)
            sf_mismatch = (sf != ref_sf).float().mean().item()
        else:
            assert qa.quant_key == kFp8StaticTensorSym
            ref_q, _ = ops.scaled_fp8_quant(
                ref_y, linear.input_scale, group_shape=(-1, -1)
            )
            dist = (_fp8_codes(qa.data) - _fp8_codes(ref_q)).abs()
            sf_mismatch = 0.0
        if exact:
            assert dist.max().item() == 0
            assert sf_mismatch == 0.0
        else:
            assert dist.max().item() <= 1
            assert (dist > 0).float().mean().item() <= 1e-4
            assert sf_mismatch <= 1e-4

    for nvfp4 in (False, True):
        for num_tokens in (1, 17, 333, FALLBACK_TOKENS):
            run_one(nvfp4, True, num_tokens)
        run_one(nvfp4, False, 17)
    for allreduce in (False, True):
        run_one(None, allreduce, 17)
    cleanup_dist_env_and_memory()


def test_forward_fused():
    if current_platform.device_count() < WORLD_SIZE:
        pytest.skip(f"Need >= {WORLD_SIZE} GPUs")
    spawn(
        _worker,
        args=(WORLD_SIZE, str(get_open_port())),
        nprocs=WORLD_SIZE,
        join=True,
    )
