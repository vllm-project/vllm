# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy
from types import SimpleNamespace

import pytest
import torch

from tests.v1.worker.test_mamba_utils import (
    _make_dual_layer_state,
    _make_gpu_ctx,
    _make_kv_cache_config,
    _TestConfig,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.model_executor.warmup.mamba_triton_warmup import (
    _warm_batch_memcpy_kernel,
    _warm_mamba_align_state_copy_kernels,
    _warm_short_conv_prefill_kernel,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


@pytest.fixture
def request_jit_monitor(monkeypatch):
    from triton import knobs

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        _causal_conv1d_fwd_kernel,
    )
    from vllm.v1.worker.mamba_utils import (
        postprocess_mamba_fused_kernel,
        precopy_mamba_align_fused_kernel,
    )

    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", None)
    for kernel in (
        _causal_conv1d_fwd_kernel,
        precopy_mamba_align_fused_kernel,
        postprocess_mamba_fused_kernel,
    ):
        device_caches = copy(kernel.device_caches)
        device_caches.clear()
        monkeypatch.setattr(kernel, "device_caches", device_caches)

    def reject_compile(**kwargs):
        pytest.fail(f"Unexpected request-time JIT: {kwargs['fn'].name}")

    def activate():
        monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", reject_compile)

    return activate


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_mamba_batch_memcpy_kernel_compiles_on_gpu() -> None:
    _warm_batch_memcpy_kernel(torch.device("cuda"))
    torch.accelerator.synchronize("cuda")


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA is required")
@pytest.mark.parametrize("num_speculative_blocks", [0, 1, 2, 3])
def test_short_conv_warmup_covers_mixed_prefill_views(
    request_jit_monitor, num_speculative_blocks: int
) -> None:
    """Null-block warmup covers strided and offset views without cache writes."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    device = torch.device("cuda")
    state = torch.randn(
        8, 2 + num_speculative_blocks, 96, dtype=torch.bfloat16, device=device
    ).transpose(-1, -2)
    original = state.clone()
    weight = torch.randn(96, 3, dtype=torch.bfloat16, device=device)
    _warm_short_conv_prefill_kernel(state, weight, None, num_speculative_blocks)
    assert torch.equal(state, original)

    indices = torch.arange(4, dtype=torch.int32, device=device)[:, None].repeat(
        1, 1 + num_speculative_blocks
    )
    request_jit_monitor()
    causal_conv1d_fn(
        torch.randn(6, 96, dtype=torch.bfloat16, device=device).t(),
        weight,
        None,
        state,
        torch.arange(4, dtype=torch.int32, device=device) * 2,
        cache_indices=indices[1:, 0],
        has_initial_state=torch.zeros(3, dtype=torch.bool, device=device),
        activation=None,
    )
    torch.accelerator.synchronize("cuda")


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA is required")
@pytest.mark.parametrize("num_reqs", [1, 3])
def test_align_copy_warmup_preserves_cache_and_avoids_request_jit(
    request_jit_monitor, num_reqs: int
) -> None:
    """Warmup must leave state intact and cover real boundary migrations."""
    device = torch.device("cuda")
    config = _TestConfig(num_reqs=num_reqs)
    kv_cache_config = _make_kv_cache_config(config, ["layer_0"])
    ctx = _make_gpu_ctx(config, kv_cache_config, device)
    conv_before, ssm_before, conv, ssm, _, forward_context = _make_dual_layer_state(
        config, device
    )
    block_table = torch.arange(16, dtype=torch.int32, device=device).repeat(
        config.max_num_reqs, 1
    )
    block_table += (
        4 * torch.arange(config.max_num_reqs, dtype=torch.int32, device=device)[:, None]
    )
    copy_funcs = {
        MambaAttentionBackendEnum.MAMBA2: (get_conv_copy_spec, get_temporal_copy_spec)
    }
    runner = SimpleNamespace(
        device=device,
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        kv_cache_config=kv_cache_config,
        compilation_config=SimpleNamespace(static_forward_context=forward_context),
        input_batch=SimpleNamespace(
            block_table=[SimpleNamespace(get_device_tensor=lambda n: block_table[:n])]
        ),
        _get_mamba_bufs=lambda: SimpleNamespace(postprocess_align=ctx),
        _get_mamba_state_copy_funcs=lambda: copy_funcs,
    )
    _warm_mamba_align_state_copy_kernels(runner)
    assert torch.equal(conv, conv_before)
    assert torch.equal(ssm, ssm_before)

    zeros = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    ones = torch.ones_like(zeros)
    request_jit_monitor()
    ctx.run_fused_precopy(num_reqs, ones, zeros, zeros, idx_mapping=None)
    src_blocks = block_table[:num_reqs, 0].long()
    dst_blocks = block_table[:num_reqs, 1].long()
    assert torch.equal(conv[dst_blocks], conv_before[src_blocks])
    assert torch.equal(ssm[dst_blocks], ssm_before[src_blocks])
    ctx.run_fused_postprocess(
        num_reqs, ones, ones * 2, ones * config.block_size, zeros, zeros
    )
    postprocess_src_blocks = block_table[:num_reqs, 2].long()
    assert torch.equal(conv[src_blocks], conv_before[postprocess_src_blocks])
    assert torch.equal(ssm[src_blocks], ssm_before[postprocess_src_blocks])
