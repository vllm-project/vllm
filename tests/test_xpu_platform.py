# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

# XPUPlatform imports these extensions at module import time. Skip this
# platform-specific test when the XPU runtime is not installed so generic CPU,
# CUDA, and ROCm test collection remains unaffected.
pytest.importorskip("vllm_xpu_kernels._C")
pytest.importorskip("vllm_xpu_kernels._moe_C")
pytest.importorskip("vllm_xpu_kernels._xpu_C")

from vllm.config.compilation import CompilationMode
from vllm.platforms.xpu import XPUPlatform


def test_uva_offload_disables_static_triton_launcher(monkeypatch):
    """The XPU UVA workaround uses the key consumed by PyTorch's bundler."""
    pass_config = SimpleNamespace(
        fuse_gemm_comms=False,
        fuse_allreduce_rms=False,
        fuse_attn_quant=False,
        fuse_act_padding=False,
        fuse_rope_kvcache=False,
        fuse_rope_kvcache_cat_mla=False,
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            compile_sizes=[],
            cudagraph_mode=object(),
            pass_config=pass_config,
            mode=CompilationMode.DYNAMO_TRACE_ONCE,
            inductor_compile_config={},
        ),
        offload_config=SimpleNamespace(
            offload_backend="uva",
            prefetch=SimpleNamespace(offload_group_size=0),
            uva=SimpleNamespace(cpu_offload_gb=1),
        ),
        parallel_config=SimpleNamespace(worker_cls="auto"),
        kv_transfer_config=None,
        shutdown_timeout=1,
    )

    monkeypatch.setattr("vllm.envs.VLLM_XPU_ENABLE_XPU_GRAPH", False)
    monkeypatch.setattr("vllm.envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", False)
    monkeypatch.setattr("vllm.utils.torch_utils.supports_xpu_graph", lambda: False)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    XPUPlatform.check_and_update_config(config)

    assert config.compilation_config.inductor_compile_config == {
        "use_static_cuda_launcher": False,
        "use_static_triton_launcher": False,
    }
