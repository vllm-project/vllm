# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the cumem sleep-mode + custom all-reduce IPC crash.

Background
----------
When sleep mode is enabled, vLLM activates the CuMemAllocator, which backs
model activation tensors with the CUDA Virtual Memory Management API
(``cuMemCreate`` + ``cuMemMap``). The custom all-reduce kernel tries to
register the CUDA-graph activation buffers for cross-rank IPC using
``cudaIpcGetMemHandle`` (``csrc/custom_all_reduce.cuh`` ->
``get_graph_buffer_ipc_meta``). That legacy IPC API only works on
``cudaMalloc`` allocations and returns the opaque CUDA error
``invalid argument`` on VMM-backed pointers. The result on a TP>1 +
sleep-mode deployment is a hard crash during cudagraph capture
(``custom_all_reduce.cuh:455 'invalid argument'``) followed by a wedged
engine while ``/health`` still returns 200.

The fix auto-disables custom all-reduce whenever the cumem allocator is
active, mirroring the existing auto-disable for ``VLLM_BATCH_INVARIANT`` and
multi-node deployments.

Pre-fix:  this test FAILS — ``disable_custom_all_reduce`` stays ``False``
          (its constructed default) so the kernel would later crash at
          ``cudaIpcGetMemHandle`` during cudagraph capture.
Post-fix: this test PASSES — ``VllmConfig.__post_init__`` flips
          ``disable_custom_all_reduce`` to ``True``.
"""

import pytest

from vllm.config import ModelConfig, ParallelConfig, VllmConfig


@pytest.fixture(autouse=True)
def _force_custom_allreduce_supported(monkeypatch):
    """Make the platform report custom all-reduce as supported.

    ``ParallelConfig._verify_args`` unconditionally sets
    ``disable_custom_all_reduce=True`` when
    ``current_platform.use_custom_allreduce()`` is ``False`` (the CPU / base
    platform default). That platform gate would otherwise mask whether the
    cumem auto-disable in ``VllmConfig.__post_init__`` is the thing flipping
    the flag — and would make the over-broad-guard negative test fail on a
    CPU-only CI runner. Forcing it ``True`` isolates the assertions to the
    behavior under test on any host.
    """
    monkeypatch.setattr(
        "vllm.config.parallel.current_platform.use_custom_allreduce",
        lambda: True,
    )


def _build_config(
    *,
    enable_cumem_allocator: bool,
    tensor_parallel_size: int,
    disable_custom_all_reduce: bool = False,
) -> VllmConfig:
    model_config = ModelConfig("facebook/opt-125m")
    # Simulate what enable_sleep_mode does on a CUDA box without needing a GPU
    # in CI: sleep mode forces the cumem allocator on.
    model_config.enable_cumem_allocator = enable_cumem_allocator
    parallel_config = ParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        disable_custom_all_reduce=disable_custom_all_reduce,
    )
    return VllmConfig(model_config=model_config, parallel_config=parallel_config)


def test_cumem_allocator_auto_disables_custom_all_reduce():
    """cumem (sleep mode) must auto-disable custom all-reduce on TP>1.

    This is the load-bearing assertion. Without the fix, the constructed
    config leaves ``disable_custom_all_reduce`` at ``False`` and the engine
    crashes at ``cudaIpcGetMemHandle`` during cudagraph capture.
    """
    cfg = _build_config(enable_cumem_allocator=True, tensor_parallel_size=2)
    assert cfg.parallel_config.disable_custom_all_reduce is True


def test_cumem_allocator_auto_disable_is_idempotent():
    """An explicit disable_custom_all_reduce=True is preserved (no-op path)."""
    cfg = _build_config(
        enable_cumem_allocator=True,
        tensor_parallel_size=2,
        disable_custom_all_reduce=True,
    )
    assert cfg.parallel_config.disable_custom_all_reduce is True


def test_no_cumem_leaves_custom_all_reduce_enabled():
    """Without the cumem allocator the auto-disable must NOT fire.

    Guards against the fix being too broad (it must only trigger for the
    VMM-backed sleep-mode path, not for ordinary TP deployments).
    """
    cfg = _build_config(enable_cumem_allocator=False, tensor_parallel_size=2)
    assert cfg.parallel_config.disable_custom_all_reduce is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
