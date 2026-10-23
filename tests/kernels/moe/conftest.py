# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.distributed import cleanup_dist_env_and_memory
from vllm.v1.worker.workspace import reset_workspace_manager

NEEDS_CLEAN_ENTRY = frozenset(
    {
        # Fail unless they start from a clean distributed and memory state,
        # which they used to get only as a side effect of the preceding test
        # cleaning up after itself.
        "test_moe_layer.py",
        "test_modular_oai_triton_moe.py",
        "test_ocp_mx_moe.py",
        "test_zero_expert_moe.py",
        # Execute no test on ROCm, where this was measured, so nothing is known
        # about how they behave without the per-test cleanup. Cleaned on entry
        # as a precaution; drop once they have been seen green without it.
        "test_batched_deepgemm.py",
        "test_deepep_deepgemm_moe.py",
        "test_deepep_moe.py",
        "test_deepep_v2_moe.py",
        "test_deepgemm.py",
        "test_grouped_topk.py",
        "test_mxfp4_moe.py",
        "test_shared_fused_moe_routed_transform.py",
        "test_silu_mul_per_token_group_quant_fp8_colmajor.py",
        "test_situ_mul_fp8_quant.py",
    }
)


def pytest_addoption(parser):
    parser.addoption(
        "--subtests", action="store", type=str, default=None, help="subtest ids"
    )


@pytest.fixture
def subtests(request):
    return request.config.getoption("--subtests")


@pytest.fixture()
def should_do_global_cleanup_after_test() -> bool:
    """Drop the per-test global cleanup for this directory.

    A full ``gc.collect()`` plus an emptied caching allocator after each of the
    ~1450 small kernel tests per shard costs more than the tests themselves.
    The tests that depend on isolation get a clean state on entry instead, in
    the hook below.
    """
    return False


def pytest_runtest_setup(item):
    # Not tryfirst: the skipping plugin evaluates skip marks in a tryfirst
    # hook, so tests about to be skipped never reach this, while fixture setup
    # still runs afterwards.
    if item.path.name in NEEDS_CLEAN_ENTRY or "dist_init" in item.fixturenames:
        reset_workspace_manager()
        cleanup_dist_env_and_memory()


@pytest.fixture
def trtllm_workspace_check(monkeypatch, workspace_init):
    """Opt into workspaces and compare real buffered/unbuffered kernel calls."""
    import flashinfer.fused_moe as fused_moe
    import torch

    from vllm.model_executor.layers.fused_moe.experts import trtllm_moe_base

    monkeypatch.setenv("VLLM_FLASHINFER_MOE_USE_WORKSPACE", "1")
    monkeypatch.setattr(trtllm_moe_base, "fi_moe_largest_bucket", lambda config: 32)
    buffers = []

    def install(
        kernel_name,
        quant_mode,
        *,
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=16,
        **size_kwargs,
    ):
        kernel = getattr(fused_moe, kernel_name)

        def compare(**kwargs):
            buffer = kwargs.get("workspace_buffer")
            assert buffer is not None
            required = fused_moe.trtllm_fused_moe_workspace_size(
                32,
                hidden_size,
                intermediate_size,
                num_experts,
                kwargs["top_k"],
                quant_mode=quant_mode,
                **size_kwargs,
            )
            assert buffer.nbytes >= required
            buffers.append(buffer)
            result = kernel(**kwargs)
            outputs = (result,) if isinstance(result, torch.Tensor) else result
            snapshots = [tensor.clone() for tensor in outputs]
            reference = kernel(**{**kwargs, "workspace_buffer": None})
            references = (
                (reference,) if isinstance(reference, torch.Tensor) else reference
            )
            for actual, expected in zip(snapshots, references, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            return result

        monkeypatch.setattr(fused_moe, kernel_name, compare)

    yield install
    assert len(buffers) == 3
    assert all(buffer is buffers[0] for buffer in buffers)
