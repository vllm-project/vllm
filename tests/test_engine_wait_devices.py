# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for scoping the ``VllmRunner`` memory-settle wait to the
devices an engine actually occupies, so it never blocks on GPUs pinned to a
different, concurrently-running engine.
"""

from tests.utils import multi_gpu_test


@multi_gpu_test(num_gpus=2)
def test_two_engines_do_not_cross_wait_on_memory(vllm_runner):
    """Two engines pinned to distinct devices must not wait on each other's
    memory during the bounded memory-settle guard.

    Engine A stays resident on visible device 0. While it is alive, engine B is
    built on visible device 1 and then torn down. Both of B's memory-settle
    waits -- the pre-construction wait (XPU/ROCm) and the teardown wait (every
    platform) -- must be scoped to B's own device, never device 0. Otherwise
    the bounded wait blocks on A's still-occupied GPU and times out with
    ``ValueError: Memory of devices ... not free``.

    Instead of relying only on that slow (240 s) timeout, this spies on
    ``wait_for_memory_to_settle`` and asserts every wait B issues is scoped to
    ``[1]``. The regression then fails fast and stays meaningful even on CUDA,
    where the wait itself is a no-op but the ``devices`` argument still carries
    the scope.

    ``device_ids`` are visible ordinals within whatever set the launcher's mask
    (``ZE_AFFINITY_MASK`` / ``CUDA_VISIBLE_DEVICES``) assigned to this process,
    so the test does not depend on which physical GPUs CI happens to hand out.
    """
    from tests import utils as test_utils
    from vllm import SamplingParams

    prompts = ["The capital of France is"]
    params = SamplingParams(max_tokens=1, temperature=0.0)

    engine_a_device = 0
    engine_b_device = 1
    capturing_b = False
    b_wait_scopes: list[list[int] | None] = []
    original_wait = test_utils.wait_for_memory_to_settle

    def spy_wait(*, threshold_ratio=0.1, timeout_s=240, devices=None):
        if capturing_b:
            b_wait_scopes.append(devices)
            assert devices == [engine_b_device], (
                f"engine B's memory-settle wait was scoped to {devices}, "
                f"expected [{engine_b_device}]; None -- or anything including "
                f"device {engine_a_device} -- means the wait would block on "
                "engine A's still-occupied GPU."
            )
        return original_wait(
            threshold_ratio=threshold_ratio, timeout_s=timeout_s, devices=devices
        )

    test_utils.wait_for_memory_to_settle = spy_wait
    try:
        with vllm_runner(
            "facebook/opt-125m",
            device_ids=[engine_a_device],
            max_model_len=256,
            gpu_memory_utilization=0.3,
            enforce_eager=True,
        ) as engine_a:
            out_a = engine_a.get_llm().generate(prompts, params)
            assert out_a and out_a[0].outputs

            # Only B's construction and teardown waits must be scoped to [1].
            capturing_b = True
            try:
                with vllm_runner(
                    "facebook/opt-125m",
                    device_ids=[engine_b_device],
                    max_model_len=256,
                    gpu_memory_utilization=0.3,
                    enforce_eager=True,
                ) as engine_b:
                    out_b = engine_b.get_llm().generate(prompts, params)
                    assert out_b and out_b[0].outputs
            finally:
                capturing_b = False

        assert b_wait_scopes, (
            "engine B never invoked the memory-settle wait; the scoping was "
            "not exercised."
        )
    finally:
        test_utils.wait_for_memory_to_settle = original_wait
