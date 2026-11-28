# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.distributed import cleanup_dist_env_and_memory

_SKIPPED = pytest.StashKey[bool]()


@pytest.fixture(autouse=True)
def reset_default_torch_device():
    """Several kernel tests call torch.set_default_device without restoring
    it, which poisons subsequent tests in the same pytest run (e.g. CPU
    tensors silently created on CUDA). Restore the factory default after
    every test.
    """
    yield
    torch.set_default_device(None)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item):
    report = yield
    # An xfailed test also reports as skipped, but its body ran.
    if report.skipped and not hasattr(report, "wasxfail"):
        item.stash[_SKIPPED] = True
    return report


@pytest.fixture(autouse=True)
def cleanup_fixture(request, should_do_global_cleanup_after_test: bool):
    """Skip the global cleanup after a test that skipped itself.

    Many kernel tests skip from their body on unsupported hardware, so they
    still paid its full `gc.collect()`, which grows to seconds per test late
    in a long shard.
    """
    yield
    if should_do_global_cleanup_after_test and not request.node.stash.get(
        _SKIPPED, False
    ):
        cleanup_dist_env_and_memory()
