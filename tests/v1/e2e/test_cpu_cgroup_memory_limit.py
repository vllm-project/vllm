# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the CPU backend's cgroup-aware KV-cache sizing.

Only meaningful under a real container memory cgroup -- see the
"CPU-Cgroup Memory Clamp Test" Buildkite step
(.buildkite/hardware_tests/cpu.yaml), which runs this through
.buildkite/scripts/hardware_ci/run-cpu-test.sh with CONTAINER_MEMORY_LIMIT
set and VLLM_CPU_KVCACHE_SPACE explicitly unset.

With no explicit --kv-cache-memory-bytes / VLLM_CPU_KVCACHE_SPACE, the CPU
worker's fraction-of-available-memory auto-sizing path
(CPUWorker.determine_available_memory) must size off the container's
memory.max, not the host's/NUMA-node's full RAM -- otherwise it
over-allocates and gets OOM-killed on a memory-limited K8s pod or CI
container (see vllm/utils/cpu_resource_utils.py::get_memory_node_info).
"""

import os

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.utils.cpu_resource_utils import get_memory_node_info

pytestmark = pytest.mark.cpu_model

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

_LIMIT_ENV = "CONTAINER_MEMORY_LIMIT"


def _require_container_memory_limit() -> int:
    raw = os.environ.get(_LIMIT_ENV)
    if not raw:
        pytest.skip(
            f"{_LIMIT_ENV} not set; only meaningful under a memory-capped "
            "container (see the CPU-Cgroup Memory Clamp Test Buildkite step)"
        )
        raise AssertionError("unreachable")  # pytest.skip always raises
    return int(raw)


def test_cgroup_limit_is_detected_not_host_ram():
    """get_memory_node_info() must report (at most) the container's own
    cap, not the CI host's much larger physical RAM."""
    limit_bytes = _require_container_memory_limit()
    info = get_memory_node_info()
    assert info.total_memory <= limit_bytes, (
        f"reported total_memory={info.total_memory} exceeds the container's "
        f"{limit_bytes}-byte cgroup limit -- the cgroup clamp did not apply "
        "and this run sized off host RAM instead"
    )


def test_auto_sized_kv_cache_does_not_oom_under_cgroup_limit():
    """Boot a tiny model with no explicit KV-cache size: the fraction-of-
    available-memory auto-sizing path must clamp to the container's cgroup
    limit instead of trying to claim a fraction of host RAM, which would
    get this process (or a sibling, e.g. dockerd in a DinD sidecar)
    OOM-killed before it ever reaches this assertion."""
    _require_container_memory_limit()
    assert "VLLM_CPU_KVCACHE_SPACE" not in os.environ, (
        "this test only exercises the auto-sizing path; "
        "VLLM_CPU_KVCACHE_SPACE must be unset"
    )
    llm = LLM(
        model="facebook/opt-125m",
        dtype="bfloat16",
        max_model_len=256,
        enforce_eager=True,
    )
    outputs = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=8))
    assert outputs[0].outputs[0].text
