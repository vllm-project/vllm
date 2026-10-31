"""
vLLM utility functions shared by the simulator and collector hooks.
"""

import typing

from vllm_simulator.simulation.types import SchedulerConfig

if typing.TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request


def resolve_scheduler_config(vllm_config: "VllmConfig") -> SchedulerConfig:
    """Create simulator SchedulerConfig from vLLM's VllmConfig."""
    import vllm

    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config

    return SchedulerConfig(
        kv_cache_dtype=cache_config.cache_dtype,
        mem_fraction_static=cache_config.gpu_memory_utilization,
        tp_size=parallel_config.tensor_parallel_size,
        pp_size=parallel_config.pipeline_parallel_size,
        dp_size=parallel_config.data_parallel_size,
        backend_name="vllm",
        backend_version=vllm.__version__,
    )


def kv_hit_split(request: "Request", num_computed_tokens: int) -> tuple[int, int]:
    """(local, external) prefix hit tokens of a newly scheduled request.

    The split is recorded on prefill_stats at the first schedule; without it
    (e.g. after a preemption) every computed token counts as local.
    """
    prefill_stats = request.prefill_stats
    if prefill_stats is not None and prefill_stats.num_prompt_tokens:
        return (
            prefill_stats.num_local_cached_tokens,
            prefill_stats.num_external_cached_tokens,
        )
    return max(num_computed_tokens, 0), 0
