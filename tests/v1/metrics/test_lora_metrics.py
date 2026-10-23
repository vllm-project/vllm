# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrometheusStatLogger's LoRA metrics, fed LoRALoadEvents directly."""

import prometheus_client
import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.lora.load_event import LoRALoadEvent, LoRALoadTiming
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

pytestmark = pytest.mark.cpu_test

LORA_FAMILIES = {
    "vllm:lora_requests_info",
    "vllm:num_gpu_loaded_lora_adapters",
    "vllm:num_cpu_loaded_lora_adapters",
    "vllm:max_gpu_lora_adapters",
    "vllm:max_cpu_lora_adapters",
    "vllm:lora_adapter_loaded",
    "vllm:lora_adapter_load_seconds",
}


@pytest.fixture
def make_logger():
    def make(
        engine_indexes: list[int] | None = None, enable_lora: bool = True
    ) -> PrometheusStatLogger:
        lora_config = LoRAConfig(max_loras=2, max_cpu_loras=3) if enable_lora else None
        config = VllmConfig(
            model_config=ModelConfig(model="Qwen/Qwen3-0.6B"), lora_config=lora_config
        )
        return PrometheusStatLogger(config, engine_indexes)

    yield make
    unregister_vllm_metrics()


def _samples(name: str):
    for metric in prometheus_client.REGISTRY.collect():
        if metric.name == name:
            yield from metric.samples


def _loaded_series() -> dict[tuple[str, str, str], float]:
    """(adapter_name, level, pinned) -> value for every live series."""
    return {
        (s.labels["adapter_name"], s.labels["level"], s.labels["pinned"]): s.value
        for s in _samples("vllm:lora_adapter_loaded")
    }


def _gauge(name: str, engine: str = "0") -> float:
    (value,) = (s.value for s in _samples(name) if s.labels["engine"] == engine)
    return value


def _load_seconds() -> dict[tuple[str, str], tuple[float, float]]:
    """(engine, transition) -> (count, sum) for the load histogram."""
    count, total = {}, {}
    for s in _samples("vllm:lora_adapter_load_seconds"):
        key = (s.labels["engine"], s.labels["transition"])
        if s.name.endswith("_count"):
            count[key] = s.value
        elif s.name.endswith("_sum"):
            total[key] = s.value
    return {key: (count[key], total[key]) for key in count}


def _exposition() -> str:
    """The LoRA families as /metrics serves them, with samples sorted within
    each family."""
    lines = []
    for block in prometheus_client.generate_latest().decode().split("# HELP ")[1:]:
        help_line, type_line, *samples = block.splitlines()
        if help_line.split(" ", 1)[0] in LORA_FAMILIES:
            lines += ["# HELP " + help_line, type_line, *sorted(samples)]
    return "".join(line + "\n" for line in lines)


EXPOSITION = """\
# HELP vllm:lora_requests_info Running stats on lora requests. DEPRECATED: encodes adapter names into comma-separated label values; superseded by vllm:lora_adapter_loaded, vllm:num_gpu_loaded_lora_adapters and vllm:num_cpu_loaded_lora_adapters.
# TYPE vllm:lora_requests_info gauge
# HELP vllm:num_gpu_loaded_lora_adapters Number of LoRA adapters loaded into GPU slots.
# TYPE vllm:num_gpu_loaded_lora_adapters gauge
vllm:num_gpu_loaded_lora_adapters{engine="0",model_name="Qwen/Qwen3-0.6B"} 2.0
# HELP vllm:num_cpu_loaded_lora_adapters Number of LoRA adapters resident in the worker's CPU cache (superset of the GPU-loaded set).
# TYPE vllm:num_cpu_loaded_lora_adapters gauge
vllm:num_cpu_loaded_lora_adapters{engine="0",model_name="Qwen/Qwen3-0.6B"} 3.0
# HELP vllm:max_gpu_lora_adapters Number of GPU LoRA slots (max_loras).
# TYPE vllm:max_gpu_lora_adapters gauge
vllm:max_gpu_lora_adapters{engine="0",model_name="Qwen/Qwen3-0.6B"} 2.0
# HELP vllm:max_cpu_lora_adapters Number of CPU cache LoRA slots (max_cpu_loras).
# TYPE vllm:max_cpu_lora_adapters gauge
vllm:max_cpu_lora_adapters{engine="0",model_name="Qwen/Qwen3-0.6B"} 3.0
# HELP vllm:lora_adapter_loaded Whether a LoRA adapter is loaded in the worker's adapter caches. The series exists (value 1) while the adapter is resident; 'level' is 'gpu' when the adapter is active in a GPU slot and 'cpu' when it is only in the host cache. 'rank' is the adapter's LoRA rank.
# TYPE vllm:lora_adapter_loaded gauge
vllm:lora_adapter_loaded{adapter_name="alpha",engine="0",level="gpu",model_name="Qwen/Qwen3-0.6B",pinned="true",rank="64"} 1.0
vllm:lora_adapter_loaded{adapter_name="delta",engine="0",level="cpu",model_name="Qwen/Qwen3-0.6B",pinned="false",rank="0"} 1.0
vllm:lora_adapter_loaded{adapter_name="gamma",engine="0",level="gpu",model_name="Qwen/Qwen3-0.6B",pinned="false",rank="16"} 1.0
# HELP vllm:lora_adapter_load_seconds Histogram of LoRA adapter transition time in seconds. 'transition' is 'load' for a read from disk into the CPU cache and 'activate' for a move from the CPU cache into a GPU slot.
# TYPE vllm:lora_adapter_load_seconds histogram
vllm:lora_adapter_load_seconds_bucket{engine="0",le="+Inf",model_name="Qwen/Qwen3-0.6B",transition="activate"} 1.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="+Inf",model_name="Qwen/Qwen3-0.6B",transition="load"} 3.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="0.01",model_name="Qwen/Qwen3-0.6B",transition="activate"} 0.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="0.01",model_name="Qwen/Qwen3-0.6B",transition="load"} 0.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="0.025",model_name="Qwen/Qwen3-0.6B",transition="activate"} 0.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="0.025",model_name="Qwen/Qwen3-0.6B",transition="load"} 0.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="0.05",model_name="Qwen/Qwen3-0.6B",transition="activate"} 1.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="0.05",model_name="Qwen/Qwen3-0.6B",transition="load"} 0.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="0.1",model_name="Qwen/Qwen3-0.6B",transition="activate"} 1.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="0.1",model_name="Qwen/Qwen3-0.6B",transition="load"} 0.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="0.25",model_name="Qwen/Qwen3-0.6B",transition="activate"} 1.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="0.25",model_name="Qwen/Qwen3-0.6B",transition="load"} 1.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="0.5",model_name="Qwen/Qwen3-0.6B",transition="activate"} 1.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="0.5",model_name="Qwen/Qwen3-0.6B",transition="load"} 1.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="1.0",model_name="Qwen/Qwen3-0.6B",transition="activate"} 1.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="1.0",model_name="Qwen/Qwen3-0.6B",transition="load"} 1.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="10.0",model_name="Qwen/Qwen3-0.6B",transition="activate"} 1.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="10.0",model_name="Qwen/Qwen3-0.6B",transition="load"} 2.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="2.5",model_name="Qwen/Qwen3-0.6B",transition="activate"} 1.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="2.5",model_name="Qwen/Qwen3-0.6B",transition="load"} 2.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="5.0",model_name="Qwen/Qwen3-0.6B",transition="activate"} 1.0
vllm:lora_adapter_load_seconds_bucket{engine="0",le="5.0",model_name="Qwen/Qwen3-0.6B",transition="load"} 2.0
vllm:lora_adapter_load_seconds_count{engine="0",model_name="Qwen/Qwen3-0.6B",transition="activate"} 1.0
vllm:lora_adapter_load_seconds_count{engine="0",model_name="Qwen/Qwen3-0.6B",transition="load"} 3.0
vllm:lora_adapter_load_seconds_sum{engine="0",model_name="Qwen/Qwen3-0.6B",transition="activate"} 0.03
vllm:lora_adapter_load_seconds_sum{engine="0",model_name="Qwen/Qwen3-0.6B",transition="load"} 13.7
"""  # noqa: E501


def test_exposition_is_pinned(make_logger):
    """Routers parse these names, labels, values and buckets."""
    logger = make_logger()
    logger.record_lora_load_event(
        LoRALoadEvent(
            gpu_adapters=["alpha", "beta"],
            cpu_adapters=["alpha", "beta", "gamma"],
            pinned_adapters=["alpha"],
            ranks={"alpha": 64, "beta": 8, "gamma": 16},
            loads=[
                LoRALoadTiming("alpha", "load", 0.2),
                LoRALoadTiming("alpha", "activate", 0.03),
                LoRALoadTiming("gamma", "load", 1.5),
            ],
        )
    )
    logger.record_lora_load_event(
        LoRALoadEvent(
            gpu_adapters=["alpha", "gamma"],
            cpu_adapters=["alpha", "delta", "gamma"],
            pinned_adapters=["alpha"],
            ranks={"alpha": 64, "gamma": 16},
            loads=[LoRALoadTiming("delta", "load", 12.0)],
        )
    )

    assert _exposition() == EXPOSITION


def test_eviction_removes_the_series(make_logger):
    """A snapshot replaces the previous one: evicted adapters must not keep
    reading as resident, and a tier change must not leave the old series."""
    logger = make_logger()
    logger.record_lora_load_event(
        LoRALoadEvent(gpu_adapters=["alpha", "beta"], cpu_adapters=["alpha", "beta"])
    )

    logger.record_lora_load_event(
        LoRALoadEvent(gpu_adapters=["gamma"], cpu_adapters=["beta", "gamma"])
    )

    assert _loaded_series() == {
        ("beta", "cpu", "false"): 1.0,
        ("gamma", "gpu", "false"): 1.0,
    }
    assert _gauge("vllm:num_gpu_loaded_lora_adapters") == 1
    assert _gauge("vllm:num_cpu_loaded_lora_adapters") == 2


def test_empty_snapshot_clears_everything(make_logger):
    logger = make_logger()
    logger.record_lora_load_event(
        LoRALoadEvent(gpu_adapters=["alpha"], cpu_adapters=["alpha"])
    )

    logger.record_lora_load_event(LoRALoadEvent())

    assert _loaded_series() == {}
    assert _gauge("vllm:num_gpu_loaded_lora_adapters") == 0
    assert _gauge("vllm:num_cpu_loaded_lora_adapters") == 0


def test_engines_are_tracked_independently(make_logger):
    logger = make_logger(engine_indexes=[0, 1])

    logger.record_lora_load_event(
        LoRALoadEvent(gpu_adapters=["alpha"], cpu_adapters=["alpha"]), engine_idx=0
    )
    logger.record_lora_load_event(
        LoRALoadEvent(gpu_adapters=["beta"], cpu_adapters=["beta"]), engine_idx=1
    )
    logger.record_lora_load_event(LoRALoadEvent(), engine_idx=0)

    assert _loaded_series() == {("beta", "gpu", "false"): 1.0}
    assert _gauge("vllm:num_gpu_loaded_lora_adapters", engine="0") == 0
    assert _gauge("vllm:num_gpu_loaded_lora_adapters", engine="1") == 1


def test_load_timings_land_in_the_histogram_per_engine_and_transition(make_logger):
    logger = make_logger(engine_indexes=[0, 1])
    logger.record_lora_load_event(
        LoRALoadEvent(
            gpu_adapters=["alpha"],
            cpu_adapters=["alpha"],
            loads=[
                LoRALoadTiming("alpha", "load", 0.4),
                LoRALoadTiming("alpha", "activate", 0.05),
            ],
        ),
        engine_idx=0,
    )
    logger.record_lora_load_event(
        LoRALoadEvent(loads=[LoRALoadTiming("beta", "load", 1.5)]), engine_idx=1
    )

    observed = _load_seconds()
    assert observed[("0", "load")] == (1.0, pytest.approx(0.4))
    assert observed[("0", "activate")] == (1.0, pytest.approx(0.05))
    assert observed[("1", "load")] == (1.0, pytest.approx(1.5))
    assert observed[("1", "activate")] == (0.0, 0.0)


def test_load_event_without_lora_config_is_a_noop(make_logger):
    """Engines started without --enable-lora register no LoRA metrics."""
    logger = make_logger(enable_lora=False)

    logger.record_lora_load_event(LoRALoadEvent(cpu_adapters=["alpha"]))

    assert _exposition() == ""
