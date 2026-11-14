from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class SchedulerConfig:
    # vLLM cache_config.cache_dtype string (e.g. "auto", "fp8", "fp8_e4m3").
    kv_cache_dtype: str = "auto"
    mem_fraction_static: Optional[float] = None

    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    pp_size: int = 1

    # framework backend
    backend_name: str = "vllm"
    backend_version: Optional[str] = None

    @property
    def attn_tp_size(self) -> int:
        return self.tp_size // self.dp_size

    @property
    def attn_dp_size(self) -> int:
        return self.dp_size

    @property
    def moe_tp_size(self) -> int:
        return self.tp_size // self.ep_size

    @property
    def moe_ep_size(self) -> int:
        return self.ep_size


class SimulationMode(Enum):
    BLOCKING = "BLOCKING"
    NON_BLOCKING = "NON_BLOCKING"


@dataclass(slots=True)
class RequestStats:
    rid: str = ""
    last_event_time: float = 0.0
    input_length: int = 1
    output_length: int = 1

    # Prefix cache hits: local prefix cache vs KV connector.
    local_kv_hit_len: int = 0
    ext_kv_hit_len: int = 0

    queue_start: float = -1
    queue_end: float = -1
    created_time: float = -1
    gen_token_latencies: list[float] = field(default_factory=list)


@dataclass(slots=True)
class IterationStats:
    # [extend_length, past_kv_length] per scheduled request.
    requests: list[list[int]] = field(default_factory=list)
    # RPC-1 only, matching the predictor's training label.
    forward_latency: float = 0.0
    sample_tokens_latency: float = 0.0
    logprobs_latency: float = 0.0
    total_tokens: int = 0
    # Engine-step wall time, excluding predictor work and idle waits.
    cpu_overhead: float = 0.0

    @property
    def full_step_latency(self) -> float:
        return (
            self.forward_latency
            + self.sample_tokens_latency
            + self.logprobs_latency
            + self.cpu_overhead
        )
