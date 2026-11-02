import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from vllm_simulator.simulation.types import SchedulerConfig
from vllm_simulator.spec.accelerator import AcceleratorInfo

if typing.TYPE_CHECKING:
    from vllm.config import ModelConfig


@dataclass
class ScheduleRequest:
    extend_length: int = 0
    past_kv_length: int = 0


@dataclass
class ScheduleBatch:
    reqs: list[ScheduleRequest] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"batch_size={len(self.reqs)},reqs={[(req.extend_length, req.past_kv_length) for req in self.reqs]}"

    def request_info(self) -> list[list[int, int]]:
        # The request information organized in the format `(input_len, past_kv_len)`
        return [[req.extend_length, req.past_kv_length] for req in self.reqs]

    @property
    def batch_size(self) -> int:
        return len(self.reqs)

    def is_empty(self) -> bool:
        return len(self.reqs) == 0

    def is_decode(self) -> bool:
        for req in self.reqs:
            if req.extend_length > 1:
                return False
        return True


class InferTimePredictor(ABC):
    # Extra per-token cost of logprobs requests, in seconds.
    logprobs_cost_s_per_token: float = 0.0

    def __init__(
        self,
        model: "ModelConfig",
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        *args,
        **kwargs,
    ):
        self.model: "ModelConfig" = model
        self.hw: AcceleratorInfo = hw
        self.config: SchedulerConfig = config

    @abstractmethod
    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        # Return the forward (RPC-1) time in seconds.
        pass

    def predict_sample_tokens_time(self, total_tokens: int) -> float:
        """Return the sample_tokens (RPC-2) time in seconds; 0 if not modelled."""
        return 0.0

    def predict_step_time(self, batch: ScheduleBatch) -> tuple[float, float]:
        """Return the forward and sample_tokens times for one engine step."""
        total_tokens = sum(req.extend_length for req in batch.reqs)
        return (
            self.predict_infer_time(batch),
            self.predict_sample_tokens_time(total_tokens),
        )
