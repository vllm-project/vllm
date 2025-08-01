from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar, Optional

from dllm import constants


class VllmInstanceStatus(Enum):
    UNREADY = auto()
    RUNNING = auto()
    SUBPROCESS_EXITED = auto()
    HEALTHCHECK_FAILED = auto()


class Role(Enum):
    PREFILL = 0
    DECODE = 1
    MIXED = 2


class SchedulerPolicy(Enum):
    ROUND_ROBIN = 0


@dataclass
class VllmInstanceInfo:
    id: str
    uri: str
    role: Role
    status: VllmInstanceStatus = VllmInstanceStatus.UNREADY
    dp_master_ip: str = ""
    dp_master_port: int = 0


@dataclass
class DispatchResult:
    prefill_uri: Optional[str]
    decode_uri: Optional[str]


@dataclass
class MetricsInfo:
    num_running_requests: int = 0
    num_waiting_requests: int = 0
    device_usage_percent: float = 0.0

    METRIC_NAME_MAPPING: ClassVar[dict] = {
        "num_running_requests": constants.NUM_RUNNING_REQUESTS,
        "num_waiting_requests": constants.NUM_WAITING_REQUESTS,
        "device_usage_percent": constants.DEVICE_USAGE_PERCENT,
    }