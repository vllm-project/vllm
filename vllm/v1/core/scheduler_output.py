from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from vllm.v1.core.scheduler import SchedulerOutput


#TODO: Move this file
class ExecutorMsgType(Enum):
    WORK = auto()
    TERMINATE = auto()


@dataclass
class ExecutorMsg:
    """A directive from the core process to its worker processes.
    
	Wraps SchedulerOutput with a message type to distinguish between
	regular work assignments and termination orders."""
    message_type: ExecutorMsgType
    payload: Optional[SchedulerOutput]
