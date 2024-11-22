from enum import Enum, auto
from typing import Optional

import msgspec

from vllm.v1.core.scheduler import SchedulerOutput


#TODO: Move this file
class ExecutorMsgType(Enum):
    TOIL = auto()
    TERMINATE = auto()


class ExecutorMsg(msgspec.Struct,
                  array_like=True,
                  omit_defaults=True,
                  gc=False):
    """A directive from the core process to its worker processes.
    
	Wraps SchedulerOutput with a message type to distinguish between
	regular work assignments and termination orders."""
    message_type: ExecutorMsgType
    payload: Optional[SchedulerOutput]
