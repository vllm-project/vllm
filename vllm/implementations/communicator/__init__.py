from enum import Enum, auto


class CommunicatorType(Enum):
    PYNCCL = auto()


def get_communicator_class(communicator_type: CommunicatorType) -> type:
    # lazy init
    # only import the communicator when it is needed
    if communicator_type == CommunicatorType.PYNCCL:
        from vllm.implementations.communicator.nccl.pynccl.pynccl_communicator import (  # noqa
            NCCLCommunicator)
        return NCCLCommunicator
    else:
        raise ValueError(
            f"Communicator type {communicator_type} not regonized.")
