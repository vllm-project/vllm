from enum import Enum, auto


class CoordinatorType(Enum):
    TORCH_DISTRIBUTED = auto()


def get_coordinator_class(coordinator_type: CoordinatorType) -> type:
    # lazy init
    # only import the coordinator when it is needed
    if coordinator_type == CoordinatorType.TORCH_DISTRIBUTED:
        from vllm.implementations.coordinator.torch_distributed.torch_distributed_coordinator import (  # noqa
            TorchDistributedCoordinator)
        return TorchDistributedCoordinator
    else:
        raise ValueError(f"Coordinator type {coordinator_type} not regonized.")
