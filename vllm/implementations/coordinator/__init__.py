from vllm.interfaces.coordinator import Coordinator


def get_coordinator_class(name: str) -> Coordinator:
    # lazy init
    # only import the coordinator when it is needed
    if name == "torch_distributed":
        from vllm.implementations.coordinator.torch_distributed.torch_distributed_coordinator import (  # noqa
            TorchDistributedCoordinator)
        return TorchDistributedCoordinator
    else:
        raise ValueError(f"Coordinator type {name} not regonized.")
