from vllm.implementations.communicator import (CommunicatorType,
                                               get_communicator_class)
from vllm.implementations.coordinator import (CoordinatorType,
                                              get_coordinator_class)
from vllm.interfaces.communicator import Communicator
from vllm.interfaces.coordinator import Coordinator
from vllm.interfaces.launcher import DistributedTask


class GlobalCoordinatorDistributedTask(DistributedTask):

    def run(self, *, coordinator_type: CoordinatorType,
            communicator_type: CommunicatorType, **kwargs):
        coordinator_cls = get_coordinator_class(coordinator_type)
        communicator_cls = get_communicator_class(communicator_type)
        self.coordinator: Coordinator = coordinator_cls()
        self.coordinator.initialize()
        self.communicator: Communicator = communicator_cls(self.coordinator)
        self.post_init_distributed(**kwargs)

    def post_init_distributed(self, **kwargs):
        """Subclasses can override this method to do whatever they want.
        They can use `self.coordinator` for global communication over the 
        whole process group.
        They can use `self.communicator` for communication between devices.
        """
        return
