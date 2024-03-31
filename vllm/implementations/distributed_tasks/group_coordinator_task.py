from typing import List

from vllm.implementations.communicator import (CommunicatorType,
                                               get_communicator_class)
from vllm.implementations.coordinator import (CoordinatorType,
                                              get_coordinator_class)
from vllm.interfaces.communicator import Communicator
from vllm.interfaces.coordinator import Coordinator
from vllm.interfaces.launcher import DistributedTask


class GroupCoordinatorDistributedTask(DistributedTask):

    def run(self, *, coordinator_type: CoordinatorType,
            communicator_type: CommunicatorType, groups: List[List[int]],
            **kwargs):
        coordinator_cls = get_coordinator_class(coordinator_type)
        communicator_cls = get_communicator_class(communicator_type)
        self.global_coordinator: Coordinator = coordinator_cls()
        self.global_coordinator.initialize()

        self.group_coordinator: Coordinator = coordinator_cls(groups)
        self.group_coordinator.initialize()

        self.communicator: Communicator = communicator_cls(
            self.group_coordinator)
        self.post_init_distributed(**kwargs)

    def post_init_distributed(self, **kwargs):
        """Subclasses can override this method to do whatever they want.
        They can use `self.global_coordinator` for global communication
         over the whole process group.
        They can use `self.group_coordinator` for communication within a
         subgroup.
        They can use `self.communicator` for communication between devices
         in a subgroup.
        """
        return
