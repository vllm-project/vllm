# coordinator interface, as proposed in
# https://github.com/vllm-project/vllm/issues/3587
# `Coordinator` is responsible for communicating **tiny control messages**
# between multiple processes. This functionality is usually provided by
# PyTorch (gloo backend) or MPI, implemented using CPU.
# Put it simple, this is for control-plane communication.

from abc import ABC, abstractmethod
from typing import List


class Coordinator(ABC):
    """This is the abstract interface for the coordinator.
    The least requirement for the coordinator is to provide:
    1. The world size of the distributed environment.
    2. The rank of the current process inside the world.
    3. The local rank of the current process inside the node.
    4. The local world size inside the node.
    5. The current group the process belongs to.

    Note that if the `group` is provided, the coordinator should only
    synchronize the processes in the group. If you want to not only
    coordinate **all** the processes but also coordinate **subgroups**
    of processes, you can create multiple coordinators. In that case,
    the coordinator with only one group must be initialized first.

    To avoid confusion in argument passing, all arguments are set
    to be keyword-only.

    Usually subclasses need to implement the following methods:
    1. `__init__`: Initialize the coordinator, only set the necessary
     attributes with sanity checks.
    2. `initialize`: Initialize the coordinator. This is set to be a
     separate method for lazy initialization. In addition, subclasses
     should call this method after their `initialize` method.
    3. `barrier`: Synchronize all the processes.
    4. `broadcast`: Broadcast a message from the source process to all
     other processes.
    """

    def __init__(self, *, rank: int, world_size: int, local_rank: int,
                 local_world_size: int, groups: List[List[int]], **kwargs):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        self._initialize = False
        self.groups = groups
        self.group = [g for g in groups if self.rank in g][0]

    def initialize(self):
        """Initialize the coordinator. This is set to be a separate method
        so that the coordinator can be initialized after the object is created.

        This method is supposed to be called by all the participating processes.
        """
        self._initialize = True

    def is_initialized(self) -> bool:
        """Check if the coordinator has been initialized."""
        return self._initialize

    def get_world_size(self) -> int:
        """Get the world size of the distributed environment."""
        return self.world_size

    def get_rank(self) -> int:
        """Get the rank of the current process inside the world."""
        return self.rank

    def is_master(self) -> bool:
        """Check if the current process is the master process."""
        return self.rank == 0

    def get_local_world_size(self) -> int:
        """Get the local world size inside the node."""
        return self.local_world_size

    def get_local_rank(self) -> int:
        """Get the local rank of the current process inside the node."""
        return self.local_rank

    def is_local_master(self) -> bool:
        """Check if the current process is the local master process."""
        return self.local_rank == 0

    def get_group_size(self) -> int:
        """Get the size of the group."""
        return len(self.group)

    def get_group_rank(self) -> int:
        """Get the rank of the current process inside the group."""
        return self.group.index(self.rank)

    def is_group_master(self) -> bool:
        """Check if the current process is the group master process."""
        return self.get_group_rank() == 0

    def get_group_master_rank(self) -> int:
        """Get the rank of the group master process."""
        return self.group[0]

    @abstractmethod
    def barrier(self):
        """Synchronize all the processes."""
        raise NotImplementedError

    @abstractmethod
    def broadcast(self, message: bytearray, src: int = 0) -> None:
        """Broadcast a message from the source process to all other processes.
        Note that the message type is explicitly set to `bytearray`, to
        indicate that this is a tiny control message.

        Note: this is an in-place operation, the message is modified in-place.
        """
        raise NotImplementedError

    @abstractmethod
    def __del__(self):
        """Release the resources."""
        raise NotImplementedError
