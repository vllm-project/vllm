# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, List
import abc

if TYPE_CHECKING:
    # Third Party
    pass


class OffloadServerInterface(metaclass=abc.ABCMeta):
    """Abstract interface for offload server."""

    @abc.abstractmethod
    def offload(
        self,
        hashes: List[int],
        slot_mapping: List[int],
        offsets: List[int],
    ) -> bool:
        """
        Perform offload for the given hashes and block IDs.

        Args:
            hashes: The hashes to offload.
            slot_mapping: The slot ids to offload.
            offsets: Number of tokens in each block.

        Returns:
            Whether the offload was successful.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Close the offload server and clean up resources."""
        raise NotImplementedError
