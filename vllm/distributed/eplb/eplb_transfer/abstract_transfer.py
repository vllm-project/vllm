# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod


class BaseTransfer(ABC):
    """
    Abstract base class for a transfer component responsible for managing
    expert weights and their transfer/update mechanisms in an Expert Parallel
    (EP) system.

    Concrete implementations will define how weights are prepared for sending,
    prepared for receiving, actually sent/received, and updated locally.
    """

    @abstractmethod
    def prepare_send(self, expert_send_info, layer_id):
        """
        Abstract method: Prepares the necessary data structures or buffers
        for sending expert weights or related information to other ranks.

        This might involve packing weights, creating communication requests,
        or setting up send buffers.
        """
        pass

    @abstractmethod
    def prepare_recv(self, expert_recv_info, updated_expert_map):
        """
        Abstract method: Prepares the necessary data structures or buffers
        for receiving expert weights or related information from other ranks.

        This might involve allocating memory for incoming weights or setting
        up receive buffers.
        """
        pass

    @abstractmethod
    def update_expert_map_and_weight(self):
        """
        Abstract method: Applies the received expert weights or updates
        the local expert weights based on some logic (e.g., after a transfer).

        This might involve copying data from a receive buffer to the actual
        expert weight tensors.
        """
        pass
