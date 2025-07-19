# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid
from collections import defaultdict
from typing import Optional


class FakeNixlWrapper:
    """Mock implementation of NixlWrapper for testing.
    
    We don't inherit from nixl._api.nixl_agent because nixl may not be
    installed.
    """

    AGENT_METADATA = b"fake_agent_metadata"
    REMOTE_AGENT_NAME = "remote_agent"

    def __init__(self, agent_name: str, *args, **kwargs):
        self._cycles_before_xfer_done = 0
        self._check_xfer_state_cycles: defaultdict[int, int] = defaultdict(
            lambda: 0)

    def get_reg_descs(self, caches_data, memory_type: str) -> list:
        return [str(uuid.uuid4()) for _ in caches_data]

    def register_memory(self, descs) -> None:
        pass

    def get_xfer_descs(self, blocks_data, memory_type: str) -> list:
        return [str(uuid.uuid4()) for _ in blocks_data]

    def prep_xfer_dlist(self, agent_name: str, descs: list) -> int:
        return uuid.uuid4().int

    def get_agent_metadata(self) -> bytes:
        return self.AGENT_METADATA

    def add_remote_agent(self, agent_metadata: bytes) -> str:
        return self.REMOTE_AGENT_NAME

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        # Used to collect done_sending, which we don't test yet.
        return {}

    def check_xfer_state(self, handle: int) -> str:
        if self._check_xfer_state_cycles[
                handle] >= self._cycles_before_xfer_done:
            return "DONE"
        self._check_xfer_state_cycles[handle] += 1
        return "PROC"

    def release_xfer_handle(self, handle: int) -> None:
        pass

    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        pass

    def make_prepped_xfer(self,
                          xfer_type: str,
                          local_xfer_side_handle: int,
                          local_block_descs_ids: list[int],
                          remote_xfer_side_handle: int,
                          remote_block_descs_ids: list[int],
                          notif_msg: Optional[bytes] = None) -> int:
        return uuid.uuid4().int

    def transfer(self, handle: int) -> str:
        return "PROC"

    ############################################################
    # Follow are for changing the behavior during testing.
    ############################################################

    def set_cycles_before_xfer_done(self, cycles: int):
        """Set the number of cycles before a transfer is considered done."""
        self._cycles_before_xfer_done = cycles
