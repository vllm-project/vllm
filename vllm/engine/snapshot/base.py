# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseSnapshotProvider(ABC):
    @abstractmethod
    def trigger(self) -> None:
        """Cloud-specific logic to snapshot the Pod/GPU."""
        pass
