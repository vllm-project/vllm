# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch


class Torch25CustomGraphPass(ABC):  # noqa (redefinition)
    """
    This class replaces CustomGraphPass from torch==2.6 when using torch<2.6.
    It conforms to the 2.6 interface but also supports pickling, as that's what
    the inductor code cache uses to determine the cache key before 2.6.
    (in 2.6 and above, uuid() is used.)

    Subclasses can just "pretend" that uuid is used.
    """

    @abstractmethod
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        """
        Implementation of the custom pass.
        """

    @abstractmethod
    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom pass implementation.
        Return None to skip inductor code caching entirely.
        """

    def __getstate__(self):
        """
        Pickling is used instead of uuid() in torch<2.6. Just return uuid()
         to enable subclasses to only have to implement uuid.
        """
        return self.uuid()

    def __setstate__(self, state):
        raise ValueError("Cannot unpickle CustomGraphPass because pickling"
                         " is used for cache key uuid. Use torch>=2.6 with"
                         " native uuid support for custom passes.")
