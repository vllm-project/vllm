"""Parameters for speculative deocding."""
from typing import Any, Optional

import copy
import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

class SpecDecodeParams:
    """
    Parameters for Speculative Decoding choices and future features.

    Note that this class should be used internally. For online
    serving, it is recommended to not allow users to use this class but
    instead provide another layer of abstraction to prevent users from
    accessing unauthorized Speculative Decoding proposers.

    Args:
        proposer_name: Name of proposer to used for SpecDecodeWorker.
        proposer_id: The id of proposer to used for SpecDecodeWorker.
    """

    def __init__(
        self,
        proposer_name: str,
    ) -> None:
        self.proposer_name = proposer_name
        self._verify_args()
        
    
    def _verify_args(self) -> None:
        if self.proposer_name is None:
            raise ValueError(f"proposer_name must be provided.")
    def clone(self) -> "SpecDecodeParams":
        return copy.deepcopy(self)
    
    def get_proposer(self) -> str:
        return self.proposer_name
    
    def set_proposer(self, proposer_name: str) -> None:
        self.proposer_name = proposer_name

    def __repr__(self) -> str:
        return (
            f"SpecDecodeParams(proposer_name={self.proposer_name})")
