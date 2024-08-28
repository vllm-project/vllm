"""Parameters for speculative decoding."""
import copy


class SpecDecodeParams:
    """
    Parameters for Speculative Decoding choices and future features.

    Args:
        proposer_name: Name of proposer to be used for SpecDecodeWorker.
    """

    def __init__(
        self,
        proposer_name: str,
    ) -> None:
        self.proposer_name = proposer_name
        self._verify_args()

    def _verify_args(self) -> None:
        if not isinstance(self.proposer_name, str) or not self.proposer_name:
            raise ValueError("proposer_name (a non-empty string) must be "
                             "provided.")

    def clone(self) -> "SpecDecodeParams":
        return copy.deepcopy(self)

    def get_proposer(self) -> str:
        return self.proposer_name

    def set_proposer(self, proposer_name: str) -> None:
        if not isinstance(proposer_name, str) or not proposer_name:
            raise ValueError("proposer_name (a non-empty string) must be "
                             "provided.")
        self.proposer_name = proposer_name

    def __repr__(self) -> str:
        return (f"SpecDecodeParams(proposer_name={self.proposer_name})")
