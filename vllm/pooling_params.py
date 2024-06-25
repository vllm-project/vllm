from typing import Any, Optional


class PoolingParams:
    """Pooling parameters for pooling.

    Attributes:
        additional_data: Any additional data needed for pooling.
    """

    def __init__(self, additional_data: Optional[Any] = None):
        self.additional_data = additional_data

    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return PoolingParams(additional_data=self.additional_data, )

    def __repr__(self) -> str:
        return (f"PoolingParams("
                f"additional_metadata={self.additional_data})")
