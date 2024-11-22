from typing import Optional


class CachingParams:
    """Pooling parameters for pooling.

    Attributes:
        additional_data: Any additional data needed for pooling.
    """

    def __init__(self,
                 expired_at: Optional[float] = None,
                 ttl: Optional[float] = None):
        if expired_at is None and ttl is None:
            raise ValueError("expired_at and ttl must specify one")
        self.expired_at = expired_at
        self.ttl = ttl

    def clone(self) -> "CachingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return CachingParams(self.expired_at, self.ttl)

    def __repr__(self) -> str:
        return (f"PoolingParams("
                f"expired_at={self.expired_at}, "
                f"ttl={self.ttl})")
