"""Lab 06: Error Handling - Tests"""

import pytest
from solution import CircuitBreaker


def test_circuit_breaker_init():
    """Test circuit breaker initialization."""
    cb = CircuitBreaker(failure_threshold=3)
    assert cb.state == "CLOSED"
    assert cb.failures == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
