"""Lab 04: Batch Processing - Tests"""

import pytest
from solution import create_batches, process_batch


def test_create_batches():
    """Test batch creation."""
    data = list(range(10))
    batches = list(create_batches(data, 3))
    assert len(batches) == 4
    assert batches[0] == [0, 1, 2]
    assert batches[-1] == [9]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
