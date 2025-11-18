"""Lab 07: Performance Monitoring - Tests"""

import pytest
from solution import MetricsCollector


def test_metrics_collector():
    """Test metrics collection."""
    collector = MetricsCollector()
    start = collector.start_request()
    collector.end_request(start)
    assert len(collector.latencies) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
