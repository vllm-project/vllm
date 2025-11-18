"""Lab 10: Production Deployment - Tests"""

import pytest
from fastapi.testclient import TestClient
from starter import app


client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200


def test_metrics():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
