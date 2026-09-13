import pytest

def validate_sampling_bounds(temperature: float, top_p: float, top_k: int):
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0.0, 1.0]")
    if top_k < -1:
        raise ValueError("top_k must be >= -1")
    return True

def test_sampling_bounds_valid():
    assert validate_sampling_bounds(0.0, 1.0, -1) is True
    assert validate_sampling_bounds(0.7, 0.95, 50) is True

def test_sampling_bounds_invalid():
    with pytest.raises(ValueError, match="temperature"):
        validate_sampling_bounds(-0.1, 1.0, 50)
    with pytest.raises(ValueError, match="top_p"):
        validate_sampling_bounds(0.7, 0.0, 50)
    with pytest.raises(ValueError, match="top_k"):
        validate_sampling_bounds(0.7, 0.9, -2)
