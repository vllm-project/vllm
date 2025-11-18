"""
Lab 03: Custom Sampling Strategies - Tests
"""

import pytest
from unittest.mock import Mock, patch
from solution import (
    create_greedy_params,
    create_temperature_params,
    create_topk_params,
    create_topp_params,
    create_beam_search_params
)


class TestSamplingParams:
    """Tests for sampling parameter creation."""

    @patch('solution.SamplingParams')
    def test_greedy_params(self, mock_class):
        """Test greedy params have temperature=0."""
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        result = create_greedy_params()

        mock_class.assert_called_once_with(temperature=0.0, max_tokens=100)

    @patch('solution.SamplingParams')
    def test_temperature_params(self, mock_class):
        """Test temperature params."""
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        result = create_temperature_params(0.5)

        mock_class.assert_called_once_with(temperature=0.5, max_tokens=100)

    @patch('solution.SamplingParams')
    def test_topk_params(self, mock_class):
        """Test top-k params."""
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        result = create_topk_params(k=10)

        mock_class.assert_called_once_with(temperature=1.0, top_k=10, max_tokens=100)

    @patch('solution.SamplingParams')
    def test_topp_params(self, mock_class):
        """Test top-p params."""
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        result = create_topp_params(p=0.9)

        mock_class.assert_called_once_with(temperature=1.0, top_p=0.9, max_tokens=100)

    @patch('solution.SamplingParams')
    def test_beam_search_params(self, mock_class):
        """Test beam search params."""
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        result = create_beam_search_params(n=4)

        mock_class.assert_called_once_with(
            best_of=4,
            use_beam_search=True,
            temperature=0.0,
            max_tokens=100
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
