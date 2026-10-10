# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pandas as pd
import pytest

from vllm.benchmarks.sweep.plot import (
    PlotBinner,
    PlotEqualTo,
    PlotFilterBase,
    PlotFilters,
    PlotGreaterThan,
    PlotGreaterThanOrEqualTo,
    PlotLessThan,
    PlotLessThanOrEqualTo,
    PlotNotEqualTo,
)


class TestPlotFilters:
    """Test PlotFilter functionality including 'inf' edge case."""

    def setup_method(self):
        """Create sample DataFrames for testing."""
        # DataFrame with numeric values
        self.df_numeric = pd.DataFrame(
            {
                "request_rate": [1.0, 5.0, 10.0, 50.0, 100.0],
                "value": [10, 20, 30, 40, 50],
            }
        )

        # DataFrame with float('inf') - note: string "inf" values are coerced
        # to float when loading data, so we only test with float('inf')
        self.df_inf_float = pd.DataFrame(
            {
                "request_rate": [1.0, 5.0, 10.0, float("inf"), float("inf")],
                "value": [10, 20, 30, 40, 50],
            }
        )

        self.df_boolean = pd.DataFrame(
            {
                "enable_prefix_caching": [True, False],
                "request_rate": [10, 20],
            }
        )

    @pytest.mark.parametrize(
        "target,expected_count",
        [
            ("5.0", 1),
            ("10.0", 1),
            ("1.0", 1),
        ],
    )
    def test_equal_to_numeric(self, target, expected_count):
        """Test PlotEqualTo with numeric values."""
        filter_obj = PlotEqualTo("request_rate", target)
        result = filter_obj.apply(self.df_numeric)
        assert len(result) == expected_count

    def test_equal_to_inf_float(self):
        """Test PlotEqualTo with float('inf')."""
        filter_obj = PlotEqualTo("request_rate", "inf")
        result = filter_obj.apply(self.df_inf_float)
        # Should match both float('inf') entries because float('inf') == float('inf')
        assert len(result) == 2

    @pytest.mark.parametrize(
        "target,expected_count",
        [
            ("5.0", 4),  # All except 5.0
            ("1.0", 4),  # All except 1.0
        ],
    )
    def test_not_equal_to_numeric(self, target, expected_count):
        """Test PlotNotEqualTo with numeric values."""
        filter_obj = PlotNotEqualTo("request_rate", target)
        result = filter_obj.apply(self.df_numeric)
        assert len(result) == expected_count

    def test_not_equal_to_inf_float(self):
        """Test PlotNotEqualTo with float('inf')."""
        filter_obj = PlotNotEqualTo("request_rate", "inf")
        result = filter_obj.apply(self.df_inf_float)
        # Should exclude float('inf') entries
        assert len(result) == 3

    def test_equal_to_boolean(self):
        """Boolean filter targets should match JSON boolean columns."""
        filter_obj = PlotFilters.parse_str("enable_prefix_caching==true")

        result = filter_obj.apply(self.df_boolean)

        assert result["request_rate"].tolist() == [10]

    def test_not_equal_to_boolean(self):
        """Boolean inequality should exclude only the matching rows."""
        filter_obj = PlotFilters.parse_str("enable_prefix_caching!=true")

        result = filter_obj.apply(self.df_boolean)

        assert result["request_rate"].tolist() == [20]

    def test_boolean_target_is_literal_for_string_column(self):
        """String labels named true/false must keep string comparisons."""
        df = pd.DataFrame({"label": ["true", "false"]})

        result = PlotFilters.parse_str("label==true").apply(df)

        assert result["label"].tolist() == ["true"]

    def test_boolean_filter_with_missing_values(self):
        """Nullable boolean columns keep missing rows out of either match."""
        df = pd.DataFrame(
            {
                "enabled": pd.Series([True, False, None], dtype="boolean"),
                "request_rate": [10, 20, 30],
            }
        )

        equal = PlotFilters.parse_str("enabled==false").apply(df)
        not_equal = PlotFilters.parse_str("enabled!=false").apply(df)

        assert equal["request_rate"].tolist() == [20]
        assert not_equal["request_rate"].tolist() == [10]

    @pytest.mark.parametrize(
        "target,expected_count",
        [
            ("10.0", 2),  # 1.0, 5.0
            ("50.0", 3),  # 1.0, 5.0, 10.0
            ("5.0", 1),  # 1.0
        ],
    )
    def test_less_than(self, target, expected_count):
        """Test PlotLessThan with numeric values."""
        filter_obj = PlotLessThan("request_rate", target)
        result = filter_obj.apply(self.df_numeric)
        assert len(result) == expected_count

    @pytest.mark.parametrize(
        "target,expected_count",
        [
            ("10.0", 3),  # 1.0, 5.0, 10.0
            ("5.0", 2),  # 1.0, 5.0
        ],
    )
    def test_less_than_or_equal_to(self, target, expected_count):
        """Test PlotLessThanOrEqualTo with numeric values."""
        filter_obj = PlotLessThanOrEqualTo("request_rate", target)
        result = filter_obj.apply(self.df_numeric)
        assert len(result) == expected_count

    @pytest.mark.parametrize(
        "target,expected_count",
        [
            ("10.0", 2),  # 50.0, 100.0
            ("5.0", 3),  # 10.0, 50.0, 100.0
        ],
    )
    def test_greater_than(self, target, expected_count):
        """Test PlotGreaterThan with numeric values."""
        filter_obj = PlotGreaterThan("request_rate", target)
        result = filter_obj.apply(self.df_numeric)
        assert len(result) == expected_count

    @pytest.mark.parametrize(
        "target,expected_count",
        [
            ("10.0", 3),  # 10.0, 50.0, 100.0
            ("5.0", 4),  # 5.0, 10.0, 50.0, 100.0
        ],
    )
    def test_greater_than_or_equal_to(self, target, expected_count):
        """Test PlotGreaterThanOrEqualTo with numeric values."""
        filter_obj = PlotGreaterThanOrEqualTo("request_rate", target)
        result = filter_obj.apply(self.df_numeric)
        assert len(result) == expected_count

    @pytest.mark.parametrize(
        "filter_str,expected_var,expected_target,expected_type",
        [
            ("request_rate==5.0", "request_rate", "5.0", PlotEqualTo),
            ("request_rate!=10.0", "request_rate", "10.0", PlotNotEqualTo),
            ("request_rate<50.0", "request_rate", "50.0", PlotLessThan),
            ("request_rate<=50.0", "request_rate", "50.0", PlotLessThanOrEqualTo),
            ("request_rate>10.0", "request_rate", "10.0", PlotGreaterThan),
            ("request_rate>=10.0", "request_rate", "10.0", PlotGreaterThanOrEqualTo),
            ("request_rate==inf", "request_rate", "inf", PlotEqualTo),
            ("request_rate!='inf'", "request_rate", "inf", PlotNotEqualTo),
        ],
    )
    def test_parse_str(self, filter_str, expected_var, expected_target, expected_type):
        """Test parsing filter strings."""
        filter_obj = PlotFilterBase.parse_str(filter_str)
        assert isinstance(filter_obj, expected_type)
        assert filter_obj.var == expected_var
        assert filter_obj.target == expected_target

    def test_parse_str_inf_edge_case(self):
        """Test parsing 'inf' string in filter."""
        filter_obj = PlotFilterBase.parse_str("request_rate==inf")
        assert isinstance(filter_obj, PlotEqualTo)
        assert filter_obj.var == "request_rate"
        assert filter_obj.target == "inf"

    def test_parse_str_preserves_operator_in_quoted_target(self):
        """Only the first operator separates a filter's variable and target."""
        filter_obj = PlotFilterBase.parse_str('model=="foo==bar"')

        assert filter_obj.var == "model"
        assert filter_obj.target == "foo==bar"

    def test_parse_str_strips_variable_whitespace(self):
        """Whitespace around a filter expression should not alter its column."""
        filter_obj = PlotFilterBase.parse_str(" request_rate >= 10.0 ")

        assert filter_obj.var == "request_rate"
        assert filter_obj.target == "10.0"

    def test_parse_multiple_filters(self):
        """Test parsing multiple filters."""
        filters = PlotFilters.parse_str("request_rate>5.0,value<=40")
        assert len(filters) == 2
        assert isinstance(filters[0], PlotGreaterThan)
        assert isinstance(filters[1], PlotLessThanOrEqualTo)

    def test_parse_empty_filter(self):
        """Test parsing empty filter string."""
        filters = PlotFilters.parse_str("")
        assert len(filters) == 0

    def test_parse_binner_strips_whitespace(self):
        """Whitespace around a bin expression should not break parsing."""
        binner = PlotBinner.parse_str(" request_rate % 5 ")

        assert binner.var == "request_rate"
        assert binner.bin_size == 5
