# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.v1.structured_output.backend_outlines import validate_regex_is_buildable

pytestmark = pytest.mark.cpu_test


# ================================================
# Test validate_regex_is_buildable functionality
# ================================================


class TestValidateRegexIsBuildable:
    class TestPR60359Regressions:
        """This class implements regression tests for PR #60359.
        Initial bug report issue #60344.
        validate_regex_is_buildable misses \b and backref inside groups
        So, we check here that these features are rejected by us.
        """

        @pytest.mark.parametrize(
            "regex",
            [
                # top-level word boundary \b
                r"a\b",
                # \b inside capturing group
                r"(a\b)",
                # word boundary \b in nested group
                r"((a\b))",
                # top-level backref
                r"(a)\1",
                # backref in nested group
                r"((a)\2)",
                # positive lookahead
                r"(a(?=b))",
                # inline flag group + word boundary
                r"(?i:a\b)",
                # counted repeat over a group containing \b
                r"(?:(a\b)){2}",
                # lazy repeat + word boundary
                r"(?:a\b)*?",
                # possessive repeat + word boundary
                r"(?:a\b)*+",
                # atomic group + word boundary
                r"(?>a\b)",
            ],
        )
        def test_unsupported_regex(self, regex):
            with pytest.raises(VLLMValidationError):
                validate_regex_is_buildable(regex)

        @pytest.mark.parametrize(
            "regex",
            [
                # [\b] is character class, not word-boundary \b
                r"[\b]",
                r"(a[\b]c)",
                # escaped backslash + "b", not word-boundary \b
                r"a\\b",
                r"(a\\bc)",
                # no \b or backref here, must be ok
                r"(abc)",
                # BRANCH and MAX_REPEAT must be ok
                r"(a|b)+c",
                r"(?:ab)*?c",
            ],
        )
        def test_supported_regex(self, regex):
            validate_regex_is_buildable(regex)
