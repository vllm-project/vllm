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

    class TestUniversalStartState:
        """A trailing anchor is only a problem when nothing before it has to
        consume a character. The check must agree with outlines_core, which
        is what compiles the regex for the request."""

        # every match of the prefix consumes a character, so `$` is fine
        BUILDABLE = [
            r'[^"]+$',
            r"[^a]$",
            r"[^,]{2}$",
            r"\d+?$",
            r".+?$",
            r"a++$",
            r"(a|b){2}?$",
            r"(?:[^a]|b)$",
            r"abc$",
            r"[^\n]+\n$",
        ]
        # the prefix can match the empty string, so the anchor comes first
        NOT_BUILDABLE = [
            r"^abc",
            r"a*$",
            r"\d*$",
            r"(a|)$",
            r"(a|b?)$",
            r"(?:a?)$",
            r"[^a]*?^b",
        ]

        @pytest.mark.parametrize("regex", BUILDABLE)
        def test_buildable(self, regex):
            validate_regex_is_buildable(regex)

        @pytest.mark.parametrize("regex", NOT_BUILDABLE)
        def test_not_buildable(self, regex):
            with pytest.raises(VLLMValidationError):
                validate_regex_is_buildable(regex)

        @pytest.mark.parametrize("regex", BUILDABLE + NOT_BUILDABLE)
        def test_matches_outlines_core(self, regex):
            from outlines_core import Index, Vocabulary

            vocab = Vocabulary(99, {c: [i] for i, c in enumerate('abc",1\n')})
            try:
                Index(regex, vocab)
                core_ok = True
            except Exception as e:
                assert "universal start state" in str(e), e
                core_ok = False
            try:
                validate_regex_is_buildable(regex)
                ours_ok = True
            except VLLMValidationError:
                ours_ok = False
            assert ours_ok == core_ok
