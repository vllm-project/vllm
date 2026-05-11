# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

from vllm.kernels.helion.case_key import CaseKey


class TestCaseKey:
    """Test suite for CaseKey class."""

    def test_construction_with_dict(self):
        key = CaseKey({"intermediate": 2048, "numtokens": 256})
        assert key["intermediate"] == 2048
        assert key["numtokens"] == 256

    def test_empty_construction_raises(self):
        with pytest.raises(TypeError, match="at least one key-value pair"):
            CaseKey()
        with pytest.raises(TypeError, match="at least one key-value pair"):
            CaseKey({})

    def test_default_construction(self):
        key = CaseKey.default()
        assert len(key) == 0
        assert key.is_default()

    def test_non_default_is_not_default(self):
        key = CaseKey({"intermediate": 2048})
        assert not key.is_default()

    def test_hashable_and_equality(self):
        a = CaseKey({"intermediate": 2048, "numtokens": 256})
        b = CaseKey({"numtokens": 256, "intermediate": 2048})
        assert a == b
        assert hash(a) == hash(b)
        assert a != CaseKey({"intermediate": 4096})
        assert CaseKey.default() == CaseKey.default()

        configs = {
            CaseKey.default(): "default_config",
            a: "a_config",
        }
        assert configs[b] == "a_config"
        assert configs[CaseKey.default()] == "default_config"

    def test_str_is_sorted_json(self):
        assert str(CaseKey({"z": 1, "a": 2})) == '{"a":2,"z":1}'
        assert str(CaseKey.default()) == "{}"

    def test_immutable(self):
        key = CaseKey({"intermediate": 2048})
        with pytest.raises(TypeError, match="immutable"):
            key["intermediate"] = 4096
        with pytest.raises(TypeError, match="immutable"):
            del key["intermediate"]
        with pytest.raises(TypeError, match="immutable"):
            key.update({"numtokens": 256})
        with pytest.raises(TypeError, match="immutable"):
            key.clear()
